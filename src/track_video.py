#!/usr/bin/env python3
"""
Multi-Object Tracking for Pet Detection

Tracks individual pets across video frames with persistent IDs, trajectory
recording, and movement analysis.

Features:
- Persistent object IDs across frames
- Trajectory visualization
- Movement analysis (speed, distance)
- Re-identification for lost tracks
- Track history recording
- Unique pet counting

Example Usage:
    # Basic tracking
    python src/track_video.py --video input.mp4 --output tracked.mp4
    
    # With trajectory visualization
    python src/track_video.py --video input.mp4 --show-trajectories \\
        --trajectory-length 30
    
    # Live tracking statistics
    python src/track_video.py --video input.mp4 --display --show-stats
    
    # Save track data
    python src/track_video.py --video input.mp4 --save-tracks tracks.json
"""

import argparse
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics.models.yolo import YOLO


@dataclass
class TrackInfo:
    """Information about a tracked object."""
    
    track_id: int
    class_id: int
    class_name: str
    positions: deque = field(default_factory=lambda: deque(maxlen=100))
    first_seen: int = 0
    last_seen: int = 0
    total_frames: int = 0
    
    def add_position(self, bbox: np.ndarray, frame_idx: int) -> None:
        """Add a new position to the trajectory."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.positions.append((center_x, center_y, frame_idx))
        self.last_seen = frame_idx
        self.total_frames += 1
    
    def get_distance_traveled(self) -> float:
        """Calculate total distance traveled."""
        if len(self.positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.positions)):
            x1, y1, _ = self.positions[i-1]
            x2, y2, _ = self.positions[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        return total_distance
    
    def get_average_speed(self) -> float:
        """Calculate average speed in pixels per frame."""
        if self.total_frames < 2:
            return 0.0
        
        distance = self.get_distance_traveled()
        return distance / self.total_frames


class PetTracker:
    """Multi-object tracker for pet detection."""
    
    def __init__(
        self,
        model_path: str = "models/yolov8n-pets.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        tracker: str = "botsort.yaml"
    ):
        """
        Initialize pet tracker.
        
        Args:
            model_path: Path to YOLOv8 model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for tracking
            tracker: Tracker configuration (botsort.yaml or bytetrack.yaml)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker = tracker
        
        self.tracks: Dict[int, TrackInfo] = {}
        self.frame_count = 0
        self.class_names = {0: 'dog', 1: 'cat'}
        
        # Update class names from model if available
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame with tracking.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (annotated_frame, detections_with_tracks)
        """
        # Run tracking
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker,
            persist=True,
            verbose=False
        )[0]
        
        detections = []
        
        # Parse results
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # type: ignore
            confidences = results.boxes.conf.cpu().numpy()  # type: ignore
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # type: ignore
            
            # Get track IDs if available
            if results.boxes.id is not None:
                track_ids = results.boxes.id.cpu().numpy().astype(int)  # type: ignore
            else:
                track_ids = [-1] * len(boxes)
            
            for bbox, conf, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
                # Update track info
                if track_id != -1:
                    if track_id not in self.tracks:
                        self.tracks[track_id] = TrackInfo(
                            track_id=track_id,
                            class_id=int(class_id),
                            class_name=self.class_names.get(int(class_id), f"class_{class_id}"),
                            first_seen=self.frame_count
                        )
                    
                    self.tracks[track_id].add_position(bbox, self.frame_count)
                
                detections.append({
                    'track_id': int(track_id),
                    'class_id': int(class_id),
                    'class_name': self.class_names.get(int(class_id), f"class_{class_id}"),
                    'confidence': float(conf),
                    'bbox': bbox.tolist()
                })
        
        self.frame_count += 1
        
        return frame, detections
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        return {
            'total_frames': self.frame_count,
            'unique_tracks': len(self.tracks),
            'active_tracks': sum(1 for t in self.tracks.values() 
                               if self.frame_count - t.last_seen < 30),
            'tracks_by_class': self._count_tracks_by_class()
        }
    
    def _count_tracks_by_class(self) -> Dict[str, int]:
        """Count tracks by class."""
        counts = defaultdict(int)
        for track in self.tracks.values():
            counts[track.class_name] += 1
        return dict(counts)
    
    def save_tracks(self, output_path: Path) -> None:
        """Save track data to JSON file."""
        track_data = {}
        
        for track_id, track in self.tracks.items():
            track_data[track_id] = {
                'track_id': track.track_id,
                'class_name': track.class_name,
                'first_seen': track.first_seen,
                'last_seen': track.last_seen,
                'total_frames': track.total_frames,
                'distance_traveled': track.get_distance_traveled(),
                'average_speed': track.get_average_speed(),
                'trajectory': [(float(x), float(y), int(f)) 
                              for x, y, f in track.positions]
            }
        
        with open(output_path, 'w') as f:
            json.dump(track_data, f, indent=2)
        
        print(f"✓ Track data saved to: {output_path}")


def draw_tracks(
    frame: np.ndarray,
    detections: List[Dict],
    tracker: PetTracker,
    show_trajectories: bool = True,
    trajectory_length: int = 30,
    show_ids: bool = True
) -> np.ndarray:
    """
    Draw tracking visualization on frame.
    
    Args:
        frame: Input frame
        detections: List of detections with track IDs
        tracker: PetTracker instance
        show_trajectories: Whether to show trajectories
        trajectory_length: Number of points in trajectory
        show_ids: Whether to show track IDs
        
    Returns:
        Annotated frame
    """
    # Colors for different classes
    colors = {
        'dog': (255, 100, 0),   # Blue
        'cat': (0, 200, 0)      # Green
    }
    
    # Draw trajectories first (behind bounding boxes)
    if show_trajectories:
        for det in detections:
            track_id = det['track_id']
            if track_id in tracker.tracks:
                track = tracker.tracks[track_id]
                color = colors.get(track.class_name, (0, 255, 255))
                
                # Draw trajectory line
                points = list(track.positions)[-trajectory_length:]
                if len(points) > 1:
                    pts = np.array([(int(x), int(y)) for x, y, _ in points], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)
                    
                    # Draw trajectory points
                    for i, (x, y, _) in enumerate(points):
                        alpha = i / len(points)  # Fade older points
                        radius = 2 + int(alpha * 3)
                        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    
    # Draw bounding boxes and labels
    for det in detections:
        bbox = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        track_id = det['track_id']
        
        x1, y1, x2, y2 = map(int, bbox)
        color = colors.get(class_name, (0, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        if show_ids and track_id != -1:
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
        else:
            label = f"{class_name} {confidence:.2f}"
        
        # Draw label background
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - 10),
            (x1 + label_width + 10, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    
    return frame


def draw_statistics(
    frame: np.ndarray,
    tracker: PetTracker,
    fps: float
) -> np.ndarray:
    """Draw tracking statistics on frame."""
    stats = tracker.get_statistics()
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw statistics
    y_offset = 40
    cv2.putText(frame, "TRACKING STATISTICS", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(frame, f"Unique Pets: {stats['unique_tracks']}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(frame, f"Active Tracks: {stats['active_tracks']}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Class counts
    for class_name, count in stats['tracks_by_class'].items():
        y_offset += 25
        cv2.putText(frame, f"  {class_name}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def track_video(
    video_path: str,
    output_path: Optional[str] = None,
    model_path: str = "models/yolov8n-pets.pt",
    conf_threshold: float = 0.25,
    tracker: str = "botsort.yaml",
    display: bool = False,
    show_trajectories: bool = True,
    trajectory_length: int = 30,
    show_stats: bool = True,
    save_tracks: Optional[str] = None
) -> Dict:
    """
    Track pets in video.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        model_path: Path to YOLOv8 model
        conf_threshold: Confidence threshold
        tracker: Tracker name (botsort.yaml or bytetrack.yaml)
        display: Show video while processing
        show_trajectories: Draw trajectories
        trajectory_length: Number of points in trajectory
        show_stats: Show statistics overlay
        save_tracks: Path to save track data JSON
        
    Returns:
        Tracking statistics
    """
    # Initialize tracker
    pet_tracker = PetTracker(
        model_path=model_path,
        conf_threshold=conf_threshold,
        tracker=tracker
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*60}")
    print(f"TRACKING VIDEO: {Path(video_path).name}")
    print(f"{'='*60}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Tracker: {tracker}")
    print(f"{'='*60}\n")
    
    # Initialize video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame, detections = pet_tracker.process_frame(frame)
            
            # Draw tracking visualization
            frame = draw_tracks(
                frame,
                detections,
                pet_tracker,
                show_trajectories=show_trajectories,
                trajectory_length=trajectory_length
            )
            
            # Draw statistics
            if show_stats:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                frame = draw_statistics(frame, pet_tracker, current_fps)
            
            # Write frame
            if writer is not None:
                writer.write(frame)
            
            # Display
            if display:
                cv2.imshow('Pet Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    
    # Calculate final statistics
    elapsed_time = time.time() - start_time
    final_stats = pet_tracker.get_statistics()
    final_stats['processing_time'] = elapsed_time
    final_stats['avg_fps'] = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRACKING SUMMARY")
    print(f"{'='*60}")
    print(f"Frames Processed: {frame_count}")
    print(f"Unique Pets Tracked: {final_stats['unique_tracks']}")
    print(f"Processing Time: {elapsed_time:.2f}s")
    print(f"Average FPS: {final_stats['avg_fps']:.2f}")
    
    for class_name, count in final_stats['tracks_by_class'].items():
        print(f"  {class_name}: {count} unique individuals")
    
    if output_path:
        print(f"\n✓ Output saved to: {output_path}")
    
    # Save track data
    if save_tracks:
        pet_tracker.save_tracks(Path(save_tracks))
    
    print(f"{'='*60}\n")
    
    return final_stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Track pets in video with persistent IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tracking
  python src/track_video.py --video input.mp4 --output tracked.mp4
  
  # With trajectory visualization
  python src/track_video.py --video input.mp4 --show-trajectories \\
      --trajectory-length 50
  
  # Live display with statistics
  python src/track_video.py --video input.mp4 --display --show-stats
  
  # Save track data for analysis
  python src/track_video.py --video input.mp4 --save-tracks tracks.json
  
  # Use ByteTrack (faster)
  python src/track_video.py --video input.mp4 --tracker bytetrack.yaml
        """
    )
    
    # Input/Output
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--save-tracks', help='Save track data to JSON file')
    
    # Model parameters
    parser.add_argument('--model', default='models/yolov8n-pets.pt',
                        help='Model path (default: models/yolov8n-pets.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--tracker', default='botsort.yaml',
                        choices=['botsort.yaml', 'bytetrack.yaml'],
                        help='Tracker to use (default: botsort.yaml)')
    
    # Visualization
    parser.add_argument('--display', action='store_true',
                        help='Display video while processing')
    parser.add_argument('--show-trajectories', action='store_true', default=True,
                        help='Draw movement trajectories')
    parser.add_argument('--trajectory-length', type=int, default=30,
                        help='Number of points in trajectory (default: 30)')
    parser.add_argument('--show-stats', action='store_true', default=True,
                        help='Show statistics overlay')
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_tracked.mp4")
    
    try:
        track_video(
            video_path=args.video,
            output_path=args.output,
            model_path=args.model,
            conf_threshold=args.conf,
            tracker=args.tracker,
            display=args.display,
            show_trajectories=args.show_trajectories,
            trajectory_length=args.trajectory_length,
            show_stats=args.show_stats,
            save_tracks=args.save_tracks
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
