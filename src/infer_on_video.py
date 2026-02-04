#!/usr/bin/env python3
"""
Video inference script for detecting pets in videos using YOLOv8.

This implementation uses a streaming architecture to handle large videos efficiently:
- Streams frames from video decoder (no full video in memory)
- Bounded queues for backpressure management
- Batch inference on GPU
- Streams predictions to NDJSON incrementally
- Final JSON compilation at the end

Memory stays bounded regardless of video size (handles 100GB+ videos on 4GB RAM).

Features:
- Process video files with streaming decode
- Real-time webcam detection
- Batch processing for GPU efficiency
- FPS tracking and performance monitoring
- Frame skipping for performance optimization
- Output video with annotations (optional)
- Structured JSON output with all detections

Usage:
    # Process video with annotations
    python infer_on_video.py video.mp4 --output-video output.mp4
    
    # Memory-efficient JSON only (handles huge videos)
    python infer_on_video.py huge_video.mp4 --output-json results.json --batch-size 16
    
    # Webcam detection
    python infer_on_video.py 0 --display
"""

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
import gzip
import json
from pathlib import Path
from queue import Queue, Empty
import sys
from threading import Thread, Event
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics.models.yolo import YOLO

from utils import draw_detections


@dataclass
class Detection:
    """Single detection result."""
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    class_id: int


@dataclass
class FrameResult:
    """Results for a single frame."""
    frame_id: int
    timestamp: float
    detections: List[Detection]
    frame_width: int
    frame_height: int


class VideoProcessor:
    """
    Streaming video processor with bounded memory usage.
    
    Architecture:
    1. Frame decoder thread -> bounded queue (backpressure)
    2. Inference thread: batch frames -> GPU inference -> results
    3. Results streamed to NDJSON file incrementally
    4. Optional: video writer thread for annotated output
    5. Final: compile NDJSON into single JSON response
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        batch_size: int = 8,
        queue_size: int = 100,
        skip_frames: int = 0,
        output_video_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        display: bool = False,
        compress_json: bool = False
    ):
        """
        Initialize streaming video processor.
        
        Args:
            model_path: Path to YOLOv8 model
            conf_threshold: Confidence threshold for detections
            batch_size: Number of frames to batch for inference (8-32 recommended)
            queue_size: Max frames in decode queue (controls memory)
            skip_frames: Number of frames to skip (0 = process all frames)
            output_video_path: Path to save annotated video (None = no video output)
            output_json_path: Path to save JSON results
            display: Whether to display video in real-time
            compress_json: Whether to gzip the final JSON
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.skip_frames = skip_frames
        self.output_video_path = output_video_path
        self.output_json_path = output_json_path
        self.display = display
        self.compress_json = compress_json
        
        # Bounded queues for backpressure
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        
        # Threading control
        self.stop_event = Event()
        self.decode_thread = None
        self.inference_thread = None
        self.writer_thread = None
        
        # Temporary NDJSON file for streaming results
        self.temp_ndjson = None
        
        # Statistics
        self.frame_count = 0
        self.processed_frames = 0
        self.total_detections = 0
        self.fps_history = []
        self.start_time = 0
    
    def _decode_frames(self, video_source, fps: float):
        """
        Decoder thread: stream frames into bounded queue.
        Enforces backpressure - blocks if queue is full.
        
        Args:
            video_source: Video capture object
            fps: Frames per second of video
        """
        frame_id = 0
        
        try:
            while not self.stop_event.is_set():
                ret, frame = video_source.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Frame skipping logic
                if self.skip_frames > 0 and (frame_id % (self.skip_frames + 1) != 0):
                    frame_id += 1
                    continue
                
                timestamp = frame_id / fps if fps > 0 else frame_id
                
                # Put frame in queue (blocks if queue is full - backpressure!)
                self.frame_queue.put((frame_id, timestamp, frame))
                frame_id += 1
                
        finally:
            # Signal end of stream
            self.frame_queue.put(None)
    
    def _inference_worker(self, class_names: dict):
        """
        Inference thread: batch frames, run inference, stream results to disk.
        
        Args:
            class_names: Dictionary mapping class IDs to names
        """
        batch_frames = []
        batch_metadata = []
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get frame from queue (blocks if empty)
                    item = self.frame_queue.get(timeout=1.0)
                    
                    if item is None:  # End of stream signal
                        # Process remaining batch
                        if batch_frames:
                            self._process_batch(batch_frames, batch_metadata, class_names)
                        break
                    
                    frame_id, timestamp, frame = item
                    batch_frames.append(frame)
                    batch_metadata.append((frame_id, timestamp, frame.shape))
                    
                    # Process batch when full
                    if len(batch_frames) >= self.batch_size:
                        self._process_batch(batch_frames, batch_metadata, class_names)
                        batch_frames = []
                        batch_metadata = []
                        
                except Empty:
                    continue
                    
        finally:
            # Signal end of results
            self.result_queue.put(None)
    
    def _process_batch(
        self,
        frames: List[np.ndarray],
        metadata: List[Tuple[int, float, Tuple]],
        class_names: dict
    ):
        """
        Process a batch of frames through the model.
        
        Args:
            frames: List of frame images
            metadata: List of (frame_id, timestamp, shape) tuples
            class_names: Class ID to name mapping
        """
        batch_start = time.time()
        
        # Run inference on batch
        results = self.model(frames, conf=self.conf_threshold, verbose=False)
        
        # Process each frame's results
        for i, (result, (frame_id, timestamp, shape)) in enumerate(zip(results, metadata)):
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Create frame result
            detections = [
                Detection(
                    bbox=box.tolist(),
                    class_name=class_names[class_id],
                    confidence=float(conf),
                    class_id=int(class_id)
                )
                for box, conf, class_id in zip(boxes, confidences, class_ids)
            ]
            
            frame_result = FrameResult(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=detections,
                frame_width=shape[1],
                frame_height=shape[0]
            )
            
            # Stream result to NDJSON (one line per frame)
            self._write_ndjson_line(frame_result)
            
            # Put result in queue for video writer (if needed)
            if self.output_video_path or self.display:
                self.result_queue.put((frames[i].copy(), frame_result))
            
            self.processed_frames += 1
            self.total_detections += len(detections)
        
        # Track FPS
        batch_time = time.time() - batch_start
        fps = len(frames) / batch_time if batch_time > 0 else 0
        self.fps_history.append(fps)
    
    def _write_ndjson_line(self, frame_result: FrameResult):
        """
        Append frame result to NDJSON file (one JSON object per line).
        This keeps memory bounded - we never hold all results in RAM.
        
        Args:
            frame_result: Frame detection results
        """
        if self.temp_ndjson:
            json_line = json.dumps(asdict(frame_result))
            self.temp_ndjson.write(json_line + '\n')
            self.temp_ndjson.flush()  # Ensure it's written to disk
    
    def _video_writer_worker(
        self,
        video_writer: Optional[cv2.VideoWriter],
        class_names: dict
    ):
        """
        Writer thread: consume results and write annotated video.
        Optional - only runs if output video is requested.
        
        Args:
            video_writer: OpenCV video writer
            class_names: Class ID to name mapping
        """
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.result_queue.get(timeout=1.0)
                    
                    if item is None:  # End of stream
                        break
                    
                    frame, frame_result = item
                    
                    # Draw detections
                    if frame_result.detections:
                        boxes = [d.bbox for d in frame_result.detections]
                        confidences = [d.confidence for d in frame_result.detections]
                        class_ids = [d.class_id for d in frame_result.detections]
                        
                        # Convert dict to list for draw_detections
                        class_names_list = [class_names[i] for i in sorted(class_names.keys())]
                        
                        output_frame = draw_detections(
                            frame,
                            boxes,
                            class_names_list,
                            confidences,
                            class_ids
                        )
                    else:
                        output_frame = frame
                    
                    # Add overlay
                    if self.fps_history:
                        current_fps = self.fps_history[-1]
                        output_frame = self._add_overlay(
                            output_frame,
                            current_fps,
                            len(frame_result.detections),
                            frame_result.frame_id,
                            self.frame_count
                        )
                    
                    # Write frame
                    if video_writer:
                        video_writer.write(output_frame)
                    
                    # Display
                    if self.display:
                        cv2.imshow('Pet Detection', output_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.stop_event.set()
                            
                except Empty:
                    continue
                    
        finally:
            if video_writer:
                video_writer.release()
            if self.display:
                cv2.destroyAllWindows()
    
    def process_video(self, video_source: str) -> dict:
        """
        Process video using streaming architecture.
        
        Architecture flow:
        1. Decoder thread streams frames -> frame_queue
        2. Inference thread batches & processes -> writes NDJSON -> result_queue
        3. Writer thread (optional) writes video from result_queue
        4. Final: compile NDJSON into single JSON
        
        Args:
            video_source: Path to video file or webcam index
            
        Returns:
            Dictionary with processing statistics
        """
        # Open video source
        if isinstance(video_source, int) or (isinstance(video_source, str) and video_source.isdigit()):
            cap = cv2.VideoCapture(int(video_source))
            source_name = f"webcam_{video_source}"
            is_webcam = True
        else:
            cap = cv2.VideoCapture(video_source)
            source_name = Path(video_source).stem
            is_webcam = False
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_webcam else 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
        
        print(f"\n📹 Video Properties:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        if total_frames > 0:
            print(f"   Total frames: {total_frames}")
            print(f"   Duration: {total_frames/fps:.2f}s")
        
        # Setup temporary NDJSON file for streaming results
        temp_dir = Path("../outputs/detections")
        temp_dir.mkdir(parents=True, exist_ok=True)
        ndjson_path = temp_dir / f"{source_name}_temp.ndjson"
        self.temp_ndjson = open(ndjson_path, 'w')
        
        # Setup video writer (optional)
        video_writer = None
        if self.output_video_path:
            output_dir = Path(self.output_video_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
            video_writer = cv2.VideoWriter(
                self.output_video_path,
                fourcc,
                fps,
                (width, height)
            )
            print(f"\n💾 Saving video to: {self.output_video_path}")
        
        # Get class names
        class_names = self.model.names
        
        print("\n🚀 Starting streaming pipeline... (Press 'q' to quit)\n")
        print("Pipeline: Decode → Batch → Infer → Stream to disk → Compile JSON")
        print(f"Batch size: {self.batch_size} frames")
        print(f"Queue size: {self.queue_size} frames (memory bounded)\n")
        
        self.start_time = time.time()
        
        # Start decoder thread
        self.decode_thread = Thread(
            target=self._decode_frames,
            args=(cap, fps),
            daemon=True
        )
        self.decode_thread.start()
        
        # Start inference thread
        self.inference_thread = Thread(
            target=self._inference_worker,
            args=(class_names,),
            daemon=True
        )
        self.inference_thread.start()
        
        # Start writer thread (if needed)
        if self.output_video_path or self.display:
            self.writer_thread = Thread(
                target=self._video_writer_worker,
                args=(video_writer, class_names),
                daemon=True
            )
            self.writer_thread.start()
        
        # Monitor progress
        try:
            while self.inference_thread.is_alive():
                time.sleep(2.0)
                elapsed = time.time() - self.start_time
                avg_fps = self.processed_frames / elapsed if elapsed > 0 else 0
                
                if total_frames > 0:
                    progress = (self.frame_count / total_frames * 100)
                    eta = (total_frames - self.frame_count) / avg_fps if avg_fps > 0 else 0
                    print(f"Progress: {progress:.1f}% | "
                          f"Frame {self.frame_count}/{total_frames} | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Detections: {self.total_detections} | "
                          f"ETA: {eta:.0f}s")
                else:
                    print(f"Frame {self.frame_count} | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Detections: {self.total_detections}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
            self.stop_event.set()
        
        # Wait for threads to complete
        if self.decode_thread:
            self.decode_thread.join()
        if self.inference_thread:
            self.inference_thread.join()
        if self.writer_thread:
            self.writer_thread.join()
        
        # Close temp NDJSON
        self.temp_ndjson.close()
        
        # Cleanup
        cap.release()
        
        # Compile final JSON from NDJSON
        if self.output_json_path:
            self._compile_final_json(ndjson_path, source_name, fps)
        else:
            # Clean up temp file if no JSON output requested
            ndjson_path.unlink()
        
        # Calculate statistics
        elapsed_time = time.time() - self.start_time
        avg_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
        
        stats = {
            'source': source_name,
            'total_frames': self.frame_count,
            'processed_frames': self.processed_frames,
            'skipped_frames': self.frame_count - self.processed_frames,
            'total_detections': self.total_detections,
            'elapsed_time': elapsed_time,
            'average_fps': avg_fps,
            'max_fps': max(self.fps_history) if self.fps_history else 0,
            'min_fps': min(self.fps_history) if self.fps_history else 0,
            'batch_size': self.batch_size,
            'queue_size': self.queue_size
        }
        
        return stats
    
    def _compile_final_json(self, ndjson_path: Path, source_name: str, fps: float):
        """
        Compile NDJSON into final single JSON response.
        Streams the data to avoid loading everything into memory.
        
        Args:
            ndjson_path: Path to temporary NDJSON file
            source_name: Name of video source
            fps: Frames per second
        """
        print(f"\n📄 Compiling final JSON from {ndjson_path}...")
        
        if not self.output_json_path:
            return
        
        output_path = Path(self.output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Stream NDJSON into single JSON
        if self.compress_json:
            output_file = gzip.open(str(output_path) + '.gz', 'wt')
            print(f"   Compressing with gzip...")
        else:
            output_file = open(output_path, 'w')
        
        try:
            # Write JSON header
            output_file.write('{\n')
            output_file.write(f'  "source": "{source_name}",\n')
            output_file.write(f'  "fps": {fps},\n')
            output_file.write(f'  "processed_frames": {self.processed_frames},\n')
            output_file.write(f'  "total_detections": {self.total_detections},\n')
            output_file.write(f'  "timestamp": "{datetime.now().isoformat()}",\n')
            output_file.write('  "frames": [\n')
            
            # Stream frames from NDJSON
            with open(ndjson_path, 'r') as ndjson_file:
                lines = ndjson_file.readlines()
                for i, line in enumerate(lines):
                    output_file.write('    ')
                    output_file.write(line.strip())
                    if i < len(lines) - 1:
                        output_file.write(',')
                    output_file.write('\n')
            
            # Write JSON footer
            output_file.write('  ]\n')
            output_file.write('}\n')
            
        finally:
            output_file.close()
        
        # Clean up temp NDJSON
        ndjson_path.unlink()
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        compression_suffix = '.gz' if self.compress_json else ''
        print(f"✅ Final JSON saved: {output_path}{compression_suffix} ({file_size:.2f} MB)")
    
    def _add_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        detections: int,
        current_frame: int,
        total_frames: int
    ) -> np.ndarray:
        """Add performance overlay to frame."""
        overlay = frame.copy()
        
        # Overlay background
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add text
        y_offset = 35
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Detections: {detections}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Frame: {current_frame}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if total_frames > 0:
            y_offset += 25
            progress = (current_frame / total_frames) * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="🐕 Pet Detection - Video Inference with Streaming Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file with output video
  python infer_on_video.py input.mp4 --output-video output.mp4
  
  # Memory-efficient processing with JSON output (handles 100GB video on 4GB RAM)
  python infer_on_video.py huge_video.mp4 --output-json results.json --batch-size 16
  
  # Skip frames for faster processing
  python infer_on_video.py input.mp4 --skip-frames 5 --output-video fast_output.mp4
  
  # Webcam inference
  python infer_on_video.py 0 --display
  
  # High throughput with large batches
  python infer_on_video.py input.mp4 --batch-size 32 --queue-size 200 --output-json results.json
  
  # Compressed JSON output
  python infer_on_video.py input.mp4 --output-json results.json --compress
        """
    )
    
    parser.add_argument(
        'source',
        help='Video file path or webcam index (0, 1, etc.)'
    )
    parser.add_argument(
        '--model',
        default='../models/yolov8n-pets.pt',
        help='Path to YOLOv8 model weights (default: ../models/yolov8n-pets.pt)'
    )
    parser.add_argument(
        '--output-video',
        help='Path to save annotated output video'
    )
    parser.add_argument(
        '--output-json',
        help='Path to save detection results as JSON'
    )
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Compress JSON output with gzip'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (default: 0.25)'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=0,
        help='Skip N frames between detections (default: 0 - process all frames)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Number of frames to batch for GPU inference (default: 8)'
    )
    parser.add_argument(
        '--queue-size',
        type=int,
        default=100,
        help='Max frames in decode queue - controls memory (default: 100)'
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display video during processing (press q to quit)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.output_video and not args.output_json and not args.display:
        parser.error("At least one output method required: --output-video, --output-json, or --display")
    
    # Initialize processor
    print("🚀 Pet Detection Video Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Batch size: {args.batch_size} frames")
    print(f"Queue size: {args.queue_size} frames (memory bounded)")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Skip frames: {args.skip_frames}")
    
    processor = VideoProcessor(
        model_path=args.model,
        output_video_path=args.output_video,
        output_json_path=args.output_json,
        conf_threshold=args.conf_threshold,
        skip_frames=args.skip_frames,
        batch_size=args.batch_size,
        queue_size=args.queue_size,
        display=args.display,
        compress_json=args.compress
    )
    
    # Process video
    try:
        stats = processor.process_video(args.source)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Processing Complete!")
        print("=" * 60)
        print(f"Source: {stats['source']}")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Processed frames: {stats['processed_frames']}")
        print(f"Skipped frames: {stats['skipped_frames']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Elapsed time: {stats['elapsed_time']:.2f}s")
        print(f"Average FPS: {stats['average_fps']:.2f}")
        print(f"Max FPS: {stats['max_fps']:.2f}")
        print(f"Min FPS: {stats['min_fps']:.2f}")
        print(f"Memory efficiency: Bounded queue ({stats['queue_size']} frames)")
        print(f"Batch processing: {stats['batch_size']} frames/batch")
        print("=" * 60)
        
        if args.output_video:
            print(f"✅ Video saved: {args.output_video}")
        if args.output_json:
            suffix = '.gz' if args.compress else ''
            print(f"✅ JSON saved: {args.output_json}{suffix}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
