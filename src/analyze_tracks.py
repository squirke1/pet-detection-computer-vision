#!/usr/bin/env python3
"""
Track Analysis and Heatmap Generation

Analyzes tracking data from videos to generate insights and visualizations:
- Density heatmaps showing where pets spend time
- Zone analytics with dwell time and visit counts
- Temporal patterns and activity timelines
- Interactive HTML reports
- Path frequency analysis

Example Usage:
    # Generate heatmap
    python src/analyze_tracks.py --tracks tracks.json --heatmap heatmap.png
    
    # Zone analytics
    python src/analyze_tracks.py --tracks tracks.json --zones zones.yaml \\
        --report zone_report.html
    
    # Temporal analysis
    python src/analyze_tracks.py --tracks tracks.json --temporal \\
        --output activity_timeline.png
    
    # Complete analysis
    python src/analyze_tracks.py --tracks tracks.json --all \\
        --output-dir analytics/
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle


@dataclass
class Zone:
    """Defines a region of interest."""
    
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int] = (255, 255, 0)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the zone."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def get_center(self) -> Tuple[float, float]:
        """Get zone center point."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def get_area(self) -> int:
        """Get zone area in pixels."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class ZoneStatistics:
    """Statistics for a zone."""
    
    zone_name: str
    total_visits: int = 0
    unique_tracks: int = 0
    total_frames: int = 0
    average_dwell_time: float = 0.0
    track_visits: Dict[int, int] = field(default_factory=dict)


class TrackAnalyzer:
    """Analyzes tracking data and generates visualizations."""
    
    def __init__(self, tracks_file: Path, video_size: Optional[Tuple[int, int]] = None):
        """
        Initialize track analyzer.
        
        Args:
            tracks_file: Path to tracking JSON file
            video_size: Video dimensions (width, height). Auto-detected if None.
        """
        self.tracks_file = tracks_file
        self.tracks = self._load_tracks()
        self.video_size = video_size or self._detect_video_size()
        self.zones: List[Zone] = []
        
    def _load_tracks(self) -> Dict:
        """Load tracking data from JSON file."""
        with open(self.tracks_file, 'r') as f:
            return json.load(f)
    
    def _detect_video_size(self) -> Tuple[int, int]:
        """Auto-detect video size from trajectory data."""
        max_x, max_y = 0, 0
        
        for track_data in self.tracks.values():
            for x, y, _ in track_data.get('trajectory', []):
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        
        # Add 10% padding
        width = int(max_x * 1.1)
        height = int(max_y * 1.1)
        
        return (width, height)
    
    def generate_heatmap(
        self,
        output_path: Path,
        blur_radius: int = 15,
        colormap: str = 'jet',
        class_filter: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate density heatmap from trajectories.
        
        Args:
            output_path: Path to save heatmap image
            blur_radius: Gaussian blur radius for smoothing
            colormap: Matplotlib colormap name
            class_filter: Filter by class name (e.g., 'dog', 'cat')
            
        Returns:
            Heatmap array
        """
        width, height = self.video_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate positions
        for track_id, track_data in self.tracks.items():
            # Filter by class if specified
            if class_filter and track_data.get('class_name') != class_filter:
                continue
            
            trajectory = track_data.get('trajectory', [])
            for x, y, _ in trajectory:
                x_int, y_int = int(x), int(y)
                if 0 <= x_int < width and 0 <= y_int < height:
                    heatmap[y_int, x_int] += 1
        
        # Apply Gaussian blur for smoothing
        if blur_radius > 0:
            heatmap = cv2.GaussianBlur(heatmap, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap=colormap, interpolation='bilinear')
        plt.colorbar(label='Normalized Density')
        plt.title('Pet Activity Heatmap')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Heatmap saved to: {output_path}")
        
        return heatmap
    
    def load_zones(self, zones_file: Path) -> None:
        """Load zones from YAML file."""
        import yaml
        
        with open(zones_file, 'r') as f:
            zones_config = yaml.safe_load(f)
        
        self.zones = []
        for zone_data in zones_config.get('zones', []):
            self.zones.append(Zone(
                name=zone_data['name'],
                x1=zone_data['x1'],
                y1=zone_data['y1'],
                x2=zone_data['x2'],
                y2=zone_data['y2'],
                color=tuple(zone_data.get('color', [255, 255, 0]))
            ))
        
        print(f"✓ Loaded {len(self.zones)} zones")
    
    def analyze_zones(self) -> Dict[str, ZoneStatistics]:
        """Analyze track activity within defined zones."""
        if not self.zones:
            raise ValueError("No zones loaded. Call load_zones() first.")
        
        zone_stats = {zone.name: ZoneStatistics(zone_name=zone.name) 
                     for zone in self.zones}
        
        # Analyze each track
        for track_id, track_data in self.tracks.items():
            trajectory = track_data.get('trajectory', [])
            track_id_int = int(track_id)
            
            # Track which zones this track has visited
            visited_zones = set()
            current_zone = None
            zone_entry_frame = None
            
            for x, y, frame in trajectory:
                # Find which zone(s) contain this point
                in_zone = None
                for zone in self.zones:
                    if zone.contains_point(x, y):
                        in_zone = zone.name
                        break
                
                # Track zone transitions
                if in_zone != current_zone:
                    # Exiting previous zone
                    if current_zone is not None and zone_entry_frame is not None:
                        stats = zone_stats[current_zone]
                        dwell_frames = frame - zone_entry_frame
                        stats.total_frames += dwell_frames
                    
                    # Entering new zone
                    if in_zone is not None:
                        stats = zone_stats[in_zone]
                        if in_zone not in visited_zones:
                            stats.unique_tracks += 1
                            visited_zones.add(in_zone)
                        
                        if track_id_int not in stats.track_visits:
                            stats.track_visits[track_id_int] = 0
                            stats.total_visits += 1
                        
                        stats.track_visits[track_id_int] += 1
                        zone_entry_frame = frame
                    
                    current_zone = in_zone
        
        # Calculate average dwell times
        for stats in zone_stats.values():
            if stats.total_visits > 0:
                stats.average_dwell_time = stats.total_frames / stats.total_visits
        
        return zone_stats
    
    def visualize_zones(
        self,
        output_path: Path,
        heatmap: Optional[np.ndarray] = None
    ) -> None:
        """Visualize zones on top of heatmap or blank canvas."""
        width, height = self.video_size
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Show heatmap as background if provided
        if heatmap is not None:
            ax.imshow(heatmap, cmap='jet', alpha=0.6, interpolation='bilinear')
        else:
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_aspect('equal')
        
        # Draw zones
        for zone in self.zones:
            rect = Rectangle(
                (zone.x1, zone.y1),
                zone.x2 - zone.x1,
                zone.y2 - zone.y1,
                linewidth=2,
                edgecolor=np.array(zone.color) / 255.0,
                facecolor='none',
                label=zone.name
            )
            ax.add_patch(rect)
            
            # Add zone label
            center_x, center_y = zone.get_center()
            ax.text(
                center_x, center_y, zone.name,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        ax.set_title('Zone Definitions')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.legend(loc='upper right')
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Zone visualization saved to: {output_path}")
    
    def plot_zone_statistics(
        self,
        zone_stats: Dict[str, ZoneStatistics],
        output_path: Path
    ) -> None:
        """Create bar charts for zone statistics."""
        zone_names = list(zone_stats.keys())
        visits = [stats.total_visits for stats in zone_stats.values()]
        unique_tracks = [stats.unique_tracks for stats in zone_stats.values()]
        dwell_times = [stats.average_dwell_time for stats in zone_stats.values()]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Total visits
        axes[0].bar(zone_names, visits, color='steelblue', alpha=0.8)
        axes[0].set_title('Total Visits per Zone')
        axes[0].set_ylabel('Number of Visits')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Unique tracks
        axes[1].bar(zone_names, unique_tracks, color='coral', alpha=0.8)
        axes[1].set_title('Unique Individuals per Zone')
        axes[1].set_ylabel('Number of Unique Tracks')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Average dwell time
        axes[2].bar(zone_names, dwell_times, color='mediumseagreen', alpha=0.8)
        axes[2].set_title('Average Dwell Time per Zone')
        axes[2].set_ylabel('Frames')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Zone statistics plot saved to: {output_path}")
    
    def analyze_temporal_patterns(
        self,
        output_path: Path,
        bin_size: int = 30
    ) -> None:
        """Analyze and visualize activity patterns over time."""
        # Find frame range
        max_frame = 0
        for track_data in self.tracks.values():
            if track_data.get('last_seen', 0) > max_frame:
                max_frame = track_data['last_seen']
        
        # Create bins
        num_bins = (max_frame // bin_size) + 1
        activity = np.zeros(num_bins)
        
        # Count active tracks per bin
        for track_data in self.tracks.values():
            first_frame = track_data.get('first_seen', 0)
            last_frame = track_data.get('last_seen', 0)
            
            first_bin = first_frame // bin_size
            last_bin = last_frame // bin_size
            
            for bin_idx in range(first_bin, last_bin + 1):
                if bin_idx < num_bins:
                    activity[bin_idx] += 1
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bins = np.arange(num_bins) * bin_size
        ax.plot(bins, activity, linewidth=2, color='steelblue')
        ax.fill_between(bins, activity, alpha=0.3, color='steelblue')
        
        ax.set_title('Activity Timeline')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Active Tracks')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Temporal analysis saved to: {output_path}")
    
    def generate_report(
        self,
        output_path: Path,
        zone_stats: Optional[Dict[str, ZoneStatistics]] = None
    ) -> None:
        """Generate interactive HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pet Tracking Analytics Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        
        .section {{
            margin: 40px 0;
        }}
        
        .section-title {{
            font-size: 1.8rem;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .track-details {{
            background: #f8f9ff;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        
        .track-id {{
            font-weight: bold;
            color: #667eea;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🐾 Pet Tracking Analytics Report</h1>
        <p class="subtitle">Generated from: {self.tracks_file.name}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Tracks</div>
                <div class="stat-value">{len(self.tracks)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Distance</div>
                <div class="stat-value">{sum(t.get('distance_traveled', 0) for t in self.tracks.values()):.0f}px</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Speed</div>
                <div class="stat-value">{np.mean([t.get('average_speed', 0) for t in self.tracks.values()]):.2f}px/f</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Video Size</div>
                <div class="stat-value">{self.video_size[0]}x{self.video_size[1]}</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">Track Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Track ID</th>
                        <th>Class</th>
                        <th>First Frame</th>
                        <th>Last Frame</th>
                        <th>Duration</th>
                        <th>Distance</th>
                        <th>Avg Speed</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add track rows
        for track_id, track_data in sorted(self.tracks.items(), key=lambda x: int(x[0])):
            html_content += f"""
                    <tr>
                        <td class="track-id">#{track_id}</td>
                        <td>{track_data.get('class_name', 'unknown')}</td>
                        <td>{track_data.get('first_seen', 0)}</td>
                        <td>{track_data.get('last_seen', 0)}</td>
                        <td>{track_data.get('total_frames', 0)} frames</td>
                        <td>{track_data.get('distance_traveled', 0):.1f}px</td>
                        <td>{track_data.get('average_speed', 0):.2f}px/f</td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
"""
        
        # Add zone statistics if available
        if zone_stats:
            html_content += """
        <div class="section">
            <h2 class="section-title">Zone Analytics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Zone Name</th>
                        <th>Total Visits</th>
                        <th>Unique Tracks</th>
                        <th>Avg Dwell Time</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for zone_name, stats in zone_stats.items():
                html_content += f"""
                    <tr>
                        <td><strong>{zone_name}</strong></td>
                        <td>{stats.total_visits}</td>
                        <td>{stats.unique_tracks}</td>
                        <td>{stats.average_dwell_time:.1f} frames</td>
                    </tr>
"""
            
            html_content += """
                </tbody>
            </table>
        </div>
"""
        
        html_content += f"""
        <div class="footer">
            <p>Report generated on {Path.cwd().name}</p>
            <p>Pet Detection Computer Vision v3.1</p>
        </div>
    </div>
</body>
</html>
"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved to: {output_path}")
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics from all tracks."""
        total_distance = sum(t.get('distance_traveled', 0) for t in self.tracks.values())
        speeds = [t.get('average_speed', 0) for t in self.tracks.values()]
        durations = [t.get('total_frames', 0) for t in self.tracks.values()]
        
        classes = defaultdict(int)
        for track_data in self.tracks.values():
            classes[track_data.get('class_name', 'unknown')] += 1
        
        return {
            'total_tracks': len(self.tracks),
            'total_distance': total_distance,
            'average_speed': np.mean(speeds) if speeds else 0,
            'average_duration': np.mean(durations) if durations else 0,
            'classes': dict(classes),
            'video_size': self.video_size
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze tracking data and generate visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate heatmap
  python src/analyze_tracks.py --tracks tracks.json --heatmap heatmap.png
  
  # Zone analytics
  python src/analyze_tracks.py --tracks tracks.json --zones zones.yaml \\
      --zone-stats zone_stats.png --zone-viz zones.png
  
  # Temporal analysis
  python src/analyze_tracks.py --tracks tracks.json --temporal timeline.png
  
  # Complete analysis
  python src/analyze_tracks.py --tracks tracks.json --all \\
      --output-dir analytics/
  
  # Generate HTML report
  python src/analyze_tracks.py --tracks tracks.json --report report.html
        """
    )
    
    # Input
    parser.add_argument('--tracks', required=True, help='Path to tracking JSON file')
    parser.add_argument('--video-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help='Video dimensions (auto-detected if not specified)')
    
    # Output options
    parser.add_argument('--heatmap', help='Generate heatmap (output path)')
    parser.add_argument('--zones', help='Path to zones YAML file')
    parser.add_argument('--zone-stats', help='Plot zone statistics (output path)')
    parser.add_argument('--zone-viz', help='Visualize zones (output path)')
    parser.add_argument('--temporal', help='Temporal analysis (output path)')
    parser.add_argument('--report', help='Generate HTML report (output path)')
    parser.add_argument('--all', action='store_true', help='Generate all analyses')
    parser.add_argument('--output-dir', default='outputs/analytics', help='Output directory for --all')
    
    # Heatmap options
    parser.add_argument('--blur', type=int, default=15, help='Blur radius for heatmap')
    parser.add_argument('--colormap', default='jet', help='Matplotlib colormap')
    parser.add_argument('--class-filter', help='Filter by class name')
    
    # Temporal analysis options
    parser.add_argument('--bin-size', type=int, default=30, help='Frame bin size for temporal analysis')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    video_size = tuple(args.video_size) if args.video_size else None
    analyzer = TrackAnalyzer(Path(args.tracks), video_size=video_size)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRACK ANALYSIS")
    print(f"{'='*60}")
    summary = analyzer.get_summary_statistics()
    print(f"Total Tracks: {summary['total_tracks']}")
    print(f"Total Distance: {summary['total_distance']:.1f} pixels")
    print(f"Average Speed: {summary['average_speed']:.2f} px/frame")
    print(f"Average Duration: {summary['average_duration']:.1f} frames")
    print(f"Classes: {summary['classes']}")
    print(f"Video Size: {summary['video_size'][0]}x{summary['video_size'][1]}")
    print(f"{'='*60}\n")
    
    # Handle --all flag
    if args.all:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args.heatmap = str(output_dir / 'heatmap.png')
        args.temporal = str(output_dir / 'temporal.png')
        args.report = str(output_dir / 'report.html')
        
        if args.zones:
            args.zone_stats = str(output_dir / 'zone_stats.png')
            args.zone_viz = str(output_dir / 'zones.png')
    
    # Generate heatmap
    heatmap = None
    if args.heatmap:
        heatmap = analyzer.generate_heatmap(
            Path(args.heatmap),
            blur_radius=args.blur,
            colormap=args.colormap,
            class_filter=args.class_filter
        )
    
    # Zone analysis
    zone_stats = None
    if args.zones:
        analyzer.load_zones(Path(args.zones))
        zone_stats = analyzer.analyze_zones()
        
        # Print zone statistics
        print(f"\n{'='*60}")
        print("ZONE STATISTICS")
        print(f"{'='*60}")
        for zone_name, stats in zone_stats.items():
            print(f"\n{zone_name}:")
            print(f"  Total Visits: {stats.total_visits}")
            print(f"  Unique Tracks: {stats.unique_tracks}")
            print(f"  Avg Dwell Time: {stats.average_dwell_time:.1f} frames")
        print(f"{'='*60}\n")
        
        if args.zone_stats:
            analyzer.plot_zone_statistics(zone_stats, Path(args.zone_stats))
        
        if args.zone_viz:
            analyzer.visualize_zones(Path(args.zone_viz), heatmap=heatmap)
    
    # Temporal analysis
    if args.temporal:
        analyzer.analyze_temporal_patterns(
            Path(args.temporal),
            bin_size=args.bin_size
        )
    
    # Generate report
    if args.report:
        analyzer.generate_report(Path(args.report), zone_stats=zone_stats)
    
    print("\n✅ Analysis complete!")
    
    return 0


if __name__ == '__main__':
    exit(main())
