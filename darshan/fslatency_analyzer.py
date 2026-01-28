#!/usr/bin/env python3
"""
Analyze multiple Darshan .darshan files to measure I/O latency variation over time.
Optimized for identifying optimal candidates for Client-Side I/O Latency.
Includes statistical analysis for Jitter and Tail Latency.
"""

import sys
import subprocess
import re
import os
import fnmatch
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Try to import scientific stack
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT_DEPS = True
except ImportError:
    HAS_PLOT_DEPS = False

# --- Constants ---
POSIX_COUNTERS = {
    'POSIX_READS': ('reads', int),
    'POSIX_WRITES': ('writes', int),
    'POSIX_BYTES_READ': ('read_bytes', int),
    'POSIX_BYTES_WRITTEN': ('write_bytes', int),
    'POSIX_F_READ_TIME': ('read_time', float),
    'POSIX_F_WRITE_TIME': ('write_time', float),
    'POSIX_F_META_TIME': ('meta_time', float),
}

def run_darshan_parser(darshan_file):
    """Run darshan-parser and return the output."""
    try:
        return subprocess.check_output(['darshan-parser', str(darshan_file)], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Error processing {darshan_file}: {e}")
        return None

def extract_metadata(darshan_file, parser_output):
    """Extract timestamp and PID using a priority-based approach."""
    ts_match = re.search(r'(?:start_time|end_time):\s+(\d+)', parser_output)
    timestamp = int(ts_match.group(1)) if ts_match else int(os.path.getmtime(darshan_file))
    pid_match = re.search(r'id\d+-(\d+)_', str(darshan_file))
    pid = pid_match.group(1) if pid_match else "unknown"
    return timestamp, pid

def parse_posix_data(output):
    """Efficiently parse POSIX module data."""
    io_data = defaultdict(lambda: {
        'reads': 0, 'writes': 0, 'read_bytes': 0, 'write_bytes': 0,
        'read_time': 0.0, 'write_time': 0.0, 'meta_time': 0.0,
        'filename': '', 'mount_pt': '', 'record_id': ''
    })
    
    for line in output.splitlines():
        if not line or line.startswith('#'): continue
        parts = line.split()
        if len(parts) < 6 or parts[0] != 'POSIX': continue
        
        _, _, record_id, counter, value = parts[:5]
        path_info = parts[5:]
        
        key = (record_id, path_info[0])
        entry = io_data[key]
        if not entry['filename']:
            entry.update({
                'filename': path_info[0],
                'mount_pt': path_info[1] if len(path_info) > 1 else "",
                'record_id': record_id
            })
        if counter in POSIX_COUNTERS:
            attr, func = POSIX_COUNTERS[counter]
            entry[attr] += func(value)
    return io_data

def get_fs_root(path, mount_pt=None):
    if mount_pt and mount_pt.strip(): return mount_pt.strip()
    p = Path(path)
    # Fallback to root or top level directory if no mount point is found
    return str(p.parents[len(p.parents)-2]) if len(p.parents) > 1 else '/'

def extract_candidates(io_data, max_size, min_ops, fs_pattern=None):
    candidates = []
    for (record_id, filename), data in io_data.items():
        fs_root = get_fs_root(filename, data['mount_pt'])
        if fs_pattern and not (fnmatch.fnmatch(filename, fs_pattern) or fnmatch.fnmatch(fs_root, fs_pattern)):
            continue

        for mode in ['READ', 'WRITE']:
            prefix = mode.lower()
            ops, bytes_val, time_val = data[f'{prefix}s'], data[f'{prefix}_bytes'], data[f'{prefix}_time']
            if ops >= min_ops and bytes_val > 0:
                avg_size = bytes_val / ops
                if avg_size <= max_size:
                    candidates.append({
                        'filename': filename, 'mount_pt': data['mount_pt'],
                        'norm_filename': os.path.basename(filename),
                        'record_id': record_id, 'type': mode, 'count': ops,
                        'avg_size': avg_size, 'latency_ms': (time_val / ops) * 1000.0,
                        'meta_time_ms': data['meta_time'] * 1000.0
                    })
    return candidates

def select_best_per_fs(results):
    global_stats = defaultdict(list)
    for entry in results:
        for c in entry['candidates']:
            # Use string key to match what plotting functions expect
            key = f"{c['norm_filename']}|{c['type']}"
            global_stats[key].append(c)

    best_per_fs = {}
    fs_groups = defaultdict(list)
    for key, occurrences in global_stats.items():
        sample = occurrences[0]
        root = get_fs_root(sample['filename'], sample['mount_pt'])
        fs_groups[root].append({
            'key': key, 'freq': len(occurrences),
            'avg_size': sum(o['avg_size'] for o in occurrences) / len(occurrences),
            'count': sum(o['count'] for o in occurrences),
            'last_real_filename': occurrences[-1]['filename'],
            'last_mount_pt': occurrences[-1]['mount_pt'],
            'last_record_id': occurrences[-1]['record_id']
        })

    for root, probes in fs_groups.items():
        # Prioritize frequency, then smaller average size (closer to metadata-like latency)
        probes.sort(key=lambda x: (-x['freq'], x['avg_size']))
        best_per_fs[root] = probes[0]
    return best_per_fs

def calculate_advanced_stats(latencies):
    """Calculates Jitter and Tail Latency metrics."""
    if not latencies: return {}
    lats = np.array(latencies)
    # Jitter = Mean Absolute Deviation of successive differences
    jitter = np.mean(np.abs(np.diff(lats))) if len(lats) > 1 else 0
    return {
        'avg': np.mean(lats),
        'p95': np.percentile(lats, 95),
        'p99': np.percentile(lats, 99),
        'std': np.std(lats),
        'jitter': jitter,
        'cov': (np.std(lats) / np.mean(lats)) * 100 if np.mean(lats) > 0 else 0
    }

def plot_validation(results, selected_stat, output_dir):
    """Generates a 3-panel validation plot for a specific candidate key with stats."""
    if not HAS_PLOT_DEPS:
        return

    os.makedirs(output_dir, exist_ok=True)
    selected_key = selected_stat['key']
    base_name, io_type = selected_key.rsplit('|', 1)
    mount_pt = selected_stat['last_mount_pt']
    rec_id = selected_stat['last_record_id']

    data_points = []
    for entry in results:
        for c in entry['candidates']:
            if f"{c['norm_filename']}|{c['type']}" == selected_key:
                data_points.append({
                    'time': entry['timestamp'],
                    'dt_obj': datetime.fromtimestamp(entry['timestamp']),
                    'latency': c['latency_ms'],
                    'size': c['avg_size'],
                    'pid': entry['pid']
                })

    if not data_points: return

    times = [d['dt_obj'] for d in data_points]
    latencies = [d['latency'] for d in data_points]
    sizes = [d['size'] for d in data_points]
    pids = [d['pid'] for d in data_points]
    raw_times = [d['time'] for d in data_points]
    intervals = [0] + [raw_times[i] - raw_times[i-1] for i in range(1, len(raw_times))]
    
    # Calculate stats
    stats = calculate_advanced_stats(latencies)
    std_latency = stats['std']
    std_size = np.std(sizes) if len(sizes) > 1 else 0
    std_int = np.std(intervals[1:]) if len(intervals) > 1 else 0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Latency Plot
    ax1.plot(times, latencies, color='gray', alpha=0.3, zorder=1)
    for i, (t, lat, pid) in enumerate(zip(times, latencies, pids)):
        line = ax1.errorbar(t, lat, yerr=std_latency, fmt='o', ms=8, alpha=0.8, zorder=2, capsize=3)
        color = line[0].get_color()
        ax1.text(t, lat, f" {pid}", fontsize=7, color=color)
        
        ax2.errorbar(t, sizes[i], yerr=std_size, fmt='o', ms=8, color=color, alpha=0.8, capsize=3)
        ax2.text(t, sizes[i], f" {pid}", fontsize=7, color=color)
        
        ax3.errorbar(t, intervals[i], yerr=std_int if i > 0 else 0, fmt='s', ms=8, color=color, alpha=0.8, capsize=3)
        ax3.text(t, intervals[i], f" {pid}", fontsize=7, color=color)

    # Add Stats Box to the Latency axis
    stats_text = (
        f"Stats (ms):\n"
        f"Mean: {stats['avg']:.2f}\n"
        f"StdDev: {stats['std']:.2f}\n"
        f"Jitter: {stats['jitter']:.2f}\n"
        f"P95: {stats['p95']:.2f}\n"
        f"P99: {stats['p99']:.2f}\n"
        f"CoV: {stats['cov']:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(1.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    ax1.set_ylabel('Latency (ms)')
    ax1.set_title(f'FS: {mount_pt} | Record: {rec_id} | File: {base_name} ({io_type})\nLatency Error Bars = Global StdDev')
    ax1.grid(True, alpha=0.2)
    
    ax2.set_ylabel('Avg Size (Bytes)')
    ax2.set_title('Avg Size Consistency')
    ax2.grid(True, alpha=0.2)
    
    ax3.set_ylabel('Interval Δt (sec)')
    ax3.set_title('Measurement Frequency')
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    # Adjust layout to make room for stats box
    fig.subplots_adjust(right=0.88)
    
    fs_tag = mount_pt.replace('/', '_').strip('_') or 'root'
    safe_name = base_name.replace('.', '_').replace('_', '-')
    out_file = os.path.join(output_dir, f'val_{fs_tag}_{io_type}_{safe_name}.png')
    plt.savefig(out_file, dpi=120)
    plt.close()
    print(f"  Individual plot generated: {out_file}")

def plot_fs_summary(results, best_candidates_stats, output_dir):
    """Generates a final summary plot overlaying latencies of all shared filesystems."""
    if not HAS_PLOT_DEPS:
        return

    plt.figure(figsize=(14, 7))
    found_data = False
    
    for fs_root, stat in best_candidates_stats.items():
        selected_key = stat['key']
        base_name, io_type = selected_key.rsplit('|', 1)
        
        times = []
        latencies = []
        for entry in results:
            for c in entry['candidates']:
                if f"{c['norm_filename']}|{c['type']}" == selected_key:
                    times.append(datetime.fromtimestamp(entry['timestamp']))
                    latencies.append(c['latency_ms'])
        
        if times:
            found_data = True
            std_latency = np.std(latencies) if len(latencies) > 1 else 0
            plt.errorbar(times, latencies, yerr=std_latency, marker='o', capsize=3, alpha=0.7,
                         label=f"{fs_root} ({base_name}) {io_type}")

    if not found_data:
        plt.close()
        return

    plt.xlabel('Time')
    plt.ylabel('Latency (ms)')
    plt.title('Shared Filesystem Latency Variation (Error Bars = StdDev)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'fs_latency_summary.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"\nFinal Summary Plot generated: {out_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Darshan logs per filesystem')
    parser.add_argument('darshan_files', nargs='*', help='Darshan log files')
    parser.add_argument('--list', '-l', help='File with list of Darshan logs')
    parser.add_argument('--max-size', '-s', type=int, default=4096, help='Max I/O size (bytes)')
    parser.add_argument('--min-ops', '-m', type=int, default=10, help='Min operation count')
    parser.add_argument('--plot', '-p', action='store_true', help='Generate plots')
    parser.add_argument('--output-dir', '-o', default='plots', help='Output directory')
    parser.add_argument('--read-only', action='store_true')
    parser.add_argument('--write-only', action='store_true')
    parser.add_argument('--fs-pattern', help='Filter by filename or FS root pattern')
    
    args = parser.parse_args()
    
    if args.plot and not HAS_PLOT_DEPS:
        print("Error: Matplotlib or Numpy not found. Plotting disabled.")
        args.plot = False

    io_filter = 'READ' if args.read_only else 'WRITE' if args.write_only else None
    
    log_files = args.darshan_files or []
    if args.list:
        with open(args.list) as f:
            log_files.extend([line.strip() for line in f if line.strip()])
    
    if not log_files:
        parser.print_help()
        sys.exit(1)

    print(f"Analyzing {len(log_files)} logs...")
    all_results = []
    for log in log_files:
        if not os.path.exists(log):
            print(f"Warning: File {log} not found.")
            continue
        out = run_darshan_parser(log)
        if not out: continue
        ts, pid = extract_metadata(log, out)
        io_data = parse_posix_data(out)
        candidates = extract_candidates(io_data, args.max_size, args.min_ops, args.fs_pattern)
        if io_filter: 
            candidates = [c for c in candidates if c['type'] == io_filter]
        if candidates: 
            all_results.append({'timestamp': ts, 'pid': pid, 'log_file': log, 'candidates': candidates})
    
    all_results.sort(key=lambda x: x['timestamp'])
    if not all_results:
        print("No suitable I/O patterns found.")
        return

    best_candidates = select_best_per_fs(all_results)
    
    print(f"\nFound candidates for {len(best_candidates)} filesystems:")
    for fs, stat in best_candidates.items():
        # Split correctly from the string key
        fname, mode = stat['key'].split('|')
        print(f"  [{fs}] Best probe: {fname} ({mode}) - Frequency: {stat['freq']} logs")
        if args.plot:
            plot_validation(all_results, stat, args.output_dir)

    if args.plot and best_candidates:
        plot_fs_summary(all_results, best_candidates, args.output_dir)

if __name__ == '__main__':
    main()
