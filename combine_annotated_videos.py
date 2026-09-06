import os
import glob
import subprocess
import cv2
import shutil
from pathlib import Path

def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def main():
    source_dir = "two_phase_video_output"
    output_video = "combined_annotated_videos.mp4"
    temp_dir = "temp_video_concat"
    
    # 1. Find all annotated videos
    search_pattern = os.path.join(source_dir, "**", "*_annotated.mp4")
    videos = sorted(glob.glob(search_pattern, recursive=True))
    
    if not videos:
        print(f"No annotated videos found in {source_dir}")
        return
        
    print(f"Found {len(videos)} videos to combine.")
    
    # 2. Set target dimensions for smaller output
    target_w, target_h = 1280, 720
        
    print(f"Target resolution: {target_w}x{target_h}. All videos will be scaled and padded to fit to drastically reduce file size.")
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # 3. Standardize all videos to max resolution and H.264
    temp_files = []
    for i, v in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] Standardizing {v}...")
        temp_file = os.path.join(temp_dir, f"temp_{i:03d}.mp4")
        temp_files.append(temp_file)
        
        # Extract original FPS to halve it (taking every second frame)
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        target_fps = fps / 2.0
        
        # Pad video to target_w:target_h, keep aspect ratio, center it
        # -crf 28 provides higher compression for smaller file size
        # -r halves the framerate to optimize processing and size
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-stats",
            "-i", v,
            "-r", str(target_fps),
            "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
            "-c:v", "libx264", "-crf", "28", "-preset", "fast",
            "-c:a", "aac",
            temp_file
        ]
        subprocess.run(cmd, check=True)
        
    # 4. Create concat list
    concat_list_path = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for tf in temp_files:
            # ffmpeg concat demuxer requires relative or absolute paths, properly escaped
            # using absolute paths to be safe
            abs_path = os.path.abspath(tf)
            f.write(f"file '{abs_path}'\n")
            
    print(f"\nConcatenating all {len(temp_files)} videos...")
    
    # 5. Concatenate using stream copy (lossless and fast since they are now standard)
    concat_cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-stats",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_video
    ]
    subprocess.run(concat_cmd, check=True)
    
    # 6. Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print(f"\nSuccess! Combined video saved to: {output_video}")

if __name__ == "__main__":
    main()
