#!/usr/bin/env python3
"""
Recursively extract frames from videos while preserving the input folder structure.

Example:
    python process.py ./data/videos --output_root ./data/frames

Input:
    /data/videos/classA/clip1.mp4
Output:
    /data/frames/classA/clip1/00000.jpg
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2


VIDEO_EXTS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively find videos and extract frames while keeping directory structure."
    )
    parser.add_argument("input_root", type=str, help="Root directory to recursively search videos.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Output root for extracted frames. Default: <input_root>_frames",
    )
    parser.add_argument("--image_ext", type=str, default="jpg", choices=["jpg", "png"], help="Frame image format.")
    parser.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality [1,100].")
    parser.add_argument("--zero_pad", type=int, default=5, help="Frame index zero padding width.")
    parser.add_argument("--every_n", type=int, default=1, help="Keep one frame every N frames.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frame folders.")
    return parser.parse_args()


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def collect_videos(input_root: Path, output_root: Path) -> List[Path]:
    videos: List[Path] = []
    for p in input_root.rglob("*"):
        if not is_video_file(p):
            continue
        # Prevent recursive processing if output_root is placed under input_root.
        if output_root in p.parents:
            continue
        videos.append(p)
    videos.sort()
    return videos


def make_output_dir(video_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = video_path.relative_to(input_root)
    rel_no_suffix = rel.with_suffix("")  # classA/clip1
    return output_root / rel_no_suffix


def extract_frames(
    video_path: Path,
    out_dir: Path,
    image_ext: str = "jpg",
    jpg_quality: int = 95,
    zero_pad: int = 5,
    every_n: int = 1,
) -> Tuple[int, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, "open_failed"

    write_params: List[int] = []
    if image_ext == "jpg":
        quality = max(1, min(100, int(jpg_quality)))
        write_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    src_idx = 0
    dst_idx = 0
    written = 0
    frame_step = max(1, int(every_n))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if src_idx % frame_step == 0:
            frame_name = f"{dst_idx:0{zero_pad}d}.{image_ext}"
            frame_path = out_dir / frame_name
            saved = cv2.imwrite(str(frame_path), frame, write_params)
            if not saved:
                cap.release()
                return written, "write_failed"
            dst_idx += 1
            written += 1

        src_idx += 1

    cap.release()
    return written, "ok"


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise RuntimeError(f"input_root does not exist or is not a directory: {input_root}")

    output_root = Path(args.output_root).resolve() if args.output_root else input_root.parent / f"{input_root.name}_frames"
    output_root.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(input_root, output_root)
    if not videos:
        print(f"[Done] No video files found under: {input_root}")
        return

    print(f"[Info] input_root : {input_root}")
    print(f"[Info] output_root: {output_root}")
    print(f"[Info] found videos: {len(videos)}")

    total_written = 0
    done = 0
    skipped = 0
    failed = 0

    for i, video_path in enumerate(videos, start=1):
        out_dir = make_output_dir(video_path, input_root, output_root)

        if out_dir.exists():
            has_frames = any(p.is_file() and p.suffix.lower() in {".jpg", ".png"} for p in out_dir.iterdir())
            if has_frames and not args.overwrite:
                skipped += 1
                print(f"[{i}/{len(videos)}] skip existing: {video_path}")
                continue
            if args.overwrite:
                shutil.rmtree(out_dir, ignore_errors=True)

        written, status = extract_frames(
            video_path=video_path,
            out_dir=out_dir,
            image_ext=args.image_ext,
            jpg_quality=args.jpg_quality,
            zero_pad=args.zero_pad,
            every_n=args.every_n,
        )

        if status == "ok":
            done += 1
            total_written += written
            print(f"[{i}/{len(videos)}] ok   {video_path} -> {out_dir} ({written} frames)")
        else:
            failed += 1
            print(f"[{i}/{len(videos)}] fail {video_path} ({status})")

    print(
        f"[Summary] done={done} skipped={skipped} failed={failed} "
        f"videos={len(videos)} total_frames={total_written}"
    )


if __name__ == "__main__":
    main()
