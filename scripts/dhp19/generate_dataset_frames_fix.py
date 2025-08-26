
"""
Convert DHP19 *_events.h5 into per-camera .npy frames with standard names.
Standalone: no project imports required.
"""
import argparse
import glob
import os
import re
import h5py
import numpy as np

def get_standard_path(subject: int, session: int, movement: int, frame: int, cam: int, postfix: str = ""):
    # 与 DHP19Core.get_standard_path 一致
    return f"S{subject}_session_{session}_mov_{movement}_frame_{frame}_cam_{cam}{postfix}.npy"

def parse_ssm_from_filename(fname: str):
    # 兼容多位数：匹配 S{num}_session_{num}_mov_{num}
    bname = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"S(\d+)_session_(\d+)_mov_(\d+)", bname)
    if not m:
        raise ValueError(f"Cannot parse subject/session/mov from filename: {fname}")
    sub, sess, mov = map(int, m.groups())
    return sub, sess, mov

def convert_raw_frame_and_save(h5_path: str, out_dir: str):
    sub, sess, mov = parse_ssm_from_filename(h5_path)
    with h5py.File(h5_path, "r") as f:
        # 期望 DVS 形状: (num_frames, H, W, 4)
        dvs = f["DVS"]
        num_frames, H, W, num_cams = dvs.shape
        for cam in range(num_cams):
            frames = dvs[..., cam]  # (num_frames, H, W)
            for ind in range(num_frames):
                frame = frames[ind, :, :]  # (H, W)
                out_name = get_standard_path(sub, sess, mov, ind, cam, "")
                out_path = os.path.join(out_dir, out_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, frame)

def main():
    parser = argparse.ArgumentParser(description="Convert DHP19 *_events.h5 to per-camera .npy frames")
    parser.add_argument("--input_dir", required=True, help="Folder containing *_events.h5")
    parser.add_argument("--out_dir", required=True, help="Output folder for .npy frames")
    args = parser.parse_args()

    x_paths = sorted(glob.glob(os.path.join(args.input_dir, "*events.h5")))
    if not x_paths:
        raise FileNotFoundError(f"No *events.h5 found in: {args.input_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    for x in x_paths:
        # 跳过 label 文件（如果命名包含 _label.h5）
        if x.endswith("_label.h5"):
            continue
        convert_raw_frame_and_save(x, args.out_dir)

    print(f"Done. Saved frames to: {args.out_dir}")

if __name__ == "__main__":
    main()
