# generate_joints_standalone.py
import argparse
import glob
import os
import re
from os.path import join
import h5py
import numpy as np
import cv2
from tqdm import tqdm

def parse_ssm_from_label_filename(fname: str):
    # 支持 S{num}_session_{num}_mov_{num}_..._label.h5
    b = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"S(\d+)_session_(\d+)_mov_(\d+)", b)
    if not m:
        raise ValueError(f"Cannot parse subject/session/mov from filename: {fname}")
    return tuple(map(int, m.groups()))

def decompose_projection_matrix(P: np.ndarray):
    """P: 3x4 投影矩阵 -> (K:3x3, R:3x3, [R|t]:3x4)"""
    # OpenCV 返回的 t 是齐次坐标 (4x1)，需要除以最后一维
    K, R, t_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2, 2]
    t = (t_h[:3] / t_h[3]).reshape(3, 1)
    extrinsic = np.hstack([R, t])  # 3x4
    return K, extrinsic

def main():
    ap = argparse.ArgumentParser(description="Generate *_2dhm.npz from DHP19 *_events_label.h5")
    ap.add_argument("--input_dir", required=True, help="Folder with *_events_label.h5")
    ap.add_argument("--out_dir",   required=True, help="Where to write *_2dhm.npz")
    ap.add_argument("--p_matrices_dir", required=True, help="rootDataFolder/P_matrices")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取四个相机的投影矩阵（按 DHP19 的相机顺序映射）
    P1 = np.load(join(args.p_matrices_dir, "P1.npy"))
    P2 = np.load(join(args.p_matrices_dir, "P2.npy"))
    P3 = np.load(join(args.p_matrices_dir, "P3.npy"))
    P4 = np.load(join(args.p_matrices_dir, "P4.npy"))
    # 与你原脚本一致的映射顺序（注意这行如果官方说明不同可相应调整）
    p_mats = [P4, P1, P3, P2]  # for cam indices 0,1,2,3

    label_paths = sorted(glob.glob(join(args.input_dir, "*label.h5")))
    if not label_paths:
        raise FileNotFoundError(f"No *label.h5 found in: {args.input_dir}")

    print(f"N of files: {len(label_paths)}")

    for x_path in tqdm(label_paths):
        sub, sess, mov = parse_ssm_from_label_filename(x_path)
        with h5py.File(x_path, "r") as f:
            # DHP19 labels：通常 'XYZ' 维度为 (num_frames, J, 3) 或 (num_frames, 3*J)
            XYZ = f["XYZ"][...]  # 形状依数据而定
            num_frames = XYZ.shape[0]

            for cam in range(4):
                K, extrinsic = decompose_projection_matrix(p_mats[cam])  # 3x3, 3x4
                for ind in range(num_frames):
                    xyz = XYZ[ind]
                    # 保持与 DHP19Core.get_standard_path 一致的命名
                    out_name = f"S{sub}_session_{sess}_mov_{mov}_frame_{ind}_cam_{cam}_2dhm.npz"
                    out_path = join(args.out_dir, out_name)
                    np.savez(out_path, xyz=xyz, M=extrinsic, camera=K)

    print(f"Done. Saved npz labels to: {args.out_dir}")

if __name__ == "__main__":
    main()
