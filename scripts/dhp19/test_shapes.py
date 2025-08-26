import glob, os, re
import numpy as np

frames_dir = r"C:\code\EventHPE\outDatasetFolder\frames"
labels_dir = r"C:\code\EventHPE\outDatasetFolder\labels"

# 找一个样本（相机0）
npy_paths = sorted(glob.glob(os.path.join(frames_dir, "S*_session_*_mov_*_frame_*_cam_0.npy")))
if not npy_paths:
    raise FileNotFoundError("No npy frames found. Check frames_dir.")

f = npy_paths[0]
print("frame file:", f)
img = np.load(f)  # (H,W)
print("frame shape:", img.shape, "dtype:", img.dtype, "min/max:", img.min(), img.max())

# 匹配对应的标签 npz：把 .npy 替换为 _2dhm.npz
b = os.path.basename(f)
m = re.match(r"(S\d+_session_\d+_mov_\d+_frame_\d+_cam_\d+)\.npy", b)
if not m:
    raise RuntimeError("Unexpected frame filename pattern.")
npz_name = m.group(1) + "_2dhm.npz"
npz_path = os.path.join(labels_dir, npz_name)
print("label file:", npz_path)
lab = np.load(npz_path)

xyz = lab["xyz"]
K   = lab["camera"]
M   = lab["M"]
print("xyz shape:", xyz.shape, "K:", K.shape, "M:", M.shape)

# DHP19Core 会执行 swapaxes(0,1) 在 xyz 上，确保这个操作后形状合理（多数情况期望 (3, J)）
xyz_swapped = np.swapaxes(xyz, 0, 1)
print("xyz after swapaxes(0,1):", xyz_swapped.shape)
