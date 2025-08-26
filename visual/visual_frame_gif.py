import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import re

frames_dir = r"C:\code\EventHPE\outDatasetFolder\frames"
out_gif = "event_sequence_cam0_first10.gif"

# 只选 cam_0 的文件
files = [f for f in os.listdir(frames_dir) if f.endswith(".npy") and "cam_0" in f]

# 按 frame 索引排序
files.sort(key=lambda x: int(re.search(r"frame_(\d+)_cam_0", x).group(1)))

# 只取前10个
files = files[:10]

fig, ax = plt.subplots()

def update(i):
    ax.clear()
    frame = np.load(os.path.join(frames_dir, files[i]))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Frame {i} (cam_0)")
    ax.axis("off")

ani = animation.FuncAnimation(fig, update, frames=len(files), interval=200)  # 200ms一帧

ani.save(out_gif, writer="pillow")
print(f"已保存为 {out_gif}")
