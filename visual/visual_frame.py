import numpy as np
import matplotlib.pyplot as plt

# 文件路径
file_path = r"C:\code\EventHPE\outDatasetFolder\frames\S15_session_4_mov_5_frame_149_cam_1.npy"

# 加载事件帧
frame = np.load(file_path)

print("Frame shape:", frame.shape)
print("Data type:", frame.dtype)
print("Min/Max:", frame.min(), frame.max())

# 可视化
plt.figure(figsize=(6, 6))
plt.imshow(frame, cmap="gray")  # 如果是灰度图
plt.colorbar(label="Pixel Value")
plt.title("Event Frame Visualization")
plt.axis("off")
plt.show()

arr = np.load(file_path)

print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("Min/Max:", arr.min(), arr.max())
