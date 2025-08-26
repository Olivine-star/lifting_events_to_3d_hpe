import numpy as np

file_path = r"C:\code\EventHPE\outDatasetFolder\labels\S1_session_1_mov_1_frame_0_cam_0_2dhm.npz"

data = np.load(file_path)
print("包含的键：", data.files)

for key in data.files:
    print(key, data[key].shape, data[key].dtype)
    print(data[key])
