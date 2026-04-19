import torch
import cv2
import os
import time
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_folder = "input"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Sobel kernel (edge detection)
kernel = torch.tensor([[[[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]]], dtype=torch.float32)

# Load images
images = []
filenames = []

for file in os.listdir(input_folder):
    path = os.path.join(input_folder, file)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        continue
        
    img = cv2.resize(img, (256, 256))
    images.append(img)
    filenames.append(file)

print(f"Total images loaded: {len(images)}")

# Convert to tensor batch
images_np = np.array(images)
images_tensor = torch.tensor(images_np, dtype=torch.float32).unsqueeze(1)

# ---------------- CPU Execution ----------------
start = time.time()
cpu_output = torch.nn.functional.conv2d(images_tensor, kernel, padding=1)
cpu_time = time.time() - start

# ---------------- GPU Execution ----------------
images_gpu = images_tensor.to(device)
kernel_gpu = kernel.to(device)

start = time.time()
gpu_output = torch.nn.functional.conv2d(images_gpu, kernel_gpu, padding=1)

if device.type == "cuda":
    torch.cuda.synchronize()

gpu_time = time.time() - start

gpu_output = gpu_output.cpu()

# Save outputs
for i in range(len(filenames)):
    output_img = gpu_output[i].squeeze().detach().numpy()
    output_img = np.clip(output_img, 0, 255)
    
    cv2.imwrite(os.path.join(output_folder, filenames[i]), output_img)

# Save log
os.makedirs("results", exist_ok=True)
with open("results/execution_log.txt", "w") as f:
    f.write(f"Total Images: {len(images)}\n")
    f.write(f"CPU Time: {cpu_time}\n")
    f.write(f"GPU Time: {gpu_time}\n")

print("Processing complete!")
print("CPU Time:", cpu_time)
print("GPU Time:", gpu_time)