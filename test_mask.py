import torch
import cv2
import os
import yaml
import time

from matplotlib import pyplot as plt

from model.network import Net

mask_dir = "./data/DVRK/5.jpg"
config_dir = "./config.yml"
model_dir = "./dht_r50_nkl_d97b97138.pth"

assert os.path.isfile(config_dir)
CONFIGS = yaml.full_load(open(config_dir))

model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], backbone=CONFIGS["MODEL"]["BACKBONE"])
model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
# model.compile()
if os.path.isfile(model_dir):
    checkpoint = torch.load(model_dir)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
else:
    print("=> no pretrained model found at '{}'".format(model_dir))

# Read and preprocess the input image
# img = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
# img = cv2.GaussianBlur(img, (13, 13), 0)
# img = cv2.Canny(img, 50, 150, apertureSize=3)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.imread(mask_dir)
cv2.imshow("Input Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

# shrink the image
size = (img.shape[2] // 2, img.shape[3] // 2)
img = torch.nn.functional.interpolate(img, size=size, mode='bilinear', align_corners=False)

with torch.no_grad():
    key_points = model(img)

    start_time = time.time()
    key_points = model(img)
    end_time = time.time()

print("Time taken for inference: {:.4f} seconds".format(end_time - start_time))
print(key_points.size())

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First subplot: Key Points Heatmap
key_points_np = key_points.squeeze(0).cpu().numpy() * 100
key_points_np = key_points_np.transpose(1, 2, 0)  # Change to HWC format
im1 = ax1.imshow(key_points_np, cmap='jet')
ax1.axis('off')
ax1.set_title("Raw Heatmap")
# Add a colorbar to the first subplot
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)

# Second subplot: Binary Key Points
key_points_np = (key_points).sigmoid().squeeze(0).cpu().numpy() 
# binary_mask = key_points_np > CONFIGS['MODEL']['THRESHOLD']
# key_points_np[~binary_mask] = 0.0
key_points_np = key_points_np.transpose(1, 2, 0)  # Change to HWC format
im2 = ax2.imshow(key_points_np, cmap='jet')
ax2.axis('off')
ax2.set_title("Softmax Heatmap")
# Add a colorbar to the second subplot
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046)

# Add main title and adjust layout
plt.suptitle("Key Points Visualization", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
plt.show()