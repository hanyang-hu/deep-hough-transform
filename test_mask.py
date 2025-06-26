import torch
import cv2
import os
import yaml
import time

from torchvision import transforms
from matplotlib import pyplot as plt

from model.network import Net

mask_dir = "./data/DVRK/2.jpg"
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
    print("=> loaded checkpoint '{}'".format(model_dir))
else:
    print("=> no pretrained model found at '{}'".format(model_dir))

# Read and preprocess the input image
# img = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
# img = cv2.GaussianBlur(img, (13, 13), 0)
# img = cv2.Canny(img, 50, 150, apertureSize=3)
# # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# img = cv2.imread(mask_dir, cv2.IMREAD_COLOR)
# img = cv2.resize(img, (400, 400))
# cv2.imshow("Input Image", img)

# Read image as a PIL Image
from PIL import Image
img = Image.open(mask_dir).convert('RGB')

transform = transforms.Compose(
    [
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

cv2.waitKey(0)
cv2.destroyAllWindows()

img = transform(img).cuda()
img = img.unsqueeze(0)  # Add batch dimension
print(img.shape)

model.eval()

with torch.no_grad():
    start_time = time.time()
    key_points = model(img)
    # key_points = torch.sigmoid(key_points)
    end_time = time.time()

print("Time taken for inference: {:.4f} seconds".format(end_time - start_time))
print(key_points.size())

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First subplot: Key Points Heatmap
key_points_np = key_points.squeeze().cpu().numpy()
im1 = ax1.imshow(key_points_np, cmap='jet')
ax1.axis('off')
ax1.set_title("Raw Heatmap")
# Add a colorbar to the first subplot
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)

from skimage.measure import label, regionprops

# Second subplot: Binary Key Points
key_points = torch.sigmoid(key_points)  # Apply sigmoid to get probabilities
key_points_np = key_points.squeeze().cpu().numpy()
binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
im2 = ax2.imshow(key_points_np, cmap='jet')
kmap_label = label(binary_kmap, connectivity=1)
props = regionprops(kmap_label)
plist = []
for prop in props:
    plist.append(prop.centroid)
print("Number of lines detected: ", len(plist))
# draw stars at the centroids
for point in plist:
    ax2.plot(point[1], point[0], 'y*', markersize=10)  # Note: (y, x) for plotting

ax2.axis('off')
ax2.set_title("Sigmoid Heatmap")
# Add a colorbar to the second subplot
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046)

# Add main title and adjust layout
plt.suptitle("Key Points Visualization", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
plt.show()