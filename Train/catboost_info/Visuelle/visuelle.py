import os
import glob
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms, models
from torch import nn

# Đường dẫn tới thư mục ảnh
image_base_dir = r"E:\visuelle2\visuelle2\images"

# Tìm tất cả ảnh trong thư mục con
image_paths = glob.glob(os.path.join(image_base_dir, '**', '*.jpg'), recursive=True)
image_paths += glob.glob(os.path.join(image_base_dir, '**', '*.png'), recursive=True)

# Transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ResNet18 pretrained
resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Bỏ lớp fully connected cuối
resnet.eval()

# Trích đặc trưng
features = []
image_names = []

for img_path in tqdm(image_paths):
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)

        if not isinstance(img_tensor, torch.Tensor):
            raise ValueError("Image transform failed to return a tensor")

        img_tensor = img_tensor.unsqueeze(0)  # [1, 3, 224, 224]

        with torch.no_grad():
            feature = resnet(img_tensor).squeeze().numpy()  # [512]

        features.append(feature)
        rel_path = os.path.relpath(img_path, image_base_dir).replace("\\", "/")
        image_names.append(rel_path)

    except Exception as e:
        print(f"❌ Lỗi ảnh {img_path}: {e}")
        features.append(np.zeros(512))
        rel_path = os.path.relpath(img_path, image_base_dir).replace("\\", "/")
        image_names.append(rel_path)

# Lưu thành DataFrame
df_features = pd.DataFrame(features)
df_features.columns = [f'img_feat_{i}' for i in range(512)]
df_features['image_path'] = image_names

# Xuất CSV
df_features.to_csv("image_features_visuelle.csv", index=False)
print("✅ Đã lưu xong file image_features_visuelle.csv")
