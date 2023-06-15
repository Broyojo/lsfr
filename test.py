import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import L1Loss
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm

from transformer import ViTForImageReconstruction

torch.set_default_device("cuda")

images = []

for root, dirs, files in tqdm(os.walk("./lfw/lfw-deepfunneled/lfw-deepfunneled/")):
    for file in files:
        if file.endswith(".jpg"):
            file_path = os.path.join(root, file)
            with Image.open(file_path) as image:
                images.append(np.array(image))

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()

model = ViTForImageReconstruction(d_embed=768, n_layers=12, n_head=12, dropout=0.0, img_size=250, patch_size=25)
print(model)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

model.load_state_dict(torch.load("./image_reconstruction_transformer.pt"))

plt.imshow(images[100])
plt.show()

image = to_tensor(images[100]).unsqueeze(0).cuda()
output = model(image)
image = to_image(output[0])
image.save(f"test.png")