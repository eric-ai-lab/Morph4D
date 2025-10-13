# Authors: Hui Ren (rhfeiyang.github.io)
import torch
from torchvision import transforms
import os


from PIL import Image
image_folder = "/public/home/renhui/code/4d/feature-4dgs/data/davis_dev/train/images"
new_folder = "/public/home/renhui/code/4d/feature-4dgs/data/davis_dev/train/images_224"
os.makedirs(new_folder, exist_ok=True)
# resize and crop
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

for image_file in os.listdir(image_folder):
    img = Image.open(os.path.join(image_folder, image_file))
    img = transform(img)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = (img * 255).astype('uint8')
    img = Image.fromarray(img)
    img.save(os.path.join(new_folder, image_file))
