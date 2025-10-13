# Authors: Hui Ren (rhfeiyang.github.io)
import os
import glob
import shutil
# dataset_root = "./data/nerfies"
# image_files = glob.glob(os.path.join(dataset_root, "*", "rgb/4x", "*.png"))
# print(image_files)

# for image_file in image_files:
#     if int(image_file.split(".")[0].split("_")[-1]) >=90:
#         continue
#     if "right1" in image_file:
#         subfolder = "right1"
#     elif "left1" in image_file:
#         subfolder = "left1"
#     parent_dir = os.path.dirname(image_file)
#     save_dir = parent_dir.replace("rgb/4x", subfolder )
#     save_dir = os.path.join(save_dir, "images")
#     os.makedirs(save_dir, exist_ok=True)
#     shutil.copy(image_file, save_dir)
#
# camera_files = glob.glob(os.path.join(dataset_root, "*", "camera", "*.json"))
# for camera_file in camera_files:
#     if int(camera_file.split(".")[0].split("_")[-1]) >=90:
#         continue
#     if "right1" in camera_file:
#         subfolder = "right1"
#     elif "left1" in camera_file:
#         subfolder = "left1"
#     parent_dir = os.path.dirname(camera_file)
#     save_dir = parent_dir.replace("camera", f"camera_{subfolder}" )
#     os.makedirs(save_dir, exist_ok=True)
#     shutil.copy(camera_file, save_dir)

left_views={"camera":[], "image":[]}
left_ids = []
right_views={"camera":[], "image":[]}
dataset_dir = "./data/nerfies/tail"
camera_dir = os.path.join(dataset_dir, "camera")
rgb_dir = os.path.join(dataset_dir, "rgb/4x")

camera_files = glob.glob(os.path.join(camera_dir, "*.json"))
for camera_file in camera_files:
    if "right1" in camera_file:
        subfolder = "right1"
    elif "left1" in camera_file:
        subfolder = "left1"
    name = os.path.basename(camera_file).split(".")[0]
    id = int(name.split("_")[-1])
    if id>=90:
        continue
    image_pair = os.path.join(rgb_dir, f"{name}.png")
    if os.path.exists(image_pair):
        if subfolder == "right1":
            # if not int(name.split("_")[-1]) in left_ids:
            #     continue
            right_views["camera"].append(camera_file)
            right_views["image"].append(image_pair)
        elif subfolder == "left1":
            left_views["camera"].append(camera_file)
            left_views["image"].append(image_pair)
            left_ids.append(int(name.split("_")[-1]))

# save
for i, (camera_file, image_file) in enumerate(zip(left_views["camera"], left_views["image"])):
    save_dir = os.path.join(dataset_dir, "left1", "images")
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(image_file, save_dir)
    save_dir = os.path.join(dataset_dir, "camera_left1")
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(camera_file, save_dir)

saved_right_ids = []
for i, (camera_file, image_file) in enumerate(zip(right_views["camera"], right_views["image"])):
    if not int(os.path.basename(camera_file).split(".")[0].split("_")[-1]) in left_ids:
        continue
    saved_right_ids.append(int(os.path.basename(camera_file).split(".")[0].split("_")[-1]))
    save_dir = os.path.join(dataset_dir, "right1", "images")
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(image_file, save_dir)
    save_dir = os.path.join(dataset_dir, "camera_right1")
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(camera_file, save_dir)

print(f"{len(saved_right_ids)} images saved")