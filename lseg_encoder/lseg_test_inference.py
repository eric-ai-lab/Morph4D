import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import argparse
from torch.nn import functional as F
import torchvision.transforms as transforms
import yaml
import matplotlib.animation as animation
import imageio
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib_4d')))
from autoencoder.model import Feature_heads
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='inference with reconstructed feat of sam2')
    parser.add_argument('--rendered_results_path', type=str, default='/home/shijie/Desktop/work/feature-4dgs/output/flower/native_feat_davis.yaml_compactgs_mixfeat_nomotion_channel32_dep=uni_gt_cam=False_lrfeat=0.01_reversed=False_20241111_004723/final_viz/50_round_moving/rendered_results.pth')
    parser.add_argument("--labels", type=str, default="default", help="segment using specified labels",)
    #parser.add_argument("--rendered_root",type=str, default="/home/shijie/Desktop/work/feature-4dgs/data/davis_dev/train/code_output/log/native_feat_davis.yaml_compactgs_mixfeat_channel64_dep=uni_gt_cam=False_lrfeat=0.01_20241102_195618")
    parser.add_argument("--head_config", type=str, default="../configs/default_config.yaml")
    args = parser.parse_args()
    args.rendered_root = os.path.dirname(os.path.dirname(os.path.dirname(args.rendered_results_path)))
    args.semantic_head_path = os.path.join(args.rendered_root,"finetune_semantic_heads.pth")
    return args

def run_ffmpeg(input_pattern, output_video):
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', '12',  # Set the desired frame rate
        '-i', input_pattern,  # Input pattern for the images
        '-c:v', 'libx264',  # Specify the video codec
        '-pix_fmt', 'yuv420p',  # Set pixel format
        output_video  # Output video file name with path
    ]
    subprocess.run(ffmpeg_command)

if __name__ == "__main__":
    args = parse_args()

    save_dir = os.path.join(args.rendered_root,"lseg_results")
    os.makedirs(save_dir, exist_ok=True)

    rendered_results = args.rendered_results_path
    render_dicts = torch.load(rendered_results,weights_only=True)
    print(len(render_dicts)) # frames
    # print(render_dicts[0].keys())
    # print(render_dicts[0]["rgb"].shape) # [3, 480, 480]
    # print(render_dicts[0]["feature_map"].shape) # [channels, 64, 64]

    with open(args.head_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    head_config = config["Head"]
    feature_heads = Feature_heads(head_config).to("cuda")
    state_dict = torch.load(args.semantic_head_path,weights_only=True)
    feature_heads.load_state_dict(state_dict)
    feature_heads.eval()

    img_save_dir = os.path.join(save_dir,"renders")
    os.makedirs(img_save_dir, exist_ok=True)
    feat_save_dir = os.path.join(save_dir,"saved_features")
    os.makedirs(feat_save_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()

    for frame_idx in range(len(render_dicts)):
        rgb_feat = render_dicts[frame_idx]["rgb"]  # [3, 480, 480]
        image = to_pil(rgb_feat)  # Convert tensor to PIL image
        img_save_path = os.path.join(img_save_dir, f"{frame_idx:05d}.png")
        image.save(img_save_path)

        rendered_feat = render_dicts[frame_idx]["feature_map"] # [channels, 64, 64]
        rendered_feat = F.interpolate(rendered_feat[None], size=(480, 480), mode='bilinear', align_corners=False)[0].permute(1,2,0) # [480, 480, channels]
        rendered_feat = feature_heads.decode("langseg" , rendered_feat).permute(2,0,1).cpu() # [512, 480, 480] gt shape
        feat_save_path = os.path.join(feat_save_dir, f"{frame_idx:05d}_fmap_CxHxW.pt")
        torch.save(rendered_feat, feat_save_path)

    command = [
        "python", "-u", "segmentation.py",
        "--data", args.rendered_root,
        "--label_src", args.labels
    ]
    subprocess.run(command)

    # Define the folder containing the PNG images and the output video file name
    video_save_dir = os.path.join(save_dir,"segmentation")
    input_folder = os.path.join(video_save_dir,"lseg_results")
    output_video1 = os.path.join(video_save_dir, 'lseg.mp4')
    output_video2 = os.path.join(video_save_dir, 'lseg_viz.mp4')
    output_video3 = os.path.join(video_save_dir, 'lseg_legend.mp4')

    # Change to the input directory
    os.chdir(input_folder)

    run_ffmpeg('%05d_fmap_CxHxW.png', output_video1)
    run_ffmpeg('%05d_fmap_CxHxW.png_vis.png', output_video2)
    run_ffmpeg('%05d_fmap_CxHxW.png_legend.png', output_video3)