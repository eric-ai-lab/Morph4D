# Authors: Hui Ren (rhfeiyang.github.io)
import os
import json
import argparse

import numpy as np

from sam2_segmentation import segmentation
import os.path as osp
import torch
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, help="path to exp dir")
    parser.add_argument("--gt_root", type=str)
    parser.add_argument('--point',
                        type=int,
                        nargs='+',
                        help='two values x, y as a input point')
    parser.add_argument("--tto", action="store_true", help="use tto")
    args = parser.parse_args()
    return args

def calculate_accuracy(gt, pred):
    accs = []
    for i in range(len(gt)):
        # numpy
        acc = (gt[i][0] == pred[i][0]).sum().item() / gt[i][0].size
        accs.append(acc)

    return sum(accs)/len(accs)

def calculate_iou(gt, pred):
    ious = []
    for i in range(len(gt)):
        # numpy
        intersection = np.logical_and(gt[i][0], pred[i][0]).sum().item()
        union = np.logical_or(gt[i][0], pred[i][0]).sum().item()
        if union == 0:
            continue
        iou = intersection / union
        ious.append(iou)

    return sum(ious)/len(ious)

def main(exp_root, gt_root, point, tto):
    prefix = "tto_" if tto else ""
    # feature_config = osp.join(exp_root, "feature_config.yaml")
    # feature_head_ckpt_path = osp.join(exp_root, "finetune_semantic_heads.pth")
    # rendered_feature_path = osp.join(exp_root, "tto_test_latent_features.pth")


    gt_dir = osp.join(gt_root, f"sam2_segmentation_point_{point[0]}_{point[1]}")

    with open(osp.join(gt_dir, f"sam2_prompt.json"), "r") as f:
        sam2_prompt = json.load(f)
    gt_point = sam2_prompt["point"]
    assert gt_point == point, f"gt point {gt_point} does not match input point {point}"
    print(f"Point: {point}")
    save_dir=osp.join(exp_root, f"{prefix}test_rgb_segmentation")
    pred_seg = osp.join(save_dir, f"sam2_segmentation_point_{point[0]}_{point[1]}/sam2_masks.pth")
    if not osp.exists(pred_seg):
        segmentation(point=point, video_dir=osp.join(exp_root, f"{prefix}test"),save_dir=save_dir)

    gt_seg = osp.join(gt_dir, "sam2_masks.pth")
    gt_seg = torch.load(gt_seg)
    pred_seg = torch.load(pred_seg)

    accuracy = calculate_accuracy(gt_seg, pred_seg)
    iou = calculate_iou(gt_seg, pred_seg)

    print(f"Accuracy: {accuracy}")
    print(f"IoU: {iou}")
    # save
    with open(osp.join(save_dir, f"sam2_segmentation_point_{point[0]}_{point[1]}/sam2_metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy, "iou": iou}, f)

    print(f"results saved to {save_dir}")




if __name__ == "__main__":
    args = parse_args()
    main(args.exp_root, args.gt_root, args.point, args.tto)