import os
import sys
import cv2

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os.path as osp
import argparse
from torch.nn import functional as F
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib_4d')))
from autoencoder.model import Feature_heads
import matplotlib.animation as animation
import imageio
from tqdm import tqdm
import re
import json
import time
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.axis('off')
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=600):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=4))

np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

############################################################################################
def extract_numbers(s):
    numbers = re.findall(r'\d+', s)
    # concat number
    numbers = ''.join(numbers)
    return int(numbers)


def parse_args():
    parser = argparse.ArgumentParser(description='inference with reconstructed feat of sam2')
    parser.add_argument("--video_dir", type=str, default=None, help='path to the video')
    parser.add_argument('--rendered_results_path', type=str, default=None, help="path to the result (MorphoSim output)")
    parser.add_argument('--frame', type=int, default=0, help='starting frame for segmentation and propagation')
    parser.add_argument('--point', 
                    type=int,
                    nargs='+',
                    help='two values x, y as a input point')

    parser.add_argument('--box', 
                    type=int,
                    nargs='+',
                    help='four values x1, y1 as top left and x2, y2 corner as a bottom right corner')
    parser.add_argument("--head_config", type=str, default="../configs/default_config.yaml")
    parser.add_argument("--save_dir", type=str, default=None, help="path to save the results")
    args = parser.parse_args()

    args.semantic_head_path = None

    if args.video_dir is None:
        args.video_dir = os.path.dirname(args.rendered_results_path)+"_rgb"
    if args.rendered_results_path is not None:
        args.rendered_root = os.path.dirname(os.path.dirname(os.path.dirname(args.rendered_results_path)))
        args.semantic_head_path = os.path.join(args.rendered_root,"finetune_semantic_heads.pth")
    assert  args.video_dir is not None or args.rendered_results_path is not None, "Please provide either video_dir or rendered_results_path"
    return args


def segmentation(semantic_head_path=None,video_dir=None, rendered_results_path=None, rendered_feature_path=None , head_config="../configs/default_config.yaml", ann_frame_idx=0, point=None, box=None, save_dir=None):


    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        # pred_iou_thresh=0.7,
        # stability_score_thresh=0.92,
        # stability_score_offset=0.7,
        # crop_n_layers=1,
        # box_nms_thresh=0.7,
        # crop_n_points_downscale_factor=2,
        min_mask_region_area=25,
        # use_m2m=True,
    )

    # video_dir = args.video_dir # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: extract_numbers(os.path.splitext(p)[0]))
    w,h = Image.open(os.path.join(video_dir, frame_names[0])).size
    ############################################################################################
    start_time = time.time()
    # rendered_results_path = args.rendered_results_path
    if rendered_results_path is not None:
        render_dicts = torch.load(rendered_results_path,weights_only=True)

        with open(head_config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        head_config = config["Head"]
        feature_heads = Feature_heads(head_config).to("cuda")
        state_dict = torch.load(semantic_head_path,weights_only=True)
        feature_heads.load_state_dict(state_dict)
        feature_heads.eval()

        sam2_rendered_feat = {}
        for frame_idx in range(len(render_dicts)):
            rendered_feat = render_dicts[frame_idx]["feature_map"]

            rendered_feat = F.interpolate(rendered_feat[None], size=(64, 64), mode='bilinear', align_corners=False)[0]
            rendered_feat = rendered_feat.permute(1,2,0)
            rendered_feat = feature_heads.decode("sam2" , rendered_feat)
            rendered_feat = rendered_feat.permute(2,0,1)
            rendered_feat = rendered_feat[None, ...]

            sam2_rendered_feat[frame_idx] = {}
            sam2_rendered_feat[frame_idx]['backbone_fpn'] = [rendered_feat]
            sam2_rendered_feat[frame_idx]['vision_pos_enc'] = [[]]

        # gt_feats = predictor.extract_features(video_path=args.video_dir)
        # sam2_rendered_feat=gt_feats

        rgb_feats = torch.zeros(len(sam2_rendered_feat), 3, h,w)

        inference_state = predictor.novel_view_inference_init( sam2_feats=sam2_rendered_feat, rgb_feats=rgb_feats)
        save_dir = os.path.dirname(rendered_results_path) if save_dir is None else save_dir

    elif rendered_feature_path is not None:
        render_dicts = torch.load(rendered_feature_path,weights_only=True) # tensor T X C X H X W
        with open(head_config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        head_config = config["Head"]
        feature_heads = Feature_heads(head_config).to("cuda")
        state_dict = torch.load(semantic_head_path,weights_only=True)
        feature_heads.load_state_dict(state_dict)
        feature_heads.eval()

        sam2_rendered_feat = {}
        for frame_idx in range(len(render_dicts)):
            rendered_feat = render_dicts[frame_idx]
            rendered_feat = rendered_feat.to("cuda")
            rendered_feat = F.interpolate(rendered_feat[None], size=(64, 64), mode='bilinear', align_corners=False)[0]
            rendered_feat = rendered_feat.permute(1,2,0)
            rendered_feat = feature_heads.decode("sam2" , rendered_feat)
            rendered_feat = rendered_feat.permute(2,0,1)
            rendered_feat = rendered_feat[None, ...]

            sam2_rendered_feat[frame_idx] = {}
            sam2_rendered_feat[frame_idx]['backbone_fpn'] = [rendered_feat]
            sam2_rendered_feat[frame_idx]['vision_pos_enc'] = [[]]

        rgb_feats = torch.zeros(len(sam2_rendered_feat), 3, h,w)
        inference_state = predictor.novel_view_inference_init( sam2_feats=sam2_rendered_feat, rgb_feats=rgb_feats)
        save_dir = os.path.dirname(rendered_feature_path) if save_dir is None else save_dir

    else:
        inference_state = predictor.init_state(video_path=video_dir)
        save_dir = video_dir if save_dir is None else save_dir

    ############################################################################################

    # ann_frame_idx = args.frame  # the frame index we interact with
    # points =  np.array([])
    # labels = np.array([])
    # box = np.array([])
    if point is None and box is not None: # only box input
        box = np.array(box)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=0,
            box=box
        )
        # show the results on the current (interacted) frame
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_box(box, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.show()
        save_dir = os.path.join(save_dir, f"sam2_segmentation_box_{box[0]}_{box[1]}_{box[2]}_{box[3]}")
        video_save_path = os.path.join(save_dir, "sam2_segmentation_box.mp4")
        img_save_path = os.path.join(save_dir, "sam2_segmentation_box")
        os.makedirs(img_save_path, exist_ok=True)
    
    elif point is not None and box is None: # only point input
        points = np.array([point], dtype=np.float32)
        labels = np.array([1], np.int32) # for labels, `1` means positive click and `0` means negative click
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=0,
            points=points,
            labels=labels,
        )
        # show the results on the current (interacted) frame
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.show()
        save_dir = os.path.join(save_dir, f"sam2_segmentation_point_{point[0]}_{point[1]}")
        video_save_path = os.path.join(save_dir, "sam2_segmentation_point.mp4")
        img_save_path = os.path.join(save_dir, "sam2_segmentation_point")
        os.makedirs(img_save_path, exist_ok=True)

    elif point is not None and box is not None: # both box and point input
        points = np.array([point], dtype=np.float32)
        labels = np.array([1], np.int32) # for labels, `1` means positive click and `0` means negative click
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=0,
            points=points,
            labels=labels,
        )
        box = np.array(box)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            box=box
        )
        # show the results on the current (interacted) frame
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        show_box(box, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        show_mask((out_mask_logits[1] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[1])
        plt.show()
        save_dir = os.path.join(save_dir, f"sam2_segmentation_point_{point[0]}_{point[1]}_box_{box[0]}_{box[1]}_{box[2]}_{box[3]}")
        video_save_path = os.path.join(save_dir, "sam2_segmentation_point&box.mp4")
        img_save_path = os.path.join(save_dir, "sam2_segmentation_point&box")
        os.makedirs(img_save_path, exist_ok=True)
        
    else: # no box and no input, automatic mask segmentation
        image = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
        image = np.array(image.convert("RGB"))
        masks = mask_generator.generate(image)

        for i in range(len(masks)):
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id= i,
                mask=masks[i]["segmentation"]
            )
        # show the results on the current (interacted) frame
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        for i in range(len(masks)):
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[i])
        plt.show()
        save_dir = os.path.join(save_dir, "sam2_segmentation_automask")
        video_save_path = os.path.join(save_dir, "sam2_segmentation_automask.mp4")
        img_save_path = os.path.join(save_dir, "sam2_segmentation_automask")
        os.makedirs(img_save_path, exist_ok=True)

    ###########################################################################################

    #run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits, pix_feats in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    end_time = time.time()
    compute_time = end_time - start_time
    # save inference time
    with open(os.path.join(save_dir, "sam2_inference_time.json"), "w") as f:
        json.dump({"inference_time": compute_time}, f)
    print(f"inference time: {compute_time}")

    # save prompt info as json
    with open(os.path.join(save_dir, "sam2_prompt.json"), "w") as f:
        print(f"point saved: {point}, box: {box} to {save_dir}")
        json.dump({"point": point, "box": box}, f)

    # save masks
    save_path = os.path.join(save_dir, "sam2_masks.pth")
    torch.save(video_segments, save_path)

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")
    viz_results = []

    for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
        plt.axis('off')
        fig = plt.figure(figsize=(9, 6), dpi=100)
        #plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        
        if out_frame_idx == ann_frame_idx:
            if point is not None:
                show_points(points, labels, plt.gca())
            if box is not None:
                show_box(box, plt.gca())

        # take as image
        output_img = os.path.join(img_save_path, f"{out_frame_idx:05d}.jpg")
        plt.savefig(output_img, bbox_inches='tight', pad_inches=0)
        img = Image.open(output_img).convert("RGB")
            
        viz_results.append(img)
        plt.close()

    # save as video
    imageio.mimsave(video_save_path, viz_results)
    print(f"video saved to {video_save_path}")

if __name__ == "__main__":
    args = parse_args()
    segmentation(semantic_head_path=args.semantic_head_path, video_dir=args.video_dir, rendered_results_path=args.rendered_results_path, head_config=args.head_config, ann_frame_idx=args.frame, point=args.point, box=args.box, save_dir = args.save_dir)
