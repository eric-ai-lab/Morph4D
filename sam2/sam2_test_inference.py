import os
import sys
import cv2
import time

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
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


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
def parse_args():
    parser = argparse.ArgumentParser(description='inference with reconstructed feat of sam2')
    parser.add_argument('--rendered_results_path', type=str, default=None, help='path to the result (MorphoSim output)')
    # parser.add_argument("--ori_feat_path", type=str, default=None)
    # parser.add_argument("--semantic_head_path", type=str, default=None)
    # parser.add_argument("--video_dir", type=str, default=None)
    # parser.add_argument("--rendered_root", type=str, default=None)
    parser.add_argument("--head_config", type=str, default="../configs/default_config.yaml")
    args = parser.parse_args()

    args.video_dir = os.path.dirname(args.rendered_results_path)+"_rgb"
    args.rendered_root = os.path.dirname(os.path.dirname(os.path.dirname(args.rendered_results_path)))
    args.semantic_head_path = os.path.join(args.rendered_root,"finetune_semantic_heads.pth")
    return args

if __name__ == "__main__":
    args = parse_args()

    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    )

    ############################################################################################

    # sam2_ref_path = args.ori_feat_path
    # sam2_ref_feats = torch.load(sam2_ref_path,weights_only=True)
    # vision_pos_enc = sam2_ref_feats[0]['vision_pos_enc']
    # print(len(sam2_feats)) # frames (70)
    # print(sam2_feats[0].keys()) # ['vision_pos_enc', 'backbone_fpn']
    # print(sam2_feats[0]['vision_pos_enc'][0].shape) # [1, 256, 64, 64]
    # print(sam2_feats[0]['backbone_fpn'][0].shape) # [1, 256, 64, 64]

    rendered_results = args.rendered_results_path
    render_dicts = torch.load(rendered_results,weights_only=True)
    print(len(render_dicts)) # frames
    print(render_dicts[0].keys())
    # print(render_dicts[0]["rgb"].shape) # [3, 480, 480]
    # print(render_dicts[0]["feature_map"].shape) # [256, 64, 64]

    video_dir = args.video_dir
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    with open(args.head_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    head_config = config["Head"]
    feature_heads = Feature_heads(head_config).to("cuda")
    state_dict = torch.load(args.semantic_head_path,weights_only=True)
    feature_heads.load_state_dict(state_dict)
    feature_heads.eval()

    # Record the start time
    start_time = time.time()

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

    rgb_feats = torch.zeros(len(sam2_rendered_feat), 3, 480, 480)
    
    inference_state = predictor.novel_view_inference_init( sam2_feats=sam2_rendered_feat, rgb_feats=rgb_feats)

    ############################################################################################

    ann_frame_idx = 0  # the frame index we interact with
    
    points = np.array([[100, 240]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=0,
        points=points,
        labels=labels,
    )
    # # show the results on the current (interacted) frame
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {ann_frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    # show_points(points, labels, plt.gca())
    # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    # plt.show()

    # points = np.array([[300, 200]], dtype=np.float32)
    # labels = np.array([1], np.int32)
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=0,
    #     obj_id=1,
    #     points=points,
    #     labels=labels,
    # )
    # show_points(points, labels, plt.gca())
    # show_mask((out_mask_logits[1] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[1])
    # plt.show()

    # points = np.array([[100, 200], [100, 230]], dtype=np.float32)
    # labels = np.array([1, 1], np.int32)
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=2,
    #     points=points,
    #     labels=labels,
    # )
    # show_points(points, labels, plt.gca())
    # show_mask((out_mask_logits[2] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[2])
    # plt.show()

    ###########################################################################################

    #run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits, pix_feats in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.4f} seconds")

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")
    viz_results = []

    img_save_path = os.path.join(os.path.dirname(args.rendered_results_path), "sam2_segmentation_from_features") 
    os.makedirs(img_save_path, exist_ok=True)

    for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
        plt.axis('off')
        fig = plt.figure(figsize=(9, 6), dpi=100)
        #plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        
        if out_frame_idx == ann_frame_idx:
            show_points(points, labels, plt.gca())

        # take as image
        output_img = os.path.join(img_save_path, f"{out_frame_idx:05d}.jpg")
        plt.savefig(output_img, bbox_inches='tight', pad_inches=0)
        img = Image.open(output_img).convert("RGB")

        viz_results.append(img)
        plt.close()

    # save as video
    video_save_path = os.path.join(os.path.dirname(args.rendered_results_path), "sam2_semantic_segmentation_from_features.mp4")
    imageio.mimsave(video_save_path, viz_results)
    print(f"video saved to {video_save_path}")