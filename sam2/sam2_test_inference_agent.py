import os
import sys

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
import cv2, base64
import random 
import time 
import requests 
import json 

global api_key
api_key = os.getenv("OPENAI_API_KEY", "")

def gptv_query(transcript=None, temp=0.):
    max_tokens = 512
    wait_time = 10

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        'model': 'gpt-4o', # 'gpt-4-vision-preview', # , # 'gpt-4-vision-preview',
        'max_tokens':max_tokens, 
        'temperature': temp,
        'top_p': 0.5,
        'messages':[]
    }
    if transcript is not None:
        data['messages'] = transcript

    response_text, retry, response_json = '', 0, None
    while len(response_text)<2:
        retry += 1
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data)) 
            response_json = response.json()
        except Exception as e:
            if random.random()<1: print(e)
            time.sleep(wait_time)
            continue
        if response.status_code != 200:
            print(response.headers,response.content)
            if random.random()<0.01: print(f"The response status code for is {response.status_code} (Not OK)")
            time.sleep(wait_time)
            data['temperature'] = min(data['temperature'] + 0.2, 1.0)
            continue
        if 'choices' not in response_json:
            time.sleep(wait_time)
            continue
        response_text = response_json["choices"][0]["message"]["content"]
    return response_json["choices"][0]["message"]["content"]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_img(image_path):
    base64_image = encode_image(image_path)
    image_meta = "data:image/png;base64" if 'png' in image_path else "data:image/jpeg;base64"
    img_dict = {
        "type": "image_url",
        "image_url": {
          "url": f"{image_meta},{base64_image}",
          "detail": "low"
        }
    }
    return img_dict

def gptv_reflection_prompt_selectbest(user_prompt, image_path, args):
    transcript = [{ "role": "system", "content": [] }, {"role": "user", "content": []}]
    # System prompt
    transcript[0]["content"].append("You are a helpful assistant. I want to find an object in the Image. Please output the width, height image-coordinate in the body of the object. Only output 2 numbers.")
    # You need to output the width, height image-coordinate of the user-specified object in the image. The input image is of resolution (W=480, H=480). Only output two numbers width and height for the answer.
    # transcript[0]["content"].append("From scale 1 to 10, decide how similar each image is to the user imagined IDEA of the scene.")

    # ## Example & Query prompt
    # if args.select_fewshot:
    #     transcript[-1]["content"] = transcript[-1]["content"] + prepare_fewshot_selectbest(user_prompt, img_prompt, listofimages, args)

    transcript[-1]["content"] = transcript[-1]["content"] + ["Give the coordinate of the car"]
    transcript[-1]["content"].append(load_img(image_path))

    # transcript[-1]["content"].append("Let's think step by step. Check all aspects to see how well these images strictly follow the content in IDEA, including having correct object counts, attributes, entities, relationships, sizes, appearance, and all other descriptions in the IDEA. Then give a score for each input images. Finally, consider the scores and select the image with the best overall quality with image index 0 to %d wrapped with <START> and <END>. Only wrap single image index digits between <START> and <END>."%(num_img-1))

    response = gptv_query(transcript)

    # if '<START>' not in response or '<END>' not in response: ## one format retry
    #     response = gptv_query(transcript, temp=0.1)
    # if args.verbose:
    #     print('gptv_reflection_prompt_selectbest\n %s\n'%(response))
    # if '<START>' not in response or '<END>' not in response:
    #     return random.randint(0,num_img-1), response
    # prompts = response.split('<START>')[1]
    # prompts = prompts.strip().split('<END>')[0]
    return response

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
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
            import cv2
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
    parser.add_argument('--rendered_results_path', type=str, default=None, help='path to the result .pth file (MorphoSim output)')
    parser.add_argument("--ori_feat_path", type=str, default=None)
    parser.add_argument("--rendered_root", type=str, default=None)
    parser.add_argument("--head_config", type=str, default="../configs/default_config.yaml")
    args = parser.parse_args()
    args.semantic_head_path = os.path.join(args.rendered_root,"finetune_semantic_heads.pth")
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.rendered_results_path is None or args.ori_feat_path is None or args.rendered_root is None:
        raise ValueError("Please set --rendered_results_path, --ori_feat_path, and --rendered_root to valid MorphoSim paths")

    image_path = os.path.join(args.rendered_root, "final_viz", "training_moving_rgb", "0010.jpg")
    image = plt.imread(image_path)
    
    plt.imshow(image)
    plt.show()
    response = gptv_reflection_prompt_selectbest(None, image_path, args)
    print('response:', response)
    # response = response.split('(')[-1].split(')')[0]
    # print(response.split(',')[0].toint())
    # print(response.split(',')[1].toint())

    plt.imshow(image)
    plt.show()

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

    # data_dir = "/home/shijie/Desktop/work/feature-4dgs/data/davis_dev/lucia/code_output"

    sam2_path = args.ori_feat_path
    sam2_gt_feats = torch.load(sam2_path,weights_only=True)
    vision_pos_enc = sam2_gt_feats[0]['vision_pos_enc']
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

    with open(args.head_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    head_config = config["Head"]
    feature_heads = Feature_heads(head_config).to("cuda")
    state_dict = torch.load(args.semantic_head_path,weights_only=True)
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
        sam2_rendered_feat[frame_idx]['vision_pos_enc'] = [vision_pos_enc]

    rgb_feats = torch.zeros(len(sam2_rendered_feat), 3, 480, 480)

    inference_state = predictor.novel_view_inference_init( sam2_feats=sam2_rendered_feat, rgb_feats=rgb_feats)

    #video_dir = os.path.join(args.rendered_root, "final_viz/training_moving_rgb")
    video_dir = os.path.join(args.rendered_root, "final_viz/40_round_freezing_rgb")

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    ############################################################################################

    ann_frame_idx = 0  # the frame index we interact with

    #points = np.array([[200, 250]], dtype=np.float32)
    points = np.array([[200, 200]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
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

    ############################################################################################

    #run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results

    first_frame_pix_feats = [obj['cond_frame_outputs'][0]["pix_feats"].detach().cpu() for obj in inference_state['temp_output_dict_per_obj'].values() ]
    first_frame_pix_feats = torch.cat(first_frame_pix_feats, dim=0)
    all_pix_feats = {0: first_frame_pix_feats}

    #all_pix_feats={}
    for out_frame_idx, out_obj_ids, out_mask_logits, pix_feats in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if out_frame_idx not in all_pix_feats:
            all_pix_feats[out_frame_idx] = pix_feats.detach().cpu()

    all_pix_feats=[all_pix_feats[i] for i in range(len(all_pix_feats))]
    all_pix_feats=torch.cat(all_pix_feats, dim=0)

    print(all_pix_feats.shape)

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")
    viz_results = []

    for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
        fig = plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        # take as image
        plt.savefig(f"frame_{out_frame_idx}.jpg")
        with open(f"frame_{out_frame_idx}.jpg", "rb") as f:
            img = Image.open(f).convert("RGB")
        os.remove(f"frame_{out_frame_idx}.jpg")

        viz_results.append(img)
        plt.close()

    # save as video
    video_save_path = os.path.join(args.rendered_root, "sam2_agent_semantic_segmentation.mp4")
    imageio.mimsave(video_save_path, viz_results)
    print(f"video saved to {video_save_path}")