# Authors: Hui Ren (rhfeiyang.github.io)
import sys
sys.path.append("internvideo_chat_feature")
import argparse
import torch
import numpy as np
from torch.nn import functional as F
from transformers import AutoTokenizer
from internvideo_chat_feature.modeling_videochat2 import InternVideo2_VideoChat2
import yaml
import os
from lib_4d.autoencoder.model import Feature_heads
import time ### timer

def get_indices_split(num_frames, num_segments):
    # results = set()
    seg_size = num_frames // num_segments
    offsets_first = np.array([int(seg_size * i) for i in range(num_segments)])
    offsets = [i + offsets_first for i in range(seg_size)]
    if offsets[-1][-1] < num_frames-1:
        new_offset_reverse = [num_frames-1-seg_size*i for i in range(num_segments)]
        offsets.append(np.array(new_offset_reverse[::-1]))
    return offsets

def parse_args():
    parser = argparse.ArgumentParser(description='inference with reconstructed feat of internvideo')
    parser.add_argument('--rendered_results_path', type=str, default=os.path.join('output','bmx-bumps','32_channel','final_viz','3D_moving','rendered_results.pth'), help='path to the result')
    parser.add_argument("--ori_feat_path", type=str,default=os.path.join('data','davis_dev','bmx-bumps','preprocess','semantic_features','internvideo_feats.pth'),)
    # parser.add_argument("--semantic_head_path", type=str, default="/home/shijie/Desktop/work/feature-4dgs/data/davis_dev/flamingo/code_output/log/native_feat_davis.yaml_compactgs_mixfeat_nomotion_channel64_dep=uni_gt_cam=False_lrfeat=0.01_20241106_094007/finetune_semantic_heads.pth")
    parser.add_argument("--head_config", type=str, default="configs/default_config.yaml")
    args =  parser.parse_args()
    args.rendered_root = os.path.dirname(os.path.dirname(os.path.dirname(args.rendered_results_path)))
    args.semantic_head_path = os.path.join(args.rendered_root,"finetune_semantic_heads.pth")
    return args


@torch.no_grad()
def main(args):
    access_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN') or ""
    rendered_results = torch.load(args.rendered_results_path,weights_only=True)
    ori_feats = torch.load(args.ori_feat_path,weights_only=True)
    cls_feat = ori_feats["cls_feat"].view(1,1,1408)
    print(rendered_results.keys())

    ###

    with open(args.head_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    head_config = config["Head"]
    feature_heads = Feature_heads(head_config).to("cuda")
    state_dict = torch.load(args.semantic_head_path,weights_only=True)
    feature_heads.load_state_dict(state_dict)
    feature_heads.eval()
    ###
    all_video_feat = []
    for vid, result in rendered_results.items():
        # first resize to 16*16
        feat = F.interpolate(result["feature_map"][None, ...], size=(16, 16), mode='area')[0]
        feat = feat.permute(1,2,0)
        feat = feature_heads.decode("internvideo" , feat)
        all_video_feat.append(feat)
    all_video_feat = torch.stack(all_video_feat)
    split = get_indices_split(all_video_feat.size(0), 8)[0]
    sampled_video_feat = all_video_feat[split]
    # sampled_video_feat = ori_feats['video_feat'][split]

    sampled_video_feat_flatten = sampled_video_feat.view(1,-1,1408)

    gathered_result = torch.cat([cls_feat.cuda(), sampled_video_feat_flatten.cuda()], dim=1)

    model = InternVideo2_VideoChat2.from_pretrained("OpenGVLab/InternVideo2-Chat-8B",token=access_token).cuda()

    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)
    ### timer
    start_time = time.time()
    ###
    chat_history= []
    response, chat_history = model.chat(tokenizer, '', 'Give a detailed description of the video.',
                                        media_type='video', media_tensor=None, chat_history= chat_history,
                                        return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'What direction is the cow moving towards?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'Which way is the cow facing?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'Is the cow moving to the left or right?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'Is the cow facing to the left or right?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'Is the cow moving to the left?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'Is the cow moving to the right?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'From the camera perspective, is the cow moving to the left?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'From the camera perspective, is the cow moving to the right?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'Is the boy moving uphill or downhill?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'Is the boy pedalling his bike?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'How many bike riders are there?', 
    #                                     media_type='video', media_tensor=None, chat_history= chat_history, 
    #                                     return_history=True,generation_config={'do_sample':False, "feat":gathered_result})
    # print(response)

    ### timer
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.6f} seconds")
    ###


if __name__ == "__main__":
    args = parse_args()
    main(args)
