# Authors: Hui Ren (rhfeiyang.github.io)
import os
import sys
os.chdir(os.path.dirname(__file__))
token = os.environ['HF_TOKEN']
import torch
from transformers import AutoTokenizer, AutoModel
# from modeling_videochat2 import InternVideo2_VideoChat2
# os.environ["http_proxy"] = "http://localhost:7895"
# os.environ["https_proxy"] = "http://localhost:7895"

from modeling_videochat2 import InternVideo2_VideoChat2
from transformers import AutoTokenizer, AutoModel, AutoConfig


model = InternVideo2_VideoChat2.from_pretrained("OpenGVLab/InternVideo2-Chat-8B",).cuda()



tokenizer =  AutoTokenizer.from_pretrained("OpenGVLab/InternVideo2-Chat-8B", trust_remote_code=True, use_fast=False,)


# config = AutoConfig.from_pretrained(
#     './',
#     trust_remote_code=True,
# )

# model = InternVideo2_VideoChat2(config=config).cuda()



# model = AutoModel.from_pretrained(
#     './',
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     cache_dir="./",
#     local_files_only=True,
# ).cuda()
#get encoder
encoder = model.vision_encoder.cuda()


import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
decord.bridge.set_bridge("torch")

def get_index(num_frames, num_segments, start=None):
    seg_size = float(num_frames - 1) / num_segments
    if start is None:
        start = int(seg_size / 2)

    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def get_indices_split(num_frames, num_segments):
    # results = set()
    seg_size = num_frames // num_segments
    offsets_first = np.array([int(seg_size * i) for i in range(num_segments)])
    offsets = [i + offsets_first for i in range(seg_size)]
    if offsets[-1][-1] < num_frames-1:
        new_offset_reverse = [num_frames-1-seg_size*i for i in range(num_segments)]
        offsets.append(np.array(new_offset_reverse[::-1]))

    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    # frame_indices = get_index(num_frames, num_segments, start=0)
    frame_indices_list = get_indices_split(num_frames, num_segments)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])
    frames_list=[]
    for frame_indices in frame_indices_list:
        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)
        frames = transform(frames)
        frames_list.append((frames, frame_indices))

    T_, C, H, W = frames_list[0][0].shape

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames_list, msg, num_frames
    else:
        return frames_list, num_frames

video_path = "yoga.mp4"
# sample uniformly 8 frames from the video
video_tensor_list, num_frames = load_video(video_path, num_segments=8, return_msg=False)

video_feat_all = torch.zeros(1, num_frames, 16, 16, 1408, dtype=torch.bfloat16).cuda()
video_feat_list=[]
cls_feat_list=[]
for data in video_tensor_list:
    video_tensor, frame_id = data
    frame_id = torch.tensor(frame_id)
    video_tensor = video_tensor.to("cuda")

    ## sanity check
    # chat_history= []
    # response, chat_history = model.chat(tokenizer, '', 'describe the action step by step.', media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False, "feat":None})
    # print(response)
    # # The video shows a woman performing yoga on a rooftop with a beautiful view of the mountains in the background. She starts by standing on her hands and knees, then moves into a downward dog position, and finally ends with a standing position. Throughout the video, she maintains a steady and fluid movement, focusing on her breathing and alignment. The video is a great example of how yoga can be practiced in different environments and how it can be a great way to connect with nature and find inner peace.



    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    # specify ret_map=True to get the feat
    vid_feat, cls_feat=encoder(video_tensor, ret_map=True)


    ## test: restore feature
    # chat_history= []
    # vid_feat_flatten = vid_feat.view(vid_feat.size(0), -1, 1408)
    # restore_feat = torch.cat([cls_feat,vid_feat_flatten], dim=1)
    # response, chat_history = model.chat(tokenizer, '', 'describe the action step by step.', media_type='video', media_tensor=None, chat_history= chat_history, return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
    # The video shows a woman performing yoga on a rooftop with a beautiful view of the mountains in the background. She starts by standing on her hands and knees, then moves into a downward dog position, and finally ends with a standing position. Throughout the video, she maintains a steady and fluid movement, focusing on her breathing and alignment. The video is a great example of how yoga can be practiced in different environments and how it can be a great way to connect with nature and find inner peace.


    video_feat_all[:, frame_id] = vid_feat
    cls_feat_list.append(cls_feat)

cls_feat_avg = torch.stack(cls_feat_list).mean(dim=0).unsqueeze(0)

result = {
    "video_feat": video_feat_all.cpu(),
    "cls_feat": cls_feat_avg,
}
torch.save(result, "yoga_feat.pth")



# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'describe the action step by step.', media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
# print(response)
# # The video shows a woman performing yoga on a rooftop with a beautiful view of the mountains in the background. She starts by standing on her hands and knees, then moves into a downward dog position, and finally ends with a standing position. Throughout the video, she maintains a steady and fluid movement, focusing on her breath and alignment. The video is a great example of how yoga can be practiced in different environments and how it can be a great way to connect with nature and find inner peace.
# # The woman in the video is performing a yoga pose on a rooftop. She starts by standing on her hands and knees, then she moves her legs to the side and stretches her arms out. She then moves her arms to the side and stretches her legs out. Finally, she moves her arms to the side and stretches her legs out again.
#
# response, chat_history = model.chat(tokenizer, '', 'What is she wearing?', media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
# # The woman in the video is wearing a black tank top and grey yoga pants.
# print(response)