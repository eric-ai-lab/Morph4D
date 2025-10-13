import torch
from modeling_videochat2 import InternVideo2_VideoChat2
import numpy as np
import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge("torch")
from torchvision import transforms
#import tokenizer
from transformers import AutoTokenizer
import time ### timer

access_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN') or ""

tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)
model = InternVideo2_VideoChat2.from_pretrained("OpenGVLab/InternVideo2-Chat-8B",token=access_token).cuda()

encoder = model.vision_encoder.cuda()

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

#video_path = "./output/blackswan/32_channel/final_viz/25_fixed_moving_rgb.mp4"
video_path = "./output/bmx-bumps/32_channel/final_viz/3D_moving_rgb.mp4"
#video_path = "./data/davis_dev/blackswan/preprocess/video.mp4"

# sample uniformly 8 frames from the video
video_tensor_list, num_frames = load_video(video_path, num_segments=33, return_msg=False)
### timer
start_time = time.time()
###
print("num_frames:", num_frames)
print("video_tensor_list", len(video_tensor_list))

video_feat_all = torch.zeros(1, num_frames, 16, 16, 1408, dtype=torch.bfloat16).cuda()
video_feat_list=[]
cls_feat_list=[]
for data in video_tensor_list:
    video_tensor, frame_id = data
    frame_id = torch.tensor(frame_id)
    video_tensor = video_tensor.to("cuda")
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    # specify ret_map=True to get the feat
    vid_feat, cls_feat=encoder(video_tensor, ret_map=True)
    
    # gather the features to form all frames
    vid_feat = vid_feat.to(video_feat_all.dtype)
    video_feat_all[:, frame_id] = vid_feat
    cls_feat_list.append(cls_feat)

# cls_feat_avg = torch.stack(cls_feat_list).mean(dim=0).unsqueeze(0)

# result = {
#     "video_feat": video_feat_all.cpu(),
#     "cls_feat": cls_feat_avg,
# }
#torch.save(result, "tmp_video_feat.pth")

# just taking the last one feat, for convenient test - Hui
chat_history= []
vid_feat_flatten = vid_feat.view(vid_feat.size(0), -1, 1408)
restore_feat = torch.cat([cls_feat,vid_feat_flatten], dim=1) # 1 2049 1408

# just pass in "feat":restore_feat in generation_config
# response, chat_history = model.chat(tokenizer, '', 'describe the action step by step.', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})

response, chat_history = model.chat(tokenizer, '', 'Give a detailed description of the video.', 
                                    media_type='video', media_tensor=None, chat_history= chat_history, 
                                    return_history=True,generation_config={'do_sample':False, "feat":restore_feat})

print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'What direction is the cow moving towards?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'Which way is the cow facing?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'is the cow moving to the left or right?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'is the cow facing to the left or right?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'Is the cow moving to the left?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'Is the cow moving to the right?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'From the camera perspective, is the cow moving to the left?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'From the camera perspective, is the cow moving to the right?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'Is the boy moving uphill or downhill?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'Is the boy pedalling his bike?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

# chat_history= []
# response, chat_history = model.chat(tokenizer, '', 'How many bike riders are there?', 
#                                     media_type='video', media_tensor=None, chat_history= chat_history, 
#                                     return_history=True,generation_config={'do_sample':False, "feat":restore_feat})
# print(response)

### timer
end_time = time.time()
print(f"Execution Time: {end_time - start_time:.6f} seconds")
###