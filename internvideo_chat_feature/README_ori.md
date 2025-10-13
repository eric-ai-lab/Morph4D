---
license: mit
pipeline_tag: video-classification
extra_gated_prompt: >-
  You agree to not use the model to conduct experiments that cause harm to human
  subjects.
extra_gated_fields:
  Name: text
  Company/Organization: text
  Country: text
  E-Mail: text
language:
- en
tags:
- video
---

# InternVideo2-Chat-8B

[\[📂 GitHub\]](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2)   [\[📜 Tech Report\]](https://arxiv.org/abs/2403.15377) [\[🗨️ Chat Demo\]](https://vchat.opengvlab.com/)  

To further enrich the semantics embedded in **InternVideo2** and improve its user-friendly in human communications, we tune InternVideo2 by incorporating it into a VideoLLM with a LLM and a video BLIP. We employ the progressive learning scheme in [VideoChat](https://arxiv.org/abs/2311.17005) by using InternVideo2 as the video encoder and train a video blip for
communicating with open-sourced LLM. In training, the video encoder will be updated. Detailed training recipts are in [VideoChat](https://arxiv.org/abs/2311.17005).

The BaseLLM of this model is Mistral-7B.**Before using it, please ensure that you have obtained the access permission of Mistral-7B**, if not yet obtained, please go to[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) to obtain the access permission and add your `HF_token` to the environment variable. 

## 📈 Performance
| Model |  MVBench | VideoMME(w/o sub)| 
| ---   |  ---     |   ---            |
|[InternVideo2-Chat-8B](https://huggingface.co/OpenGVLab/InternVideo2-Chat-8B)| 60.3 | 41.9    |
|[InternVideo2-Chat-8B-HD](https://huggingface.co/OpenGVLab/InternVideo2_chat_8B_HD) | 65.4 | 46.1|
|InternVideo2-Chat-8B-HD-F16 | 67.5 | 49.4|
|[InternVideo2-Chat-8B-InternLM](https://huggingface.co/OpenGVLab/InternVideo2_Chat_8B_InternLM2_5)| 61.9| 49.1|

## 🚀 How to use the model

1. Apply for the permission of this project and the base LLM permission 

2. Fill the HF user access token into the environment variable

```shell
export HF_TOKEN=hf_....
```
If you don't know how to obtain the token starting with "hf_", please refer to: [How to Get HF User access Token](https://huggingface.co/docs/hub/security-tokens#user-access-tokens)

3. make sure to have `transformers >= 4.39.0, peft==0.5.0`

```
pip install transformers==4.39.1
pip install peft==0.5.0
pip install timm easydict einops
```

Install the requisite Python packages from [pip_requirements](https://huggingface.co/OpenGVLab/InternVideo2_chat_8B_HD/blob/main/requirements.txt) 
   
4. Inference with Video input

```Python
import os
token = os.environ['HF_TOKEN']
import torch

tokenizer =  AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)

from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained(
    'OpenGVLab/InternVideo2-Chat-8B',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).cuda()

from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
decord.bridge.set_bridge("torch")

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

video_path = "yoga.mp4"
# sample uniformly 8 frames from the video
video_tensor = load_video(video_path, num_segments=8, return_msg=False)
video_tensor = video_tensor.to(model.device)

chat_history= []
response, chat_history = model.chat(tokenizer, '', 'describe the action step by step.', media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
print(response)
# The video shows a woman performing yoga on a rooftop with a beautiful view of the mountains in the background. She starts by standing on her hands and knees, then moves into a downward dog position, and finally ends with a standing position. Throughout the video, she maintains a steady and fluid movement, focusing on her breath and alignment. The video is a great example of how yoga can be practiced in different environments and how it can be a great way to connect with nature and find inner peace.

response, chat_history = model.chat(tokenizer, '', 'What is she wearing?', media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
# The woman in the video is wearing a black tank top and grey yoga pants.
print(response)
```

## ✏️ Citation
If this work is helpful for your research, please consider citing InternVideo and VideoChat.

```
@article{wang2024internvideo2,
  title={Internvideo2: Scaling video foundation models for multimodal video understanding},
  author={Wang, Yi and Li, Kunchang and Li, Xinhao and Yu, Jiashuo and He, Yinan and Wang, Chenting and Chen, Guo and Pei, Baoqi and Zheng, Rongkun and Xu, Jilan and Wang, Zun and others},
  journal={arXiv preprint arXiv:2403.15377},
  year={2024}
}

@article{li2023videochat,
  title={Videochat: Chat-centric video understanding},
  author={Li, KunChang and He, Yinan and Wang, Yi and Li, Yizhuo and Wang, Wenhai and Luo, Ping and Wang, Yali and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2305.06355},
  year={2023}
}
```