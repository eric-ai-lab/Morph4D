<div align="center">

<h3>MorphoSim: An Interactive, Controllable, and Editable Language-guided 4D World Simulator</h3>

👤 Xuehai He* · Shijie Zhou* · Thivyanth Venkateswaran · Kaizhi Zheng · Ziyu Wan · Achuta Kadambi · Xin Eric Wang

🌐 <a href="https://morphosim.github.io/">Project Website</a> &nbsp;|&nbsp; 🔗 <a href="https://arxiv.org/abs/2510.04390">arXiv</a> &nbsp;|&nbsp; 🎬 <a href="#demo-video">Demo Video</a>

</div>



# TODO
- [ ] Gradio Demo
- [ ] Huggingface setup
- [ ] Release training code
- [ ] Release inference code
- [ ] Release demo


# Environment Setup

```
bash env.sh
conda activate morphosim
```

# Download Data
make weights folder under root:

```
mkdir weights
cd weights
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
mv models raft_models
mv models.zip raft_models.zip
```
weights folder should be structured as such:
```
weights
├── dpt_large-midas-2f21e586.pt
└── raft_models
    ├── raft-chairs.pth
    ├── raft-kitti.pth
    ├── raft-sintel.pth
    ├── raft-small.pth
    └── raft-things.pth
```

nvidia dataset download command:
```
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/data.zip
```
nvidia gt download command:
```
wget --no-check-certificate https://huggingface.co/datasets/rhfeiyang/feature-4dgs_resources/resolve/main/nvidia_gt.zip
```

dropbox download link to DAVIS dataset:
```
https://drive.google.com/file/d/1vtG3gVT38HII-6n62ok5TTjruNFEP3vk/view?usp=sharing
```
structure data folder as follows:
```
data
├── davis_dev
│   ├── bear
│   ├── blackswan
│   ├── bmx-bumps
│   ├── bmx-trees
│   ├── boat
│   ├── breakdance
│   ├── breakdance-flare
│   ├── bus
│   ├── camel
...
├── nvidia_dev
│   │── gt
│   ├── Balloon1
│   ├── Balloon2
│   ├── Jumping
│   ├── Playground
│   ├── Skating
│   ├── Truck
│   ├── Umbrella
```


### Debugging
if after environment setup you encounter issues with numpy version 2.x.x in vid24d, run:

```
pip uninstall numpy
```
until all numpy versions are uninstalled in vid24d, then run:
```
pip install numpy==1.26.4
```

Pip version >= 24.2  do not support our version of ninja and decord. Downgrade pip to version 24.1.2
```
python -m pip install pip==24.1.2
```

`co-tracker version issue! Code will automatically download main branch but we expect 2v1_release branch! Not fixed!`Shijie: fixed!

## Checkpoint preparation
### LSeg model preparation
Download the LSeg model file `demo_e200.ckpt` using gdown and place it under the folder: `lseg_encoder`.
```
gdown 1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb
mv demo_e200.ckpt lseg_encoder
```





# Run training and inference

Run this script, from all the following individual steps for InternVideo2, SAM2, and LSeg feature extraction to training (run.py) to final visualization (viz.py).

```
bash run_all.sh
```

## E2E training + inference

Generate an editable 4D scene with prompt "A teal robot is cooking food in a kitchen with steam rising from pots"
```
bash run.sh e2e
```

## First prepare all the 2D priors for a certain dataset

```
python prepare.py --config ./configs/wild/prepare_davis.yaml --src ./data/davis_dev/train
```
or
```
python prepare.py --config ./configs/nvidia/prepare_nvidia.yaml --src ./data/nvidia_dev/Balloon1
```

## Run guided 2d video generation
Reffering to code here:
https://github.com/eric-ai-lab/Mojito


## Run InternVideo2 semantic feature extraction

```
cd internvideo_chat_feature

python internvideo_extract_feat.py --video_path <absolute_path_to_dataset>

cd ..
```
As <absolute_path_to_dataset> pass the path directly to the video.mp4 file and it will generate a file named "video_feat.pth" in the same directory.

OR pass in the path to a directory containing "images" folder, it will extract features from all the images in the folder.
Additionally, a file named "video.mp4" will be generated in the same directory for previewing. 

example: ".../data/davis_dev/train/code_output"


## Run SAM2 semantic feature extraction 

```
cd sam2
```

Download the checkpoints:
```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```
If this doesn't work when you are using docker on Windows:
```
cd checkpoints

sed -i 's/\r$//' download_ckpts.sh

bash download_ckpts.sh

cd ..
```

Feature extraction: 

```
cd sam2

python sam2_extract_feat.py --video_path <absolute_path_to_dataset>

cd ..
```
## Run Lang-seg semantic feature extraction

Install:
Download the LSeg model file `demo_e200.ckpt` from [the Google drive](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing) and place it under the folder: `lseg_encoder`.
```
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
pip install pytorch-lightning==2.4.0
pip install lightning
pip install git+https://github.com/openai/CLIP.git
```

```
cd lseg_encoder

python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ../data/DATASET_NAME/rgb_feature_langseg --test-rgb-dir ../data/DATASET_NAME/images --workers 0

cd ..
```
If you run into errors try specifying `CUDA_VISIBLE_DEVICES=0`

## Compile semantic feature rasterizer

* see `lib_4d/render_helper.py`, `lib_render/gs3d/diff_gaussian_rasterization-alphadep-new_feature`, `lib_render/gs3d/gauspl_renderer_native.py`

validated forward rendering is working, by randomly inited feature
See `TODO` in `lib_4d/prior2d_utils.py` 
  lib_4d/prior2d_utils.py
### Install customized package:
```
pip install lib_render/gs3d/diff_gaussian_rasterization-alphadep-new_feature
```

## Extract features together
``` Davis:
bash scripts/preprocess/preprocess_davis.sh <data_root>
```
``` Nvidia:
bash scripts/preprocess/preprocess_nvidia.sh <data_root>
```

## Then run MorphoSim reconstruction

```
python run.py --config ./configs/wild/davis.yaml --src ./data/davis_dev/train
```
or
```
python run.py --config ./configs/nvidia/nvidia.yaml --src ./data/nvidia_dev/Balloon1
```

see `scripts` for more commands

Support two GS rendering backend, set env var `GS_BACKEND` to `native` or `gof`, default is `native`

Visualize: see `viz_main` called in `run.py`

### Run together
See `scripts/run_davis.sh` and `scripts/run_nvidia.sh`, pass in the parameters like dataset_name, feat_dim


# Final Visualization
```
python viz.py --config ./configs/wild/davis.yaml --root <path_to_log_folder> --save <output_folder>
```
example:

```
python viz.py --config ./configs/wild/davis.yaml --root data/davis_dev/soccerball/code_output/log/native_feat... --save data/davis_dev/soccerball/final_viz
```

# Inference
## InternVideo2 Q&A
```
python inference_rendered_result.py 
```
## SAM2 Segmentation
```
cd sam2
python sam2_segmentation.py
# or
python sam2_test_inference.py
# or
python sam2_test_inference_agent.py
cd ..
```
## LangSeg
```
cd lseg_encoder
python lseg_test_inference.py
cd ..
```
## Editing
Color editing
``` 
python viz_agent.py --config ./configs/wild/davis.yaml --root output/cows/32_channel --user_prompt 'Make the cow purple'
```
* 'Make the cow more/less [color]'
* 'Make the cow black and white'
* 'Make the cow more/less saturated'
* 'Increase/decrease the brightness of the cow'

Object removing
```
python viz_agent.py --config ./configs/wild/davis.yaml --root output/cows/32_channel --user_prompt 'delete the cow'
```

Object extracting
```
python viz_agent.py --config ./configs/wild/davis.yaml --root output/cows/32_channel --user_prompt 'extract the cow'
```




## Demo video

<video controls width="720" muted playsinline poster="morphosim_thumbnail.jpg">
  <source src="MorphoSim.mp4" type="video/mp4">
  <a href="MorphoSim.mp4">direct link</a>.
</video>


## Citation
```bibtex
@misc{he2025morphosiminteractivecontrollableeditable,
      title={MorphoSim: An Interactive, Controllable, and Editable Language-guided 4D World Simulator}, 
      author={Xuehai He and Shijie Zhou and Thivyanth Venkateswaran and Kaizhi Zheng and Ziyu Wan and Achuta Kadambi and Xin Eric Wang},
      year={2025},
      eprint={2510.04390},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.04390}, 
}
```
