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

### Environment (CUDA 11.8 example)

```bash
# Conda (recommended)
conda create -n morphosim python=3.10 -y
conda activate morphosim

# PyTorch + CUDA 11.8
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip3 install -U xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu118
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Project deps
python -m pip install pip==24.1.2
pip install -r requirements.txt

# Optional extra deps used in scripts
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
pip install torch_geometric
```

### Configure credentials and paths

- Create a `.env` file in the repo root and set tokens as needed:
  - `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` for Hugging Face models
  - `OPENROUTER_API_KEY` (if using OpenRouter)
  - Adjust `DATA_ROOT` and `OUTPUT_ROOT` if your data/output locations differ

### Data layout

```
./data
└── davis_dev
    └── train
        └── preprocess
            ├── images
            └── semantic_features
```

### Prepare Features

```bash
# Prepare dataset
python prepare.py --config ./configs/wild/prepare_davis.yaml --src ./data/davis_dev/train

# Extract features
python internvideo_chat_feature/internvideo_extract_feat.py --video_path ./data/davis_dev/train/preprocess
python sam2/sam2_extract_feat.py --video_path ./data/davis_dev/train/preprocess
cd lseg_encoder && \
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt \
  --widehead --no-scaleinv \
  --outdir ./data/davis_dev/train/preprocess/semantic_features/rgb_feature_langseg \
  --test-rgb-dir ./data/davis_dev/train/preprocess/images --workers 0 && cd ..

# Train and visualize
python run.py --config ./configs/wild/davis.yaml --src ./data/davis_dev/train
python viz.py --config ./configs/wild/davis.yaml --root ./output/train/32_channel/final_viz --save ./output/train/final_viz

```

### Run directional guided video generation first
Reffering to code here:
https://github.com/eric-ai-lab/Mojito


### Editing with viz_agent.py (three edit types)

`viz_agent.py` supports three operations via natural language prompts:
- Color change (color_func)
- Deletion (remove an object)
- Extraction (isolate an object)

General form:
```bash
python viz_agent.py \
  --config ./configs/wild/davis.yaml \
  --root ./output/<scene>/32_channel/final_viz \
  --user_prompt "<instruction>"
```

- Color change example:
```bash
python viz_agent.py --config ./configs/wild/davis.yaml \
  --root ./output/cows/32_channel/final_viz \
  --user_prompt "change the cow color to purple"
```

- Deletion example:
```bash
python viz_agent.py --config ./configs/wild/davis.yaml \
  --root ./output/cows/32_channel/final_viz \
  --user_prompt "delete the cow"
```

- Extraction example:
```bash
python viz_agent.py --config ./configs/wild/davis.yaml \
  --root ./output/cows/32_channel/final_viz \
  --user_prompt "extract the cow"
```

Notes:
- Set `OPENROUTER_API_KEY` in your environment if using the default `openrouter` provider, or configure Azure variables if using `--api xh-gpt4.1`.
- Outputs are saved under `--output_root` (default `./output`) in a subfolder named `agentic_edit`.

### Individual steps

- Prepare dataset:
```bash
python prepare.py --config ./configs/wild/prepare_davis.yaml --src ./data/davis_dev/train
```




- InternVideo2 features:
```bash
cd internvideo_chat_feature
python internvideo_extract_feat.py --video_path ./data/davis_dev/train/preprocess
cd ..
```
- SAM2 features:
```bash
cd sam2
python sam2_extract_feat.py --video_path ./data/davis_dev/train/preprocess
cd ..
```
- LangSeg features:
```bash
cd lseg_encoder
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt \
  --widehead --no-scaleinv \
  --outdir ./data/davis_dev/train/preprocess/semantic_features/rgb_feature_langseg \
  --test-rgb-dir ./data/davis_dev/train/preprocess/images --workers 0
cd ..
```
- Train and visualize:
```bash
python run.py --config ./configs/wild/davis.yaml --src ./data/davis_dev/train
python viz.py --config ./configs/wild/davis.yaml --root ./output/train/32_channel/final_viz --save ./output/train/final_viz
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
