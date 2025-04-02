conda create -n morph4d python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate morph4d

which python
which pip

# conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# conda install xformers -c xformers -y # for efficient transformer in diffusion guidance
pip3 install -U xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
pip install ninja
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -r requirements.txt
# conda install pyg -c pyg -y
# conda install pytorch-cluster -c pyg -y
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu124.html ### sz
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html ### sz
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu124.html ### sz
pip install torch_geometric

# install gaussian splatting
pip install lib_render/gs3d/simple-knn

# * install GOF
pip install lib_render/gs3d/gof-diff-gaussian-rasterization
# * install old GS renderor
# conda install -c conda-forge glm -y
# export CFLAGS="-I/$CONDA_PREFIX/include"
pip install lib_render/gs3d/diff_gaussian_rasterization-alphadep-new_feature
# --global-option=build_ext  --global-option="-I/$CONDA_PREFIX/include"

# # install NVDiffRast
# pip install git+https://github.com/NVlabs/nvdiffrast/
# pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch

# pip install -e lib_guidance/mvdream/extern/MVDream

# for later depth anything
pip install git+https://github.com/huggingface/transformers.git

# eval
pip install lpips pytorch_msssim

# # for dot
# pip install tqdm matplotlib einops einshape scipy timm lmdb av mediapy
pip install matplotlib==3.9.2

# for featup
# pip install git+https://github.com/openai/CLIP.git
# pip install git+https://github.com/mhapip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu124.htmlmilton723/CLIP.git
# pip install git+https://github.com/mhamilton723/FeatUp

# # spatracker
# pip install flow_vis  cupy-cuda11x

# for evaluate the iphone dataset with JAX from the uiuc generalizable paper
# pip install -r jax_requirements.txt

# * LR feature
# conda install -c "nvidia/label/cuda-11.8.0" libcusolver-dev -y
conda install -c "nvidia/label/cuda-12.4" libcusolver-dev -y

# Install cuDNN for transformer_engine
conda install -c conda-forge cudnn -y

# Install Transformer Engine (with error handling)
pip install transformer_engine[pytorch]==1.11 || echo "Transformer Engine installation failed, continuing anyway"

# Install other packages for Cosmos
pip install av \
    better-profanity \
    einops==0.7.0 \
    einx==0.1.3 \
    huggingface-hub>=0.26.2 \
    hydra-core \
    imageio[ffmpeg] \
    iopath \
    loguru \
    mediapy \
    nltk \
    peft \
    pillow \
    sentencepiece \
    termcolor \
    git+https://github.com/NVlabs/Pytorch_Retinaface.git@b843f45
    
pip install mistral_inference

pip install --extra-index-url https://pypi.nvidia.com cuml-cu12

pip install viser

pip install roma

pip install cupy-cuda12x

pip install opt_einsum

pip install ml_dtypes

pip install protobuf

pip install 'flash-attn<2.5.7'

# for lseg
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
pip install pytorch-lightning==2.4.0
pip install lightning
pip install git+https://github.com/openai/CLIP.git
