DATA_NAME=${DATA_NAME:-train}
DATA_ROOT=${DATA_ROOT:-./data/davis_dev}
DATA_PATH="$DATA_ROOT/$DATA_NAME"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python prepare.py --config ./configs/wild/prepare_davis.yaml --src $DATA_PATH

cd internvideo_chat_feature
python internvideo_extract_feat.py --video_path $DATA_PATH/preprocess
cd ..

cd sam2
python sam2_extract_feat.py --video_path $DATA_PATH/preprocess
cd ..

cd lseg_encoder
CUDA_VISIBLE_DEVICES=2 python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir $DATA_PATH/preprocess/semantic_features/rgb_feature_langseg --test-rgb-dir $DATA_PATH/preprocess/images --workers 0
cd ..

CUDA_VISIBLE_DEVICES=2 python run.py --config ./configs/wild/davis.yaml --src $DATA_PATH
