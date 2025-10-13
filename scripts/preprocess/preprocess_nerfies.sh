data_root=$1

python prepare.py --config ./configs/nerfies/prepare_nerfies.yaml --src ${data_root}

cd internvideo_chat_feature
python internvideo_extract_feat.py --video_path ${data_root}/preprocess
cd ..

cd sam2
python sam2_extract_feat.py --video_path ${data_root}/preprocess
cd ..

cd lseg_encoder
CUDA_VISIBLE_DEVICES=0 python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ${data_root}/preprocess/semantic_features/rgb_feature_langseg --test-rgb-dir ${data_root}/preprocess/images --workers 0
cd ..