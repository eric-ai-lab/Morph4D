export GS_BACKEND=native_feat32
dataset_name=${1}
exp_root=${2}


data_root=/public/home/renhui/code/4d/feature-4dgs/output/${dataset_name}
gt_root=/public/home/renhui/code/4d/feature-4dgs/data/nvidia_dev/gt/${dataset_name}
python test.py --config ./configs/nvidia/nvidia.yaml --root ${exp_root} --src ${data_root} --tto

echo "start to evaluate lseg metric"
cd lseg_encoder
python segmentation_metric.py \
--backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv \
--student-feature-dir ${exp_root}/tto_test_lseg  \
--teacher-feature-dir ${gt_root}/preprocess/semantic_features/rgb_feature_langseg \
--test-rgb-dir ${gt_root} --workers 0 \
--eval-mode test
cd ..


 cd lseg_encoder
 python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ${exp_root}/tto_test_rgb_lseg --test-rgb-dir ${exp_root}/tto_test --workers 0
 cd ..

 echo "start to evaluate rgb lseg metric"
 cd lseg_encoder
 python segmentation_metric.py \
 --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv \
 --student-feature-dir ${exp_root}/tto_test_rgb_lseg  \
 --teacher-feature-dir ${gt_root}/preprocess/semantic_features/rgb_feature_langseg \
 --test-rgb-dir ${gt_root} --workers 0 \
 --eval-mode test
 cd ..



