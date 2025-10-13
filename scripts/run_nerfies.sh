dataset_name=${1}
mode=${2:-"mlp_all_32"}
lr_semantic_feature=0.01
feat_dim=${3:-"32"}
resume_ckpt=${4:-None}
export GS_BACKEND=native_feat${feat_dim}
export MKL_SERVICE_FORCE_INTEL=1
# Optional: define in env/.env if needed
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
: "${WANDB_API_KEY:=}"
: "${HF_TOKEN:=}"
# change to your own path

data_root=/public/home/renhui/code/4d/feature-4dgs/data/nerfies/${dataset_name}/left1
gt_root=/public/home/renhui/code/4d/feature-4dgs/data/nerfies/${dataset_name}/right1
if [ ! -d ${gt_root}/preprocess/semantic_features/rgb_feature_langseg ]; then
    echo "not exist ${gt_root}/preprocess/semantic_features/rgb_feature_langseg"
    exit 1
fi

if [ ${resume_ckpt} == "None" ]; then
    add_cmd=""
else
  add_cmd=" --sta_scf_dir ${resume_ckpt} --sta_gs_dir ${resume_ckpt} --dyn_scf_dir ${resume_ckpt}"
fi

main_cmd="
python run.py --lr_semantic_feature ${lr_semantic_feature} --feature_config configs/${mode}.yaml --config ./configs/nerfies/nerfies.yaml --src ${data_root} --comment ${mode} --gt_cam --save_dir output/nerfies/${dataset_name}
"

main_cmd=${main_cmd}${add_cmd}

echo ${main_cmd}

EXP_LOG_DIR=$(${main_cmd} | grep '^EXP_LOG_DIR:' | sed 's/^EXP_LOG_DIR://')

# if success, run the following command
if [ $? -eq 0 ]; then
    echo "run success"
    exp_root=${EXP_LOG_DIR}
    echo "exp_root: ${exp_root}"
    if [ ! -d ${exp_root}/tto_test_lseg ]; then
        echo "not exist ${exp_root}/tto_test_lseg"
    else
        echo "start to evaluate"
        cd lseg_encoder
        python segmentation_metric.py \
        --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv \
        --student-feature-dir ${exp_root}/tto_test_lseg  \
        --teacher-feature-dir ${gt_root}/preprocess/semantic_features/rgb_feature_langseg \
        --test-rgb-dir ${gt_root}/images --workers 0 \
        --eval-mode test
        cd ..
    fi
else
    echo "run failed"
fi