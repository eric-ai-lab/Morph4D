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
if [ ${resume_ckpt} == "None" ]; then
    add_cmd=""
else
  add_cmd=" --sta_scf_dir ${resume_ckpt} --sta_gs_dir ${resume_ckpt} --dyn_scf_dir ${resume_ckpt}"
fi

main_cmd="
python run.py --lr_semantic_feature ${lr_semantic_feature} --feature_config configs/${mode}.yaml --config ./configs/wild/davis.yaml --src ${dataset_name} --comment ${mode}
"

main_cmd=${main_cmd}${add_cmd}

echo ${main_cmd}

# EXP_LOG_DIR=$(${main_cmd} | grep '^EXP_LOG_DIR:' | sed 's/^EXP_LOG_DIR://')