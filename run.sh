#!/bin/bash



if [ -n "$USE_ENV" ]; then
    ENV_NAME=$USE_ENV
else
    ENV_NAME=feature4x
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Base log directory
LOG_BASE_DIR=$(readlink -f $(dirname $0))/logs
mkdir -p $LOG_BASE_DIR

# Function to run a command with logging in nested directories
function run_with_log() {
    local cmd="$1"
    local log_dir="$2"
    
    # Create nested log directory
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local full_log_dir="${LOG_BASE_DIR}/${log_dir}"
    mkdir -p $full_log_dir
    
    local log_file="${full_log_dir}/${timestamp}.log"
    
    echo "Running command: $cmd"
    echo "Logging to: $log_file"
    
    # Use script command to capture all terminal output
    script -q -c "$cmd" $log_file
    
    # Check exit status
    local status=$?
    if [ $status -ne 0 ]; then
        echo "Command failed with status $status. See log: $log_file"
        return $status
    fi
    
    echo "Command completed successfully. Log: $log_file"
    return 0
}

# Individual command functions
function run_prepare() {
    local DATA_PATH=$1
    echo "Running prepare.py step..."
    run_with_log "python prepare.py --config ./configs/wild/prepare_davis.yaml --src $DATA_PATH" "prepare"
}

function run_internvideo() {
    local DATA_PATH=$1
    echo "Running internvideo feature extraction..."
    cd internvideo_chat_feature
    run_with_log "python internvideo_extract_feat.py --video_path $DATA_PATH/preprocess" "internvideo"
    cd ..
}

function run_sam2() {
    local DATA_PATH=$1
    echo "Running SAM2 feature extraction..."
    cd sam2
    run_with_log "python sam2_extract_feat.py --video_path $DATA_PATH/preprocess" "sam2"
    cd ..
}

function run_lseg() {
    local DATA_PATH=$1
    echo "Running LSEG encoding..."
    cd lseg_encoder
    run_with_log "python encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir $DATA_PATH/preprocess/semantic_features/rgb_feature_langseg --test-rgb-dir $DATA_PATH/preprocess/images --workers 0" "lseg"
    cd ..
}

function run_final() {
    local DATA_PATH=$1
    echo "Running final step..."
    run_with_log "python run.py --config ./configs/wild/davis.yaml --src $DATA_PATH" "run"
}

# Main function to run all steps
function run_all() {
    DATA_NAME=train
    DATA_PATH=data/davis_dev/$DATA_NAME
    # Make DATA_PATH absolute
    DATA_PATH=$(realpath $DATA_PATH)
    
    # Main log file
    mkdir -p "${LOG_BASE_DIR}/run_all"
    LOG_FILE="${LOG_BASE_DIR}/run_all/$(date +"%Y%m%d_%H%M%S").log"
    echo "Starting run_all process at $(date)" | tee $LOG_FILE
    
    # Run each step, checking for errors after each
    run_prepare $DATA_PATH
    if [ $? -ne 0 ]; then
        echo "Prepare step failed, stopping pipeline" | tee -a $LOG_FILE
        return 1
    fi
    
    run_internvideo $DATA_PATH
    if [ $? -ne 0 ]; then
        echo "Internvideo step failed, stopping pipeline" | tee -a $LOG_FILE
        return 1
    fi
    
    run_sam2 $DATA_PATH
    if [ $? -ne 0 ]; then
        echo "SAM2 step failed, stopping pipeline" | tee -a $LOG_FILE
        return 1
    fi
    
    run_lseg $DATA_PATH
    if [ $? -ne 0 ]; then
        echo "LSEG step failed, stopping pipeline" | tee -a $LOG_FILE
        return 1
    fi
    
    run_final $DATA_PATH
    if [ $? -ne 0 ]; then
        echo "Final step failed" | tee -a $LOG_FILE
        return 1
    fi
    
    echo "All steps completed successfully at $(date)" | tee -a $LOG_FILE
    return 0
}

# Run individual steps
function prepare() {
    DATA_NAME=train
    DATA_PATH=data/davis_dev/$DATA_NAME
    # Make DATA_PATH absolute
    DATA_PATH=$(realpath $DATA_PATH)
    run_prepare $DATA_PATH
}

function internvideo() {
    DATA_NAME=train
    DATA_PATH=data/davis_dev/$DATA_NAME
    # Make DATA_PATH absolute
    DATA_PATH=$(realpath $DATA_PATH)
    run_internvideo $DATA_PATH
}

function sam2() {
    DATA_NAME=train
    DATA_PATH=data/davis_dev/$DATA_NAME
    # Make DATA_PATH absolute
    DATA_PATH=$(realpath $DATA_PATH)
    run_sam2 $DATA_PATH
}

function lseg() {
    DATA_NAME=train
    DATA_PATH=data/davis_dev/$DATA_NAME
    # Make DATA_PATH absolute
    DATA_PATH=$(realpath $DATA_PATH)
    run_lseg $DATA_PATH
}

function run() {
    DATA_NAME=train
    DATA_PATH=data/davis_dev/$DATA_NAME
    # Make DATA_PATH absolute
    DATA_PATH=$(realpath $DATA_PATH)
    run_final $DATA_PATH
}

function install() {
    run_with_log "bash install.sh" "install"
}

function env() {
    run_with_log "bash env.sh" "env"
}

# Display Python executable
which python

# GPU setup - check if gpu_selector is available
if python -c "import gpu_selector" &> /dev/null; then
    echo "GPU selector package found, using dynamic GPU selection"
    # Use gpu_selector to select GPUs
    function set_gpus() {
        export CUDA_VISIBLE_DEVICES=$(get_sorted_gpus 1)
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    }
    set_gpus 1
else
    echo "gpu_selector package not available, skipping dynamic GPU selection"
    # CUDA_VISIBLE_DEVICES is expected to be set before calling the script
fi

# Function to generate 4D scene from text prompt
function generate_4d_from_prompt() {
    local PROMPT=${1:-"A teal robot is cooking food in a kitchen with steam rising from pots"}
    local MOVE_ANGLE_DEG=${2:-10.0}
    local GPU_ID=${3:-0}
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Create unique identifier and output paths
    local TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    local RUN_ID="${TIMESTAMP}_$(echo "$PROMPT" | tr ' ' '_' | cut -c1-30)"
    local VIDEO_SAVE_NAME="cosmos_${RUN_ID}"
    local OUTPUT_DIR="output/cosmos_scenes"
    local DATA_DIR="data/prompt_generated/${VIDEO_SAVE_NAME}"
    
    # Get absolute paths
    local CURRENT_DIR=$(pwd)
    local COSMOS_DIR=$(realpath "../Cosmos")
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$DATA_DIR/preprocess/images"
    
    echo "Step 1/3: Generating video from prompt using Cosmos"
    echo "Prompt: $PROMPT"
    
    # Change to Cosmos directory and generate video
    cd "$COSMOS_DIR" || { echo "Cannot find Cosmos directory!"; return 1; }
    
    # Run Cosmos to generate video from prompt
    run_with_log "PYTHONPATH=$COSMOS_DIR python cosmos1/models/diffusion/inference/text2world.py \
        --checkpoint_dir checkpoints \
        --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Text2World \
        --prompt \"$PROMPT\" \
        --video_save_name \"$VIDEO_SAVE_NAME\" \
        --video_save_folder \"$CURRENT_DIR/$OUTPUT_DIR\" \
        --offload_tokenizer \
        --offload_diffusion_transformer \
        --offload_text_encoder_model \
        --offload_prompt_upsampler \
        --offload_guardrail_models \
        --height 512 \
        --width 896 \
        --num_video_frames 121 \
        --num_steps 25" "cosmos_generation"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate video with Cosmos"
        cd "$CURRENT_DIR"
        return 1
    fi
    
    # Check if video was generated
    VIDEO_PATH="$CURRENT_DIR/$OUTPUT_DIR/${VIDEO_SAVE_NAME}.mp4"
    if [ ! -f "$VIDEO_PATH" ]; then
        echo "Error: Video file not found at $VIDEO_PATH"
        cd "$CURRENT_DIR"
        return 1
    fi
    
    # Return to feature-4dgs directory
    cd "$CURRENT_DIR"
    
    echo "Step 2/3: Converting video to frames"
    run_with_log "python prepare.py --config ./configs/wild/prepare_davis.yaml --src \"$DATA_DIR\" --video_path \"$VIDEO_PATH\"" "frames_conversion"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to convert video to frames"
        return 1
    fi
    
    # Create required directories
    mkdir -p "$DATA_DIR/preprocess/semantic_features/rgb_feature_langseg"
    
    echo "Step 3/3: Running 4D scene generation pipeline"
    
    # Extract features using internvideo
    echo "Step 3.1/5: Running InternVideo feature extraction"
    cd internvideo_chat_feature
    run_with_log "python internvideo_extract_feat.py --video_path \"$CURRENT_DIR/$DATA_DIR/preprocess/images\"" "internvideo"
    cd "$CURRENT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract InternVideo features"
        return 1
    fi
    
    # Extract features using SAM2
    echo "Step 3.2/5: Running SAM2 feature extraction"
    cd sam2
    run_with_log "python sam2_extract_feat.py --video_path \"$CURRENT_DIR/$DATA_DIR/preprocess/images\"" "sam2"
    cd "$CURRENT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract SAM2 features"
        return 1
    fi
    
    # Extract features using LSEG
    echo "Step 3.3/5: Running LSEG encoding"
    cd lseg_encoder
    run_with_log "python encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv \
        --outdir \"$CURRENT_DIR/$DATA_DIR/preprocess/semantic_features/rgb_feature_langseg\" \
        --test-rgb-dir \"$CURRENT_DIR/$DATA_DIR/preprocess/images\"" "lseg"
    cd "$CURRENT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run LSEG encoding"
        return 1
    fi
    
    # Run final 4D scene generation
    echo "Step 3.4/5: Running 4D scene generation"
    run_with_log "python run.py --config ./configs/wild/davis.yaml --src \"$DATA_DIR\"" "generate_4d"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate 4D scene"
        return 1
    fi
    
    # Generate visualizations 
    echo "Step 3.5/5: Generating visualizations"
    # Find the latest output directory
    local OUTPUT_GS_DIR=$(find "$DATA_DIR" -type d -name "gs_*" 2>/dev/null | sort | tail -n 1)
    if [ -z "$OUTPUT_GS_DIR" ]; then
        echo "Warning: No gs_* directory found for visualization"
        OUTPUT_GS_DIR="$DATA_DIR"
    fi
    
    run_with_log "python viz.py --config ./configs/wild/davis.yaml \
        --root \"$OUTPUT_GS_DIR\" \
        --save \"$DATA_DIR/viz\" \
        --N 5" "visualize_4d"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate visualizations"
        return 1
    fi
    
    echo "Done! Results available at:"
    echo "- Raw video: $VIDEO_PATH"
    echo "- 4D scene data: $DATA_DIR"
    echo "- Visualizations: $DATA_DIR/viz"
    
    return 0
}

# Add a simple wrapper function for e2e processing
function e2e() {
    generate_4d_from_prompt "A teal robot is cooking food in a kitchen with steam rising from pots" 10.0 0
}

# Execute function based on command line argument
$@
