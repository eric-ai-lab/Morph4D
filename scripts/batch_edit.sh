#!/usr/bin/env bash
set -euo pipefail


# # skip edit
# bash scripts/batch_edit.sh \
#   --data-root /home/xuehai/mojito/MoSca/data/VLM4D/videos_synthetic_frames \
#   --gpu-ids "1,2" \
#   --sequences "synth_001 synth_005 synth_009 synth_010 synth_011 synth_013 synth_018 synth_019" \
#   --skip-edit \
#   --resume-stages

# Optional: set via environment or .env
: "${HF_TOKEN:=}"


# # resume edit
# python /home/xuehai/mojito/feature-4dgs/scripts/batch_edit.py \
#   --data-root /home/xuehai/mojito/feature-4dgs/data/VLM4D/videos_synthetic_frames \
#   --gpu-ids "2" \
#   --sequences "synth_001 synth_005 synth_009 synth_010 synth_011 synth_013 synth_018 synth_019" \
#   --resume-edit \
#   --auto-prompts \
#   --num-color-variants 2 \
#   --resume-stages



python "${REPO_ROOT}/scripts/batch_edit.py" \
  --data-root ./data/veo3 \
  --gpu-ids "2" \
  --sequences "Robot_Learning_and_Walking_Video Robot_Walking_and_Learning_Video" \
  --auto-prompts \
  --num-color-variants 2 \
  --resume-stages


# normal 
python "${REPO_ROOT}/scripts/batch_edit.py" \
  --data-root ./data/VLM4D/videos_synthetic_frames \
  --gpu-ids "1,2" \
  --sequences "synth_020 synth_023 synth_030 synth_032 synth_036 synth_042 synth_052 synth_054 synth_059 synth_001 synth_005 synth_009 synth_010 synth_011 synth_013 synth_018 synth_019" \
  --auto-prompts \
  --num-color-variants 2 \
  --resume-stages

# # resume edit
# python /home/xuehai/mojito/feature-4dgs/scripts/batch_edit.py \
#   --data-root /home/xuehai/mojito/feature-4dgs/data/veo3 \
#   --gpu-ids "2" \
#   --sequences "Robot_Learning_and_Walking_Video Robot_Walking_and_Learning_Video" \
#   --resume-edit \
#   --auto-prompts \
#   --num-color-variants 2 \
#   --resume-stages

# Thin wrapper to call the Python orchestrator (keeps existing CLI usage working)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

python "${REPO_ROOT}/scripts/batch_edit.py" "$@"

# synth_001 robotic arm
# synth_005 robot car
# synth_009 snowplow
# synth_010 robot archaeologist
# synth_011 bus
# synth_013 jellyfish
# synth_018 exploration rover
# synth_019 robot
# synth_020 robot dog
# synth_023 robotic arm
# synth_030 mars-rover
# synth_032 bus
# synth_036 robot
# synth_042 vehicle
# synth_052 garbage-truck
# synth_054 vehicle
# synth_059 roomba

python "${REPO_ROOT}/scripts/batch_edit.py" \
  --data-root ./data/VLM4D/videos_synthetic_frames \
  --gpu-ids "1" \
  --sequences "synth_019" \
  --edit-kinds remove \
  --resume-stages

python "${REPO_ROOT}/scripts/batch_edit.py" \
  --data-root ./data/VLM4D/videos_synthetic_frames \
  --gpu-ids "1" \
  --sequences "synth_019" \
  --edit-kinds "color,remove,extract" \
  --colors "blue,green" \
  --thr-min 0.86 --thr-max 0.94 \
  --gpt-temp 0.1 \
  --num-prompts 8 \
  --agent-verbose \
  --resume-stages

python scripts/batch_edit.py   --data-ro
python /home/xuehai/mojito/feature-4dgs/scripts/batch_edit.py --data-root ./data/VLM4D/videos_synthetic_frames   --sequences "synth_001 synth_005 synth_009 synth_010 synth_011 synth_013 synth_018 synth_019"   --resume-edit --edit-kinds "color,remove,extract" --thr_min 0.60 --thr_max 0.99