#!/bin/bash
#SBATCH --job-name=RankLLM
#SBATCH --output=zout_rankllm_%j.out
#SBATCH --error=zerr_rankllm_%j.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
##SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=ahw@a100

# --- Environment Setup ---
module purge
module load pytorch-gpu/py3/2.3.0

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate llama

set -x

# --- High-Performance Storage Setup (Jean Zay) ---
FAST_STORAGE="${JOBSCRATCH:-${SLURM_TMPDIR:-/tmp}}"
MODEL_SHORT="qwen4"
# MODEL_SHORT="qwen30" # Uncomment to use 30B model

if [ "$MODEL_SHORT" == "qwen4" ]; then
    MODEL_DIR_NAME="Qwen3-4B-Instruct-2507"
elif [ "$MODEL_SHORT" == "qwen30" ]; then
    MODEL_DIR_NAME="Qwen3-30B-A3B-Instruct-2507"
elif [ "$MODEL_SHORT" == "zephyr" ]; then
    MODEL_DIR_NAME="rank_zephyr_7b_v1_full"
elif [ "$MODEL_SHORT" == "vicuna" ]; then
    MODEL_DIR_NAME="rank_vicuna_7b_v1"
elif [ "$MODEL_SHORT" == "monot5" ]; then
    MODEL_DIR_NAME="monot5-base-msmarco-10k"
else
    echo "Unknown model short name: $MODEL_SHORT"
    MODEL_DIR_NAME=None
fi

SOURCE_MODELS_DIR="./models"
LOCAL_MODELS_DIR="${FAST_STORAGE}/models"

echo "Setting up execution environment on: ${FAST_STORAGE}"


if [ "${MODEL_DIR_NAME}" != "None" ]; then
    # Create local model directory
    mkdir -p "${LOCAL_MODELS_DIR}"

    # Copy model to fast storage
    echo "Transferring model ${MODEL_DIR_NAME} from shared storage to fast scratch..."
    cp -r "${SOURCE_MODELS_DIR}/${MODEL_DIR_NAME}" "${LOCAL_MODELS_DIR}/"

    # Configure Python script to use the local copy
    export LLM_MODELS_ROOT="${LOCAL_MODELS_DIR}"
fi

# --- Execution ---
echo "Starting RankLLM"

# Run from root directory so imports work as expected
python -u all_in_one_rankllm.py \
    --dataset trec20 \
    --bm25_topk 100 \
    --filter_topk 20 \
    --filter_method custom_llm \
    --custom_model "${MODEL_SHORT}" \
    --ranking_method custom_llm \
    --stage filter 
    # --load_filtered_rundict \
    # --listwise_model zephyr \
    # --listwise_window_size full \
    # --bert_model BAAI/bge-m3 \


echo "Training finished."
# --- Cleanup ---
