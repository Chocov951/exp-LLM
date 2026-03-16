#!/bin/bash
#SBATCH --job-name=RankLLM
#SBATCH --output=zout_rankllm_%j.out
#SBATCH --error=zerr_rankllm_%j.err
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=ahw@v100

# --- Environment Setup ---
module purge
module load cuda/12.2.0
module load python

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate llama
export JAVA_HOME=/lustre/fswork/projects/rech/ahw/uep39vh/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

set -x

# --- High-Performance Storage Setup (Jean Zay) ---
FAST_STORAGE="${JOBSCRATCH:-${SLURM_TMPDIR:-/tmp}}"
MODEL_SHORT="qwen4"
# MODEL_SHORT="qwen30" # Uncomment to use 30B model

if [ "$MODEL_SHORT" == "qwen4" ]; then
    MODEL_DIR_NAME="Qwen3-4B-Instruct-2507"
elif [ "$MODEL_SHORT" == "qwen30" ]; then
    MODEL_DIR_NAME="Qwen3-30B-A3B-Instruct-2507"
elif [ "$MODEL_SHORT" == "qwen72" ]; then
    MODEL_DIR_NAME="Qwen2.5-72B-Instruct"
elif [ "$MODEL_SHORT" == "smollm" ]; then
    MODEL_DIR_NAME="SmolLM3-3B"
elif [ "$MODEL_SHORT" == "gpt" ]; then
    MODEL_DIR_NAME="gpt-oss-20b"
elif [ "$MODEL_SHORT" == "deepseek" ]; then
    MODEL_DIR_NAME="DeepSeek-V3.2"
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
echo "Starting RankLLM (custom_llm_vllm backend)"

# Run from root directory so imports work as expected
python -u all_in_one_rankllm.py --dataset "trec19" --bm25_topk 100 --filter_topk 20 --filter_method "custom_llm_vllm" --custom_model "${MODEL_SHORT}" --ranking_method "custom_llm_vllm" --stage both

echo "Training finished."
# --- Cleanup ---