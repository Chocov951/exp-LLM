import os

# Définir les plages de paramètres
datasets = ["trec20", "trec19"]
model_names = ["qwen30","smollm","gpt"]
bm25_topks = [100]
reject_numbers = [10,20,30]
filter_methods = ["custom_llm_vllm"]
# Chemin du dossier de sortie
output_dir = "slurm_files/"

# Supprimer le dossier de sortie s'il existe puis le recréer
if os.path.exists(output_dir):
    os.system(f"rm -r {output_dir}")
os.makedirs(output_dir)

# Modèle de script SLURM
slurm_template = """#!/bin/bash
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
FAST_STORAGE="${{JOBSCRATCH:-${{SLURM_TMPDIR:-/tmp}}}}"
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
LOCAL_MODELS_DIR="${{FAST_STORAGE}}/models"

echo "Setting up execution environment on: ${{FAST_STORAGE}}"


if [ "${{MODEL_DIR_NAME}}" != "None" ]; then
    # Create local model directory
    mkdir -p "${{LOCAL_MODELS_DIR}}"

    # Copy model to fast storage
    echo "Transferring model ${{MODEL_DIR_NAME}} from shared storage to fast scratch..."
    cp -r "${{SOURCE_MODELS_DIR}}/${{MODEL_DIR_NAME}}" "${{LOCAL_MODELS_DIR}}/"

    # Configure Python script to use the local copy
    export LLM_MODELS_ROOT="${{LOCAL_MODELS_DIR}}"
fi

# --- Execution ---
echo "Starting RankLLM (custom_llm_vllm backend)"

# Run from root directory so imports work as expected
python -u all_in_one_rankllm.py --dataset {dataset} --bm25_topk {bm25_topk} --filter_topk {reject_number} --filter_method {filter_method} --custom_model "${{MODEL_SHORT}}" --ranking_method {filter_method} --stage both

echo "Training finished."
# --- Cleanup ---

"""

# Générer les fichiers SLURM
for bm in bm25_topks:
    for rj in reject_numbers:
        for dataset in datasets:
            for model_name in model_names:
                for filter_method in filter_methods:
                    script_content = slurm_template.format(
                        model=model_name, dataset=dataset, bm25_topk=bm, reject_number=rj, filter_method=filter_method)
                    script_filename = f"slurm_{dataset}_{model_name}_{filter_method}_filter{rj}_top{bm}.slurm"
                    script_filepath = os.path.join(output_dir, script_filename)
                    with open(script_filepath, "w") as script_file:
                        script_file.write(script_content)
                       

# Exécuter les scripts SLURM
for script_filename in os.listdir(output_dir):
    script_filepath = os.path.join(output_dir, script_filename)
    os.system(f"sbatch {script_filepath}")
os.system("watch -n 2 -d squeue --me")