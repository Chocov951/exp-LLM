import os
import csv
import shutil

# Parameter space
datasets = ["trec20", "trec19"]
model_names = ["qwen4", "smollm", "gpt", "qwen30", "zephyr", "vicuna"]
bm25_topks = [100]
reject_numbers = [10, 20, 30]
filter_methods = ["Pointwise", "BERT-all-MiniLM-L12-v2", "BERT-bge-m3"]

output_dir = "slurm_files"
tasks_csv = os.path.join(output_dir, "tasks.csv")
array_slurm = os.path.join(output_dir, "rankllm_array.slurm")


def _normalize_text(value):
    return (value or "").strip().lower()


def _safe_int(value):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _extract_model_from_rerank_method(rerank_method):
    rerank = _normalize_text(rerank_method)

    if rerank.startswith("listwise-"):
        parts = rerank.split("-")
        return parts[1] if len(parts) > 1 else None

    if rerank.startswith("customllm-vllm-"):
        parts = rerank.split("-")
        return parts[-1] if len(parts) > 2 else None

    return None


def load_existing_combinations(csv_path):
    """
    Build a set of existing (filter_method, model_name, filter_size) tuples.
    A row is considered existing only if ndcg@5 is present and not N/A.
    """
    existing = set()
    if not os.path.exists(csv_path):
        print(f"[INFO] CSV not found: {csv_path}. All combinations will be generated for this dataset.")
        return existing

    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filter_method = _normalize_text(row.get("filter_method"))
            model_name = _extract_model_from_rerank_method(row.get("rerank_method"))
            filter_size = _safe_int(row.get("filter_size"))

            if filter_method and model_name and filter_size is not None:
                ndcg5 = str(row.get("ndcg@5", "N/A")).strip()
                if ndcg5 and ndcg5 != "N/A":
                    existing.add((filter_method, model_name, filter_size))

    print(f"[INFO] Loaded {len(existing)} existing combinations from {csv_path}.")
    return existing


def build_tasks():
    existing_by_dataset = {
        dataset: load_existing_combinations(os.path.join("rankllm_results", f"results_{dataset}.csv"))
        for dataset in datasets
    }

    tasks = []
    skipped_count = 0

    for bm in bm25_topks:
        for rj in reject_numbers:
            for dataset in datasets:
                for model_name in model_names:
                    for filter_method in filter_methods:
                        combo_key = (_normalize_text(filter_method), _normalize_text(model_name), rj)
                        if combo_key in existing_by_dataset.get(dataset, set()):
                            skipped_count += 1
                            print(
                                f"[SKIP] Already present in results_{dataset}.csv: "
                                f"filter_method={filter_method}, model={model_name}, filter_size={rj}"
                            )
                            continue

                        # Keeping your existing logic exactly as written
                        if filter_method != "CustomLLM-qwen30":
                            rej = 100
                        else:
                            rej = rj

                        rundict = (
                            f"rankllm_results/rundicts/"
                            f"filtered_{dataset}_{filter_method}_filter{rej}_top{bm}.json"
                        )

                        tasks.append({
                            "dataset": dataset,
                            "model": model_name,
                            "bm25_topk": bm,
                            "filter_topk": rj,
                            "filter_method": filter_method,
                            "rej_for_filename": rej,
                            "rundict": rundict,
                        })

    print(f"[INFO] Prepared {len(tasks)} tasks, skipped {skipped_count} existing combinations.")
    return tasks


def write_tasks_csv(tasks):
    fieldnames = [
        "dataset",
        "model",
        "bm25_topk",
        "filter_topk",
        "filter_method",
        "rej_for_filename",
        "rundict",
    ]

    with open(tasks_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(tasks)

    print(f"[INFO] Wrote task table: {tasks_csv}")


def write_array_script():
    script = r"""#!/bin/bash
#SBATCH --job-name=RankLLM
#SBATCH --output=zout_rankllm_%A_%a.out
#SBATCH --error=zerr_rankllm_%A_%a.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --hint=nomultithread
#SBATCH --account=ahw@a100

module purge
module load cuda/12.2.0

eval "$(conda shell.bash hook)"
conda activate llama
export JAVA_HOME=/lustre/fswork/projects/rech/ahw/uep39vh/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

set -euo pipefail
set -x

# Ensure all relative paths resolve from the submission directory
cd "$SLURM_SUBMIT_DIR"

FAST_STORAGE="${JOBSCRATCH:-${SLURM_TMPDIR:-/tmp}}"
SOURCE_MODELS_DIR="./models"
LOCAL_MODELS_DIR="${FAST_STORAGE}/models"
TASKS_FILE="slurm_files/tasks.csv"

# Read the row matching this array task id.
# +2 because:
# - csv has a header line
# - SLURM_ARRAY_TASK_ID starts at 0
TASK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$TASKS_FILE")

if [ -z "${TASK_LINE}" ]; then
    echo "No task line found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Handle CSV files written with CRLF so the last field is not polluted by '\r'.
TASK_LINE="${TASK_LINE%$'\r'}"

IFS=',' read -r DATASET MODEL_SHORT BM25_TOPK FILTER_TOPK FILTER_METHOD REJ_FOR_FILENAME RUNDICT <<< "${TASK_LINE}"

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
    exit 1
fi

echo "Setting up execution environment on: ${FAST_STORAGE}"
mkdir -p "${LOCAL_MODELS_DIR}"

if [ ! -d "${LOCAL_MODELS_DIR}/${MODEL_DIR_NAME}" ]; then
    echo "Transferring model ${MODEL_DIR_NAME} from shared storage to fast scratch..."
    cp -r "${SOURCE_MODELS_DIR}/${MODEL_DIR_NAME}" "${LOCAL_MODELS_DIR}/"
fi

export LLM_MODELS_ROOT="${LOCAL_MODELS_DIR}"

echo "Starting RankLLM array task ${SLURM_ARRAY_TASK_ID}"
echo "DATASET=${DATASET}"
echo "MODEL_SHORT=${MODEL_SHORT}"
echo "BM25_TOPK=${BM25_TOPK}"
echo "FILTER_TOPK=${FILTER_TOPK}"
echo "FILTER_METHOD=${FILTER_METHOD}"
echo "RUNDICT=${RUNDICT}"

python -u all_in_one_rankllm.py \
  --dataset "${DATASET}" \
  --bm25_topk "${BM25_TOPK}" \
  --filter_topk "${FILTER_TOPK}" \
  --load_filtered_rundict "${RUNDICT}" \
  --listwise_model "custom_llm" \
  --custom_model "${MODEL_SHORT}" \
  --ranking_method listwise \
  --stage rerank \
  --listwise_window_size full

echo "Task finished."
"""
    with open(array_slurm, "w", encoding="utf-8") as f:
        f.write(script)

    print(f"[INFO] Wrote array script: {array_slurm}")


def main():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tasks = build_tasks()
    if not tasks:
        print("[INFO] No tasks to submit.")
        return

    write_tasks_csv(tasks)
    write_array_script()

    max_parallel = 10
    print()
    print("[READY]")
    print(f"Number of tasks: {len(tasks)}")
    print("Submit with:")
    print(f"sbatch --array=0-{len(tasks)-1}%{max_parallel} {array_slurm}")


if __name__ == "__main__":
    main()