#!/bin/bash
#SBATCH --job-name=RANLLM
#SBATCH --output=zout_ranllm%j.out
#SBATCH --error=zerr_ranllm%j.err
##SBATCH --partition=prepost
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
##SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=ahw@a100
 
module purge
 
module load pytorch-gpu/py3/2.3.0
conda activate llama
export JAVA_HOME=/lustre/fswork/projects/rech/ahw/uep39vh/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

 
set -x

srun python -u all_in_one.py --model_name 'qwen72' --dataset trec20 --bm25_topk 100 --reject_number 20