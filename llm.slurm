#!/bin/bash
#SBATCH --job-name=RANLLM
#SBATCH --output=zout_ranllm%j.out
#SBATCH --error=zerr_ranllm%j.err
##SBATCH --partition=prepost
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
 
module purge
 
module load pytorch-gpu/py3/2.3.0
conda activate llama
 
set -x
 
srun llamafactory-cli train train_llm_bloom.json
srun python -u test_llm.py --model_name bloom --dataset scifact