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

srun python -u llm_train_test.py --train True --overwrite_output_dir True --model_name bloom --dataset scifact --epochs 50 --gradient_accumulation_steps 1 --learning_rate 5e-5