#!/bin/bash
#SBATCH --job-name=BERTTEST
#SBATCH --output=zout_berttest%j.out
#SBATCH --error=zerr_berttest%j.err
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

srun python -u bert_test.py