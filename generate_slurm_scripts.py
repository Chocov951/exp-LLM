import os

# Définir les plages de paramètres
learning_rates = [1e-5, 5e-5, 1e-4]
epochs = [40]
datasets = ["iso_none"]
model_names = ["bloom", "bloomz560", "llama"]
save_steps = 1 # Fréquence de sauvegarde du modèle en nombre d'époques
steps_per_epoch = 189 # Nombre de pas par époque

# Chemin du dossier de sortie
output_dir = "slurm_files/"

# Supprimer le dossier de sortie s'il existe puis le recréer
if os.path.exists(output_dir):
    os.system(f"rm -r {output_dir}")
os.makedirs(output_dir)

# Modèle de script SLURM
slurm_template = """#!/bin/bash
#SBATCH --job-name=RANLLM
#SBATCH --output=out/zout_ranllm%j.out
#SBATCH --error=err/zerr_ranllm%j.err
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

srun python -u llm_train_test.py --train True --model_name {model_name} --dataset {dataset} --epochs {epochs} --learning_rate {learning_rate} --save_steps {save_steps} --steps_per_epoch {steps_per_epoch}
"""

# Générer les fichiers SLURM
for lr in learning_rates:
    for epoch in epochs:
        for dataset in datasets:
            for model_name in model_names:
                script_content = slurm_template.format(
                    learning_rate=lr, epochs=epoch, dataset=dataset, model_name=model_name, save_steps=save_steps, steps_per_epoch=steps_per_epoch)
                script_filename = f"slurm_lr{lr}_epochs{epoch}_dataset{dataset}_model{model_name}.slurm"
                script_filepath = os.path.join(output_dir, script_filename)
                with open(script_filepath, "w") as script_file:
                    script_file.write(script_content)

# Exécuter les scripts SLURM
for script_filename in os.listdir(output_dir):
    script_filepath = os.path.join(output_dir, script_filename)
    os.system(f"sbatch {script_filepath}")
os.system("watch -n 2 -d squeue --me")