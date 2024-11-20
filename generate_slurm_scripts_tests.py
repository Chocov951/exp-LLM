import os

# Définir les plages de paramètres
learning_rates = [1e-5, 5e-5, 1e-4]
epochs = [40]
datasets = ["iso_none"]
model_names = ["bloom", "bloomz560", "llama"]
test_cpts = [10,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
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
#SBATCH --output=out/{dataset}/test/zout_ranllm%j.out
#SBATCH --error=err/{dataset}/test/zerr_ranllm%j.err
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

srun python -u llm_train_test.py --test_cpt {test_cpt} --steps_per_epoch {steps_per_epoch} --model_name {model_name} --dataset {dataset} --epochs {epochs} --learning_rate {learning_rate}
"""

# Générer les fichiers SLURM
for lr in learning_rates:
    for epoch in epochs:
        for dataset in datasets:
            for model_name in model_names:
                for test_cpt in test_cpts:
                    script_content = slurm_template.format(
                        learning_rate=lr, epochs=epoch, dataset=dataset, model_name=model_name, test_cpt=test_cpt, steps_per_epoch=steps_per_epoch)
                    script_filename = f"slurm_test-e{test_cpt}_dataset{dataset}_model{model_name}.slurm"
                    script_filepath = os.path.join(output_dir, script_filename)
                    if os.path.exists(script_filepath):
                        os.remove(script_filepath)
                    with open(script_filepath, "w") as script_file:
                        script_file.write(script_content)

# Exécuter les scripts SLURM
for script_filename in os.listdir(output_dir):
    script_filepath = os.path.join(output_dir, script_filename)
    os.system(f"sbatch {script_filepath}")
os.system("watch -n 2 -d squeue --me")