#!/bin/bash
#SBATCH --time=01:10:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=phi3_llm_17
#SBATCH --mem=50G

module purge
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/venvs/thesis_venv/bin/activate

#python3 -m pip install transformers tensorflow torch

python3 phi3_model_prompting.py eval long 17 eval_data_long.json
python3 phi3_model_prompting.py eval short 17 eval_data_short.json

