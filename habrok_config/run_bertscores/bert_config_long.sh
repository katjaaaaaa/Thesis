#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=bertscore
#SBATCH --mem=50G

module purge
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/venvs/thesis_venv/bin/activate

#python3 -m pip install bert_score

echo 1
python3 bertscore_calculate.py 1 long "One-shot"
echo 5
python3 bertscore_calculate.py 5 long
echo 7
python3 bertscore_calculate.py 7 long "Few-shot expl"
echo 8
python3 bertscore_calculate.py 8 long
echo 9
python3 bertscore_calculate.py 9 long
echo 10
python3 bertscore_calculate.py 10 long
echo 14
python3 bertscore_calculate.py 14 long
echo 15
python3 bertscore_calculate.py 15 long
echo 17
python3 bertscore_calculate.py 17 long
echo 20
python3 bertscore_calculate.py 20 long
