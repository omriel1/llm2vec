#! /bin/sh

#SBATCH --job-name=mntp
#SBATCH --output=/home/joberant/NLP_2324b/<user-name>/llm2vec/output/mntp.out
#SBATCH --error=/home/joberant/NLP_2324b/<user-name>/llm2vec/output/mntp.err
#SBATCH --partition=studentbatch
#SBATCH --time=600
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="a100"

python experiments/run_mntp.py nlp_course/dictalm2_mntp.json
