wikireader.py: This file makes use of the WiktionaryParser library to pull definitions of words from wiktionary. For testing purposes, it currently prints the definition of "cat" when run. The idea is for it to be used to provide rare word definitions to a BERT model. 

bert_setup.py: This file will train a bert for masked lm model from a small english dataset.

To run on atlas: srun -p gpu_low --exclude [list of compute nodes you wish to exclude seperated by commas] --gres=gpu:1 --mem=56GB --pty bash
sbatch -p gpu_low [file name]
