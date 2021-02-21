#!/bin/bash
#SBATCH -J lstm                  
#SBATCH -o lstm.out 
#SBATCH -p Antarktis                                                   

python3 hate_speech_classifier_lstm.py                