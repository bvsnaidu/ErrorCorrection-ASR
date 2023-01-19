#!/bin/bash

#source activate LM_sait

python3 /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/LM_training_initial.py --pre_trained_path  /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/bert-kor-base --filepath /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/data_kr --epochs 20 --lr 0.0001 --batch_size 64 --model_save_path /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/my_model/model_test.pt --model_eval_path /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/my_model/model_test.pt --train False
