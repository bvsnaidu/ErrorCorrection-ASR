#!/bin/bash

#source activate speech

python3 -m torch.distributed.launch --nproc_per_node=8 LM_training_Seq2SeqTrainer.py \
 --pre_trained_path  /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/bert-kor-base \
 --train_f "/home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/data_kr/train" \
 --test_f "/home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/data_kr/test"  \
 --epochs 50 \
 --lr 0.0001 \
 --batch_size 2\
 --model_save_path /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/my_model/model_test.pt \
 --model_eval_path /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/my_model/model_test.pt \
 --train True
