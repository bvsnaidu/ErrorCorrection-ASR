#!/bin/bash

#source activate speech

python3 -m torch.distributed.run --nproc_per_node=8 LM_training_multi-DDP.py \
 --pre_trained_path  /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/bert-kor-base \
 --train_f "/home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/data_kr/train" \
 --test_f "/home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/data_kr/test"  \
 --epochs 50 \
 --lr 0.0001 \
 --batch_size 4\
 --model_save_path /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/my_model/model_test.pt \
 --model_eval_path /home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/my_model/model_test.pt \
 --train True
