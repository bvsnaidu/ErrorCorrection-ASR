#!/bin/bash

source activate LM_sait

python /home/sr5/rakshith.v/asr-poc/SeqSeq/cosine_similarity.py --pre_trained_path  /home/sr5/rakshith.v/asr-poc/SeqSeq/pre_trained_path --filepath /home/sr5/rakshith.v/Error_correction/data/train_master  --batch_size 256
