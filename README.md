In order to run training :

1) Look at data_sample for training data_sample
2) Download Bert Model from "https://huggingface.co/kykim/bert-kor-base"
3) Create a folder my_model
4) Run train.sh with changed paths(look at run script) ...This will create an initial seed model at my_model/model_test.pt
5) Run traing_multi-DDP/DP.sh for training
