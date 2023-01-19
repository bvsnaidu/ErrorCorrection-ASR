from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers import BertTokenizer
from datasets import Dataset
from transformers import get_scheduler
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import random
import os
import argparse
from copy import deepcopy

#from torchtext.data.metrics import bleu_score

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._C._distributed_c10d import ReduceOp

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("/home/raghav.garg/ASR-ErrorCorrection/asr-poc/SeqSeq/")

from jiwer import wer,cer


import socket
import pdb

'''
No of Batches to Accumulate Gradients
'''
accum_grad=128

parser = argparse.ArgumentParser(description ='Error correction training and testing')
parser.add_argument('--pre_trained_path',dest='pre_trained_path',required=True,help='Specify file path of pre-trained model')
parser.add_argument('--train_f',dest='train_f',required=True,help='Data Directory')
parser.add_argument('--test_f',dest='test_f',required=True,help='Data Directory')
parser.add_argument('--epochs',dest='epochs',required=True,help='Number of Epochs')
parser.add_argument('--lr',dest='lr',required=True,help='Learning rate')
parser.add_argument('--batch_size',dest='batch_size',required=True,help='Batch size')
parser.add_argument('--model_save_path',dest='model_save_path',required=True,help='Path to save the trained model')
parser.add_argument('--model_eval_path',dest='model_eval_path',required=True,help='Model on which evaluation needs to be done')
parser.add_argument('--train',dest='train',required=True,help='True while training and False for evaluation')
parser.add_argument('--local_rank',dest='local_rank',required=True,help='True while training and False for evaluation')
args = parser.parse_args()

def map_to_length(x):
  x["incorrect_len"] = len(tokenizer(x["incorrect"]).input_ids)
  x["correct_len"] = len(tokenizer(x["correct"]).input_ids)
  x["incorrect_512"] = int(x["incorrect_len"] > 512)
  x["incorrect_128"] = int(x["incorrect_len"] > 128)
  x["correct_64"] = int(x["correct_len"] > 64)
  x["correct_256"] = int(x["correct_len"] > 256)
  return x

def compute_and_print_stats(x):
    print("printing stats for {} samples".format(len(x["incorrect_len"])))
    print(
    "Incorrect Mean: {}, %-incorrect > 512:{}, %-incorrect > 128:{}, correct Mean:{}, %-correct > 64:{}, %-correct > 256:{}".format(
        sum(x["incorrect_len"]) / len(x["incorrect_len"]),
        sum(x["incorrect_512"]) / len(x["incorrect_len"]),
        sum(x["incorrect_128"]) / len(x["incorrect_len"]), 
        sum(x["correct_len"]) / len(x["incorrect_len"]),
        sum(x["correct_64"]) / len(x["incorrect_len"]),
        sum(x["correct_256"]) / len(x["incorrect_len"]),
        )
    )

def model_load():

    model = EncoderDecoderModel.from_pretrained(model_save_path[:model_save_path.rfind('/')])
    return model

def process_data_to_model_inputs(batch):
    
    inputs = tokenizer(batch['incorrect'], padding="max_length", truncation=True, max_length=64)
    outputs = tokenizer(batch['correct'], padding="max_length", truncation=True, max_length=64)
    batch["input_ids"] = inputs.input_ids
    #batch["attention_mask"] = inputs.attention_mask
    #batch["decoder_input_ids"] = outputs.input_ids
    #batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()
    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

def prepare_data(path,test=False):
    #Need to adapt to Pytorch dataloader format later
    files_correct=[f for f in os.listdir(path+"//"+"correct/")]
    files_incorrect=[f for f in os.listdir(path+"//"+"incorrect/")]
    ip=[]
    groundTruth=[]
    for fil in files_incorrect:
        with open(path+"//"+"incorrect/"+fil,encoding="utf-8",errors="ignore") as f:
            ip += f.read().splitlines()
            #ip = ip[:1000]
    for fil in files_correct:
        with open(path+"//"+"correct/"+fil,encoding="utf-8",errors="ignore") as f:
            groundTruth += f.read().splitlines()
            #groundTruth = groundTruth[:1000]
    ip_data={'incorrect':ip,'correct':groundTruth}
    ip_data = Dataset.from_dict(ip_data)
    

    if test is not False :
    	return ip_data
    ip_data = ip_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=batch_size,
        remove_columns=["incorrect", "correct"]
    )
    ip_data.set_format(
    type="torch", columns=["input_ids",  "labels"])
    # ip_data.set_format(
    # type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
   
    #sampler = torch.utils.data.distributed.DistributedSampler(ip_data)
    #dataloader = DataLoader(ip_data, batch_size=batch_size, persistent_workers=True,num_workers=16,shuffle=True)
    #print("Len of trainloader is {} in prepare_data".format(len(dataloader)))
    #print("DATA LOADING DONE")
    #print("LEN IS "+str(len(dataloader)))
    return ip_data

   
def training(model_2, train_set, val_set,tokenizer, device,training_args , extra_args):
    
    trainer = Seq2SeqTrainer(
    model=model_s,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    )
    trainer.train()

    
    
    return model_s

def decode(batch):

    inputs = tokenizer(batch['incorrect'], padding="max_length", truncation=True, max_length=64)
    input_ids = torch.tensor(inputs.input_ids).to(device)
    attention_mask = torch.tensor(inputs.attention_mask).to(device)
    output_list = []
    for beam in range(5,6):
        outputs = model.module.generate(input_ids, num_beams=beam,no_repeat_ngram_size=2,early_stopping=True,attention_mask=attention_mask)
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_list.append(output_str)
    batch["output"] = output_list
    return batch

def evaluation(model):

    model.eval()
    model.load_state_dict(torch.load(model_eval_path))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    d = {}

    for beam_num in range(5,6):
        d[beam_num] = [[],[],[],[]]

    for batch in tqdm(test_loader) :
        reference_corpus,candidate_corpus = [], []
        results = decode(batch)
        outputs = results['output']
        labels = results['correct']
        incorrects = results['incorrect']        
        for beam_num,out in enumerate(outputs) :
            for output,label,incorrect in zip(out,labels,incorrects) :
                label = label.upper()
                incorrect = incorrect.upper()
                output = output.upper()

                if len(output) != 0 and len(label) != 0:
                    d[beam_num+5][0].append(wer(label,output))
                    d[beam_num+5][1].append(cer(label,output))

                    if len(incorrect) != 0 :
                         wer_ag = wer(label,incorrect)
                         wer_pa = wer(output,incorrect)

                         if wer_pa > 0 :
                            if wer_ag > 0 :
                                 d[beam_num+5][2].append(1)
                            else :
                                 d[beam_num+5][3].append(1)

    for beam_num in range(5,6):
        if d[beam_num][0] != []:
            d[beam_num][0] = sum(d[beam_num][0]) / len(d[beam_num][0])
        if d[beam_num][1] != [] :
            d[beam_num][1] = sum(d[beam_num][1]) / len(d[beam_num][1])
        if d[beam_num][2] != [] :
            d[beam_num][2] = sum(d[beam_num][2]) / (sum(d[beam_num][2]) + sum(d[beam_num][3]))
    return d 

        

def Print_Stats(dpath):
    print("Printing data Statistics")
    data=prepare_data(dpath, test=True)
    data_stats=data.map(map_to_length,num_proc=4)
    data_stats.map( compute_and_print_stats,batched=True,batch_size=-1)


if __name__ == "__main__":
    
    pre_trained_path = args.pre_trained_path
    train_filepath = args.train_f
    test_filepath=args.test_f
    epochs,lr,batch_size= int(args.epochs), float(args.lr),int(args.batch_size)
    model_save_path = args.model_save_path
    model_eval_path = args.model_eval_path
    train = args.train

    
    tokenizer = BertTokenizer.from_pretrained(pre_trained_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_load()
    model.to(device)

    if train == "True" :
        train_set = prepare_data(train_filepath)
        val_set = prepare_data(test_filepath)
        training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        num_train_epochs=25,
        output_dir=model_save_path[:model_save_path.rfind('/')]+"//"+"train",
        logging_steps=1000,
        save_steps=500,
        eval_steps=7500,
        warmup_steps=2000,
        save_total_limit=20,
        )
        #optimizer = optim.AdamW(model.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        num_training_steps = int(epochs *(len(train_loader)))
        #scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=int(0.1*(len(train_loader))),num_training_steps=num_training_steps)        
        model = training(model, train_set, val_set,tokenizer, device,training_args , args) 


    # f = open(model_save_path[:model_save_path.rfind('/')]+"//"+"result.txt","w")
    # print("Testing")
    # test_data = prepare_data(test_filepath, test=True) 
    # d = evaluation(model)
    # f.write("Testing"+"\n")
    # f.write(str(d)+'\n')
    # f.close()





