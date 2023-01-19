from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers import BertTokenizer
from datasets import Dataset
from transformers import get_scheduler
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
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
#parser.add_argument('--local_rank',dest='local_rank',required=True,help='True while training and False for evaluation')
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
    
    inputs = tokenizer(batch['incorrect'], padding="max_length", truncation=True, max_length=128)
    outputs = tokenizer(batch['correct'], padding="max_length", truncation=True, max_length=128)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
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
    files_correct=sorted(files_correct)
    files_incorrect=sorted(files_incorrect)
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
    type="torch", columns=["input_ids", "attention_mask",   "labels"])
    # ip_data.set_format(
    # type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
   
    #sampler = torch.utils.data.distributed.DistributedSampler(ip_data)
    dataloader = DataLoader(ip_data, batch_size=batch_size, num_workers=16,shuffle=True)
    print("Len of trainloader is {} in prepare_data".format(len(dataloader)))
    #print("DATA LOADING DONE")
    #print("LEN IS "+str(len(dataloader)))
    return dataloader

   
def training(model, train_loader, val_loader, device, optimizer, scheduler, args):

    
    writer = SummaryWriter(log_dir=model_save_path[:model_save_path.rfind('/')]+"//"+"train")
    patience = 0
    global_step = 0
    flag_tensor = torch.zeros(1).to(device)
    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        for  batch_idx,batch in enumerate(tqdm(train_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            #decoder_input_ids = batch['decoder_input_ids'].to(device)
            # decoder_attention_mask =batch['decoder_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            if ((batch_idx + 1) % accum_grad == 0) or (batch_idx + 1 == len(train_loader)):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels,)
                loss = outputs[0]
                with torch.no_grad():
                    train_loss += outputs[0].mean()
                    local_loss = outputs[0].mean()
                loss=loss/accum_grad
                loss.mean().backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
                loss = outputs[0]
                with torch.no_grad():
                    train_loss += outputs[0].mean()
                    local_loss = outputs[0].mean()
                loss=loss/accum_grad
                loss.mean().backward()

            
            global_step += 1
            if global_step % 1000 == 0:
                writer.add_scalar("Train loss",local_loss,global_step)
                writer.add_scalar("LR",scheduler.get_last_lr()[0],global_step)           
        
        train_loss = train_loss/len(train_loader)
       
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader :
                outputs = model(input_ids = batch['input_ids'].to(device),
                                attention_mask = batch['attention_mask'].to(device),
                                labels = batch['labels'].to(device))
                                #decoder_input_ids = batch['decoder_input_ids'].to(device))
                loss = outputs[0]
                val_loss += outputs[0].mean()
        val_loss = val_loss/len(val_loader)

        
        writer.add_scalars("Loss", {'train_loss':train_loss,'val_loss':val_loss}, epoch)
        
        #Histogram of gradients
        # histogram = False
        # if histogram == True :
        #     for name, weight in model.named_parameters():
        #         if weight.grad is not None :
        #             writer.add_histogram(name,weight, epoch)
        #             writer.add_histogram(f'{name}.grad',weight.grad, epoch)

        if epoch >= 1 :
            print("Epoch {} - train loss {}, val loss {}".format(epoch,train_loss, val_loss))
            f = open(args.model_save_path[:args.model_save_path.rfind('/')]+'_logs.txt','a')
            f.write("Epoch {} - train loss {}, val loss{}".format(epoch,train_loss,val_loss) + '\n')
            f.close()
            if val_loss < prev_loss :
                patience = 0
                prev_loss = val_loss 
                #save the model
                torch.save(model.state_dict(), args.model_save_path)
            else :
                patience += 1
                if patience > 4 :
                    print("No improvement over 5 epochs")
                    writer.close()
                    flag_tensor += 1
                    
        else :
            print("Epoch {} - train loss {}, val loss {}".format(epoch,train_loss, val_loss))
            f = open(args.model_save_path[:args.model_save_path.rfind('/')]+'_logs.txt','a')
            f.write("Epoch {} - train loss {}, val loss{}".format(epoch,train_loss,val_loss) + '\n')
            f.close()
            prev_loss = val_loss 
            #save the model
            torch.save(model.state_dict(), args.model_save_path)
        
        if flag_tensor == 1:
            print("Training stopped")
            return model
    
    writer.close()
    return model

def decode(batch):

    inputs = tokenizer(batch['incorrect'], padding="max_length", truncation=True, max_length=128)
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
    #torch.cuda.set_device(local_rank)
    model.to(device)
    if torch.cuda.device_count() > 1:
       print("Multi-GPU training")
       model = nn.DataParallel(model)
    if train == "True" :
        train_loader = prepare_data(train_filepath)
        val_loader = prepare_data(test_filepath)
        optimizer = optim.AdamW(model.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        num_training_steps = int(epochs *(len(train_loader)))
        scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=int(0.1*(len(train_loader))),num_training_steps=num_training_steps)        
        model = training(model, train_loader, val_loader, device, optimizer, scheduler, args) 


    f = open(model_save_path[:model_save_path.rfind('/')]+"//"+"result.txt","w")
    print("Testing")
    test_data = prepare_data(test_filepath, test=True) 
    d = evaluation(model)
    f.write("Testing"+"\n")
    f.write(str(d)+'\n')
    f.close()





