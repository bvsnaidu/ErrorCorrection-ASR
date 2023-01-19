from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers import BertTokenizer
from datasets import Dataset
from transformers import get_scheduler

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

import pandas as pd
import os
import argparse

from torchtext.data.metrics import bleu_score


parser = argparse.ArgumentParser(description ='Error correction training and testing')
parser.add_argument('--pre_trained_path',dest='pre_trained_path',required=True,help='Specify file path of pre-trained model')
parser.add_argument('--filepath',dest='filepath',required=True,help='Data Directory')
parser.add_argument('--epochs',dest='epochs',required=True,help='Number of Epochs')
parser.add_argument('--lr',dest='lr',required=True,help='Learning rate')
parser.add_argument('--batch_size',dest='batch_size',required=True,help='Batch size')
parser.add_argument('--model_save_path',dest='model_save_path',required=True,help='Path to save the trained model')
parser.add_argument('--model_eval_path',dest='model_eval_path',required=True,help='Model on which evaluation needs to be done')
parser.add_argument('--train',dest='train',required=True,help='True while training and False for evaluation')
args = parser.parse_args()


def model_load():

    pre_trained_model = EncoderDecoderModel.from_encoder_decoder_pretrained(pre_trained_path,pre_trained_path)
    pre_trained_model.config.add_cross_attention = True
    pre_trained_model.config.is_decoder = True
    pre_trained_model.config.decoder.num_beams = 5

    pre_trained_model.config.decoder_start_token_id = tokenizer.cls_token_id
    pre_trained_model.config.eos_token_id = tokenizer.sep_token_id 
    pre_trained_model.config.pad_token_id = tokenizer.pad_token_id
    pre_trained_model.config.vocab_size = pre_trained_model.config.encoder.vocab_size
    pre_trained_model.save_pretrained(model_save_path[:model_save_path.rfind('/')])
    model = EncoderDecoderModel.from_pretrained(model_save_path[:model_save_path.rfind('/')])
    #if torch.cuda.device_count() > 1:
    #    print("Multi-GPU training")
    #    model = nn.DataParallel(model)
    torch.save(model.state_dict(),model_save_path)
    return model

def process_data_to_model_inputs(batch):
    
    inputs = tokenizer(batch['incorrect'], padding="max_length", truncation=True, max_length=64)
    outputs = tokenizer(batch['correct'], padding="max_length", truncation=True, max_length=64)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()
    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

def prepare_train_data():
    #Need to adapt to Pytorch dataloader format later
    if os.path.exists(filepath+"//"+'incorrect_train.txt') :
        with open(filepath+"//"+'incorrect_train.txt') as f:
            train = f.read().split('\n')
    if os.path.exists(filepath+"//"+'correct_train.txt'):
        with open(filepath+"//"+'correct_train.txt') as f:
            groundTruth = f.read().split('\n')

    train_data={'incorrect':train,'correct':groundTruth}
    train_data = Dataset.from_dict(train_data)
    train_data = train_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=batch_size,
        remove_columns=["incorrect", "correct"]
    )
    train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    return train_loader

def prepare_test_data():

    if os.path.exists(filepath+"//"+'incorrect_test.txt') :
        with open(filepath+"//"+'incorrect_test.txt') as f:
            test = f.read().split('\n')
    if os.path.exists(filepath+"//"+'correct_test.txt'):
        with open(filepath+"//"+'correct_test.txt') as f:
            groundTruth_test = f.read().split('\n')

    test_data = {'incorrect':test,
                'correct':groundTruth_test}
    test_data =  Dataset.from_dict(test_data)
   
    return test_data


   
def training(model):

    model.train()
    #if torch.cuda.device_count() > 1:
    #    print("Multi-GPU training")
    #    model = nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    #scheduler = ReduceLROnPlateau(optimizer,'min',factor=0.1, patience=5)
    num_training_steps = epochs * int(len(train_loader))
    scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=1*int(len(train_loader)),num_training_steps=num_training_steps)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask =batch['decoder_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,labels=labels,decoder_attention_mask=decoder_attention_mask)
            loss = outputs[0]
            #print(loss)
            #print(batch["input_ids"].size())
            #exit()
            total_loss += outputs[0]
            loss.mean().backward()
            optimizer.step()
            scheduler.step()
        total_loss = total_loss/len(train_loader)
        if epoch >= 1 :
            print("Epoch {} loss {}".format(epoch,total_loss))
            f = open('logs1.txt','a') 
            f.write("Epoch {} loss {}".format(epoch,total_loss)+'\n')
            f.close()
            if total_loss < prev_loss :
                prev_loss = total_loss 
                #save the model
                torch.save(model.state_dict(), model_save_path)
        else :
            print("Epoch {} loss {}".format(epoch,total_loss))
            f = open('logs1.txt','a')
            f.write("Epoch {} loss {}".format(epoch,total_loss) + '\n')
            f.close()
            prev_loss = total_loss 
            #save the model
            torch.save(model.state_dict(), model_save_path)
    return model

def decode(batch):

    inputs = tokenizer(batch['incorrect'], padding="max_length", truncation=True, max_length=64)
    input_ids = torch.tensor(inputs.input_ids).to(device)
    attention_mask = torch.tensor(inputs.attention_mask).to(device)
    outputs = model.generate(input_ids, num_beams=5,no_repeat_ngram_size=2,early_stopping=True,attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["output"] = output_str
    return batch

def evaluation(model):

    model.eval()
    model.load_state_dict(torch.load(model_eval_path))
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    f1 = open("output_test.txt",'a')
    f2 = open("incorrect_test.txt",'a')
    f3 = open("correct_test.txt","a")
    bleu_scores = []
    for batch in test_loader :
        reference_corpus,candidate_corpus = [], []
        with torch.no_grad():
            results = decode(batch)
        print(results)
        outputs = results['output']
        labels = results['correct']
        incorrects = results['incorrect']
        for output,label,incorrect in zip(outputs,labels,incorrects) :
            f1.write(output+"\n")
            f2.write(incorrect+"\n")
            f3.write(label+"\n")
            candidate_corpus.append(tokenizer.tokenize(output))
            reference_corpus.append([tokenizer.tokenize(label)])
        bleu_scores.append(bleu_score(candidate_corpus, reference_corpus))
    f1.close()
    f2.close()
    f3.close()
    return sum(bleu_scores)/len(bleu_scores)

        



if __name__ == "__main__":
    
    pre_trained_path = args.pre_trained_path
    filepath = args.filepath
    epochs,lr,batch_size= int(args.epochs), float(args.lr),int(args.batch_size)
    model_save_path = args.model_save_path
    model_eval_path = args.model_eval_path
    train = args.train

    
    tokenizer = BertTokenizer.from_pretrained(pre_trained_path)
    model = model_load()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    print(device)
    print(train)
    print(torch.cuda.device_count())
    #if torch.cuda.device_count() > 1:
    #    print("Multi-GPU training")
    #    model = nn.DataParallel(model)

    # if train == "True" :
    #     train_loader = prepare_train_data()
    #     model = training(model)
    # test_data = prepare_test_data()
    print(evaluation(model))
