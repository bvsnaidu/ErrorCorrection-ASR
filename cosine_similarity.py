import torch
from torch.utils.data import Dataset
import argparse
from transformers import BertTokenizer,BertModel
import torch.nn.functional as F
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description ='cosine similarity between sentence')
parser.add_argument('--pre_trained_path',dest='pre_trained_path',required=True,help='Specify file path of pre-trained model')
parser.add_argument('--filepath',dest='filepath',required=True,help='Data Directory')
parser.add_argument('--batch_size',dest='batch_size',required=True,help='Batch size')
args = parser.parse_args()


def read_file(path):
	file = open(path,"r",encoding="utf-8")
	lines = file.read().split('\n')
	lines = lines[:-1]
	file.close()
	return lines

class CustomDataset(Dataset):

	def __init__(self,incorrect,correct):
		self.incorrect = incorrect
		self.correct = correct

	def __len__(self):
		return len(self.incorrect)
	
	def __getitem__(self,index):
		return self.incorrect[index],self.correct[index]

class CosineSimilarity :

	def __init__(self,pre_trained_path,device):
		self.device = device
		self.tokenizer = BertTokenizer.from_pretrained(pre_trained_path)
		self.model = BertModel.from_pretrained(pre_trained_path)
		self.model = self.model.to(self.device)

	def similarity(self,hypothesis,reference):
		encoded_hypothesis = self.tokenizer(hypothesis, padding="max_length", truncation=True, max_length=64,return_tensors='pt')
		encoded_hypothesis = encoded_hypothesis.to(self.device)
		output_hypothesis = self.model(**encoded_hypothesis)
		encoded_reference = self.tokenizer(reference, padding="max_length", truncation=True, max_length=64,return_tensors='pt')
		encoded_reference = encoded_reference.to(self.device)
		output_reference = self.model(**encoded_reference)
		similarity_scores = []
		for i in range(output_reference[1].shape[0]):
			out_reference = output_reference[1][i,:]
			out_reference = out_reference.reshape(1,768)
			out_hypothesis = output_hypothesis[1][i,:]
			out_hypothesis = out_hypothesis.reshape(1,768)
			similarity_scores.append(F.cosine_similarity(out_reference, out_hypothesis, dim=1).detach().cpu().numpy()[0])
		return similarity_scores

if __name__ == "__main__" :
	pre_trained_path = args.pre_trained_path
	filepath = args.filepath		
	batch_size = int(args.batch_size)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	incorrect_train = read_file(filepath+"//"+"incorrect_train.txt")
	correct_train = read_file(filepath+"//"+"correct_train.txt")
	cosine_similarity_train = open(filepath+"//"+"cosine_similarity_train.txt","a")
	incorrect_test = read_file(filepath+"//"+"incorrect_test.txt")
	correct_test = read_file(filepath+"//"+"correct_test.txt")
	cosine_similarity_test = open(filepath+"//"+"cosine_similarity_test.txt","a")

	sentence_similarity = CosineSimilarity(pre_trained_path,device)

	train_dataset = CustomDataset(incorrect_train,correct_train)
	train_datagen = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
	for hypothesis,reference in train_datagen :
		scores = sentence_similarity.similarity(hypothesis,reference)
		for score in scores :
			score = round(score,3)
			cosine_similarity_train.write(str(score)+"\n")
	cosine_similarity_train.close()

	test_dataset = CustomDataset(incorrect_test,correct_test)
	test_datagen = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
	for hypothesis,reference in test_datagen :
		scores = sentence_similarity.similarity(hypothesis,reference)
		for score in scores :
			score = round(score,3)
			cosine_similarity_test.write(str(score)+"\n")
	cosine_similarity_test.close()
		
			

	
	
		
	

	
