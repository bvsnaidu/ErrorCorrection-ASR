import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import pdb

parser = argparse.ArgumentParser(description ='Dataset & stats generation')
parser.add_argument('--file',dest='filepath',required=True,help='Specify file path which has excel')
parser.add_argument('--d',dest='distribution',required=True,help='Distribution between incorrect and correct samples')
parser.add_argument('--split',dest='split',required=True,help='train test split')
args = parser.parse_args()

def extract(filepath,distribution,split) :
	overall_fail_count = 0
	excels = os.listdir(filepath)
	with open('incorrect_train.txt','a') as f1, open('correct_train.txt','a') as f2, open('incorrect_test.txt','a') as f3, open('correct_test.txt','a') as f4 :
		for excel in excels :
			overall_fail_count_excel = 0
			if excel.endswith('xlsx'):
				df = pd.ExcelFile(filepath+r'/'+excel)
				sheet_names = df.sheet_names
				for sheet in sheet_names[1:]:
					df = pd.read_excel(filepath+r'/'+excel,sheet_name=sheet,index_col=0)
					df_fail = df[df.iloc[:,-1] == 'FAIL']
					df_pass = df[df.iloc[:,-1] == 'PASS']
					x_incorrect = list(df_fail.iloc[:,0])
					y_correct = list(df_fail.iloc[:,1])
					correct_example = (int(df_fail.shape[0]/float(distribution))-df_fail.shape[0])
					if df_pass.shape[0] > 1 and df_pass.shape[0] >= correct_example :
						df_pass_random = df_pass.sample(n=correct_example)
						x_correct =  list(df_pass_random.iloc[:,0])
					else :
						x_correct = []
					x = x_incorrect + x_correct
					y = y_correct + x_correct
					if len(x) > 1:
						x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=float(split), random_state=42)
						for incorrect,correct in zip(x_train,y_train):
							f1.write(incorrect+"\n")
							f2.write(correct+"\n")
						for incorrect,correct in zip(x_test,y_test):
							f3.write(incorrect+"\n")
							f4.write(correct+"\n")
					elif len(x) == 1 :
						f1.write(x[0]+'\n')
						f2.write(y[0]+'\n')
					else :
						pass
					overall_fail_count_excel += df_fail.shape[0]
					print("Excel name {} ---  sheet name {} ---  Failed cases {}".format(excel[:-5],sheet,overall_fail_count_excel))
			overall_fail_count += overall_fail_count_excel
			print("--------------------------------------")
			print("Fail count in Excel {} is {}".format(excel[:-5],overall_fail_count_excel))
		print("---------------------------------------")
		print("Overall fail count is {}".format(overall_fail_count))
				


if __name__ == "__main__" :

	extract(args.filepath,args.distribution,args.split)
