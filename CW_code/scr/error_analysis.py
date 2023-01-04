import pandas as pd
import numpy as np

def errorAnalysis(filePath):
	## Import the output predicted csv file
	##filePath = "/extra/manojgopale/AES_data/config3p1_15ktraining/batchSize_trials/size_2048/outputPredict.csv"
	
	## Load the dataFrame
	## 1st column is actual values, 2nd column is the predicted values
	df = pd.read_csv(filePath, header=None)
	
	##Create an error_df from the mis predicted  values
	error_df = df[df[0]!=df[1]].astype('category')
	
	##Create a 3rd column with actual-predicted values
	error_df[2] = error_df[0].astype(str).str.cat(error_df[1].astype(str), sep="-")
	
	##print the predicted error values
	totalCount = df[0].count()
	errorCount = error_df[2].count()
	accuracy = ((df[0].count()-error_df[2].count())/df[0].count())*100

	print("####---------####")
	print("Error values: \n%s" %(error_df[2].value_counts()))
	print("####---------####\n")

	print("Number of incorrect predictions = %s" %(errorCount))
	print("Accuracy = %s\n" %(accuracy))
