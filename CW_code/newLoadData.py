import numpy as np
import pandas as pd
import os
import copy
from sklearn.utils import shuffle
from keras.utils import np_utils

class Data:
	def processInput(self, pklPath, tracesPerKey):
		data = pd.read_pickle(pklPath)
		##Generate a randome integer to be used for shuffling
		random_int = np.random.randint(1000)
		print("random_int= %s, for shuffling data" %(random_int))
		##If an identical random integer is used during sample(frac=1, random_state), we should get the same sampling order
		## In case we dont, its still ok, since all the keys are for 1 particular key
		x_data = data.sample(frac=1, random_state=random_int).iloc[:tracesPerKey,:].trace.apply(pd.Series)
		y_data = data.sample(frac=1, random_state=random_int).iloc[:tracesPerKey,:].key.apply(lambda x: x[0])

		return x_data, y_data

	def getData(self, dataPath, tracesPerKey, dataSetType):
		##Load data for all keys
		x_data = pd.DataFrame()
		y_data = pd.DataFrame()

		for key in range(256):
			print("Started data processing for %d key\n" %(key))
			pklPath = dataPath + "/" + dataSetType + "_" + str(key) + ".pkl.zip"
			x_data_inter, y_data_inter = self.processInput(pklPath, tracesPerKey)
			x_data = pd.concat([x_data, x_data_inter], axis=0, ignore_index=True)
			y_data = pd.concat([y_data, y_data_inter], axis=0, ignore_index=True)

		return x_data, y_data
		
	def shuffleData(self, x_data, y_data):
		"""Shuffle the data, with random_state= None as default
		"""
		##NOTE: If there is any problem's with shuffling, go back to the utlis.shuffle function
		## Shuffling
		## https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
		#Mx_data, y_data = shuffle(x_data, y_data, random_state=None)

		##Commenting this for log readability
		#Mprint("\nStarted shuffling of data\nx_data[0]= %s\ny_data[0]= %s" %(x_data.iloc[0], y_data.iloc[0]))
		#Mprint(*x_data.iloc[0])
		#Mprint("\nx_data[100]= %s\ny_data[100]= %s" %(x_data.iloc[100], y_data.iloc[100]))
		#Mprint(*x_data.iloc[100])
		print("Before shuffling\ny_data[0]= %s\ny_data[100]= %s" %(y_data.iloc[0], y_data.iloc[100]))
		newIndex = np.random.permutation(x_data.index)
		## incase the index's from concat are duplicate, we will reset them before shuffling
		x_data.reset_index(drop=True, inplace=True)
		y_data.reset_index(drop=True, inplace=True)

		## the idea here is to shuffle the index keeping the order same and then reshuffling them to get the shuffled df
		## set_index sets the index's to the newIndex without changing the order.
		## we will have to sort them, so that the new order is randomized
		x_data.set_index(newIndex, inplace=True)
		y_data.set_index(newIndex, inplace=True)

		## Sorting indexes based on the index we just set
		x_data.sort_index(inplace=True)
		y_data.sort_index(inplace=True)

		#Mprint("\nFinished shuffling of data\nx_data[0]= %s\ny_data[0]= %s" %(x_data.iloc[0], y_data.iloc[0]))
		#Mprint(*x_data.iloc[0])
		#Mprint("\nx_data[100]= %s\ny_data[100]= %s" %(x_data.iloc[100], y_data.iloc[100]))
		#Mprint(*x_data.iloc[100])
		print("After shuffling\ny_data[0]= %s\ny_data[100]= %s" %(y_data.iloc[0], y_data.iloc[100]))
		return x_data, y_data
	
	def oneHotY(self, y_data):
		"""One hot the y_values in the dataset
		"""
		## One hot assignment
		n_classes = 256
		y_data_oh = np_utils.to_categorical(y_data, n_classes)
		
		print("\nOne-hot encoded for outputs\n")
		return y_data_oh
		
	def getStdParam(self, x_data):
		"""Get mean and std deviation of the x_data
		"""
		## get mean and std dev. for each column with axis=0
		x_data_mean = x_data.mean(axis=0)
		x_data_std  = x_data.std(axis=0)
		##Save initial mean and std for future use
		x_data_mean.to_csv("/xdisk/tosiron/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/summaryStats/mean_20000perkey.csv", index=False, header=False)
		x_data_std.to_csv("/xdisk/tosiron/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/summaryStats/std_20000perkey.csv", index=False, header=False)
		return x_data_mean, x_data_std
	
	def stdData(self, x_data, meanPath, stdPath):
		""" Standardize the x_data, by the mean and std deviation 
		Make sure the x_data is in numpy format
		meanPath, stdPath are the paths to the csv files storing the coressponding values
		"""
		##Mprint("Started normalizing\n")
		##M## since we are having mean and std dev. for each column, we will have to sample each row to apply them, hence axis=1
		##Mreturn x_data.apply(lambda x: (x-meanToApply)/stdToApply, axis=1).to_numpy()
		meanToApply = pd.read_csv(meanPath, header=None).to_numpy()
		stdToApply = pd.read_csv(stdPath, header=None).to_numpy()
		print("Loaded mean and std files from\n%s\n%s\n" %(meanPath, stdPath))
		## Reshaping so that it matches the standardization function and not error out
		meanToApply = meanToApply.reshape(meanToApply.shape[0], )
		stdToApply = stdToApply.reshape(stdToApply.shape[0], )

		print("Started normalizing\n")
		chunkSize = 28000
		if(x_data.shape[0]>chunkSize):
			chunkNum = int(x_data.shape[0]/chunkSize)
			for chunkIndex in range(chunkNum):
				print("chunkIndx= %s, chunkNum = %s" %(chunkIndex, chunkNum))
				if(chunkIndex != chunkNum-1): 
					x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize] = (x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize]-meanToApply)/stdToApply
				else:
					x_data[chunkIndex*chunkSize: ] = (x_data[chunkIndex*chunkSize: ] - meanToApply)/stdToApply
		else:
			print("ChunkSize less than 28000\n")
			x_data[0: ] = (x_data[0: ] - meanToApply)/stdToApply

		return x_data
		
	def fitTransform(self, x_data):
		"""
		Get the transform 
		MAke sure x_data is numpy.ndarray
		"""
		self.full_mean = x_data.mean()
		self.full_std = x_data.std(ddof=1)
		print("Full mean= %s, full std Dev = %s\n" %(self.full_mean, self.full_std))
		return (x_data-self.full_mean)/(self.full_std), self.full_mean, self.full_std
	
	def transform(self, x_data):
		"""
		Use the mean and std from fitTransform to standardize the data
		"""
		print("------transform ------\nFull mean= %s, full std Dev = %s\n" %(self.full_mean, self.full_std))
		return (x_data-self.full_mean)/(self.full_std)

