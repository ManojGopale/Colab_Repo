import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten,AveragePooling1D
from keras.utils import np_utils
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
#from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization
import pickle
import gzip
import pandas as pd
import numpy as np
import gc
import os
from datetime import date, datetime
from sklearn.utils import shuffle

from sklearn.utils import shuffle
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import clone_model
from keras.models import load_model

##Since the path to error_analysis file is already added to sys in run_3HL.py, we will directly 
##import those here

from error_analysis import errorAnalysis

## Commented for config5p4 to re-run
np.random.seed(9)

scaler = StandardScaler()

def process_inputs (dataPath):
	data = pd.read_pickle(dataPath)
	## Data is already shuffled during saving
	#dataShuffle = shuffle(data)
	## Apply to get the 1'st element of the entire key
	y_data = data.key.apply(lambda x: x[0]).values
	## For x_data, awe need to first convert the list of memmap array to separate columns of numbers .apply(pd.Series) does that
	## .values converts it to an array
	## then we apply scaler.fir_trasnform to it
	#x_data = scaler.fit_transform(data.trace.apply(pd.Series).values)

	## Without transformation
	x_data = data.trace.apply(pd.Series).values
	return x_data, y_data

## dataPath : Path where the working directory is located. 
## trainSize : Number of power traces per key to take
## testFlag : if true, then traina nd dev data won't be loaded
def getData(dataPath, moreDataPath, trainSize, trainFlag, devFlag, testFlag):
	##TODO changed devSize to 1000 for trails for more trainSize data. It was 5000 for default run
	devSize  = 1000
	testSize = 1000
	numTraces = 1500 ##Number of traces collected per key

	## Pre defining the arrays based on sizes of the data
	x_train = np.zeros((trainSize*256, numTraces))
	x_dev = np.zeros((devSize*256, numTraces))
	x_test = np.zeros((testSize*256, numTraces))

	y_train = np.zeros((trainSize*256, 1))
	y_dev = np.zeros((devSize*256, 1))
	y_test = np.zeros((testSize*256, 1))

	for index, val in enumerate(range(0,256)):
		print("Started data processing for %d key\n" %(val))
		trainStr = dataPath + "train_" + str(val) + ".pkl.zip"
		devStr   = dataPath + "dev_" + str(val) + ".pkl.zip"
		testStr  = dataPath + "test_" + str(val) + ".pkl.zip"

		##more training data path
		moreTrainStr = moreDataPath + "train_" + str(val) + ".pkl.zip"

		## Checking if the file size is 0, before processing data
		## This check is for cross config analysis, where traina nd dev are empty
		#if (os.stat(trainStr).st_size != 0):
		if (trainFlag):
			x_train_inter, y_train_inter = process_inputs(trainStr)
			## Trainsize will still be 15000, but we will take data from devSet to trainset
			x_train[trainSize*index: trainSize*(index) + 15000, : ] = x_train_inter
			y_train[trainSize*index: trainSize*(index) + 15000, 0] = y_train_inter

			if(trainSize == 28000):
				## Adding 9000 more data
				x_train_inter_more, y_train_inter_more = process_inputs(moreTrainStr)
				x_train[trainSize*(index) + 15000: (trainSize*(index) + 15000) + 9000, : ] = x_train_inter_more[0:9000, :]
				y_train[trainSize*(index) + 15000: (trainSize*(index) + 15000) + 9000, 0] = y_train_inter_more.reshape(9000,1)[0:9000, 0]

			print("Train= %s\n" %(trainFlag))
		else: 
			## Assigning the array's to 0's
			##NOTE: needs to change shape, but since we are always training, I am not changing this
			x_train[trainSize*index: trainSize*(index+1), : ] = np.zeros((trainSize,numTraces))
			y_train[trainSize*index: trainSize*(index+1), : ] = np.zeros((trainSize, 1))
			print("train= %s\n" %(trainFlag))

		#if (os.stat(devStr).st_size != 0):
		if (devFlag):
		## get the data for each sub part
			x_dev_inter, y_dev_inter     = process_inputs(devStr)
			print("x_dev_inter= %s, y_dev_inter= %s" %(x_dev_inter.shape, y_dev_inter.shape))
			x_dev[devSize*index: devSize*(index+1), : ] = x_dev_inter[0:devSize, :]
			y_dev[devSize*index: devSize*(index+1), 0 ] = y_dev_inter.reshape(5000, 1)[0:devSize, 0]
			print("Dev= %s\n" %(devFlag))
			print("x_dev= %s, y_dev= %s" %(x_dev.shape, y_dev.shape))

			if(trainSize==28000):
				## Adding 4000 traces to trainSet here
				x_train[trainSize*(index) + 15000 +9000: (trainSize*(index) + 15000) + 13000, : ] = x_dev_inter[1000:5000, :]
				y_train[trainSize*(index) + 15000 +9000: (trainSize*(index) + 15000) + 13000, 0] = y_dev_inter.reshape(5000, 1)[devSize:5000, 0]
				print("x_trainSize = %s, y_trainSize= %s" %(x_train.shape, y_train.shape))
		else:
			x_dev[devSize*index: devSize*(index+1), : ] = np.zeros((devSize, numTraces))
			y_dev[devSize*index: devSize*(index+1), : ] = np.zeros((devSize, 1))
			print("dev= %s\n" %(devFlag))

		## Test data is present so check is not performed
		if (testFlag):
			x_test_inter, y_test_inter   = process_inputs(testStr)
			x_test[testSize*index: testSize*(index+1), : ] = x_test_inter
			y_test[testSize*index: testSize*(index+1), 0 ] = y_test_inter
			print("Test= %s\n" %(testFlag))
			print("x_test= %s, y_test= %s" %(x_test.shape, y_test.shape))
		else:
			x_test[testSize*index: testSize*(index+1), : ] = np.zeros((testSize, numTraces))
			y_test[testSize*index: testSize*(index+1), : ] = np.zeros((testSize, 1))
			print("test= %s\n" %(testFlag))

		print("Finished data processing for %d key\n" %(val))
	
	## Clear variables
	x_train_inter = None
	x_dev_inter = None
	x_test_inter = None
	y_train_inter = None
	y_dev_inter = None
	y_test_inter = None
	x_train_inter_more = None
	y_train_inter_more = None
	print("\nCleared variables\n")

	##Not shuffling for debugging, should be removed
	## Shuffling
	## https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
	print("\nStarted shuffling of data\nx_train[0]= %s\ny_train[0]= %s" %(x_train[0], y_train[0]))
	print("\nx_train[12000]= %s\ny_train[12000]= %s" %(x_train[12000], y_train[12000]))
	x_train, y_train = shuffle(x_train, y_train, random_state=0)
	x_dev, y_dev = shuffle(x_dev, y_dev, random_state=0)
	x_test, y_test = shuffle(x_test, y_test, random_state=0)
	print("\nFinished shuffling of data\nx_train[0]= %s\ny_train[0]= %s" %(x_train[0], y_train[0]))
	print("\nx_train[12000]= %s\ny_train[12000]= %s" %(x_train[12000], y_train[12000]))

	##NOTE: Remove:
	#Mimport pdb; pdb.set_trace()
	## One hot assignment
	n_classes = 256
	y_train_oh = np_utils.to_categorical(y_train, n_classes)
	y_dev_oh = np_utils.to_categorical(y_dev, n_classes)
	y_test_oh = np_utils.to_categorical(y_test, n_classes)
	
	print("\nOne-hot encoded for outputs\n")
	## Standardizing train, dev and test
	x_train_mean = x_train.mean(axis=0)
	x_train_std  = x_train.std(axis=0)

	x_dev_mean = x_dev.mean(axis=0)
	x_dev_std = x_dev.mean(axis=0)

	x_test_mean = x_test.mean(axis=0)
	x_test_std = x_test.std(axis=0)

	#M## Concatenating train and dev
	#Mx_full = np.concatenate((x_train, x_dev), axis=0)
	#Mx_full_mean = x_full.mean(axis=0)
	#Mx_full_std = x_full.std(axis=0)

	## chunking the normalization process
	print("Strated normalizing\n")
	chunkSize = 28000
	chunkNum = int(len(x_train)/chunkSize)
	for chunkIndex in range(chunkNum):
		print("Train chunkIndx= %s, chunkNum = %s" %(chunkIndex, chunkNum))
		if(chunkIndex != chunkNum-1): 
			x_train[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize] = (x_train[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize]-x_train_mean)/x_train_std
		else:
			x_train[chunkIndex*chunkSize: ] = (x_train[chunkIndex*chunkSize: ] - x_train_mean)/x_train_std
			
	devChunkSize = 10000
	devChunkNum = int(len(x_dev)/devChunkSize)
	for devChunkIndex in range(devChunkNum):
		print("Dev chunkIndx= %s, chunkNum = %s" %(devChunkIndex, devChunkNum))
		if(devChunkIndex != devChunkNum-1): 
			x_dev[devChunkIndex*devChunkSize: (devChunkIndex+1)*devChunkSize] = (x_dev[devChunkIndex*devChunkSize: (devChunkIndex+1)*devChunkSize]-x_train_mean)/x_train_std
		else:
			x_dev[devChunkIndex*devChunkSize: ] = (x_dev[devChunkIndex*devChunkSize: ] - x_train_mean)/x_train_std
			

	## Need to do the same for test too
	return (x_train, y_train_oh), (x_dev, y_dev_oh), (x_test, y_test_oh)

class Classifier:
	def __init__(self, resultDir: str, modelName: str, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, numPowerTraces):
		""" Initialize parameters and sequential model for training
		"""
		self.resultDir = resultDir
		self.modelName = modelName
		self.x_train = x_train[:,0:numPowerTraces]
		self.x_dev = x_dev[:, 0:numPowerTraces]
		self.x_test = x_test[:, 0:numPowerTraces]
		self.y_train_oh = y_train_oh
		self.y_dev_oh = y_dev_oh
		self.y_test_oh = y_test_oh
		self.numPowerTraces = numPowerTraces
		
	def createFNNModel(self, hiddenLayer, actList, dropList, batchNorm, numPowerTraces):
		self.hiddenLayer = hiddenLayer
		self.actList = actList
		self.dropList = dropList
		self.batchNorm = batchNorm
		self.numPowerTraces = numPowerTraces
		self.model = Sequential()

		## Building models
		for index, layer in enumerate(hiddenLayer):
			if (index == 0):
				## 1st layer
				self.model.add(Dense(hiddenLayer[index], activation=actList[index], input_shape=(numPowerTraces,)))
			else:
				self.model.add(Dense(hiddenLayer[index], activation=actList[index]))

			if (batchNorm[index]):
				self.model.add(BatchNormalization())
			
			self.model.add(Dropout(dropList[index]))

		## Last layer is softmax layer
		self.model.add(Dense(256, activation='softmax'))

		#self.model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
		#print("Model summary\n")
		#print(self.model.summary())
		return self.model
	
	def createRNNModel(self, hiddenList, denseList, denseAct, rnnDropList, denseDropList, typeOfLayer, biDir, trainShape):
		self.model = Sequential()
		print("hiddenList= %s" %(hiddenList))
		#print("rnnDropList= %s\n recurrDropList= %s\n denseDropList= %s" %(rnnDropList, recurrDropList, denseDropList))
		print("rnnDropList= %s\ndenseDropList= %s" %(rnnDropList, denseDropList))
		for index, hiddenNum in enumerate(hiddenList):
			print("index=%s, hiddenNum=%s" %(index, hiddenNum))
			print("type\nhiddenNum=%s\nrnnDrop=%s\nbiDir=%s\n"\
			%(type(hiddenNum), type(rnnDropList[index]),type(biDir)))
			if(typeOfLayer == "GRU"):
				##Input Layer
				if(index==0):
					if(biDir):
						print("BiDir, %s, index=%s, hiddenNum=%s, trianShape=%s" %(typeOfLayer, index, hiddenNum, (1,self.numPowerTraces)))
						#self.model.add(Bidirectional(GRU(hiddenNum,dropout=rnnDropList[index], return_sequences=True), input_shape=trainShape[-2:]))
						self.model.add(Bidirectional(GRU(int(hiddenNum), return_sequences=True), input_shape=(1,self.numPowerTraces)))
					else:
						print("%s, index=%s, hiddenNum=%s, trianShape=%s" %(typeOfLayer, index, hiddenNum, (1,self.numPowerTraces)))
						#self.model.add(GRU(hiddenNum,dropout=rnnDropList[index],return_sequences=True, input_shape=trainShape[-2:]))
						self.model.add(GRU(int(hiddenNum),return_sequences=True, input_shape=(1,self.numPowerTraces)))
				else:
					if(biDir):
						print("BiDir, %s, index=%s, hiddenNum=%s" %(typeOfLayer, index, hiddenNum))
						#self.model.add(Bidirectional(GRU(hiddenNum,dropout=rnnDropList[index], return_sequences=True)))
						self.model.add(Bidirectional(GRU(int(hiddenNum), return_sequences=True)))
					else:
						print("%s, index=%s, hiddenNum=%s" %(typeOfLayer, index, hiddenNum))
						#self.model.add(GRU(hiddenNum,dropout=rnnDropList[index], return_sequences=True))
						self.model.add(GRU(int(hiddenNum), return_sequences=True))
			elif(typeOfLayer == "LSTM"):
				##Input Layer
				if(index==0):
					if(biDir):
						print("BiDir, %s, index=%s, hiddenNum=%s, trianShape=%s" %(typeOfLayer, index, hiddenNum, trainShape[-2:]))
						self.model.add(Bidirectional(LSTM(hiddenNum,dropout=rnnDropList[index], return_sequences=True), input_shape=trainShape[-2:]))
					else:
						print("%s, index=%s, hiddenNum=%s, trianShape=%s" %(typeOfLayer, index, hiddenNum, trainShape[-2:]))
						self.model.add(LSTM(hiddenNum,dropout=rnnDropList[index], return_sequences=True, input_shape=trainShape[-2:]))
				else:
					if(biDir):
						print("BiDir, %s, index=%s, hiddenNum=%s" %(typeOfLayer, index, hiddenNum))
						self.model.add(Bidirectional(LSTM(hiddenNum, dropout=rnnDropList[index], return_sequences=True)))
					else:
						print("%s, index=%s, hiddenNum=%s" %(typeOfLayer, index, hiddenNum))
						self.model.add(LSTM(hiddenNum,dropout=rnnDropList[index], return_sequences=True))
			else:
				print("typeOfLayer not GRU or LSTM\n")
		
		##Dense Layers
		print("densList=%s, denseAct=%s" %(denseList, denseAct))
		for denseIdx, denseNum in enumerate(denseList):
			print("Dense, num=%s, act=%s" %(denseNum, denseAct[denseIdx])) 
			self.model.add(Dense(denseNum,activation=denseAct[denseIdx])) 
			self.model.add(Dropout(denseDropList[denseIdx]))

		##Output Layer
		self.model.add(Dense(256, activation='softmax'))
		return self.model

	def createCNNModel(self, cnnFilters, kernelSizes, cnnActs, trainShape, batchNorm, isMaxPool, maxPoolSizes, denseList, denseActs, denseDropouts):
		self.model = Sequential()
		print("inside createCNNModel")
		print("cnnFilters=%s\nkernelSizes=%s\ncnnActs=%strainShape=%s\nbatchNorm=%s\nisMaxPool=%s\nmaxPoolSizes=%s\n" %(cnnFilters, kernelSizes, cnnActs, trainShape, batchNorm, isMaxPool, maxPoolSizes))
		for index, cnnFilter in enumerate(cnnFilters):
			if(index == 0):
				self.model.add(Conv1D(filters=cnnFilters[index], kernel_size=int(kernelSizes[index]), activation=cnnActs[index], input_shape=trainShape))
			else:
				self.model.add(Conv1D(filters=cnnFilters[index], kernel_size=int(kernelSizes[index]), activation=cnnActs[index]))
			##Add dropouts 
			if (batchNorm[index]):
				self.model.add(BatchNormalization())
			if(isMaxPool[index]):
				##Added avergae pooling for experiments
				#self.model.add(MaxPooling1D(pool_size=maxPoolSizes[index]))
				self.model.add(AveragePooling1D(pool_size=maxPoolSizes[index]))
		##Addd flatten layer
		self.model.add(Flatten())
		print("denseList=%s\ndenseActs=%s\ndenseDropouts=%s\n" %(denseList, denseActs, denseDropouts))
		for idx, denseLayer in enumerate(denseList):
			self.model.add(Dense(denseList[idx], activation=denseActs[idx]))
			self.model.add(Dropout(denseDropouts[idx]))

		##Output layer
		self.model.add(Dense(256, activation='softmax'))

		return self.model

	def train(self, batchSize):
		""" Train the model with the training data
		batchSize : batch size during trainig
		"""

		Epochs = 1000

		logFile = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(batchSize) +'.log'
		csv_logger = CSVLogger(logFile, append=True, separator="\t")
		
		earlyStop = EarlyStopping(monitor='val_categorical_accuracy', patience=10, mode='auto', verbose=1, restore_best_weights=True)
		
		##filePath = self.resultDir + '/' + self.modelName + '_checkPoint_best_model.hdf5'
		#### This file will include the epoch number when it gets saved.
		##repeatingFile = self.resultDir + '/' + self.modelName +'_{epoch:02d}_epoch_acc_{accVar:.2f}.hdf5'
		#### By default the every_10epochs will save the model at every 10 epochs
		##checkPoint = newCallBacks.ModelCheckpoint_every_10epochs(filePath, repeatingFile, self.x_test, self.y_test_oh , monitor='val_categorical_accuracy', verbose=1, save_best_only=True, every_10epochs=True)
		
		self.history = self.model.fit(self.x_train, self.y_train_oh, batch_size= batchSize, epochs=Epochs, verbose=1, shuffle= True, validation_data=(self.x_dev, self.y_dev_oh), callbacks=[csv_logger, earlyStop])

	def evaluate(self):
		""" Evaluate the model on itself
		"""

		## We should be evaluating on dev dataset as well, so commenting x_test
		#self.model_score = self.model.evaluate(self.x_test, self.y_test_oh, batch_size=2048)
		self.model_score = self.model.evaluate(self.x_dev, self.y_dev_oh, batch_size=2048)
		print("%s score = %f\n" %(self.modelName, self.model_score[1]))

		##Saving atucal vs predicted predictions
		##np.argmax returns the index where it see's 1 in the row
		#y_pred = np.argmax(self.model.predict(self.x_test, batch_size=2048), axis=1)
		y_pred = np.argmax(self.model.predict(self.x_dev, batch_size=2048), axis=1)

		## vstack will stack them in 2 rows, so we use Trasnpose to get them in column stack
		#output_predict = np.vstack((np.argmax(self.y_test_oh, axis=1), y_pred)).T
		output_predict = np.vstack((np.argmax(self.y_dev_oh, axis=1), y_pred)).T

		self.outputFile = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + "outputPredict.csv" 
		
		np.savetxt(self.outputFile, output_predict, fmt="%5.0f", delimiter=",")

		##Error Analysis of the prediction
		errorAnalysis(self.outputFile)

		return self.model_score

	def saveModel(self):
		""" Save the model
		"""
		saveStr = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + '.h5'
		print("Saving model to\n%s\n" %(saveStr))
		self.model.save(saveStr)

	def keyAccuracy(self):

		df = pd.read_csv(self.outputFile, header=None)
		
		error_df = df[df[0]!=df[1]].astype('category')
		error_df[2] = error_df[0].astype(str).str.cat(error_df[1].astype(str), sep="-")
		
		totalCount = df[0].count()
		errorCount = error_df[2].count()
		accuracy = ((df[0].count()-error_df[2].count())/df[0].count())*100
		
		
		## to get the accuracy of individual keys, we need to count the number of rows in error_df for the same key
		## dubtract it from total data elements and divide it by the total number of data elements for each.
		
		## For example
		#c1 = df[0][df[0] == 0].count()
		#error_0 = error_df[0][error_df[0]==0].count()
		#
		#acc_0 = ((c1-error_0)/c1)*100
		
		## Now to loop it
		keyAcc = pd.DataFrame(columns={'key', 'acc'})
		
		for key in range(256):
			totalKey = df[0][df[0]==key].count()
			keyErrors = error_df[0][error_df[0]==key].count()
			acc = ((totalKey-keyErrors)/totalKey)*100
			keyAcc.loc[key] = {'key': key, 'acc': acc}
			#print("key= %s, acc= %s" %(key, acc))
		
		## Save to tsv
		saveFile = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + "keyAccuracy.tsv" 
		keyAcc.to_csv(saveFile, sep='\t', header=True, index=False)
		
		##Checking for entopy calculations of the predictions
		pred = self.model.predict(self.x_dev, batch_size=2048)
		
		## np.argsort(-pred) gets the order in which the indexes will be arranged 
		## np.argsort() of above will return the rank of the indexes with their values in the coressponding indexes
		rank = np.argsort(np.argsort(-pred))
		
		## Get the actual predictions from y_dev, need to convert one-hot to actual numbers
		dev_actual = np.argmax(self.y_dev_oh, axis=1)
		
		## Get the prediction ranks for each prediction
		prediction_ranks = rank[np.arange(len(dev_actual)), dev_actual]
		
		## getting the mean will also get the accuracy for the recall_at
		## recall = 1 , will get you accuracy at one shot
		recall_1 = np.mean(prediction_ranks < 1)
		recall_10 = np.mean(prediction_ranks < 10)
		recall_25 = np.mean(prediction_ranks < 25)
		recall_40 = np.mean(prediction_ranks < 40)
		recall_50 = np.mean(prediction_ranks < 50)
	
		print("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s" %(self.modelName, recall_1, recall_10, recall_25, recall_40, recall_50))

		## Create confusion matriux for aggregated computation
		conf = confusion_matrix(df[0], df[1])
		## Get the index for each row's max value. This is the column number in each row where the max value is located
		rowArgMax = np.argmax(conf, axis=1)

		logFile = self.resultDir + '/../log/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + "keyAccuracy.log" 

		with open(logFile, 'a') as f:
			f.write("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s\n\n" %(self.modelName, recall_1, recall_10, recall_25, recall_40, recall_50))
			for row in range(256):
				if (row != rowArgMax[row]):
					print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
					f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---\n" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
				else:
					print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
					f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)\n" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))

	def getPerplexity(self, x_dev, y_dev_oh, model):
		## Predict probabilities for each dev set
		pred = model.predict(x_dev, batch_size=2048)
		
		print("len of y_dev_oh = %s" %(len(y_dev_oh.shape)))
		if(len(y_dev_oh.shape) > 2):
			y_dev_oh = y_dev_oh.reshape(-1, 256)

		dev_actual = np.argmax(y_dev_oh, axis=1)
		dev_actual = dev_actual.reshape(dev_actual.shape[0], 1) #shape should be in array form for take_along_axis
		
		pred_prob = np.take_along_axis(pred, dev_actual, 1)
		
		## loop to calculate perplexity
		product = 1
		N = pred_prob.shape[0] #Total number of dev samples
		for index in range(N):
			## Take the n'th root of each value and then multiply them
			product = product * np.power(pred_prob[index], (1/N))
		
		perplexity = 1/product
		#Mprint("perplexity of model=%s on %s of %s is %s\n" %(self.saveStr.split("/")[-1], self.configName, dataset ,perplexity))
		print("perplexity is %s\n" %(perplexity))
		return perplexity



##https://stackoverflow.com/questions/56079223/custom-keras-data-generator-with-yield
class DataGenerator(tf.keras.utils.Sequence):
	def __init__(self, files, batch_size=32, shuffle=True, random_state=42, typeOfData="train", numPowerTraces=6000, typeOfModel="FNN"):
		'Initialization'
		self.files = files ##list of files to load
		#self.labels = labels
		self.batch_size = batch_size ##Number of files in each batch
		self.shuffle = shuffle
		self.random_state = random_state
		self.typeOfData = typeOfData ##The key used to load data from files
		self.numPowerTraces = numPowerTraces ##This is used for CNN and RNN reshaping
		self.typeOfModel = typeOfModel ##Model tyep, FNN, CNN, RNN
		self.on_epoch_end()
		self.monitor_op=np.greater

	def __len__(self):
		##This can be changed to totalFiles/batchSize
		# total_files = 256
		# batch_files = 32
		return int(np.floor(len(self.files)/self.batch_size))
		#return int(np.floor(len(self.files) / self.batch_size))

	def __getitem__(self, index):
		# Generate indexes of the batch
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		files_batch = [self.files[k] for k in indexes]
		# Generate data
		if (self.typeOfData == "data"):
			x, y = self.__data_generation(files_batch)
		elif(self.typeOfData == "train"):
			x, y = self.__train_generation(files_batch)
		elif(self.typeOfData == "dev"):
			x, y = self.__dev_generation(files_batch)
		return x, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.files))
		if self.shuffle == True:
			np.random.seed(self.random_state)
			np.random.shuffle(self.indexes)
		##Save model after each epoch
		#if self.monitor_op(current, self.best):
		#if self.verbose > 0:
		#io_utils.print_msg(
		#f"\nEpoch {epoch + 1}: {self.monitor} "
		#"improved "
		#f"from {self.best:.5f} to {current:.5f}, "
		#f"saving model to {filepath}"

	def __data_generation(self, files):
		count = 0 
		for dataFile in files:
			if(count == 0 ):
				x = np.load(dataFile)["data"][:,:-1]
				y = np.load(dataFile)["data"][:,-1]
			else:
				x = np.concatenate([x, np.load(dataFile)["data"][:,:-1]], axis=0)
				y = np.concatenate([y, np.load(dataFile)["data"][:,-1]], axis=0)
			count = count + 1

		##Sine we have -128 to 127 range, we will add 256 to convert it to 0-255
		add_index = (y<0)
		y[add_index] += 256

		##Standardize + reshape
		x, y = self.__reshape(self.typeOfModel, x, y)
		#sc = StandardScaler().fit(x_train)
		#x_train = sc.transform(x_train)
		#x_train = x_train.reshape(-1,numPowerTraces,1)
		#print("x_train shape= %s" %(x_train.shape,))
		#trainShape = x_train.shape[-2:]
		return x, y
	

	def __train_generation(self, files):
		count = 0 
		for dataFile in files:
			if(count == 0 ):
				x = np.load(dataFile)["train"][:,:-1]
				y = np.load(dataFile)["train"][:,-1]
			else:
				x = np.concatenate([x, np.load(dataFile)["train"][:,:-1]], axis=0)
				y = np.concatenate([y, np.load(dataFile)["train"][:,-1]], axis=0)
			count = count + 1

		##Sine we have -128 to 127 range, we will add 256 to convert it to 0-255
		add_index = (y<0)
		y[add_index] += 256

		##Standardize + reshape
		x, y = self.__reshape(self.typeOfModel, x, y)
		#sc = StandardScaler().fit(x_train)
		#x_train = sc.transform(x_train)
		#x_train = x_train.reshape(-1,numPowerTraces,1)
		#print("x_train shape= %s" %(x_train.shape,))
		#trainShape = x_train.shape[-2:]
		return x, y
	

	def __dev_generation(self, files):
		count = 0 
		for dataFile in files:
			if(count == 0 ):
				x = np.load(dataFile)["dev"][:,:-1]
				y = np.load(dataFile)["dev"][:,-1]
			else:
				x = np.concatenate([x, np.load(dataFile)["dev"][:,:-1]], axis=0)
				y = np.concatenate([y, np.load(dataFile)["dev"][:,-1]], axis=0)
			count = count + 1

		##Sine we have -128 to 127 range, we will add 256 to convert it to 0-255
		add_index = (y<0)
		y[add_index] += 256

		##Standardize + reshape
		x, y = self.__reshape(self.typeOfModel, x, y)
		#sc = StandardScaler().fit(x_train)
		#x_train = sc.transform(x_train)
		#x_train = x_train.reshape(-1,numPowerTraces,1)
		#print("x_train shape= %s" %(x_train.shape,))
		#trainShape = x_train.shape[-2:]
		return x, y
	
	def __reshape(self, typeOfModel, x_data, y_data):
		##Commented if using standardized data
		#sc = StandardScaler().fit(x_data)
		#x_data = sc.transform(x_data)

		if(typeOfModel == "CNN"):
			x_data = x_data.reshape(-1,self.numPowerTraces,1)
		elif(typeOfModel == "RNN"):
			x_data = x_data.reshape(-1,1,self.numPowerTraces) ##batchSize, timesteps, features
			y_data = y_data.reshape(-1,1,1) ##When we use one-hot the shape should be y_data.reshape(-1,1,256)

		return x_data, y_data


