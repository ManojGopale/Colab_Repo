import sys
sys.path.insert(0, '/xdisk/tosiron/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/')
import classify_general
import time
import numpy as np
import pandas as pd
import gc, os, random, copy
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, normalize, StandardScaler
import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras import layers
from scipy import signal
import glob
import datetime


from optparse import OptionParser

parser = OptionParser()
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 7500)
parser.add_option('--devSize',
									action = 'store', type='int', dest='devSize', default = 7500)
parser.add_option('--resultDir',
									action = 'store', type='string', dest='resultDir', default = '/xdisk/tosiron/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/')
parser.add_option('--modelName',
									action = 'store', type='string', dest='modelName', default = 'chipWhispererModel')
parser.add_option('--trainFlag',
									action = 'store', type='int', dest='trainFlag', default = 1)
parser.add_option('--devFlag',
									action = 'store', type='int', dest='devFlag', default = 1)
parser.add_option('--testFlag',
									action = 'store', type='int', dest='testFlag', default = 0)
parser.add_option('--numPowerTraces',
									action = 'store', type='int', dest='numPowerTraces', default = 1500)
parser.add_option('--typeOfStd',
									action = 'store', type='string', dest='typeOfStd', default = 'col')
parser.add_option('--modelType',
									action = 'store', type='string', dest='modelType', default = 'FNN')
parser.add_option('--chipType',
									action = 'store', type='string', dest='chipType', default = 'cw_f303_1500')

(options, args) = parser.parse_args()

########
trainSize = options.trainSize
devSize = options.devSize
resultDir = options.resultDir
modelName = options.modelName
trainFlag = options.trainFlag
devFlag = options.devFlag
testFlag = options.testFlag
numPowerTraces = options.numPowerTraces
typeOfStd = options.typeOfStd
modelType = options.modelType
chipType  = options.chipType ##This is the chip data that we want to load
#######

#####------------Define class weights-----------------######
def getClassWeights(keysToWeigh):
	classWeight = dict()
	for key in range(256):
		if(key in keysToWeigh):
			##With 50, the model only learns these classes and not others.
			classWeight[key] = 25
		else:	
			classWeight[key] = 1		
	return classWeight
			
#####------------Define class weights-----------------######

#####------------Load from the dataset-----------------######
np.random.seed()

dataDir = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_f415_6000_stdScaler_1024/" ##keys-> data
#dataDir = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_f303_6000_shuffle_stdScaler_1024/" ##keys-> train, dev
#dataDir = "/xdisk/tosiron/manojgopale/xdisk/gem5DataCollection/cw_f303_6000_shuffle_onlyMean_1024/" ##keys-> data

filesPerStep = 4 ##This is the number of files to take per step
training_files = [x for x in glob.glob(dataDir + '/train_*.npz')]
#training_generator = classify_general.DataGenerator(training_files, batch_size=2, random_state=np.random.randint(1,1000), typeOfData="train", typeOfModel=modelType)
training_generator = classify_general.DataGenerator(training_files, batch_size=filesPerStep, random_state=np.random.randint(1,1000), typeOfData="data", typeOfModel=modelType)

validation_files = [x for x in glob.glob(dataDir + '/dev_*.npz')]
#validation_generator = classify_general.DataGenerator(validation_files, batch_size=16, random_state=np.random.randint(1,1000), typeOfData="dev", typeOfModel=modelType)
validation_generator = classify_general.DataGenerator(validation_files, batch_size=16, random_state=np.random.randint(1,1000), typeOfData="data", typeOfModel=modelType)
#####------------Load from the dataset-----------------######

x_train= np.random.randint(10, size=(10,3))
y_train_oh = np.random.randint(10, size=(10,3))
x_dev = np.random.randint(10, size=(10,3))
y_dev_oh = np.random.randint(10, size=(10,3))

classifier = classify_general.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_dev, y_dev_oh, numPowerTraces)
##These are the keys performing less than 25%
keysToWeigh = [35, 51, 52, 58, 67, 81, 84, 97, 102, 104, 109, 114, 119, 142, 148, 150, 164, 169, 170, 178, 181, 190, 214, 249]

if (modelType == "FNN"):
	######---------------------FNN------------------------######
	numHiddenLayers = 3
	actList = ['elu', 'elu', 'elu']
	dropList = [0.2, 0.2, 0.2]
	## Added to run random models for NN
	#MallDrop = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
	#MdropList = [allDrop[i] for i in np.random.random_integers(0, len(allDrop)-1, numHiddenLayers).tolist()]
	batchNorm = [0, 0, 0]
	batchSize = 1024
	hiddenLayer = [1024, 2048, 1024]
	#MhiddenLayer = [np.power(2,i) for i in np.random.random_integers(5,9, size=numHiddenLayers)]## runs 30 onwards with powers of 2
	model = classifier.createFNNModel(hiddenLayer, actList, dropList, batchNorm, numPowerTraces)
	######---------------------FNN------------------------######
elif (modelType == "RNN"):
	######---------------------RNN------------------------######
	print("Hardcode RNN params")
	#model = classifier.createRNNModel(hiddenList, denseList, denseAct, rnnDropList, denseDropList, typeOfLayer, biDir, trainShape)
	##run_cw_f303_1500_RNN_1
	hiddenList= [512, 256, 1024, 128]
	denseList= [256, 256]
	denseAct= ['tanh', 'selu']
	rnnDropList= [0.3, 0,  0.1, 0.3]
	denseDropList= [0, 0]
	typeOfLayer= "GRU"
	biDir= 1
	batchSize = 128
	model = classifier.createRNNModel(hiddenList, denseList, denseAct, rnnDropList, denseDropList, typeOfLayer, biDir, trainShape)
	######---------------------RNN------------------------######
elif (modelType == "CNN"):
	######---------------------CNN------------------------######
	#run_cw_f303_1500_CNN_3
	cnnFilters = [16, 8, 8] 
	kernelSizes = [64, 64, 32]
	cnnActs = ['elu', 'elu', 'elu']
	batchNorm = [0, 0, 0]
	isMaxPool = [1, 1, 1]
	maxPoolSizes = [2, 2, 2]

	denseList = [1024, 1024, 1024]
	denseActs = ['elu', 'elu', 'elu'] 
	denseDropouts = [0.1, 0.1, 0.1]

	batchSize = 1024
	trainShape = (6000,1)
	model = classifier.createCNNModel(cnnFilters, kernelSizes, cnnActs, trainShape, batchNorm, isMaxPool, maxPoolSizes, denseList, denseActs, denseDropouts)

	### HArdcode ditzler model
	#model = tf.keras.models.Sequential([
	#tf.keras.layers.Conv1D(16, 64, activation='elu', input_shape=(numPowerTraces,1), strides=1, dilation_rate=1),
	#tf.keras.layers.AveragePooling1D(strides=2), 
	##   tf.keras.layers.BatchNormalization(),
	#tf.keras.layers.Conv1D(8, 64, activation='elu', strides=1, dilation_rate=1),
	#tf.keras.layers.AveragePooling1D(strides=2),
	##   tf.keras.layers.BatchNormalization(),
	#tf.keras.layers.Conv1D(8, 32, activation='elu', strides=1, dilation_rate=1),
	#tf.keras.layers.AveragePooling1D(strides=2), 
	##   tf.keras.layers.BatchNormalization(),
	#tf.keras.layers.Flatten(),
	#tf.keras.layers.Dense(1024, activation='elu'),
	#tf.keras.layers.Dropout(0.1), 
	#tf.keras.layers.Dense(1024, activation='elu'),
	#tf.keras.layers.Dropout(0.1), 
	#tf.keras.layers.Dense(512, activation='elu'),
	#tf.keras.layers.Dropout(0.1),
	#tf.keras.layers.Dense(256, activation='softmax')
	#])
	######---------------------CNN------------------------######
else:
	print("Specify correct modelType")

## when dataEnsemble is used
#MrunLogsPath = "/xdisk/tosiron/manojgopale/extra/gem5KeyPrediction/log/dataEnsemble/allRuns.csv"
runLogsPath = "/xdisk/tosiron/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/log/allRuns.csv"
with open(runLogsPath, 'a') as f:
	## modelName must be unique like run_<someNum>
	if(modelType=="FNN"):
		f.write("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" %(modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize, trainSize, typeOfStd, chipType, keysToWeigh))
		print("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" %(modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize, trainSize, typeOfStd, chipType, keysToWeigh))
	elif(modelType=="RNN"):
		f.write("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" \
		%(modelName, hiddenList, denseList, denseAct, rnnDropList, denseDropList, typeOfLayer, biDir, batchSize, keysToWeigh))
		print("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" \
		%(modelName, hiddenList, denseList, denseAct, rnnDropList, denseDropList, typeOfLayer, biDir, batchSize, keysToWeigh))
	elif(modelType=="CNN"):
		f.write("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" \
		%(modelName, cnnFilters, kernelSizes, cnnActs, batchNorm, isMaxPool, maxPoolSizes, denseList, denseActs, denseDropouts, batchSize, keysToWeigh))
		print("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" \
		%(modelName, cnnFilters, kernelSizes, cnnActs, batchNorm, isMaxPool, maxPoolSizes, denseList, denseActs, denseDropouts, batchSize, keysToWeigh))
	else:
		print("Specify correct modelType")

#model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
#model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')
model.summary()

## Train the model
startTime = time.time()
logFile = resultDir + chipType + "/" + modelName + ".log"
csv_logger = CSVLogger(logFile, append=True, separator="\t")
checkPointPath = resultDir + '/' + chipType +'/' + modelName + "_checkPoint.h5"
tensorboardDir= resultDir + '/' + chipType +'/' + "tensorboardDir/"
tensorboardFile = tensorboardDir + modelName +  "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

##val_acc if we want to ue validation for early_stopping
#earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, mode='auto', verbose=1, restore_best_weights=True)
## save_freq -> integer, checks for model perfomance after the given steps. In ourr case it will be after we finish all steps in an epoch
stepsPerEpoch = int(1024/filesPerStep)
modelCheckpoint = ModelCheckpoint(filepath=checkPointPath, monitor="val_acc", verbose=1, save_best_only=True, save_freq=stepsPerEpoch)

tensorboard_cb = TensorBoard(\
									log_dir=tensorboardFile,\
									histogram_freq=1,\
									write_graph=False,\
									write_images=True,\
									write_steps_per_second=False,\
									update_freq='epoch',\
									profile_batch=0,\
									embeddings_freq=0,\
									embeddings_metadata=None,\
									)
#modelCheckpoint = ModelCheckpoint(filepath=checkPointPath, monitor="val_categorical_accuracy", verbose=1, save_best_only=True, period=10)
##Keys to put more weight on

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),\
							#optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0005), \
							#optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),##This lr till epoch 16, 54% train, 37% dev\
							optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\
							metrics=['acc']\
							)
#model.summary()
epochs=8
hist = model.fit(training_generator, \
								epochs=epochs, \
								validation_data=validation_generator, \
								callbacks=[csv_logger, modelCheckpoint, tensorboard_cb], 
								shuffle = True,\
								class_weight=getClassWeights(keysToWeigh),\
								verbose=1)

saveFile = resultDir + '/' + chipType +'/' + modelName + ".h5"
model.save(saveFile)

