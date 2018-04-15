#Standard Imports
import time,os,logging, matplotlib.pyplot as plt, numpy as np
from math import sqrt,ceil,trunc
import pandas as pd

#Project module import
from Demetra import TimeSeriesDataset,TimeSeriesPreprocessing

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.models import load_model
from numpy import array

#KERAS ENV GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NUMBAPRO_NVVM']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\nvvm\bin\nvvm64_31_0.dll'
os.environ['NUMBAPRO_LIBDEVICE']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\nvvm\libdevice'

#Module logging
logger = logging.getLogger("Minerva")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

def main():
	tsd = TimeSeriesDataset()
	#tsd.supervisedData4KerasLSTM("dataset",force=True) 
	exit()
	
	
	#if(True):
	#	i = 1
	#	plt.figure()
	#	for col in range(trainX[0].shape[2]):
	#		plt.subplot(trainX[0].shape[2], 1, i)
	#		plt.plot(trainX[0][0][:, col])
	#		i += 1
	#	plt.show()
	
	minerva = Minerva()
	#minerva.getModel(trainX, trainY, validX, validY, testX, testY)
	
	testX = minerva.batchCompatible(300,testX)
	
	model = load_model('batteryLSTM.h5')
	tt = time.clock()
	Yhat = model.predict(testX[0],batch_size=300)
	logger.info("Prediction completed. Elapsed %f second(s)" %  (time.clock() - tt))
	logger.info(len(Yhat))
	logger.info(Yhat.shape)
	
	if(True):
		i = 1
		plt.figure()
		for col in range(testY[0].shape[2]):
			plt.subplot(testY[0].shape[2], 1, i)
			plt.plot(Yhat[0][:, col])
			plt.plot(testY[0][0][:, col])
			i += 1
		plt.show()
	
		
class Minerva():
	"""
	Model for learning 
	"""
	def getModel(self,trainX, trainY, validX, validY, testX, testY):
		tt = time.clock()
		# battery, days, X, Y
		batch_size = 300
		epochs = 17
		# shorten list to multiple of batch_size
		trainX = self.batchCompatible(batch_size,trainX)
		trainY = self.batchCompatible(batch_size,trainY)
		validX = self.batchCompatible(batch_size,validX)
		validY = self.batchCompatible(batch_size,validY)
		testX = self.batchCompatible(batch_size,testX)
		testY = self.batchCompatible(batch_size,testY)

		logger.info(trainX[0].shape)
		logger.info(validX[0].shape)
		logger.info(testX[0].shape)
		
		hiddenStateDim = 8
		inputFeatures = trainX[0].shape[2]
		outputFeatures = trainY[0].shape[2]
		timeSteps = trainX[0].shape[1]
		model = Sequential()
		
		encoder = (
		LSTM(hiddenStateDim, 
		batch_input_shape=(batch_size, timeSteps, inputFeatures),stateful=True,
		return_sequences=True))
		
		model.add(encoder)
		
		decoder = (
		LSTM(outputFeatures,
		input_shape=(timeSteps,hiddenStateDim),
		return_sequences=True,stateful=True)
		)
		model.add(decoder)
		model.compile(loss='mse', optimizer='adam')
		
		batteries = len(trainX)
		logger.info("Training %s episodes" % batteries)
		for battery in range(batteries):
			print("Training battery %s of %s" % ((battery+1),batteries),end='\r')
			(
			model.fit(trainX[battery],trainY[battery],
			batch_size=batch_size, epochs=epochs, shuffle=False,
			validation_data=(validX[battery], validY[battery]))
			)
			model.reset_states()
		model.save('batteryLSTM.h5')  # creates a HDF5 file 'batteryLSTM.h5'
		
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
	
	def batchCompatible(self,batch_size,list):
		for i in range(len(list)):
			exceed = list[i].shape[0] % batch_size
			if(exceed > 0):
				list[i] = list[i][:-exceed]
		return list
	
main()

