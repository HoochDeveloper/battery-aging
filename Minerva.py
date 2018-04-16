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
	
	batteries = len(os.listdir(tsd.outFolder))
	testSize = ceil(batteries * 0.2)
	trainSize = batteries - testSize
	
	logger.info("Train: %d - Test: %d"  % (trainSize,testSize))
	
	#tsd.supervisedData4KerasLSTM("dataset",force=True) 
	
	minerva = Minerva()
	minerva.getModel(trainSize,tsd)
	
	#testX = minerva.batchCompatible(300,test[0])
	
	#model = load_model('batteryLSTM.h5')
	#tt = time.clock()
	#Yhat = model.predict(testX[0],batch_size=300)
	#logger.info("Prediction completed. Elapsed %f second(s)" %  (time.clock() - tt))
	#logger.info(len(Yhat))
	#logger.info(Yhat.shape)
	
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

	modelName = "modelloNuovo.h5"

	"""
	Model for learning 
	"""
	def getModel(self,trainSize,tsd):
	
		f = os.listdir(tsd.outFolder)[0]
		data = tsd.tsp.loadZip(f)
		
		tt = time.clock()
		# battery, days, X, Y
		batch_size = 150
		epochs = 11

		hiddenStateDim = 8
		inputFeatures = data[0].shape[2]
		outputFeatures = data[1].shape[2]
		timeSteps = data[0].shape[1]
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
		
		
		logger.info("Training %s batteries" % trainSize)
		for train in range(trainSize):
			f = os.listdir(tsd.outFolder)[train]
			data = tsd.tsp.loadZip(f)
			valididx = ceil(data[0].shape[0] * 0.8)
			
			trainX = data[0][:valididx]
			trainY = data[1][:valididx]
			validX = data[0][valididx:]
			validY = data[1][valididx:]
			
			
			trainX = self.batchCompatible(batch_size,trainX)
			trainY = self.batchCompatible(batch_size,trainY)
			validX = self.batchCompatible(batch_size,validX)
			validY = self.batchCompatible(batch_size,validY)
			
			logger.info(trainX.shape)
			logger.info(trainY.shape)
			logger.info(validX.shape)
			logger.info(validY.shape)
			
			
			logger.info("Training battery %s of %s" % ((train+1),trainSize))
			(
			model.fit(trainX,trainY,
			batch_size=batch_size, epochs=epochs, shuffle=False,
			validation_data=(validX, validY))
			)
			model.reset_states()
		model.save(self.modelName)  # creates a HDF5 file 'batteryLSTM.h5'
		
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
	
	def batchCompatible(self,batch_size,data):
		exceed = data.shape[0] % batch_size
		if(exceed > 0):
			data = data[:-exceed]
		return data
	
main()

