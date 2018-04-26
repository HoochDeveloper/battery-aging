#Standard Imports
import time,os,logging, matplotlib.pyplot as plt, numpy as np
from math import sqrt,ceil,trunc
import pandas as pd

#Project module import
from Demetra import EpisodedTimeSeries

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from keras.models import load_model
from keras import optimizers
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
	ets = EpisodedTimeSeries()
	#ets.timeSeries2relevantEpisodes(os.path.join(".","dataset"),force=True)
	#x_train, y_train, x_valid, y_valid = ets.scaleTrainSet()
	x_train, y_train, x_valid, y_valid = ets.loadTrainSet()	
	minerva = Minerva()
	
	if(False):
		minerva.trainModel(x_train, y_train, x_valid, y_valid)
	else:
		model = load_model(minerva.modelName)
		
		x_valid = minerva.batchCompatible(minerva.batchSize,x_valid)
			
		y_valid = minerva.batchCompatible(minerva.batchSize,y_valid)
		
		
		y_pred = model.predict(x_valid,  batch_size=minerva.batchSize)
		
		logger.info(y_pred.shape)
	
	
		for sample2plot in range(y_pred.shape[0]):
			plt.figure()
			i = 1
			for col in range(y_pred.shape[2]):
				plt.subplot(y_pred.shape[2], 1, i)
				plt.plot(y_pred[sample2plot][:, col],color="navy")
				plt.plot(y_valid[sample2plot][:, col],color="orange")
				i += 1
			plt.show()
	

		
class Minerva():

	#modelName = "modelloNuovo.h5"
	modelName = "bidirectional_episoded_deepModel.h5"
	batchSize = 150
	epochs = 15
	"""
	Model for learning 
	"""
	def trainModel(self,x_train, y_train, x_valid, y_valid,force=False):
		
		x_train = self.batchCompatible(self.batchSize,x_train)
		y_train = self.batchCompatible(self.batchSize,y_train)
		x_valid = self.batchCompatible(self.batchSize,x_valid)
		y_valid = self.batchCompatible(self.batchSize,y_valid)
		
		
		
		tt = time.clock()
		
		inputFeatures  = x_train.shape[2]
		outputFeatures = y_train.shape[2]
		
		hiddenStateDim0 = 16
		hiddenStateDim1 = 8
		hiddenStateDim2 = 4
		
		timeSteps =  x_train.shape[1]
		model = Sequential()
		
		
		model.add(Bidirectional(LSTM(hiddenStateDim0, return_sequences=True,return_state=False), input_shape=(timeSteps, inputFeatures)))
		model.add(Bidirectional(LSTM(hiddenStateDim1, return_sequences=True,return_state=False), input_shape=(timeSteps, hiddenStateDim0)))
		model.add(TimeDistributed(Dense(hiddenStateDim2, activation='relu')))
		
		model.add(Bidirectional(LSTM(hiddenStateDim1, return_sequences=True,return_state=False), input_shape=(timeSteps, hiddenStateDim2)))
		model.add(Bidirectional(LSTM(hiddenStateDim0, return_sequences=True,return_state=False), input_shape=(timeSteps, hiddenStateDim1)))
		model.add(TimeDistributed(Dense(outputFeatures, activation='relu')))

		
		model.compile(loss='mse', optimizer='adam',metrics=['mae'])
		
		model.fit(x_train, y_train,batch_size=self.batchSize, epochs=self.epochs, shuffle=False,
			validation_data=(x_valid, y_valid))
		
		model.save(self.modelName)  # creates a HDF5 file 'batteryLSTM.h5'
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		return model
		
	def batchCompatible(self,batch_size,data):
		exceed = data.shape[0] % batch_size
		if(exceed > 0):
			data = data[:-exceed]
		return data
	
		
main()

