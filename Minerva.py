#Standard Imports
import time,os,logging, matplotlib.pyplot as plt, numpy as np
from math import sqrt,ceil,trunc
import pandas as pd

#Project module import
from Demetra import TimeSeriesDataset,TimeSeriesPreprocessing

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
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
	batteries = tsd.loadEpisodedDataset("dataset",dataFile="episodedDF")
	logger.info("%d Batteries",len(batteries))
	logger.info("%d Day(s) for battery 0 ",len(batteries[0]))
	logger.info("%d Hour(s) for battery 0 in day 0" ,len(batteries[0][0]))
	logger.info(batteries[0][0][0].shape)
	
	
	data = batteries[0][0][0]#.reindex(index=padding)
	data.reset_index(drop=True,inplace=True)
	padding = range(3600)
	data = data.reindex(index=padding)
	data[tsd.timeIndex].fillna(pd.to_datetime('1900-01-01T00:00:00'),inplace=True)
	data.fillna(0,inplace=True)
	logger.info(data.shape)
	tsd.plot(data)
				
		
class Minerva():
	"""
	Model for learning 
	"""
	def getModel(self,X_train,Y_train,X_valid,Y_valid,X_test,Y_test):
		hiddenStateDim = 8
		inputFeatures = X_train.shape[1]
		outputFeatures = Y_train.shape[1]
		timeSteps = 3600
		model = Sequential()
		encoder = LSTM(hiddenStateDim, input_shape=(timeSteps,inputFeatures),stateful=True)
		model.add(encoder)
		
		decoder =  LSTM(outputFeatures,input_shape=(timeSteps,hiddenStateDim),stateful=True)
		model.add(decoder)
		model.compile(loss='mse', optimizer='adam')
		
		epochs = len(X_train)
		logger.info("Training %s episodes" % epochs)
		for epoch in range(epochs):
			print("Training epoch %s of %s" % ((epoch+1),numOfEpochs),end='\r')
			
			
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
	
	
main()

