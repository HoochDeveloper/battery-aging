#Standard Imports
import time,os,logging, matplotlib.pyplot as plt, numpy as np
from math import sqrt,ceil,trunc
import pandas as pd

#Project module import
from Demetra import EpisodedTimeSeries

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
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
	
	logger.info(x_valid.shape)
	
	#i = 1
	#plt.figure()
	#
	#for col in range(x_train.shape[2]):
	#	plt.subplot(x_train.shape[2], 1, i)
	#	plt.plot(x_train[0,:, col])
	#	i += 1
	#plt.show()
	
	
	minerva = Minerva()
	
	
	
	
	minerva.trainModel(x_train, y_train, x_valid, y_valid)
	
	if(False):
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
	
	
	#minerva.evaluateModel(tsd,"./testData")
	
	#minerva.prediction(tsd,"./testData","THING_1.gzip")
	#minerva.testModel(tsd,"./testData","THING_24.gzip")
	
	
	if(False):
	
		batteries = len(os.listdir(tsd.outFolder))
		testSize = ceil(batteries * 0.2)
		trainSize = 1#batteries - testSize
		
		logger.info("Train: %d - Test: %d"  % (trainSize,testSize))
	
		
		model = minerva.getModel(trainSize,tsd)
		
		
		f = os.listdir(tsd.outFolder)[trainSize+3]
		test = tsd.tsp.loadZip(f)
		testX = minerva.batchCompatible(minerva.batchSize,test[0])
		testY = minerva.batchCompatible(minerva.batchSize,test[1])
		logger.info("Predicting")
		tt = time.clock()
		Yhat = np.zeros([minerva.batchSize,3600,14],dtype='float32')
		for t in range(3600):
			Yhat[:,t] = model.predict_on_batch(np.expand_dims(testX[:minerva.batchSize,t,:],axis=1))
		
		#Yhat = model.predict(testX,batch_size=minerva.batchSize)
		logger.info("Prediction completed. Elapsed %f second(s)" %  (time.clock() - tt))
		i = 1
		plt.figure()
		sample2plot = 31
		for col in range(test[1].shape[2]):
			plt.subplot(test[1].shape[2], 1, i)
			plt.plot(Yhat[sample2plot][:, col])
			plt.plot(test[1][sample2plot][:, col])
			i += 1
		plt.show()
	
		
class Minerva():

	#modelName = "modelloNuovo.h5"
	modelName = "episoded_deepModel.h5"
	batchSize = 250
	epochs = 4
	learningRate = 0.001
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
		
		hiddenStateDim = 8
		
		timeSteps =  x_train.shape[1]
		model = Sequential()
		
		encoder = (
		LSTM(hiddenStateDim, 
		batch_input_shape=(self.batchSize, timeSteps, inputFeatures),
		return_sequences=True,stateful=False))
		
		model.add(encoder)
		
		hiddenEncoder = (
		LSTM(int(hiddenStateDim / 2), 
		batch_input_shape=(self.batchSize, timeSteps, inputFeatures),
		return_sequences=True,stateful=False))
		model.add(hiddenEncoder)
        
		hiddenDecoder = (
		LSTM(hiddenStateDim,
		batch_input_shape=(self.batchSize,timeSteps,int(hiddenStateDim / 2)),
		return_sequences=True,stateful=False )
		)
		model.add(hiddenDecoder)

		decoder = (
		LSTM(outputFeatures,
		batch_input_shape=(self.batchSize,timeSteps,hiddenStateDim),
		return_sequences=True,stateful=False )
		)
		model.add(decoder)
		
		
		
		#sgd = optimizers.SGD(lr=self.learningRate, decay=1e-6, momentum=0.9, nesterov=True)
		#model.compile(loss='mse', optimizer=sgd,metrics=['mae'])
		
		model.compile(loss='mse', optimizer='adam',metrics=['mae'])
		
		model.fit(x_train, y_train,batch_size=self.batchSize, epochs=10, shuffle=False,
			validation_data=(x_valid, y_valid))
		
		model.save(self.modelName)  # creates a HDF5 file 'batteryLSTM.h5'
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		return model
		
	def batchCompatible(self,batch_size,data):
		exceed = data.shape[0] % batch_size
		if(exceed > 0):
			data = data[:-exceed]
		return data
	
	def evaluateModel(self,tsd,testFolder):
		bestMae = float("inf")
		model = load_model(self.modelName)
		for f in os.listdir(testFolder):
			loss = []
			mae  = []
			logger.info("Evaluating %s" % f)
			test = tsd.tsp.loadZip(testFolder,f)
			testX = self.batchCompatible(self.batchSize,test[0])
			testY = self.batchCompatible(self.batchSize,test[1])
			for t in range(3600):
				x = np.expand_dims(testX[:,t,:],axis=1)
				y = testY[:,t,:]
				#loss, _, mae
				score = model.evaluate(x, y, batch_size=self.batchSize,verbose=0)
				loss.append(score[0])
				mae.append(score[2])
			#reset every epoch
			model.reset_states()
			logger.info('Epoch MAE test = {}'.format(np.mean(mae)))
			logger.info('Epoch loss test = {}'.format(np.mean(loss)))
			if(np.mean(mae) < bestMae):
				bestMae = np.mean(mae)
				logger.info("New best mae is %f in file %s" % (bestMae,f))
			logger.info('___________________________________')
	
	def prediction(self,tsd,testFolder,testFile):
		model = load_model(self.modelName)
		test = tsd.tsp.loadZip(testFolder,testFile)
		testX = self.batchCompatible(self.batchSize,test[0])
		testY = self.batchCompatible(self.batchSize,test[1])
		logger.info("Predicting")
		tt = time.clock()
		Yhat = np.zeros([self.batchSize,3600,14],dtype='float32')
		for t in range(3600):
			
			Yhat[:,t] = model.predict_on_batch(np.expand_dims(testX[:self.batchSize,t,:],axis=1))
		
		logger.info("Prediction completed. Elapsed %f second(s)" %  (time.clock() - tt))
		
		for k in range(4):
			i = 1
			plt.figure()
			toPlot = np.random.randint(self.batchSize)
			logger.info("Plotting %s " % toPlot)
			for col in range(test[1].shape[2]):
				plt.subplot(test[1].shape[2], 1, i)
				plt.plot(Yhat[toPlot][:, col])
				plt.plot(testY[toPlot][:, col])
				i += 1
			plt.show()
		
	
	def testModel(self,tsd,testFolder,testFile):
		
		bestMae = float("inf")
		worstMae = 0
		bestIdx = 0
		worstIdx = 0
		
		model = load_model(self.modelName)
		logger.info(model.metrics_names)
		test = tsd.tsp.loadZip(testFolder,testFile)
		
		
		
		testX = self.batchCompatible(self.batchSize,test[0])
		testY = self.batchCompatible(self.batchSize,test[1])
		logger.info("Predicting")
		tt = time.clock()
		
		
		
		numOfBatch = int( testX.shape[0] / self.batchSize )
		
		for sample in range(numOfBatch):
			batchStartIdx = self.batchSize * sample
			batchEndIdx = batchStartIdx + self.batchSize
			for t in range(3600):
				#loss, _, mae
				score = (
				model.test_on_batch(np.expand_dims(testX[batchStartIdx:batchEndIdx,t,:],axis=1),
				testY[batchStartIdx:batchEndIdx,t,:])
				)
				logger.info(len(score))
				
				#minMaeIdx = np.argmin(score[2])
				#maxMaeIdx = np.argmax(score[2])
				#if(score[2][minMaeIdx] < bestMae):
				#	bestMae = score[2][minMaeIdx] 
				#	bestIdx = minMaeIdx + batchStartIdx
				#	logger.info("New best mae %f" % bestMae )
				#if(score[2][maxMaeIdx] > worstMae):
				#	worstMae = score[2][maxMaeIdx]
				#	worstIdx = maxMaeIdx + batchStartIdx
				#	logger.info("New worst mae %f" % worstMae )
					
		self.prediction(tsd,testFolder,testFile,bestIdx)
		self.prediction(tsd,testFolder,testFile,worstIdx)
		
main()

