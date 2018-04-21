#Standard Imports
import time,os,logging, matplotlib.pyplot as plt, numpy as np
from math import sqrt,ceil,trunc
import pandas as pd

#Project module import
from Demetra import TimeSeriesDataset,TimeSeriesPreprocessing

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
	tsd = TimeSeriesDataset()
	
	batteries = len(os.listdir(tsd.outFolder))
	testSize = ceil(batteries * 0.2)
	trainSize = 1#batteries - testSize
	
	logger.info("Train: %d - Test: %d"  % (trainSize,testSize))
	
	#tsd.supervisedData4KerasLSTM("dataset",force=True) 
	
	minerva = Minerva()
	model = minerva.getModel(trainSize,tsd)
	
	
	f = os.listdir(tsd.outFolder)[trainSize+3]
	test = tsd.tsp.loadZip(f)
	#
	testX = minerva.batchCompatible(minerva.batchSize,test[0])
	testY = minerva.batchCompatible(minerva.batchSize,test[1])
	
	#logger.info("Evaluating")
	#tt = time.clock()
	#scores = model.evaluate(testX, testY, batch_size=minerva.batchSize)
	#logger.info('mse=%f' % (scores))
	#logger.info("Evaluation completed. Elapsed %f second(s)" %  (time.clock() - tt))
	if(True):
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
	modelName = "ADAM_DEEP_20180421_test_deepModel.h5"
	batchSize = 250
	epochs = 4
	learningRate = 0.001
	"""
	Model for learning 
	"""
	def getModel(self,trainSize,tsd):
		
		if(True and os.path.exists(self.modelName)):
			model = load_model(self.modelName)
			return model
		
		logger.info("Model %s does not exists. Training a new model." % self.modelName )
		f = os.listdir(tsd.outFolder)[0]
		data = tsd.tsp.loadZip(f)
		
		tt = time.clock()
		# battery, days, X, Y
		
		inputFeatures = data[0].shape[2]
		outputFeatures = data[1].shape[2]
		
		hiddenStateDim = 8
		
		timeSteps = 1#data[0].shape[1]
		model = Sequential()
		
		encoder = (
		LSTM(hiddenStateDim, 
		batch_input_shape=(self.batchSize, timeSteps, inputFeatures),
		return_sequences=True,stateful=True))
		
		model.add(encoder)
		
		hiddenEncoder = (
		LSTM(int(hiddenStateDim / 2), 
		batch_input_shape=(self.batchSize, timeSteps, inputFeatures),
		return_sequences=True,stateful=True))
		model.add(hiddenEncoder)
        
		hiddenDecoder = (
		LSTM(hiddenStateDim,
		input_shape=(timeSteps,int(hiddenStateDim / 2)),
		return_sequences=True,stateful=True )
		)
		model.add(hiddenDecoder)

		decoder = (
		LSTM(outputFeatures,
		input_shape=(1,hiddenStateDim),
		return_sequences=False,stateful=True )
		)
		model.add(decoder)
		
		#sgd = optimizers.SGD(lr=self.learningRate, decay=1e-6, momentum=0.9, nesterov=True)
		#model.compile(loss='mse', optimizer=sgd,metrics=['accuracy','mae'])
		
		model.compile(loss='mse', optimizer='adam',metrics=['accuracy','mae'])
		
		logger.info("Training %s batteries" % trainSize)
		for train in range(trainSize):
			f = os.listdir(tsd.outFolder)[train]
			data = tsd.tsp.loadZip(f)
			trainX = data[0]
			trainY = data[1]
			bachInEpoch = int( trainX.shape[0] / self.batchSize )
			trainX = self.batchCompatible(self.batchSize,trainX)
			trainY = self.batchCompatible(self.batchSize,trainY)
			logger.info("bachInEpoch %s" % bachInEpoch)
			for epoch in range(self.epochs):
				mean_tr_mae = []
				mean_tr_acc = []
				mean_tr_loss = []
				logger.info("Epoch %s" % epoch)
				for sample in range(bachInEpoch):
					batchStartIdx = self.batchSize * sample
					batchEndIdx = batchStartIdx + self.batchSize
					logger.info("Batch %s of %s" % (sample , bachInEpoch))
					for timeStep in range(trainX.shape[1]):
						tr_loss, tr_acc, tr_mae = (
								model.train_on_batch(
									np.expand_dims(trainX[batchStartIdx:batchEndIdx,timeStep,:],axis=1)
									,
									trainY[batchStartIdx:batchEndIdx,timeStep,:]
								)
						)
						mean_tr_acc.append(tr_acc)
						mean_tr_loss.append(tr_loss)
						mean_tr_mae.append(tr_mae)
				#reset every epoch
				model.reset_states()
				print('Epoch accuracy training = {}'.format(np.mean(mean_tr_acc)))
				print('Epoch MAE training = {}'.format(np.mean(mean_tr_mae)))
				print('Epoch loss training = {}'.format(np.mean(mean_tr_loss)))
				print('___________________________________')
			#end epoch for
		# end train for	
		model.save(self.modelName)  # creates a HDF5 file 'batteryLSTM.h5'
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		return model
		
	def batchCompatible(self,batch_size,data):
		exceed = data.shape[0] % batch_size
		if(exceed > 0):
			data = data[:-exceed]
		return data
	
main()

