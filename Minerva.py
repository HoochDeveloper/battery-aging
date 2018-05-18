#Standard Imports
import uuid,time,os,logging, numpy as np, sys, pandas as pd , matplotlib.pyplot as plt

#Project module import
from Demetra import EpisodedTimeSeries

#KERAS
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, RepeatVector, Input, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, Conv1D, UpSampling1D, MaxPooling1D
from keras.models import load_model
from keras import optimizers
#
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
	force = True
	ets = EpisodedTimeSeries(15,normalize=True)
	ets.buildEpisodedDataset(os.path.join(".","dataset"),force=force)
	
	type = "D"
	#ets.showEpisodes(type)
	
	ets.buildChargeSet(force=force)
	ets.buildDischargeSet(force=force)
	normalizer = ets.loadNormalizer()
	
	if(False):
		if(len(sys.argv) == 2 and sys.argv[1].lower() == 'train'):
			minerva = Minerva()
			logger.info("Training")
			x_train, y_train, x_valid, y_valid = ets.loadTrainset(type)
			#x_train, y_train, x_valid, y_valid,normalizer = ets.loadTrainset()	
			
			logger.info(x_train.shape)
			logger.info(x_valid.shape)
			
			minerva.trainSequentialModel(x_train, y_train, x_valid, y_valid)
		elif(len(sys.argv) == 2 and sys.argv[1].lower() == 'test'):
			minerva = Minerva()
			logger.info("Testing")
			x_test, y_test = ets.loadTestset(type)	
			#x_test, y_test, scaler = ets.loadTestset()	
			logger.info(x_test.shape)
			minerva.evaluateModel(x_test, y_test,True)
		else:
			logger.error("Invalid command line argument.")

		
class Minerva():

	#modelName = "Charge_LSTM_DeepModel.h5"
	modelName = "Discharge_LSTM_DeepModel.h5"
	batchSize = 75
	epochs = 25
	imgPath = "./images"
	
	def trainSequentialModel(self,x_train, y_train, x_valid, y_valid):
		
		x_train = self.batchCompatible(self.batchSize,x_train)
		y_train = self.batchCompatible(self.batchSize,y_train)
		x_valid = self.batchCompatible(self.batchSize,x_valid)
		y_valid = self.batchCompatible(self.batchSize,y_valid)
		
		tt = time.clock()
		
		inputFeatures  = x_train.shape[2]
		outputFeatures = y_train.shape[2]
		
		hiddenStateDim0 = 256
		hiddenStateDim1 = int(hiddenStateDim0 / 2) 
		hiddenStateDim2 = int(hiddenStateDim1 / 2)
		hiddenStateDim3 = int(hiddenStateDim2 / 2)
		hiddenStateDim4 = int(hiddenStateDim3 / 2)
		stateDim = 5
		
		timeCompression = 2
		drop = 0.5
		
		timesteps =  x_train.shape[1]
		model = Sequential()
		
		model.add( LSTM(hiddenStateDim0,name='EN_0',return_sequences=True,activation='tanh',input_shape=(timesteps,inputFeatures)))
		#model.add( Conv1D(hiddenStateDim1,name='EN_1',kernel_size=timeCompression, strides=2, padding='same',activation='relu'))
		model.add( LSTM(hiddenStateDim2,name='EN_2',dropout = drop,activation='tanh') )
		
		model.add( Dense(hiddenStateDim4,name='EN_3',activation='relu') )
		# rapp
		model.add( Dense(stateDim,activation='relu',name='ENCODED') )
		# end rapp
		model.add(Dense(hiddenStateDim4,name='DC_3',activation='relu') )
		
		model.add(Dense(hiddenStateDim2,name='DC_2',activation='relu') )
		
		#model.add(RepeatVector(int(timesteps/timeCompression),name='DC_TS_1') )
		model.add(RepeatVector(int(timesteps),name='DC_TS_1') )
		model.add( LSTM(hiddenStateDim1,name='DC_1',return_sequences=True,activation='tanh') )
		
		#model.add(UpSampling1D(timeCompression,name='DC_TS_0') )
		model.add( LSTM(hiddenStateDim0,name='DC_0',return_sequences=True,dropout = drop,activation='tanh') )
		
		model.add( LSTM(outputFeatures,name='decoded',return_sequences=True,activation='tanh') )
		
		
		# end model
		
		adam = optimizers.Adam(lr=0.00001)		
		model.compile(loss='mean_squared_error', optimizer=adam,metrics=['mae'])
		print(model.summary())
		
		
		model.fit(x_train, y_train,
			batch_size=self.batchSize,
			epochs=self.epochs,
			validation_data=(x_valid,y_valid)
		)
		
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		logger.info("Save model")
		model.save(self.modelName) 
		logger.info("Model saved")
		return model
				
		
	def evaluateModel(self,x_test,y_test,saveImgs = False):
		
		logger.info("Loading model...")
		model = load_model(self.modelName)
		
		logger.info("Preparing data...")
		x_test = self.batchCompatible(self.batchSize,x_test)
		y_test = self.batchCompatible(self.batchSize,y_test)
		
		logger.info("Evaluating")
		tt = time.clock()
		mse, mae = model.evaluate( x=x_test, y=y_test, batch_size=self.batchSize, verbose=0)
		logger.info("MSE %f - MAE %f Elapsed %f" % (mse,mae,(time.clock() - tt)))
		
		logger.info("Predicting")
		tt = time.clock()
		y_pred = model.predict(x_test,  batch_size=self.batchSize)
		logger.info("Elapsed %f" % (time.clock() - tt))
		for r in range(25):
			plt.figure()
			toPlot = np.random.randint(y_pred.shape[0])
			i = 1
			for col in range(y_pred.shape[2]):
				plt.subplot(y_pred.shape[2], 1, i)
				plt.plot(y_pred[toPlot][:, col],color="navy")
				plt.plot(y_test[toPlot][:, col],color="orange")
				i += 1
			if(saveImgs):
				plt.savefig(os.path.join(self.imgPath,str(uuid.uuid4())), bbox_inches='tight')
				plt.close()
			else:
				plt.show()
	
	
		
		
	def batchCompatible(self,batch_size,data):
		exceed = data.shape[0] % batch_size
		if(exceed > 0):
			data = data[:-exceed]
		return data
	
		
main()

