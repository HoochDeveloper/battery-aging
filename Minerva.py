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
	
	force = False
	ets = EpisodedTimeSeries(20,normalize=True)
	ets.buildEpisodedDataset(os.path.join(".","dataset"),force=force)
	
	minerva = Minerva(ets,Minerva.CHARGE)
	if(len(sys.argv) == 2 and sys.argv[1].lower() == 'train'):
		logger.info("Training")
		minerva.train()
	elif(len(sys.argv) == 2 and sys.argv[1].lower() == 'test'):
		logger.info("Testing")
		minerva.evaluate()
	else:
		logger.error("Invalid command line argument.")
		
class Minerva():

	CHARGE = "C"
	DISCHARGE = "D"

	modelName = "LSTM_DeepModel.h5"
	batchSize = 100
	epochs = 50
	imgPath = "./images"
	ets = None
	type = None
	
	def __init__(self,ets,type):
		self.ets = ets
		self.type = type
		
		
	def train(self,force=False):
		
		if(self.type == self.DISCHARGE):
			logger.info("Training discharge")
			self.modelName = "Discharge_LSTM_DeepModel.h5"
			self.ets.buildDischargeSet(force=force)
		elif(self.type == self.CHARGE):
			logger.info("Training charge")
			self.modelName = "Charge_LSTM_DeepModel.h5"
			self.ets.buildChargeSet(force=force)
		else:
			logger.info("Training mixed")
			self.ets.buildLearnSet(force=force)
			
		x_train, y_train, x_valid, y_valid = self.ets.loadTrainset(self.type)
		self.__trainSequentialModel(x_train, y_train, x_valid, y_valid)
	
	def evaluate(self,force=False):
		if(self.type == self.DISCHARGE):
			logger.info("Evaluating discharge")
			self.modelName = "Discharge_LSTM_DeepModel.h5"
		elif(self.type == self.CHARGE):
			logger.info("Evaluating charge")
			self.modelName = "Charge_LSTM_DeepModel.h5"
		else:
			logger.info("Evaluating mixed")
		normalizer = self.ets.loadNormalizer()
		x_test, y_test = self.ets.loadTestset(self.type)	
		self.__evaluateModel(x_test, y_test,False,normalizer)
		
	
	def __trainSequentialModel(self,x_train, y_train, x_valid, y_valid):
		
		x_train = self.__batchCompatible(self.batchSize,x_train)
		y_train = self.__batchCompatible(self.batchSize,y_train)
		x_valid = self.__batchCompatible(self.batchSize,x_valid)
		y_valid = self.__batchCompatible(self.batchSize,y_valid)
		
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
		model.add( Conv1D(hiddenStateDim1,name='EN_1',kernel_size=timeCompression, strides=2, padding='same',activation='relu'))
		model.add( LSTM(hiddenStateDim2,name='EN_2',dropout = drop,activation='tanh') )
		model.add( Dense(hiddenStateDim4,name='EN_3',activation='relu') )
		
		# rapp
		model.add( Dense(stateDim,activation='relu',name='ENCODED') )
		# end rapp
		model.add(Dense(hiddenStateDim4,name='DC_3',activation='relu') )
		model.add(Dense(hiddenStateDim2,name='DC_2',activation='relu') )
		model.add(RepeatVector(int(timesteps/timeCompression),name='DC_TS_1') )
		model.add( LSTM(hiddenStateDim1,name='DC_1',return_sequences=True,activation='tanh') )
		model.add(UpSampling1D(timeCompression,name='DC_TS_0') )
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
		model.save(os.path.join( self.ets.rootResultFolder , self.modelName )) 
		logger.info("Model saved")
		return model
				
		
	def __evaluateModel(self,x_test,y_test,saveImgs = False,scaler = None):
		
		logger.info("Loading model...")
		model = load_model(os.path.join( self.ets.rootResultFolder , self.modelName ))
		
		logger.info("Preparing data...")
		x_test = self.__batchCompatible(self.batchSize,x_test)
		y_test = self.__batchCompatible(self.batchSize,y_test)
		
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
			sid = 14
			for col in range(y_pred.shape[2]):
				plt.subplot(y_pred.shape[2], 1, i)
				if(scaler is not None):
					minvalue = scaler[sid,0]		
					maxvalue = scaler[sid,1] 
					pred = self.__scaleBack(y_pred[toPlot][:, col],minvalue,maxvalue)
					real = self.__scaleBack(y_test[toPlot][:, col],minvalue,maxvalue)
					plt.plot(pred,color="navy")
					plt.plot(real,color="orange")
					sid +=1
				else:
					plt.plot(y_pred[toPlot][:, col],color="navy")
					plt.plot(y_test[toPlot][:, col],color="orange")
				i += 1
			if(saveImgs):
				plt.savefig(os.path.join(self.imgPath,str(uuid.uuid4())), bbox_inches='tight')
				plt.close()
			else:
				plt.show()
	
	
	def __scaleBack(self,data,minvalue,maxvalue,minrange=-1,maxrange=1):
		d = ( (data - minrange) / (maxrange - minrange) ) * (maxvalue - minvalue) + minvalue
		return d
	
	def __batchCompatible(self,batch_size,data):
		exceed = data.shape[0] % batch_size
		if(exceed > 0):
			data = data[:-exceed]
		return data
	
		
main()

