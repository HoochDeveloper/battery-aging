#Standard Imports
import uuid,time,os,logging, numpy as np, sys, pandas as pd , matplotlib.pyplot as plt

#Project module import
from Demetra import EpisodedTimeSeries

#KERAS
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, RepeatVector, Input, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, Conv1D, UpSampling1D, MaxPooling1D,Reshape
from keras.models import load_model
from keras import optimizers
from keras.callbacks import EarlyStopping
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
	minerva = Minerva(Minerva.CHARGE)
	if(len(sys.argv) == 2 and sys.argv[1].lower() == 'train'):
		logger.info("Training")
		minerva.train(force)
	elif(len(sys.argv) == 2 and sys.argv[1].lower() == 'test'):
		logger.info("Testing")
		minerva.evaluate()
	else:
		logger.error("Invalid command line argument.")
		
class Minerva():

	CHARGE = "C"
	DISCHARGE = "D"

	#modelName = "LSTM_DeepModel.h5"
	modelName = "Conv_DeepModel.h5"
	batchSize = 100
	epochs = 50
	imgPath = "./images"
	ets = None
	type = None
	
	def __init__(self,type):
		self.ets = EpisodedTimeSeries(20,normalize=True)
		self.type = type
		
	def train(self,force=False):
		self.ets.buildEpisodedDataset(os.path.join(".","dataset"),force=force)
		if(self.type == self.DISCHARGE):
			logger.info("Training discharge")
			self.modelName = "Discharge_" + self.modelName
			self.ets.buildDischargeSet(force=force)
		elif(self.type == self.CHARGE):
			logger.info("Training charge")
			self.modelName = "Charge_" + self.modelName
			self.ets.buildChargeSet(force=force)
		else:
			logger.info("Training mixed")
			self.ets.buildLearnSet(force=force)
			
		x_train, y_train, x_valid, y_valid = self.ets.loadTrainset(self.type)
		self.__trainSequentialModel(x_train, y_train, x_valid, y_valid)
	
	def evaluate(self,force=False):
		if(self.type == self.DISCHARGE):
			logger.info("Evaluating discharge")
			self.modelName = "Discharge_" + self.modelName
		elif(self.type == self.CHARGE):
			logger.info("Evaluating charge")
			self.modelName = "Charge_" + self.modelName
		else:
			logger.info("Evaluating mixed")
		normalizer = self.ets.loadNormalizer()
		x_test, y_test = self.ets.loadTestset(self.type)	
		
		if(True):
			x_test = self.__conv2DCompatible(x_test)
		
		self.__evaluateModel(x_test, y_test,False,normalizer)
		
	
	def __trainSequentialModel(self,x_train, y_train, x_valid, y_valid):
		
		x_train = self.__batchCompatible(self.batchSize,x_train)
		y_train = self.__batchCompatible(self.batchSize,y_train)
		x_valid = self.__batchCompatible(self.batchSize,x_valid)
		y_valid = self.__batchCompatible(self.batchSize,y_valid)
		
		tt = time.clock()
		
		inputFeatures  = x_train.shape[2]
		outputFeatures = y_train.shape[2]
		timesteps =  x_train.shape[1]
		
		#model = self.__lstmModel(inputFeatures,outputFeatures,timesteps)
		#model = self.__newLSTM(inputFeatures,outputFeatures,timesteps)
		if(True):
			x_train = self.__conv2DCompatible(x_train)
			x_valid = self.__conv2DCompatible(x_valid)
		
		model = self.__convModel(inputFeatures,outputFeatures,timesteps)
		
		
		
		# end model
		
		adam = optimizers.Adam(lr=0.00001)		
		model.compile(loss='mean_squared_error', optimizer=adam,metrics=['mae'])
		print(model.summary())
		
		
		early = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='min')
		
		model.fit(x_train, y_train,
			batch_size=self.batchSize,
			epochs=self.epochs,
			validation_data=(x_valid,y_valid),
			callbacks=[early]
		)
		
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		logger.info("Save model")
		model.save(os.path.join( self.ets.rootResultFolder , self.modelName )) 
		logger.info("Model saved")
		return model

	
	def __lstmModel(self,inputFeatures,outputFeatures,timesteps):
	
		hiddenStateDim0 = 512
		hiddenStateDim1 = int(hiddenStateDim0 / 2) 
		hiddenStateDim2 = int(hiddenStateDim1 / 2)
		hiddenStateDim3 = int(hiddenStateDim2 / 2)
		hiddenStateDim4 = int(hiddenStateDim3 / 2)
		stateDim = 5
		
		timeCompression = 2
		drop = 0.5
	
		model = Sequential()
		
		model.add( LSTM(hiddenStateDim0,name='EN_0',return_sequences=True,activation='tanh',input_shape=(timesteps,inputFeatures)))
		model.add(Dropout(0.5))
		model.add( Conv1D(hiddenStateDim1,name='EN_1',kernel_size=timeCompression, strides=2, padding='same',activation='relu'))
		model.add( LSTM(hiddenStateDim2,name='EN_2',dropout = drop,activation='tanh') )
		model.add(Dropout(0.5))
		model.add( Dense(hiddenStateDim4,name='EN_3',activation='relu') )
		model.add(Dropout(0.5))
		# rapp
		model.add( Dense(stateDim,activation='relu',name='ENCODED') )
		# end rapp
		model.add(Dense(hiddenStateDim4,name='DC_3',activation='relu') )
		model.add(Dropout(0.5))
		model.add(Dense(hiddenStateDim2,name='DC_2',activation='relu') )
		model.add(Dropout(0.5))
		model.add(RepeatVector(int(timesteps/timeCompression),name='DC_TS_1') )
		model.add( LSTM(hiddenStateDim1,name='DC_1',return_sequences=True,activation='tanh') )
		model.add(Dropout(0.5))
		model.add(UpSampling1D(timeCompression,name='DC_TS_0') )
		model.add( LSTM(hiddenStateDim0,name='DC_0',return_sequences=True,dropout = drop,activation='tanh') )
		model.add(Dropout(0.5))
		model.add( LSTM(outputFeatures,name='decoded',return_sequences=True,activation='tanh') )
		
		return model
		
	def __convModel(self,inputFeatures,outputFeatures,timesteps):
		model = Sequential()
		
		mask = (2,2)
		#mask = 4
		
		start = 2048
		
		model.add(Conv2D(start,mask, activation='tanh', input_shape=(8, 8 ,5))) #out 7x7x10
		model.add(Conv2D(int(start/2),mask, activation='relu')) #out 6x6x10
		model.add(Dropout(0.5))
		model.add(Conv2D(int(start/4),mask, activation='relu')) #out 5x5x10
		model.add(Dropout(0.5))
		model.add(Conv2D(10,mask, activation='tanh')) #out 4x4x10
		model.add(MaxPooling2D(pool_size=mask))
		#model.add(Conv2D(int(start/16),mask, activation='tanh')) #out 3x3x10
		#model.add(Dropout(0.25))
		#model.add(Conv2D(10,mask, activation='relu')) #out 2x2x10
		#model.add(MaxPooling2D(pool_size=(3, 3)))
		#model.add(Dense(10,activation='tanh') )
		#model.add(Dense(128,activation='relu') )
		#
		#model.add(UpSampling2D((2,2)))
		#model.add(Conv2D(5, (1,1), activation='relu'))
		#model.add(Conv2D(2, (2,2), activation='relu'))
		model.add(Reshape((20, 2)))
		#model.add(Reshape((20, 2), input_shape=(2,2,10)))
		
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
		
	def __conv2DCompatible(self,data):
		data = data.reshape(data.shape[0],8, 8 ,5)
		print(data.shape)
		return data
		
main()

