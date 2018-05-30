#Standard Imports
import uuid,time,os,logging, numpy as np, sys, pandas as pd , matplotlib.pyplot as plt

from logging import handlers as loghds

#Project module import
from Demetra import EpisodedTimeSeries

#KERAS
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, RepeatVector, Input, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, Conv1D, UpSampling1D, MaxPooling1D,Reshape, Flatten
from keras.models import load_model
from keras import optimizers
from keras.callbacks import EarlyStopping
#
#KERAS ENV GPU
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['NUMBAPRO_NVVM']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\nvvm\bin\nvvm64_31_0.dll'
#os.environ['NUMBAPRO_LIBDEVICE']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\nvvm\libdevice'

#Module logging
logger = logging.getLogger("Minerva")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler) 

def main():
	force = False
	#minerva = Minerva(Minerva.CHARGE)
	minerva = Minerva(Minerva.DISCHARGE)
	
	minerva.removeme()
	
	#if(len(sys.argv) == 2 and sys.argv[1].lower() == 'train'):
	#	logger.info("Training")
	#	minerva.train(force)
	#elif(len(sys.argv) == 2 and sys.argv[1].lower() == 'test'):
	#	logger.info("Testing")
	#	minerva.evaluate()
	#else:
	#	logger.error("Invalid command line argument.")
		
class Minerva():

	CHARGE = "C"
	DISCHARGE = "D"

	#modelName = "LSTM_DeepModel.h5"
	modelName = "Functional_Conv_DeepModel.h5"
	batchSize = 100
	epochs = 200
	imgPath = "./images"
	ets = None
	type = None
	timesteps = 60
	
	def __init__(self,type):
		logFolder = "./logs"
		# creates log folder
		if not os.path.exists(logFolder):
			os.makedirs(logFolder)
		
		logPath = logFolder + "/Minerva.log"
		hdlr = loghds.TimedRotatingFileHandler(logPath,
                                       when="H",
                                       interval=1,
                                       backupCount=5)
		hdlr.setFormatter(formatter)
		logger.addHandler(hdlr)
		
		self.ets = EpisodedTimeSeries(normalize=True)
		self.type = type
		
	def removeme(self):
		self.ets.buildUniformedDataSet(os.path.join(".","dataset"),force=True)
		#episodes = self.ets.loadUniformedDataSet()
		#self.ets.seekEpisodesBlow(episodes)
		#self.ets.loadBlowDataSet()
	
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
		
		if(False):
			x_test = self.__conv2DCompatible(x_test)
		
		self.__evaluateModel(x_test, y_test,True,normalizer)
		
	
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
		if(False):
			x_train = self.__conv2DCompatible(x_train)
			x_valid = self.__conv2DCompatible(x_valid)
		
		#model = self.__convModel(inputFeatures,outputFeatures,timesteps,x_train.shape[1:])
		model = self.__functionalConvModel(inputFeatures,outputFeatures,x_train)
		
		
		# end model
		
		adam = optimizers.Adam(lr=0.000005)		
		model.compile(loss='mean_squared_error', optimizer=adam,metrics=['mae'])
		print(model.summary())
		
		
		early = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='min')
		
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
		
	def __convModel(self,inputFeatures,outputFeatures,timesteps,inShape):
		model = Sequential()
		
		mask = (2,2)
		#mask = 4
		
		start = 512
		model.add(Dense(start,activation='relu', input_shape=inShape))
		model.add(Conv2D(start,mask, activation='relu'))
		model.add(Conv2D(int(start/2),mask, activation='relu')) 
		model.add(Conv2D(int(start/4),mask, activation='relu')) 
		model.add(Conv2D(int(start/8),mask, activation='relu')) 
		#model.add(Conv2D(int(start/16),mask, activation='relu'))
		#model.add(Conv2D(int(start/32),mask, activation='relu'))
		#model.add(Conv2D(int(start/64),mask, activation='relu'))
		model.add(MaxPooling2D(pool_size=mask))
		model.add(Flatten())
		# decoder
		model.add(Dense(timesteps*outputFeatures*8, activation='relu'))
		model.add(Dense(timesteps*outputFeatures*4, activation='relu'))
		model.add(Dense(timesteps*outputFeatures*2, activation='relu'))		
		model.add(Dense(timesteps*outputFeatures, activation='tanh'))
		model.add(Reshape((timesteps, outputFeatures)))

		return model
	
	def __functionalConvModel(self,inputFeatures,outputFeatures,data):
	
		timesteps = data.shape[1]
		signals   = data.shape[2]
		width  = 20
		heigth = 20
		deepth = timesteps * signals
		
		mask = (5,5)
		mask2 = (3,3)
		poolMask = (2,2)

		inputs = Input(shape=(timesteps,signals))
		
		initParams = 4
		outParams = 1024
		
		enlarge  = Dense(width*heigth*signals,activation='relu')(inputs)
		reshape1 = Reshape((width, heigth,deepth))(enlarge)
		conv1 =    Conv2D(initParams*8,mask, activation='relu')(reshape1)
		conv2 =    Conv2D(initParams*4,mask2, activation='relu')(conv1)
		maxpool1 = MaxPooling2D(pool_size=poolMask)(conv2)
		conv3 =    Conv2D(initParams,mask2, activation='relu')(maxpool1)
		maxpool2 = MaxPooling2D(pool_size=poolMask)(conv3)
		encoded = Flatten()(maxpool2) 
		
		dec1 = Dense(outParams,activation='relu')(encoded)
		decoded = Dense(timesteps*outputFeatures, activation='tanh')(dec1)
		out = Reshape((timesteps, outputFeatures))(decoded)
		
		model = Model(inputs=inputs, outputs=out)
		return model
	
	
	def __evaluateModel(self,x_test,y_test,saveImgs = False,scaler = None):
		
		logger.info("Loading model...")
		model = load_model(os.path.join( self.ets.rootResultFolder , self.modelName ))
		
		print(model.summary())
		
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
					p = plt.plot(pred,color="navy",label="Prediction")
					r = plt.plot(real,color="orange",label="Real")
					plt.legend()
					sid +=1
				else:
					p = plt.plot(y_pred[toPlot][:, col],color="navy",label="Prediction")
					r = plt.plot(y_test[toPlot][:, col],color="orange",label="Real")
					plt.legend()
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
		
		data = data.reshape(data.shape[0],12,10,4)
		print(data.shape)
		return data
		
main()