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
from keras.callbacks import EarlyStopping, CSVLogger

#sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
#
#KERAS ENV GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NUMBAPRO_NVVM']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\nvvm\bin\nvvm64_31_0.dll'
os.environ['NUMBAPRO_LIBDEVICE']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\nvvm\libdevice'

#Module logging
logger = logging.getLogger("Minerva")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler) 

def main():

	minerva = Minerva()
	minerva.ets.buildDataSet(os.path.join(".","dataset"),force=False)
	
	minerva.ets.dataSetSummary()
	#e = minerva.ets.loadDataSet()
	#e = minerva.ets.buildBlowDataSet(monthIndex=3)
	#print(len(e)) # batteries
	#print(len(e[0])) #months
	#print(len(e[0][0])) #episode in month 0
	#print(e[0][0][0][0].shape) #discharge blow
	#print(e[0][0][0][1].shape) #charge blow
	#minerva.ets.plot(e[0][0][0][0],mode="GUI",name=None)
	#minerva.ets.plot(e[0][0][0][1],mode="GUI",name=None)
	#minerva.ets.showEpisodes(monthIndex=2,mode="GUI")
	
	
class Minerva():

	modelName = "Functional_Conv_DeepModel.h5"
	batchSize = 50
	epochs = 500
	ets = None
	
	def __init__(self):
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
		self.ets = EpisodedTimeSeries()
	

	def trainMonth(self):
		
		blows = self.ets.loadBlowEpisodes("M",index=0)
		x,y = self.ets.getXYDataSet(blows)
		x = self.__listToNpArray(x)
		y = self.__listToNpArray(y)
				
		
		xscaler = self.__getSkScaler(x)
		yscaler = self.__getSkScaler(y)
		
		X_train, X_valid, y_train, y_valid =train_test_split( x, y, test_size=0.1, random_state=42)
		
		
		xvalid = self.__skScale(X_valid,xscaler)
		yvalid = self.__skScale(y_valid,yscaler)
		xtrain = self.__skScale(X_train,xscaler)
		ytrain = self.__skScale(y_train,yscaler)
		
		fold = "Mont"
		if False:
			self.__trainSequentialModel(xtrain, ytrain, xvalid, yvalid,fold)
		else:
			blows = self.ets.loadBlowEpisodes("M",index=3)
			xtest,ytest = self.ets.getXYDataSet(blows)
			xtest = self.__listToNpArray(xtest)
			ytest = self.__listToNpArray(ytest)
			xtest = self.__skScale(xtest,xscaler)
			ytest = self.__skScale(ytest,yscaler)
			self.__evaluateModel(xtest, ytest,fold,None)
		
	
	def crossEvaluate(self):
		logger.info("Blow load")
		blows  = self.ets.loadBlowEpisodes()
		logger.info("End load blow")
		x,y = self.ets.getXYDataSet(blows)
		logger.info("End blow concat")
		
		x = self.__listToNpArray(x)
		y = self.__listToNpArray(y)
				
		
		xscaler = self.__getSkScaler(x)
		yscaler = self.__getSkScaler(y)
		
		count = 0
		fold = 4
		kf = KFold(n_splits=4, random_state=42, shuffle=True)
		for train_index, test_index in kf.split(x):
			count += 1
			if(count != fold):
				continue
			
			xtest = x[test_index]
			ytest = y[test_index]
			
			xtest  = self.__skScale(xtest,xscaler)
			ytest  = self.__skScale(ytest,yscaler)
			
			self.__evaluateModel(xtest, ytest,fold,None)
			return
		
	def crossTrain(self):
		
		logger.info("Blow load")
		blows  = self.ets.loadBlowEpisodes()
		logger.info("End load blow")
		x,y = self.ets.getXYDataSet(blows)
		logger.info("End blow concat")
		
		x = self.__listToNpArray(x)
		y = self.__listToNpArray(y)
		
		xscaler = self.__getSkScaler(x)
		yscaler = self.__getSkScaler(y)
		
		fold = 0
		kf = KFold(n_splits=4, random_state=42, shuffle=True)
		for train_index, test_index in kf.split(x):
			
			fold += 1
			X_train, X_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			X_train, X_valid, y_train, y_valid = train_test_split( X_train, y_train, test_size=0.2, random_state=42)
			
			xvalid = self.__skScale(X_valid,xscaler)
			yvalid = self.__skScale(y_valid,yscaler)
			xtrain = self.__skScale(X_train,xscaler)
			ytrain = self.__skScale(y_train,yscaler)
		
			
			self.__trainSequentialModel(xtrain, ytrain, xvalid, yvalid,fold)
			xtest  = self.__skScale(X_test,xscaler)
			ytest  = self.__skScale(y_test,yscaler)
			
			self.__evaluateModel(xtest, ytest,fold,False,None)
		
		# sample to check scaler behavior
		#self.__showNumpyArray(xtrain)
		#xtrain = self.__skScaleBack(xtrain,xscaler)
		#self.__showNumpyArray(xtrain)

	def train(self):
		xtrain,ytrain, xvalid,yvalid, xtest,ytest,xscaler,yscaler = self.__splitDataset()
		fold = "A_"
		if(False):
			self.__trainSequentialModel(xtrain, ytrain, xvalid, yvalid,fold)
		self.__evaluateModel(xtest, ytest,fold,yscaler)
		
	def __splitDataset(self):
		logger.info("Blow load")
		blows  = self.ets.loadBlowEpisodes()
		logger.info("End load blow")
		x,y = self.ets.getXYDataSet(blows)
		logger.info("End blow concat")
		
		x = self.__listToNpArray(x)
		y = self.__listToNpArray(y)
		
		xscaler = self.__getSkScaler(x)
		yscaler = self.__getSkScaler(y)
		
		X_train, X_test,  y_train, y_test =train_test_split( x, y, test_size=0.2, random_state=42)
		X_train, X_valid, y_train, y_valid =train_test_split( X_train, y_train, test_size=0.2, random_state=42)
		
		
		xvalid = self.__skScale(X_valid,xscaler)
		yvalid = self.__skScale(y_valid,yscaler)
		xtrain = self.__skScale(X_train,xscaler)
		ytrain = self.__skScale(y_train,yscaler)
		
		xtest  = self.__skScale(X_test,xscaler)
		ytest  = self.__skScale(y_test,yscaler)
		
		return xtrain,ytrain, xvalid,yvalid, xtest,ytest, xscaler,yscaler
		
		
	def __trainSequentialModel(self,x_train, y_train, x_valid, y_valid,fold):
		
		x_train = self.__batchCompatible(self.batchSize,x_train)
		y_train = self.__batchCompatible(self.batchSize,y_train)
		x_valid = self.__batchCompatible(self.batchSize,x_valid)
		y_valid = self.__batchCompatible(self.batchSize,y_valid)
		
		tt = time.clock()
		
		inputFeatures  = x_train.shape[2]
		outputFeatures = y_train.shape[2]
		timesteps =  x_train.shape[1]

		#model = self.__functionalConvModel(inputFeatures,outputFeatures,x_train)	
		
		model = self.__functionalLSTMModel(inputFeatures,outputFeatures,x_train)	
		adam = optimizers.Adam(lr=0.000005)		
		model.compile(loss='mean_squared_error', optimizer=adam,metrics=['mae'])
		print(model.summary())
		
		early = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='min')
		
		csv_logger = CSVLogger(str(fold)+'training.log')
		
		model.fit(x_train, y_train,
			batch_size=self.batchSize,
			epochs=self.epochs,
			validation_data=(x_valid,y_valid),
			callbacks=[early,csv_logger]
		)
		
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		logger.info("Save model")
		model.save(os.path.join( self.ets.rootResultFolder , str(fold)+self.modelName )) 
		logger.info("Model saved")
		return model
	
	
	def __functionalLSTMModel(self,inputFeatures,outputFeatures,data):
	
		timesteps = data.shape[1]
		signals   = data.shape[2]
	
		inputs = Input(shape=(timesteps,signals))
		
		mask = 4
		
		s = 512
		a = Conv1D(s,mask, activation='relu')(inputs)
		b = Conv1D(int(s/2),mask, activation='relu')(a)
		c = Conv1D(int(s/4),mask, activation='relu')(b)
		c1 = Conv1D(int(s/8),mask, activation='relu')(c)
		d = Flatten()(c1)
		
		e = Dense(s,activation='relu')(d)
		f = Dense(timesteps*outputFeatures, activation='tanh')(e)
		
		out = Reshape((timesteps, outputFeatures))(f)
		
		model = Model(inputs=inputs, outputs=out)
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
		
		initParams = 2
		outParams = 512
		
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
	
	
	def __evaluateModel(self,x_test,y_test,fold,scaler = None):
		
		logger.info("Loading model...")
		model = load_model(os.path.join( self.ets.rootResultFolder ,str(fold)+self.modelName ))
		
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
		
		if(scaler is not None):
			y_pred = self.__skScaleBack(y_pred,scaler)
			y_test = self.__skScaleBack(y_test,scaler)
		
		if(True):
			for r in range(25):
				plt.figure()
				toPlot = np.random.randint(y_pred.shape[0])
				i = 1
				sid = 14
				for col in range(y_pred.shape[2]):
					plt.subplot(y_pred.shape[2], 1, i)
					plt.plot(y_pred[toPlot][:, col],color="navy",label="Prediction")
					plt.plot(y_test[toPlot][:, col],color="orange",label="Real")
					plt.legend()
					i += 1
				
				plt.show()
	
	def __showNumpyArray(self,data):
		plt.figure()
		#toPlot = np.random.randint(data.shape[0])
		toPlot = 0
		i = 1
		for col in range(data.shape[2]):
			plt.subplot(data.shape[2], 1, i)
			plt.plot(data[toPlot][:, col],color="orange",label="Real")
			plt.legend()
			i += 1
		
		plt.show()
	
	def __batchCompatible(self,batch_size,data):
		exceed = data.shape[0] % batch_size
		if(exceed > 0):
			data = data[:-exceed]
		return data
		
	def __listToNpArray(self,list):
		"""
		Convert a list of dataframe [time-steps,features] in np 3d array [samples,time-steps,features]
		"""
		out = None
		if(len(list) > 0):
			out = np.zeros([len(list),list[0].shape[0],list[0].shape[1]])
			for i in range(0,len(list)):
				out[i] = list[i]
		else:
			logger.warning("Empty list, None will be returned")
		return out
		
	def __getSkScaler(self,data):
		"""
		Build a SKL scaler for the data
		data: 3d array [samples,time-steps,features]
		return: scaled 3d array
		"""
		scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
		#scaler = preprocessing.RobustScaler( quantile_range=(0.0, 100.0))
		samples = data.shape[0]
		timesteps = data.shape[1]
		features = data.shape[2]
		# the scaler need a 2d array. [<samples>,<features>]
		# so we reshape the data aggregating samples and time-steps leaving features alone
		xnorm = data.reshape(samples*timesteps,features)
		scaler.fit(xnorm)
		return scaler
	
	def __skScale(self,data,scaler):
		"""
		Apply the SKL scaler to the data
		data: 3d array [samples,time-steps,features]
		return: scaled 3d array
		"""
		samples = data.shape[0]
		timesteps = data.shape[1]
		features = data.shape[2]
		# the scaler need a 2d array. [<samples>,<features>]
		# so we reshape the data aggregating samples and time-steps leaving features alone
		xnorm = data.reshape(samples*timesteps,features)
		xnorm = scaler.transform(xnorm)
		return xnorm.reshape(samples,timesteps,features)
		
	def __skScaleBack(self,data,scaler):
		"""
		Apply the SKL scale back to the data
		data: 3d array [samples,time-steps,features]
		return: scaled 3d array
		"""
		samples = data.shape[0]
		timesteps = data.shape[1]
		features = data.shape[2]
		# the scaler need a 2d array. [<samples>,<features>]
		# so we reshape the data aggregating samples and time-steps leaving features alone
		xnorm = data.reshape(samples*timesteps,features)
		xnorm = scaler.inverse_transform(xnorm)
		return xnorm.reshape(samples,timesteps,features)


main()		