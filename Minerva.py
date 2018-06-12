#Standard Imports
import uuid,time,os,logging, numpy as np, sys, pandas as pd , matplotlib.pyplot as plt

from logging import handlers as loghds

#Project module import
from Demetra import EpisodedTimeSeries

#KERAS
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, RepeatVector, Input, Dropout, Activation, Masking, Lambda
from keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, Conv1D, UpSampling1D, MaxPooling1D,Reshape, Flatten
from keras.models import load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger
from keras.preprocessing.sequence import pad_sequences
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
	
	mode = "swab2swab" #"swabCleanDischarge"
	minerva = Minerva()
	minerva.ets.buildDataSet(os.path.join(".","dataset"),mode=mode,force=False) # creates dataset
	########################
	#Month by month prediction
	#minerva.train4month(0)
	#minerva.predict4month(3,"Month_Conv1D",plot2video = False)
	########################
	
	########################
	# All dataset
	batteries = minerva.ets.loadBlowDataSet()
	minerva.crossTrain(batteries) # Model train and cross validate
	########################

class Minerva():
	
	logFolder = "./logs"
	modelName = "Conv1D"
	#modelName = "LSTM"
	modelExt = ".h5"
	batchSize = 100
	epochs = 500
	ets = None
	
	def __init__(self):
		
		# creates log folder
		if not os.path.exists(self.logFolder):
			os.makedirs(self.logFolder)
		
		logFile = self.logFolder + "/Minerva.log"
		hdlr = loghds.TimedRotatingFileHandler(logFile,
                                       when="H",
                                       interval=1,
                                       backupCount=30)
		hdlr.setFormatter(formatter)
		logger.addHandler(hdlr)
		self.ets = EpisodedTimeSeries()
	
	
	def predict4month(self,monthIndex,name4model,plot2video = False):
		batteries = self.ets.loadBlowDataSet(monthIndexes=[monthIndex]) # blows
		logger.info("Model trained on month 0, predicting month %d" % monthIndex)
		self.__predict(batteries,name4model,plot2video)
	
	
	def train4month(self,monthIndex):
		allDataset = self.ets.loadDataSet()
		xscaler,yscaler = self.__getXYscaler(allDataset)
		batteries = self.ets.loadBlowDataSet(monthIndexes=[monthIndex])
		_,_ = self.__getXYscaler(batteries)
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		
		
		xtrain, xvalid, ytrain, yvalid = train_test_split( x, y, test_size=0.1, random_state=42)
		
		name4model = "Month_" + self.modelName
		self.__trainlModel(xtrain, ytrain, xvalid, yvalid,name4model)
		
		
		#this is for padding
		#if(False)
		#	maxLen = 20
		#	maskVal = -1.0
		#	xpad = pad_sequences(xout, maxlen=maxLen, dtype='float32', padding='post', truncating='post', value=maskVal)
		#	logger.info("X padded")
		#	ypad = pad_sequences(yout, maxlen=maxLen, dtype='float32', padding='post', truncating='post', value=maskVal)
		#	logger.info("Y padded")
		
		
		
	def crossTrain(self,batteries):
		
		xscaler,yscaler = self.__getXYscaler(batteries)
		
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)

		foldCounter = 0
		kf = KFold(n_splits=5, random_state=42, shuffle=True)
		for train_index, test_index in kf.split(x):
			
			
			trainX, testX = x[train_index], x[test_index]
			trainY, testY = y[train_index], y[test_index]
			
			# validation set
			validPerc = 0.1
			trainX, validX, trainY, validY = train_test_split( trainX, trainY, test_size=validPerc, random_state=42)
			
			name4model = "Fold_%d_%s" % (foldCounter,self.modelName)
			
			self.__trainlModel(trainX, trainY, validX, validY,name4model)
			self.__evaluateModel(testX, testY,name4model,yscaler,False)
			foldCounter += 1


	def __evaluateModel(self,testX,testY,model2load,scaler = None, plot2video = False):
		
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt))
		
		
		
		testX = self.__batchCompatible(self.batchSize,testX)
		testY = self.__batchCompatible(self.batchSize,testY)
		
		logger.info("Testing model %s with test %s" % (model2load,testX.shape))
		
		tt = time.clock()
		mse, mae = model.evaluate( x=testX, y=testY, batch_size=self.batchSize, verbose=0)
		logger.info("Test MSE %f - MAE %f Elapsed %f" % (mse,mae,(time.clock() - tt)))

		if(plot2video):
			logger.info("Predicting")
			tt = time.clock()
			y_pred = model.predict(testX,  batch_size=self.batchSize)
			logger.info("Elapsed %f" % (time.clock() - tt))
			if(scaler is not None):
				y_pred = self.__skScaleBack(y_pred,scaler)
				testY = self.__skScaleBack(testY,scaler)
			for r in range(25):
				plt.figure()
				toPlot = np.random.randint(y_pred.shape[0])
				i = 1
				sid = 14
				for col in range(y_pred.shape[2]):
					plt.subplot(y_pred.shape[2], 1, i)
					plt.plot(y_pred[toPlot][:, col],color="navy",label="Prediction")
					plt.plot(testY[toPlot][:, col],color="orange",label="Real")
					plt.legend()
					i += 1
				
				plt.show()
	
	def __predict(self,batteries,name4model,plot2video = False):
		allDataset = self.ets.loadDataSet()
		xscaler,yscaler = self.__getXYscaler(allDataset)
		_,_ = self.__getXYscaler(batteries)
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		self.__evaluateModel(x, y,name4model,yscaler,plot2video)
	
	def __trainlModel(self,x_train, y_train, x_valid, y_valid,name4model):
		
		tt = time.clock()
		logger.debug("__trainlModel - start")
		
		x_train = self.__batchCompatible(self.batchSize,x_train)
		y_train = self.__batchCompatible(self.batchSize,y_train)
		x_valid = self.__batchCompatible(self.batchSize,x_valid)
		y_valid = self.__batchCompatible(self.batchSize,y_valid)
		
		logger.info("Training model %s with train %s and valid %s" % (name4model,x_train.shape,x_valid.shape))
		
		inputFeatures  = x_train.shape[2]
		outputFeatures = y_train.shape[2]
		timesteps =  x_train.shape[1]

		model = self.__functionalModel(inputFeatures,outputFeatures,x_train)	
		
		adam = optimizers.Adam(lr=0.000005)		
		model.compile(loss='mean_squared_error', optimizer=adam,metrics=['mae'])
		
		
		early = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='min')
		
		cvsLogFile = os.path.join(self.logFolder,name4model+'.log')
		
		csv_logger = CSVLogger(cvsLogFile)
		
		model.fit(x_train, y_train,
			verbose = 0,
			batch_size=self.batchSize,
			epochs=self.epochs,
			validation_data=(x_valid,y_valid),
			callbacks=[early,csv_logger]
		)
		
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		logger.debug("Saving model...")
		model.save(os.path.join( self.ets.rootResultFolder , name4model+self.modelExt )) 
		logger.debug("Model saved")
		
		trainMse, trainMae = model.evaluate( x=x_train, y=y_train, batch_size=self.batchSize, verbose=0)
		logger.info("Train MSE %f - MAE %f" % (trainMse,trainMae))
		validMse, validMae = model.evaluate( x=x_valid, y=y_valid, batch_size=self.batchSize, verbose=0)
		logger.info("Valid MSE %f - MAE %f" % (validMse,validMae))
		logger.debug("__trainlModel - end - %f" % (time.clock() - tt) )

	def __functionalModel(self,inputFeatures,outputFeatures,data):
	
		timesteps = data.shape[1]
		signals   = data.shape[2]
		inputs = Input(shape=(timesteps,signals))
		
		# OK CONV2D
		width  = 10
		heigth = 10
		deepth = timesteps * signals
		mask = (3,3)
		mask2 = (2,2)
		poolMask = (2,2)
		initParams = 2
		outParams = 256
		enlarge  = Dense(width*heigth*signals,activation='relu')(inputs)
		reshape1 = Reshape((width, heigth,deepth))(enlarge)
		conv1 =    Conv2D(initParams*8,mask, activation='relu')(reshape1)
		conv2 =    Conv2D(initParams*4,mask2, activation='relu')(conv1)
		maxpool1 = MaxPooling2D(pool_size=poolMask)(conv2)
		conv3 =    Conv2D(initParams,mask2, activation='relu')(maxpool1)
		#maxpool2 = MaxPooling2D(pool_size=poolMask)(conv3)
		encoded = Flatten()(conv3) 
		dec1 = Dense(outParams,activation='relu')(encoded)
		decoded = Dense(timesteps*outputFeatures, activation='tanh')(dec1)
		out = Reshape((timesteps, outputFeatures))(decoded)
		
		# OK CONV1D 7308
		#encoderFilter = 32
		#encoderKernel = 8
		#encoderPool = 2
		#encodedSize = 8
		#conv1 = Conv1D(encoderFilter,encoderKernel,activation='relu')(inputs)
		#maxpool2 = MaxPooling1D(pool_size=encoderPool)(conv1)
		#flat1 = Flatten()(maxpool2) 
		#encoded = Dense(encodedSize,activation='relu')(flat1)
		#dec2 = Dense(encodedSize*4,activation='relu')(encoded)
		#decoded = Dense(timesteps*outputFeatures, activation='tanh')(dec2)
		#out = Reshape((timesteps, outputFeatures))(decoded)
		
		# OK LSTM
		#latent_dim = 16
		#encodedSize = 8
		#encoder1 = LSTM(latent_dim,return_sequences=True,activation='relu')(inputs)
		#encoder2 = LSTM(8,return_sequences=True,activation='relu')(encoder1)
		#encoder3 = LSTM(4,return_sequences=True,activation='relu')(encoder2)
		#flat1 = Flatten()(encoder3) 
		#encoded = Dense(encodedSize,activation='relu')(flat1)	
		#dec1 = Dense(encodedSize*2,activation='relu')(encoded)
		#dec2 = Dense(encodedSize*4,activation='relu')(dec1)
		#decoded = Dense(timesteps*outputFeatures, activation='tanh')(dec2)
		#out = Reshape((timesteps, outputFeatures))(decoded)
		
		
		model = Model(inputs=inputs, outputs=out)
		print(model.summary())
		
		return model
	
	def __getXYscaler(self,batteries):
		"""
		Creates the xscaler and y scaler for the dataset
		batteries: 3 layer list of dataframe [battery][month][episode] = dataframe
		"""
		tt = time.clock()
		logger.debug("__getXYscaler - start")
		x,y = self.__datasetAs2DArray(batteries)
		xscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
		xscaler.fit(x)
		yscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
		yscaler.fit(y)
		logger.debug("__getXYscaler - end - %f" % (time.clock() - tt) )
		return xscaler,yscaler
	
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
		"""
		Transform data shape 0 in a multiple of batch_size
		"""
		exceed = data.shape[0] % batch_size
		if(exceed > 0):
			data = data[:-exceed]
		return data
	
	def __datasetAs2DArray(self,batteries):
		"""
		Convert dataset to numpy 2D array
		batteries: 3 layer list of dataframe [battery][month][episode] = dataframe
		"""
		tt = time.clock()
		logger.debug("__datasetAs2DArray - start")
		x = []
		y = []
		for battery in batteries:
			for month in battery:
				for episode in month:
					episode.drop(columns=self.ets.dropX,inplace=True)
					ydf = episode[self.ets.keepY]
					for t in range(0,episode.shape[0]):
						x.append(episode.values[t])
						y.append(ydf.values[t])
		outX = np.asarray(x)
		outY = np.asarray(y)
		tt = time.clock()
		logger.debug("__datasetAs2DArray - end - %f" % (time.clock() - tt) )
		return outX,outY
	
	def __datasetAs3DArray(self,batteries,xscaler=None,yscaler=None):
		"""
		Convert the dataset list structure to numpy 3D array
		batteries: 3 layer list of dataframe [battery][month][episode] = dataframe
		if scaler are specified, data will be transformed
		"""
		tt = time.clock()
		logger.debug("__datasetAs3DArray - start")
		xlist = []
		ylist = []
		for battery in batteries:
			monthly = []
			for month in battery:
				for e in month:
					x = e.values
					y = e[self.ets.keepY].values
					if(xscaler is not None):
						x = xscaler.transform(x)
					if(yscaler is not None):
						y = yscaler.transform(y)
					xlist.append( x )
					ylist.append( y )
		
		outX = np.asarray(xlist)
		outY = np.asarray(ylist)
		logger.debug("__datasetAs3DArray - end - %f" % (time.clock() - tt) )
		return outX,outY
		
		
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