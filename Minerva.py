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
# Ex: xpad = pad_sequences(xout, maxlen=maxLen, dtype='float32', padding='post', truncating='post', value=maskVal)
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
	
	force = False
	eps1 	= 5
	eps2 	= 5
	alpha1 	= 5
	alpha2 	= 5
	
	mode = "swab2swab" #"swabCleanDischarge"
	minerva = Minerva(eps1=eps1,eps2=eps2,alpha1=alpha1,alpha2=alpha2)
	#minerva.ets.buildDataSet(os.path.join(".","dataset"),mode=mode,force=force) # creates dataset of not exists
	mode =  "GUI" #"server" # set mode to server in order to save plot to disk instead of showing on video
	
	######################### 
	# show the histogram of resistance distribution month by month
	#minerva.ets.resistanceDistribution(batteries,join=joinDischargeCharge,mode=mode)
	
	########################
	#Month by month prediction
	#name4model = "Fold_%d_%s_%d_%d_%d_%d" % (0,minerva.modelName,eps1,eps2,alpha1,alpha2)
	#minerva.train4month(0)
	minerva.predict4month(1,plot2video = False)
	minerva.predict4month(2,plot2video = False)
	minerva.predict4month(3,plot2video = False)
	
	########################
	# All dataset predictions
	#joinDischargeCharge = True # if False one episode is loaded as a tuple wiht 0 discharge blow and 1 charge blow
	#batteries = minerva.ets.loadBlowDataSet(join=joinDischargeCharge) # load the dataset
	
	
	# cross train the model
	#minerva.crossTrain(batteries) # Model train and cross validate
	# cross validate the model
	# minerva.crossValidate(batteries,mode=mode)

class Minerva():
	
	logFolder = "./logs"
	#modelName = "InceptionConv2Dlogcosh"
	modelName = "ZeroInceptionConv2Dlogcosh"
	modelExt = ".h5"
	batchSize = 100
	epochs = 500
	ets = None
	eps1   = 5
	eps2   = 5
	alpha1 = 5
	alpha2 = 5
	
	
	def __init__(self,eps1,eps2,alpha1,alpha2):
		# 
		# creates log folder
		if not os.path.exists(self.logFolder):
			os.makedirs(self.logFolder)
		
		self.eps1   = eps1
		self.eps2   = eps2
		self.alpha1 = alpha1
		self.alpha2 = alpha2
		
		logFile = self.logFolder + "/Minerva.log"
		hdlr = loghds.TimedRotatingFileHandler(logFile,
                                       when="H",
                                       interval=1,
                                       backupCount=30)
		hdlr.setFormatter(formatter)
		logger.addHandler(hdlr)
		self.ets = EpisodedTimeSeries(self.eps1,self.eps2,self.alpha1,self.alpha2)
	
	
	def predict4month(self,monthIndex,plot2video = False):
		batteries = self.ets.loadBlowDataSet(monthIndexes=[monthIndex]) # blows
		logger.info("Model trained on month 0, predicting month %d" % monthIndex)
		name4model = "Month_%s_%d_%d_%d_%d" % ( self.modelName,self.eps1,self.eps2,self.alpha1,self.alpha2 )
		self.__predict(batteries,name4model,plot2video)
	
	def train4month(self,monthIndex):
		allDataset = self.ets.loadDataSet()
		xscaler,yscaler = self.__getXYscaler(allDataset)
		batteries = self.ets.loadBlowDataSet(monthIndexes=[monthIndex])
		_,_ = self.__getXYscaler(batteries)
		
		xscaler,yscaler = None,None
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		xtrain, xvalid, ytrain, yvalid = train_test_split( x, y, test_size=0.1, random_state=42)
		name4model = "Month_%s_%d_%d_%d_%d" % ( self.modelName,self.eps1,self.eps2,self.alpha1,self.alpha2 )
		self.__trainlModel(xtrain, ytrain, xvalid, yvalid,name4model)

	def crossTrain(self,batteries):
		xscaler,yscaler = self.__getXYscaler(batteries)
		#x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		xscaler,yscaler = None, None
		x,y = self.__datasetAs3DArray(batteries)
		foldCounter = 0
		kf = KFold(n_splits=5, random_state=42, shuffle=True)
		for train_index, test_index in kf.split(x):
			trainX, testX = x[train_index], x[test_index]
			trainY, testY = y[train_index], y[test_index]
			# validation set
			validPerc = 0.1
			trainX, validX, trainY, validY = train_test_split( trainX, trainY, test_size=validPerc, random_state=42)
			name4model = "Fold_%d_%s_%d_%d_%d_%d" % (foldCounter,self.modelName,self.eps1,self.eps2,self.alpha1,self.alpha2)
			self.__trainlModel(trainX, trainY, validX, validY,name4model)
			self.__evaluateModel(testX, testY,name4model,yscaler,False)
			foldCounter += 1

	def crossValidate(self,batteries,plot=True,mode = "server"):
		xscaler,yscaler = self.__getXYscaler(batteries)
		#x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		xscaler,yscaler = None, None
		x,y = self.__datasetAs3DArray(batteries)
		foldCounter = 0
		kf = KFold(n_splits=5, random_state=42, shuffle=True)
		for train_index, test_index in kf.split(x):
			testX = x[test_index]
			testY = y[test_index]
			name4model = "Fold_%d_%s_%d_%d_%d_%d" % (foldCounter,self.modelName,self.eps1,self.eps2,self.alpha1,self.alpha2)
			self.__evaluateModel(testX, testY,name4model,mode,yscaler,plot)
			foldCounter += 1
	
	def __evaluateModel(self,testX,testY,model2load,mode,scaler = None, plot = False):
		
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt))
		
		testX = self.__batchCompatible(self.batchSize,testX)
		testY = self.__batchCompatible(self.batchSize,testY)
		
		logger.info("Testing model %s with test %s" % (model2load,testX.shape))
		
		tt = time.clock()
		mse, mae = model.evaluate( x=testX, y=testY, batch_size=self.batchSize, verbose=0)
		logger.info("Test MSE %f - MAE %f Elapsed %f" % (mse,mae,(time.clock() - tt)))
		
		if(mode == "server" ):
			plt.switch_backend('agg')
			if not os.path.exists(self.ets.episodeImageFolder):
				os.makedirs(self.ets.episodeImageFolder)
		
		if(plot):
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
				
				title = str(toPlot) +"_"+str(uuid.uuid4())
				self.ets.plotMode(mode,title)

	def __predict(self,batteries,name4model,plot = False):
		allDataset = self.ets.loadDataSet()
		xscaler,yscaler = self.__getXYscaler(allDataset)
		_,_ = self.__getXYscaler(batteries)
		xscaler,yscaler = None, None
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		self.__evaluateModel(x, y,name4model,yscaler,plot)
	
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
		
		adam = optimizers.Adam(lr=0.000001)		
		model.compile(loss='logcosh', optimizer=adam,metrics=['mae'])
	
		early = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=0, mode='min')	
		cvsLogFile = os.path.join(self.logFolder,name4model+'.log')
		csv_logger = CSVLogger(cvsLogFile)
		model.fit(x_train, y_train,
			verbose = 1,
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
		logger.info("Train MAE %f - LCH %f" % (trainMse,trainMae))
		validMse, validMae = model.evaluate( x=x_valid, y=y_valid, batch_size=self.batchSize, verbose=0)
		logger.info("Valid MAE %f - LCH %f" % (validMse,validMae))
		logger.debug("__trainlModel - end - %f" % (time.clock() - tt) )

	def __functionalModel(self,inputFeatures,outputFeatures,data):
		
		encodedSize = 8
		timesteps = data.shape[1]
		signals   = data.shape[2]
		inputs = Input(shape=(timesteps,signals))
		
		# OK CONV2D
		width  = 15
		heigth = 15
		deepth = timesteps * signals
		mask = (5,5)
		mask2 = (3,3)
		poolMask = (3,3)
		initParams = 16
		outParams = 512
		
		enlarge1  = Dense(width*heigth*signals,activation='relu')(inputs)
		reshape1 = Reshape((width, heigth,deepth))(enlarge1)
		conv1 =    Conv2D(initParams*8,mask, activation='relu')(reshape1)
		conv2 =    Conv2D(initParams*4,mask2, activation='relu')(conv1)
		maxpool1 = MaxPooling2D(pool_size=poolMask)(conv2)
		conv3 =    Conv2D(initParams,mask2, activation='relu')(maxpool1)
		
		flat1 = Flatten()(conv3)
		enlarge2  = Dense(width*heigth*deepth,activation='relu')(flat1)
		reshape2 = Reshape((width, heigth,deepth))(enlarge2)
		conv4 =    Conv2D(initParams*8,mask, activation='relu')(reshape2)
		conv5 =    Conv2D(initParams*4,mask2, activation='relu')(conv4)
		maxpool2 = MaxPooling2D(pool_size=poolMask)(conv5)
		conv6 =    Conv2D(initParams,mask2, activation='relu')(maxpool2)
		
		flat2 = Flatten()(conv6) 
		encoded = Dense(encodedSize,activation='relu',name="encoder")(flat2)
		
		decenlarge2  = Dense(width*heigth*deepth,activation='relu')(encoded)
		decreshape2 = Reshape((width, heigth,deepth))(decenlarge2)
		decconv4 =    Conv2D(initParams*8,mask, activation='relu')(decreshape2)
		decconv5 =    Conv2D(initParams*4,mask2, activation='relu')(decconv4)
		decmaxpool2 = MaxPooling2D(pool_size=poolMask)(decconv5)
		decconv6 =    Conv2D(initParams,mask2, activation='relu')(decmaxpool2)
		
		flat3 = Flatten()(decconv6) 
		dec1 = Dense(outParams,activation='relu')(flat3)
		decoded = Dense(timesteps*outputFeatures, activation='linear')(dec1)
		out = Reshape((timesteps, outputFeatures))(decoded)
		
		# OK CONV1D 7308
		#encoderFilter = 128
		#encoderKernel = 4
		#encoderPool = 2
		#kernel_initializer='glorot_uniform', bias_initializer='zeros',
		#conv1 = Conv1D(encoderFilter,encoderKernel,activation='relu',name="CV1")(inputs)
		#maxpool1 = MaxPooling1D(pool_size=encoderPool,name="MP1")(conv1)
		#
		#conv2 = Conv1D(encoderFilter,encoderKernel,activation='relu',name="CV2")(maxpool1)
		#maxpool2 = MaxPooling1D(pool_size=encoderPool,name="MP2")(conv2)
		#
		#flat1 = Flatten(name="FT1")(maxpool2) 
		#encoded = Dense(encodedSize,activation='relu',name="encoder")(flat1)
		#
		#dec1 = Dense(encodedSize*4,activation='relu',name="DC1")(encoded)
		#decoded = Dense(timesteps*outputFeatures, activation='tanh',name="DC2")(dec1)
		#out = Reshape((timesteps, outputFeatures),name="decoder")(decoded)
		
		# OK LSTM
		#latent_dim = 16
		#encodedSize = 8
		#encoder1 = LSTM(latent_dim,return_sequences=True,activation='relu')(inputs)
		#encoder2 = LSTM(8,return_sequences=True,activation='relu')(encoder1)
		#encoder3 = LSTM(4,return_sequences=True,activation='relu')(encoder2)
		#flat1 = Flatten()(encoder3) 
		#encoded = Dense(encodedSize,activation='relu',name="encoder")(flat1)	
		#dec1 = Dense(encodedSize*2,activation='relu')(encoded)
		#dec2 = Dense(encodedSize*4,activation='relu')(dec1)
		#decoded = Dense(timesteps*outputFeatures, activation='tanh')(dec2)
		#out = Reshape((timesteps, outputFeatures))(decoded)
		
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		print(autoencoderModel.summary())
		return autoencoderModel
	
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