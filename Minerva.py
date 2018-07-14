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
from sklearn.metrics import mean_absolute_error
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
	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5)
	#minerva.ets.buildDataSet(os.path.join(".","dataset"),mode=mode,force=False) # creates dataset if does not exists
	plotMode = "GUI" #"GUI" #"server" # set mode to server in order to save plot to disk instead of showing on video
	if(plotMode == "server" ):
		plt.switch_backend('agg')
		if not os.path.exists(minerva.ets.episodeImageFolder):
			os.makedirs(minerva.ets.episodeImageFolder)
			
	######################### 
	# show the histogram of resistance distribution month by month for every battery
	#logger.info("Battery resistance distribution - start")
	#minerva.ets.resistanceDistribution(batteries,join=True,mode=plotMode)
	#logger.info("Battery resistance distribution - end")
	########################
	#logger.info("Autoencoder trained on month 0 - start")
	### Train the model on first month data for all batteris
	#minerva.train4month(0,forceTrain=False)
	### Month by month prediction
	#scaleDataset = True
	#xscaler,yscaler = None, None
	#if(scaleDataset):
	#	logger.info("Loading dataset")
	#	allDataset = minerva.ets.loadDataSet()
	#	minerva.dropDatasetLabel(allDataset)
	#	logger.info("Compute scaler")
	#	xscaler,yscaler = minerva.getXYscaler(allDataset)
	#	logger.info("Scaler loaded")
	### predict for every other months
	#minerva.decode4month(1,plotMode,showImages=True,xscaler=xscaler,yscaler=yscaler)
	#minerva.decode4month(2,plotMode,showImages=True,xscaler=xscaler,yscaler=yscaler)
	#minerva.decode4month(3,plotMode,showImages=True,xscaler=xscaler,yscaler=yscaler)
	#logger.info("Autoencoder trained on month 0 - end")
	########################
	## Train on all batteries and all months
	########################
	#logger.info("Autoencoder trained all months - start")
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	#minerva.crossTrain(batteries,forceTrain=False) #  cross train the model
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	#minerva.crossValidate(batteries,plotMode=plotMode) 	# cross validate the model
	#logger.info("Autoencoder trained all months - end")
	
	#######################
	## Anomaly detection
	#######################
	#logger.info("Loading the dataset")
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	#logger.info("Anomlay detection - start")
	#model2load = "Fold_2_" + minerva.modelName + "_5_5_5_5"
	#minerva.anomalyDetect(batteries,model2load,scaleDataset=True,plotMode=plotMode)
	#logger.info("Anomlay detection - end")
	
	
	##Show encoded plot
	#model2load = "Fold_1_" + minerva.modelName + "_5_5_5_5"
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	#encodedSize =4
	#minerva.plotEncoded(batteries,model2load,scaleDataset=True,plotMode=plotMode,encodedSize=encodedSize)
	
	
	batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	model2load = "Fold_1_" + minerva.modelName + "_5_5_5_5"
	minerva.decodeAndShow(batteries,model2load,scaleDataset=True,plotMode=plotMode)
	#print("Month 1")
	#batteries = minerva.ets.loadBlowDataSet(monthIndexes=[1])
	#minerva.decodeAndShow(batteries,model2load,scaleDataset=True,plotMode=plotMode)
	#print("Month 2")
	#batteries = minerva.ets.loadBlowDataSet(monthIndexes=[2])
	#minerva.decodeAndShow(batteries,model2load,scaleDataset=True,plotMode=plotMode)
	#print("Month 3")
	#batteries = minerva.ets.loadBlowDataSet(monthIndexes=[3])
	#minerva.decodeAndShow(batteries,model2load,scaleDataset=True,plotMode=plotMode)
	
class Minerva():
	
	logFolder = "./logs"
	modelName = "FullyConnected_4_"
	modelExt = ".h5"
	batchSize = 100
	epochs = 1000
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
	
	
	def decode4month(self,monthIndex,plotMode,showImages=False,xscaler=None,yscaler=None):
		logger.info("Model trained on month 0, autoencoding for month %d" % monthIndex)
		name4model = "Month_%s_%d_%d_%d_%d" % ( self.modelName,self.eps1,self.eps2,self.alpha1,self.alpha2 )
		batteries = self.ets.loadBlowDataSet(monthIndexes=[monthIndex]) # blows
		self.dropDatasetLabel(batteries)
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		self.__evaluateModel(x, y,name4model,plotMode,yscaler,showImages)
		
		
	
	def train4month(self,monthIndex,scaleDataset=True,forceTrain=False):
		
		name4model = "Month_%s_%d_%d_%d_%d" % ( self.modelName,self.eps1,self.eps2,self.alpha1,self.alpha2 )
		if not forceTrain and os.path.exists(os.path.join( self.ets.rootResultFolder , name4model+self.modelExt )):
			logger.info("Model %s already exists, skip training" % name4model)
			return

		allDataset = self.ets.loadDataSet()
		self.dropDatasetLabel(allDataset)
		batteries = self.ets.loadBlowDataSet(monthIndexes=[monthIndex])
		self.dropDatasetLabel(batteries)
		xscaler,yscaler = None,None
		if(scaleDataset):
			xscaler,yscaler = self.getXYscaler(allDataset)
		
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		xtrain, xvalid, ytrain, yvalid = train_test_split( x, y, test_size=0.1, random_state=42)
		self.__trainlModel(xtrain, ytrain, xvalid, yvalid,name4model)

	def crossTrain(self,batteries,plotMode="server",scaleDataset=True,forceTrain=False):
		xscaler,yscaler = None, None
		self.dropDatasetLabel(batteries)
		if(scaleDataset):
			xscaler,yscaler = self.getXYscaler(batteries)
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		foldCounter = 0
		kf = KFold(n_splits=5, random_state=42, shuffle=True)
		for train_index, test_index in kf.split(x):
			name4model = "Fold_%d_%s_%d_%d_%d_%d" % (foldCounter,self.modelName,self.eps1,self.eps2,self.alpha1,self.alpha2)
			if not forceTrain and os.path.exists(os.path.join( self.ets.rootResultFolder , name4model+self.modelExt )):
				logger.info("Model %s already exists, skip training" % name4model)
				continue
				
			trainX, testX = x[train_index], x[test_index]
			trainY, testY = y[train_index], y[test_index]
			# validation set
			validPerc = 0.1
			trainX, validX, trainY, validY = train_test_split( trainX, trainY, test_size=validPerc, random_state=42)
			self.__trainlModel(trainX, trainY, validX, validY,name4model)
			self.__evaluateModel(testX, testY,name4model,plotMode,yscaler,False)
			foldCounter += 1

	def crossValidate(self,batteries,showImages=True,plotMode="server",scaleDataset=True):
		xscaler,yscaler = None, None
		self.dropDatasetLabel(batteries)
		if(scaleDataset):
			xscaler,yscaler = self.getXYscaler(batteries)
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		foldCounter = 0
		kf = KFold(n_splits=5, random_state=42, shuffle=True)
		for train_index, test_index in kf.split(x):
			testX = x[test_index]
			testY = y[test_index]
			name4model = "Fold_%d_%s_%d_%d_%d_%d" % (foldCounter,self.modelName,self.eps1,self.eps2,self.alpha1,self.alpha2)
			self.__evaluateModel(testX, testY,name4model,plotMode,yscaler,showImages)
			foldCounter += 1
	
	
	def plotEncoded(self,batteries,model2load,scaleDataset=True,plotMode="server",encodedSize=2):
		from mpl_toolkits.mplot3d import Axes3D
		logger.info("Plot encoded")
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt))
		encoder = Model(inputs=model.input, outputs=model.get_layer("ENC").output)
	
		xscaler,yscaler = None, None
		self.dropDatasetLabel(batteries)
		if(scaleDataset):
			xscaler,yscaler = self.getXYscaler(batteries)
		x,_ = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		
		
		x = self.__batchCompatible(self.batchSize,x)
		encoded = encoder.predict(x,batch_size=self.batchSize)
		print(encoded.shape)
		if(False):
			fig = plt.figure()
			x = encoded[:,0]
			y = encoded[:,1]
			
			if(encodedSize > 2):
				z = encoded[:,2]
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter(x, y, z, c='r', marker='o')
			else:
				plt.scatter(x, y, c='r', marker='o')

			plt.show()
		
		bins = [-50,-30,-20,-10,-5,-2,-1,-0.5,0,0.5,1,2,5,10,20,30,50]
		for i in range(encoded.shape[0]):
			if(encoded[i][0] >= 3):
				resistance = x[i,:,-1] / x[i,:,-2]
				self.ets.plotResistanceDistro(resistance,bins,"Anomaly Resistance %d" % i,plotMode)
	
	def decodeAndShow(self,batteries,model2load,scaleDataset=True,plotMode="server"):
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt))
		xscaler,yscaler = None, None
		self.dropDatasetLabel(batteries)
		if(scaleDataset):
			xscaler,yscaler = self.getXYscaler(batteries)
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		
		x = self.__batchCompatible(self.batchSize,x)
		y = self.__batchCompatible(self.batchSize,y)
		decoded = model.predict(x,batch_size=self.batchSize)
		
		from sklearn.metrics import mean_absolute_error
		downScaledMAE = np.zeros(y.shape[0])
		for i in range(y.shape[0]):
			mae = mean_absolute_error(y[i,:,0], decoded[i,:,0])
			downScaledMAE[i] = mae
		
		anomaltTh = 0.05
		anomaliesIdx = np.where( downScaledMAE > anomaltTh ) 
		anomalies = downScaledMAE[ anomaliesIdx ]
		
		print("Anomalies %d" % anomalies.shape[0])
		
		
		toShowReal = y[ anomaliesIdx]
		toShowDecoded = decoded[anomaliesIdx]
		
		toShowReal = self.__skScaleBack(toShowReal,yscaler)
		toShowDecoded = self.__skScaleBack(toShowDecoded,yscaler)
		
		bins = np.arange(-5,5,.5)
		my_dpi = 96
		for i in range(toShowReal.shape[0]):
			plt.figure(figsize=(1600/my_dpi, 800/my_dpi), dpi=my_dpi)
			plt.subplot(3, 1, 1)
			plt.plot(toShowReal[i,:,0],color="orange",label="Real A")
			plt.plot(toShowDecoded[i,:,0],color="navy",label="Decoded A")
			plt.legend()
			
			plt.subplot(3, 1, 2)
			plt.plot(toShowReal[i,:,1],color="orange",label="Real V")
			plt.plot(toShowDecoded[i,:,1],color="navy",label="Decoded V")
			plt.legend()
			
			resistance = toShowReal[i,:,1] / toShowReal[i,:,0]
			resistance[resistance == -np.inf] = 0
			resistance[resistance == np.inf] = 0
			weights = np.ones_like(resistance)/float(len(resistance)) # array with all 1 / len(r)
			
			plt.subplot(3, 1, 3)
			plt.hist(resistance, bins=bins,weights=weights,color="navy",label="V / A")
			plt.xticks(bins)
			plt.xticks(rotation=90)
			plt.legend()
			
			self.ets.plotMode(plotMode,None)
	
	
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
		encodedSize = 8
		
		model = self.__functionalDeepDenseModel(inputFeatures,outputFeatures,timesteps,encodedSize)
		
		adam = optimizers.Adam()		
		model.compile(loss='mae', optimizer=adam,metrics=['logcosh'])
		early = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=25, verbose=1, mode='min')	
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
	
	
	
	def __evaluateModel(self,testX,testY,model2load,plotMode,scaler=None,showImages=True,num2show=10):
		
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt))
		
		testX = self.__batchCompatible(self.batchSize,testX)
		testY = self.__batchCompatible(self.batchSize,testY)
		
		logger.info("Testing model %s with test %s" % (model2load,testX.shape))
		
		tt = time.clock()
		mse, mae = model.evaluate( x=testX, y=testY, batch_size=self.batchSize, verbose=0)
		logger.info("Test MAE %f - LCH %f Elapsed %f" % (mse,mae,(time.clock() - tt)))
		
		
		logger.info("Autoencoding")
		tt = time.clock()
		ydecoded = model.predict(testX,  batch_size=self.batchSize)
		logger.info("Elapsed %f" % (time.clock() - tt))
		if(scaler is not None):
			ydecoded = self.__skScaleBack(ydecoded,scaler)
			testY = self.__skScaleBack(testY,scaler)
			scaledMAE = self.__skMAE(testY,ydecoded)
			logger.info("Test scaled MAE %f" % scaledMAE)
			
		if(showImages):
			for r in range(num2show):
				plt.figure()
				toPlot = np.random.randint(ydecoded.shape[0])
				i = 1
				sid = 14
				for col in range(ydecoded.shape[2]):
					plt.subplot(ydecoded.shape[2], 1, i)
					plt.plot(ydecoded[toPlot][:, col],color="navy",label="Decoded")
					plt.plot(testY[toPlot][:, col],color="orange",label="Real")
					plt.legend()
					i += 1	
				title = str(toPlot) +"_"+str(uuid.uuid4())
				self.ets.plotMode(plotMode,title)



	def __functionalDeepDenseModel(self,inputFeatures,outputFeatures,timesteps,encodedSize):
			
		inputs = Input(shape=(timesteps,inputFeatures))
		
		#OK CONV1D
		#d1 = Dense(1024,activation='relu',name="D1")(inputs)
		#d2 = Dense(512,activation='relu',name="D2")(d1)
		#d3 = Dense(256,activation='relu',name="D3")(inputs) #(d2)
		d4 = Dense(128,activation='relu',name="D4")(inputs)  #(d3)
		d5 = Dense(64,activation='relu',name="D5")(d4)
		d6 = Dense(32,activation='relu',name="D6")(d5)
		
		f1 = Flatten(name="F1")(d6) 
		enc = Dense(encodedSize,activation='relu',name="ENC")(f1)
		
		d7 = Dense(32*timesteps,activation='relu',name="D7")(enc)
		r1 = Reshape((timesteps, 32),name="R1")(d7)
		d8 = Dense(64,activation='relu',name="D8")(r1)
		d9 = Dense(128,activation='relu',name="D9")(d8)
		#d10 = Dense(256,activation='relu',name="D10")(d9)
		#d11 = Dense(512,activation='relu',name="D11")(d10)
		out = Dense(outputFeatures,activation='linear',name="OUT")(d9) #(d11)
		
		
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
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		print(autoencoderModel.summary())
		return autoencoderModel
	
	def __functionalModelConv2D(self,inputFeatures,outputFeatures,timesteps,encodedSize):
		
		inputs = Input(shape=(timesteps,inputFeatures))
		width  = 15
		heigth = 15
		deepth = timesteps * inputFeatures
		mask = (5,5)
		mask2 = (3,3)
		poolMask = (3,3)
		initParams = 16
		outParams = 512
		
		enlarge1  = Dense(width*heigth*inputFeatures,activation='relu')(inputs)
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
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		print(autoencoderModel.summary())
		return autoencoderModel
	
	
	def getXYscaler(self,batteries):
		"""
		Creates the xscaler and y scaler for the dataset
		batteries: 3 layer list of dataframe [battery][month][episode] = dataframe
		"""
		tt = time.clock()
		logger.debug("getXYscaler - start")
		x,y = self.__datasetAs2DArray(batteries)
		xscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
		xscaler.fit(x)
		yscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
		yscaler.fit(y)
		logger.debug("getXYscaler - end - %f" % (time.clock() - tt) )
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
					ydf = episode[self.ets.keepY]
					for t in range(0,episode.shape[0]):
						x.append(episode.values[t])
						y.append(ydf.values[t])
		outX = np.asarray(x)
		outY = np.asarray(y)
		tt = time.clock()
		logger.debug("__datasetAs2DArray - end - %f" % (time.clock() - tt) )
		return outX,outY
	
	def dropDatasetLabel(self,batteries):
		"""
		Drop labels column and timestamp from the dataset
		"""
		tt = time.clock()
		logger.debug("dropDatasetLabel - start")
		for battery in batteries:
			for month in battery:
				for episode in month:
					episode.drop(columns=self.ets.dropX,inplace=True)
		logger.debug("dropDatasetLabel - end - %f" % (time.clock() - tt) )
	
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

	def __skMAE(self,real,decoded):
		samples = real.shape[0]
		timesteps = real.shape[1]
		features = real.shape[2]
		
		skReal = real.reshape(samples*timesteps,features)
		skDecoded = decoded.reshape(samples*timesteps,features)
		return mean_absolute_error(skReal,skDecoded)

main()		