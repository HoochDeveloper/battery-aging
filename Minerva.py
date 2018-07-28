#Standard
import uuid,time,os,logging, numpy as np, matplotlib.pyplot as plt
from logging import handlers as loghds
#Project module
from Demetra import EpisodedTimeSeries
#Kers
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D,AveragePooling1D
from keras.models import load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger
#Sklearn
from sklearn.metrics import mean_absolute_error

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

	
class Minerva():
	
	logFolder = "./logs"
	modelName = "FullyConnected_4_"
	modelExt = ".h5"
	batchSize = 200
	epochs = 500
	ets = None
	eps1   = 5
	eps2   = 5
	alpha1 = 5
	alpha2 = 5
	
	
	def __init__(self,eps1,eps2,alpha1,alpha2,plotMode = "server"):
		
		# plotMode "GUI" #"server" # set mode to server in order to save plot to disk instead of showing on video
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
		
		if(plotMode == "server" ):
			plt.switch_backend('agg')
			if not os.path.exists(self.ets.episodeImageFolder):
				os.makedirs(self.ets.episodeImageFolder)

	def trainlModelOnArray(self,x_train, y_train, x_valid, y_valid,name4model,encodedSize = 8):
		
		tt = time.clock()
		logger.debug("trainlModelOnArray - start")
		
		x_train = self.__batchCompatible(self.batchSize,x_train)
		y_train = self.__batchCompatible(self.batchSize,y_train)
		x_valid = self.__batchCompatible(self.batchSize,x_valid)
		y_valid = self.__batchCompatible(self.batchSize,y_valid)
		
		logger.info("Training model %s with train %s and valid %s" % (name4model,x_train.shape,x_valid.shape))
		
		inputFeatures  = x_train.shape[2]
		outputFeatures = y_train.shape[2]
		timesteps =  x_train.shape[1]
		
		
		#model = self.__functionalDeepDenseModel(inputFeatures,outputFeatures,timesteps,encodedSize)
		
		model = self.__functionInceptionModel(inputFeatures,outputFeatures,timesteps,encodedSize)
		
		adam = optimizers.Adam()		
		model.compile(loss='mae', optimizer=adam,metrics=['logcosh'])
		early = EarlyStopping(monitor='val_OUT_loss', min_delta=0.0001, patience=50, verbose=1, mode='min')	
		cvsLogFile = os.path.join(self.logFolder,name4model+'.log')
		csv_logger = CSVLogger(cvsLogFile)
		model.fit(x_train, [y_train,y_train],
			verbose = 0,
			batch_size=self.batchSize,
			epochs=self.epochs,
			validation_data=(x_valid,[y_valid,y_valid]),
			callbacks=[early,csv_logger]
		)
		
		logger.debug("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		logger.debug("Saving model...")
		model.save(os.path.join( self.ets.rootResultFolder , name4model+self.modelExt )) 
		logger.debug("Model saved")
		
		_ , trainMae, ptrainMae, trainLch, ptrainLch = model.evaluate( x=x_train, y=[y_train,y_train], batch_size=self.batchSize, verbose=0)
		logger.info("Train MAE %f - LCH %f" % (trainMae,trainLch))
		logger.info("Train Probe MAE %f - LCH %f" % (ptrainMae,ptrainLch))
		_ , valMae, pvalMae, valLch, pvalLch = model.evaluate( x=x_valid, y=[y_valid,y_valid], batch_size=self.batchSize, verbose=0)
		logger.info("Valid MAE %f - LCH %f" % (valMae,valLch))
		logger.info("Valid Probe MAE %f - LCH %f" % (pvalMae,pvalLch))
		logger.debug("trainlModelOnArray - end - %f" % (time.clock() - tt) )
	
	
	
	def evaluateModelOnArray(self,testX,testY,model2load,plotMode,scaler=None,showImages=True,num2show=5,phase="Test"):
		
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt))
		
		testX = self.__batchCompatible(self.batchSize,testX)
		testY = self.__batchCompatible(self.batchSize,testY)
		
		logger.debug("Validating model %s with test %s" % (model2load,testX.shape))
		
		tt = time.clock()
		_ , mae, pMae, lch, pLch = model.evaluate( x=testX, y=[testY,testY], batch_size=self.batchSize, verbose=0)
		logger.info("%s MAE %f - LCH %f Elapsed %f" % (phase,mae,lch,(time.clock() - tt)))
		logger.debug("%s Probe MAE %f - LCH %f Elapsed %f" % (phase,pMae,pLch,(time.clock() - tt)))
		
		logger.debug("Autoencoding")
		tt = time.clock()
		ydecoded,pdecoded = model.predict(testX,  batch_size=self.batchSize)
		logger.debug("Elapsed %f" % (time.clock() - tt))
		if(scaler is not None):
			ydecoded = self.__skScaleBack(ydecoded,scaler)
			testY = self.__skScaleBack(testY,scaler)
			scaledMAE = self.__skMAE(testY,ydecoded)
			logger.debug("%s scaled MAE %f" % (phase,scaledMAE))
			
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


	def __functionInceptionModel(self,inputFeatures,outputFeatures,timesteps,encodedSize):
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		
		first = Dense(64,activation='relu',name="D1")(inputs)
		
		c1 = self.__getDeepCell(first,"C1")	
		c2 = self.__getDeepCell(c1,"C2")
		
		f1 = Flatten(name="F1")(c2) 
		enc = Dense(encodedSize,activation='relu',name="ENC")(f1)
		
		d2 = Dense(64*timesteps,activation='relu',name="D2")(enc)
		r1 = Reshape((timesteps, 64),name="R1")(d2)
		
		### PROBE ###
		dprobe = Dense(32,activation='relu',name="DPROBE1")(r1)
		dprobe = Dense(16,activation='relu',name="DPROBE2")(dprobe)
		frpobe = Flatten(name="frpobe")(dprobe)
		probe = Dense(outputFeatures*timesteps,activation='linear',name="P1")(frpobe)
		outProbe = Reshape((timesteps, outputFeatures),name="OUT_P1")(probe)
		### END PROBE ###
		
		c3 = self.__getDeepCell(r1,"C3")	
		c4 = self.__getDeepCell(c3,"C4")

		f2 = Flatten(name="F2")(c4)
		d3 = Dense(outputFeatures*timesteps,activation='linear',name="D3")(f2)
		out = Reshape((timesteps, outputFeatures),name="OUT")(d3)
	
		autoencoderModel = Model(inputs=inputs, outputs=[out,outProbe])
		
		#print(autoencoderModel.summary())
		return autoencoderModel
	
	
	def __getDeepCell(self,input,prefix):
		c = input
		c = Dense(128,activation='relu',name="%s_D1" % prefix)(input)
		c = Conv1D(64,2,activation='relu',name="%s_CV1" % prefix)(c)
		c = MaxPooling1D(pool_size=2,name="%s_MP1" % prefix)(c)
		#c = Dense(64,activation='relu',name="%s_D2" % prefix)(c)
		c = Dense(32,activation='relu',name="%s_D3" % prefix)(c)
		c = Dense(16,activation='relu',name="%s_D4" % prefix)(c)
		c1 = Conv1D(16,4,activation='relu',name="%s_INCV1" % prefix)(c)
		c2 = MaxPooling1D(pool_size=4,name="%s_INMP1" % prefix)(c)
		c3 = Conv1D(16,4,activation='relu',name="%s_INCV2" % prefix)(c)
		c4 = AveragePooling1D(pool_size=4,name="%s_INAV1" % prefix)(c)
		c = concatenate([c1,c2,c3,c4], axis=1)
		return c
	
	def __functionalDeepDenseModel(self,inputFeatures,outputFeatures,timesteps,encodedSize):
			
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")

		#OK No CONV
		d3 = Dense(256,activation='relu',name="D3")(inputs)
		
		#d4 = Dense(128,activation='relu',name="D4")(d3)
		conv1 = Conv1D(128,2,activation='relu',name="CV1")(d3)
		maxpool1 = MaxPooling1D(pool_size=2,name="MP1")(conv1)
		
		d5 = Dense(64,activation='relu',name="D5")(maxpool1)
		d6 = Dense(32,activation='relu',name="D6")(d5)
		
		conv2 = Conv1D(16,2,activation='relu',name="CV2")(d6)
		maxpool2 = MaxPooling1D(pool_size=2,name="MP2")(conv2)
		
		f1 = Flatten(name="F1")(maxpool2) 
		enc = Dense(encodedSize,activation='relu',name="ENC")(f1)
		
		d7 = Dense(256*timesteps,activation='relu',name="D7")(enc)
		r1 = Reshape((timesteps, 256),name="R1")(d7)
		
		
		#d8 = Dense(128,activation='relu',name="D8")(r1)
		conv3 = Conv1D(128,2,activation='relu',name="CV3")(r1)
		maxpool3 = MaxPooling1D(pool_size=2,name="MP3")(conv3)
		
		d9 = Dense(64,activation='relu',name="D9")(maxpool3)
		d10 = Dense(32,activation='relu',name="D10")(d9)
		
		conv4 = Conv1D(16,2,activation='relu',name="CV4")(d10)
		maxpool4 = MaxPooling1D(pool_size=2,name="MP4")(conv4)
		
		f2 = Flatten(name="F2")(maxpool4)
		d11 = Dense(outputFeatures*timesteps,activation='linear',name="D11")(f2)
		out = Reshape((timesteps, outputFeatures),name="OUT")(d11)
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		#print(autoencoderModel.summary())
		return autoencoderModel
	
	def __batchCompatible(self,batch_size,data):
		"""
		Transform data shape 0 in a multiple of batch_size
		"""
		exceed = data.shape[0] % batch_size
		if(exceed > 0):
			data = data[:-exceed]
		return data
	
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
	