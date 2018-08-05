#Standard
import uuid,time,os,logging, numpy as np, matplotlib.pyplot as plt
from logging import handlers as loghds
#Project module
from Demetra import EpisodedTimeSeries
#Kers
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Flatten, Reshape, LSTM
from keras.layers import Conv1D, MaxPooling1D,AveragePooling1D
from keras.models import load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger
import tensorflow as tf
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


'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''
def huber_loss(y_true, y_pred, clip_delta=1.0):
	error = y_true - y_pred
	cond  = tf.keras.backend.abs(error) < clip_delta
	squared_loss = 0.5 * tf.keras.backend.square(error)
	linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
	return tf.where(cond, squared_loss, linear_loss)
	
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
	return tf.reduce_mean(huber_loss(y_true, y_pred, clip_delta))
	
class Minerva():
	
	logFolder = "./logs"
	modelName = "FullyConnected_4_"
	modelExt = ".h5"
	batchSize = 100
	epochs = 500
	ets = None
	eps1   = 5
	eps2   = 5
	alpha1 = 5
	alpha2 = 5
	modelHasProbe = False
	
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
		
		loss2monitor = 'val_loss'
		if(self.modelHasProbe):
			model = self.__functionInceptionModel(inputFeatures,outputFeatures,timesteps,encodedSize)
			loss2monitor = 'val_OUT_loss'
		else:
			model = self.__convModel(inputFeatures,outputFeatures,timesteps,encodedSize)
			#print(model.summary())
			#__functionalDenseModel #__convModel
		
		adam = optimizers.Adam()		
		model.compile(loss=huber_loss, optimizer=adam,metrics=['mae'])
		early = EarlyStopping(monitor=loss2monitor, min_delta=0.000001, patience=50, verbose=1, mode='min')	
		cvsLogFile = os.path.join(self.logFolder,name4model+'.log')
		csv_logger = CSVLogger(cvsLogFile)
		
		validY = y_valid
		trainY = y_train
		if(self.modelHasProbe):
			validY = [y_valid,y_valid]
			trainY = [y_train,y_train]
		
		model.fit(x_train, trainY,
			verbose = 0,
			batch_size=self.batchSize,
			epochs=self.epochs,
			validation_data=(x_valid,validY),
			callbacks=[early,csv_logger]
		)
		
		logger.debug("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		logger.debug("Saving model...")
		model.save(os.path.join( self.ets.rootResultFolder , name4model+self.modelExt )) 
		logger.debug("Model saved")
		if(self.modelHasProbe):
			_ , trainMae, ptrainMae, trainHuber, ptrainHuber = model.evaluate( x=x_train, y=[y_train,y_train], batch_size=self.batchSize, verbose=0)
			logger.info("Train Probe HL %f - MAE %f" % (ptrainMae,ptrainHuber))
		else:
			trainMae, trainHuber = model.evaluate( x=x_train, y=y_train, batch_size=self.batchSize, verbose=0)
			
		
		logger.info("Train HL %f - MAE %f" % (trainMae,trainHuber))
		
		if(self.modelHasProbe):
			_ , valMae, pvalMae, valHuber, pvalHuber = model.evaluate( x=x_valid, y=[y_valid,y_valid], batch_size=self.batchSize, verbose=0)
			logger.info("Valid Probe HL %f - MAE %f" % (pvalMae,pvalHuber))
		else:
			valMae, valHuber = model.evaluate( x=x_valid, y=y_valid, batch_size=self.batchSize, verbose=0)
		
		logger.info("Valid HL %f - MAE %f" % (valMae,valHuber))	
		logger.debug("trainlModelOnArray - end - %f" % (time.clock() - tt) )
	
	
	
	def evaluateModelOnArray(self,testX,testY,model2load,plotMode,scaler=None,showImages=True,num2show=5,phase="Test",showScatter = False):
		
		customLoss = {'huber_loss': huber_loss}
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt)
			,custom_objects=customLoss)
		
		#logger.info("There are %s parameters in model" % model.count_params()) 
		
		testX = self.__batchCompatible(self.batchSize,testX)
		testY = self.__batchCompatible(self.batchSize,testY)
		
		logger.debug("Validating model %s with test %s" % (model2load,testX.shape))
		
		tt = time.clock()
		if(self.modelHasProbe):
			_ , mae, pMae, Huber, pHuber = model.evaluate( x=testX, y=[testY,testY], batch_size=self.batchSize, verbose=0)
			logger.debug("%s Probe HL %f - MAE %f Elapsed %f" % (phase,pMae,pHuber,(time.clock() - tt)))
		else:
			Huber, mae= model.evaluate( x=testX, y=testY, batch_size=self.batchSize, verbose=0)
		
		logger.info("%s HL %f - MAE %f Elapsed %f" % (phase,Huber,mae,(time.clock() - tt)))
		
		logger.debug("Reconstruction")
		tt = time.clock()
		if(self.modelHasProbe):
			ydecoded,pdecoded = model.predict(testX,  batch_size=self.batchSize)
		else:
			ydecoded = model.predict(testX,  batch_size=self.batchSize)
		logger.debug("Elapsed %f" % (time.clock() - tt))	
		
		
		maes = np.zeros(ydecoded.shape[0], dtype='float32')
		for sampleCount in range(0,ydecoded.shape[0]):
			maes[sampleCount] = mean_absolute_error(testY[sampleCount],ydecoded[sampleCount])
			
			
			
			
		if(showScatter):
			plt.figure(figsize=(8, 6), dpi=100)
			plt.scatter(range(0,ydecoded.shape[0]), maes)
			plt.hlines(mae,0,ydecoded.shape[0], color='r')
			self.ets.plotMode(plotMode,"Scatter")
			
		if(showImages):
				
			
			unscaledDecoded = ydecoded
			unscaledTest = testY
			
			if(scaler is not None):
				ydecoded = self.__skScaleBack(ydecoded,scaler)
				testY = self.__skScaleBack(testY,scaler)
				scaledMAE = self.__skMAE(testY,ydecoded)
				logger.debug("%s scaled MAE %f" % (phase,scaledMAE))
			for r in range(num2show):
				plt.figure(figsize=(8, 6), dpi=100)
				toPlot = np.random.randint(ydecoded.shape[0])
				
				episodeMAE = mean_absolute_error(unscaledTest[toPlot],unscaledDecoded[toPlot])
				
				i = 1
				sid = 14
				for col in range(ydecoded.shape[2]):
					plt.subplot(ydecoded.shape[2], 1, i)
					plt.plot(ydecoded[toPlot][:, col],color="navy",label="Reconstructed")
					plt.plot(testY[toPlot][:, col],color="orange",label="Target")
					if(i == 1):
						plt.title("Current (A) vs Time (s)",loc="right")
					else:
						plt.title("Voltage (V) vs Time (s)",loc="right")
					plt.suptitle("Episode MAE: %f" % episodeMAE, fontsize=16)
					plt.grid()
					plt.legend()
					i += 1	
				title = str(toPlot) +"_"+str(uuid.uuid4())
				self.ets.plotMode(plotMode,title)
		
		return maes
				
	def __functionInceptionModel(self,inputFeatures,outputFeatures,timesteps,encodedSize):
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		
		first = Dense(64,activation='relu',name="D1")(inputs)
		
		c1 = self.__getInceptionCell(first,"C1")	
		c2 = self.__getInceptionCell(c1,"C2")
		
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
		
		c3 = self.__getInceptionCell(r1,"C3")	
		c4 = self.__getInceptionCell(c3,"C4")

		f2 = Flatten(name="F2")(c4)
		d3 = Dense(outputFeatures*timesteps,activation='linear',name="D3")(f2)
		out = Reshape((timesteps, outputFeatures),name="OUT")(d3)
	
		autoencoderModel = Model(inputs=inputs, outputs=[out,outProbe])
		return autoencoderModel
	
	
	def __getConvCell(self,input,prefix):
		
		convDim = 32
		convTime = 40
		
		c = Flatten(name="%s_F" % prefix)(input)
		c = Dense(convDim*convTime,activation='relu',name="%s_S_D32" % prefix)(c)
		c = Reshape((convTime, convDim),name="%s_R" % prefix)(c)
		c = Conv1D(convDim,9,activation='relu',name="%s_S_C5" % prefix)(c)
		c = MaxPooling1D(pool_size=7,name="%s_I_MP3" % prefix)(c)
		
		return c
	
	
	def __getInceptionCell(self,input,prefix):
		
		c = Dense(256,activation='relu',name="%s_S_D64" % prefix)(input)
		c = Conv1D(128,7,activation='relu',name="%s_S_C7" % prefix)(c)
		c = Dense(64,activation='relu',name="%s_S_D32" % prefix)(c)
		c = Conv1D(32,5,activation='relu',name="%s_S_C5" % prefix)(c)
		
		c1 = Conv1D(32,5,activation='relu',name="%s_I_CV5" % prefix)(c)
		c2 = MaxPooling1D(pool_size=3,name="%s_I_MP3" % prefix)(c)
		c3 = Conv1D(32,3,activation='relu',name="%s_I_CV3" % prefix)(c)
		c4 = MaxPooling1D(pool_size=5,name="%s_I_MP5" % prefix)(c)
		
		c = concatenate([c1,c2,c3,c4], axis=1,name = "%s_INCEPTION" % prefix)
		
		return c
	
	def __convModel(self,inputFeatures,outputFeatures,timesteps,encodedSize):
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = self.__getConvCell(inputs,"EC1")
		c = self.__getConvCell(inputs,"EC2")		
		
		preEncodeFlat = Flatten(name="PRE_ENCODE")(c) 
		enc = Dense(encodedSize,activation='relu',name="ENC")(preEncodeFlat)
	
		c = Dense(encodedSize*timesteps,activation='relu',name="PRE_DC")(enc)
		c = Reshape((timesteps, encodedSize),name="PRE_DC_R")(c)
		
		c = self.__getConvCell(c,"DC1")	
		c = self.__getConvCell(c,"DC2")	
		
		preDecodeFlat = Flatten(name="PRE_DECODE")(c)
		c = Dense(outputFeatures*timesteps,activation='linear',name="DECODED")(preDecodeFlat)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		#print(autoencoderModel.summary())
		return autoencoderModel
	
	def __inceptionModel(self,inputFeatures,outputFeatures,timesteps,encodedSize):
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		
		c = self.__getInceptionCell(inputs,"EC1")	
		c = self.__getInceptionCell(c,"EC2")
		#c = self.__getInceptionCell(c,"EC3")
	
		preEncodeFlat = Flatten(name="PRE_ENCODE")(c) 
		enc = Dense(encodedSize,activation='relu',name="ENC")(preEncodeFlat)
		
		c = Dense(encodedSize*timesteps,activation='relu',name="PRE_DC")(enc)
		c = Reshape((timesteps, encodedSize),name="PRE_DC_R")(c)
		
		c = self.__getInceptionCell(c,"DC1")	
		c = self.__getInceptionCell(c,"DC2")
		#c = self.__getInceptionCell(c,"DC3")
		
		preDecodeFlat = Flatten(name="PRE_DECODE")(c)
		c = Dense(outputFeatures*timesteps,activation='linear',name="DECODED")(preDecodeFlat)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		#print(autoencoderModel.summary())
		return autoencoderModel
		
	def __functionalDenseModel(self,inputFeatures,outputFeatures,timesteps,encodedSize):
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		
		start = 256
		
		d = Dense(start,activation='relu',name="D1")(inputs)
		d = Dense(int(start / 2),activation='relu',name="D2")(d)
		d = Dense(int(start / 4),activation='relu',name="D3")(d)
		d = Dense(int(start / 8),activation='relu',name="D4")(d)
		d = Dense(int(start / 16),activation='relu',name="D5")(d)
		
		d = Flatten(name="F1")(d) 
		enc = Dense(encodedSize,activation='relu',name="ENC")(d)
		
		d = Dense(start*timesteps,activation='relu',name="D6")(enc)
		d = Reshape((timesteps, start),name="R1")(d)
		
		d = Dense(int(start / 2),activation='relu',name="D7")(d)
		d = Dense(int(start / 4),activation='relu',name="D8")(d)
		d = Dense(int(start / 8),activation='relu',name="D9")(d)
		d = Dense(int(start / 16),activation='relu',name="D10")(d)
		
		
		d = Flatten(name="F2")(d)
		d = Dense(outputFeatures*timesteps,activation='linear',name="D11")(d)
		out = Reshape((timesteps, outputFeatures),name="OUT")(d)
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
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
	