#Standard
import uuid,time,os,logging, numpy as np, matplotlib.pyplot as plt
from logging import handlers as loghds
#Project module
from Demetra import EpisodedTimeSeries
#Kers
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Flatten, Reshape, LSTM
from keras.layers import Conv1D
from keras.layers import Conv2DTranspose, Conv2D, Dropout
from keras.models import load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
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


def rae_huber_loss(y_true, y_pred, clip_delta=1.0, alpha=0.3):
	error = y_true - y_pred	
	cond  = tf.keras.backend.abs(error) < clip_delta
	squared_loss = 0.5 * tf.keras.backend.square(error)
	linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
	loss = tf.where(cond, squared_loss, linear_loss)
	
	### REL ERROR
	a = tf.matmul(y_true,y_true,transpose_b=True)
	b = tf.matmul(y_pred,y_pred,transpose_b=True)
	rel_error = a - b
	
	rel_cond  = tf.keras.backend.abs(rel_error) < clip_delta
	rel_squared_loss = 0.5 * tf.keras.backend.square(rel_error)
	rel_linear_loss  = clip_delta * (tf.keras.backend.abs(rel_error) - 0.5 * clip_delta)
	rel_loss = tf.where(rel_cond, rel_squared_loss, rel_linear_loss)
	### END
	
	return (1-alpha)*tf.reduce_mean(loss,axis=[1,2]) + alpha * tf.reduce_mean(rel_loss,axis=[1,2])
	
	#return tf.where(cond, squared_loss, linear_loss)
	
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
		
		model = self.__conv2DHyperasScore(inputFeatures,outputFeatures,timesteps)
		#__denseModel #__denseModelHyperas #__denseModelHC
		#__conv2DModelHyperasClassic #__conv2DHyperasScore
		#__conv1DModelHyperasClassic #__conve1DModelHyperasScore
		
		adam = optimizers.Adam(lr=0.0005)		
		model.compile(loss=huber_loss, optimizer=adam,metrics=['mae'])
		#print(model.summary())
		path4save = os.path.join( self.ets.rootResultFolder , name4model+self.modelExt )
		checkpoint = ModelCheckpoint(path4save, monitor='val_loss', verbose=0,
			save_best_only=True, mode='min')
		
		validY = y_valid
		trainY = y_train
		
		history  = model.fit(x_train, trainY,
			verbose = 0,
			batch_size=self.batchSize,
			epochs=self.epochs,
			validation_data=(x_valid,validY)
			,callbacks=[checkpoint]
		)
		historySaveFile = name4model+"_history"
		self.ets.saveZip(self.ets.rootResultFolder,historySaveFile,history.history)
		
		logger.info("Training completed. Elapsed %f second(s)" %  (time.clock() - tt))
		
		# loading the best model for evaluation
		customLoss = {'huber_loss': huber_loss}
		model = load_model(path4save,custom_objects=customLoss)
		
		trainMae, trainHuber = model.evaluate( x=x_train, y=y_train, batch_size=self.batchSize, verbose=0)
		logger.info("Train HL %f - MAE %f" % (trainMae,trainHuber))
		valMae, valHuber = model.evaluate( x=x_valid, y=y_valid, batch_size=self.batchSize, verbose=0)
		logger.info("Valid HL %f - MAE %f" % (valMae,valHuber))	
		logger.debug("trainlModelOnArray - end - %f" % (time.clock() - tt) )
	
	def evaluateModelOnArray(self,testX,testY,model2load,plotMode,scaler=None,showImages=True,num2show=5,phase="Test",showScatter = False):
		
		customLoss = {'huber_loss': huber_loss}
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt)
			,custom_objects=customLoss)
		logger.info("There are %s parameters in model" % model.count_params()) 
		
		testX = self.__batchCompatible(self.batchSize,testX)
		testY = self.__batchCompatible(self.batchSize,testY)
		
		logger.debug("Validating model %s with test %s" % (model2load,testX.shape))
		
		tt = time.clock()
		
		Huber, mae= model.evaluate( x=testX, y=testY, batch_size=self.batchSize, verbose=0)
		
		logger.info("%s HL %f - MAE %f Elapsed %f" % (phase,Huber,mae,(time.clock() - tt)))
		
		logger.debug("Reconstruction")
		tt = time.clock()
		ydecoded = model.predict(testX,  batch_size=self.batchSize)
		logger.debug("Elapsed %f" % (time.clock() - tt))	
		
		
		maes = np.zeros(ydecoded.shape[0], dtype='float32')
		for sampleCount in range(0,ydecoded.shape[0]):
			maes[sampleCount] = mean_absolute_error(testY[sampleCount],ydecoded[sampleCount])
			
		if(False):
			layer_name = 'ENC'
			intermediate_layer_model = Model(inputs=model.input,
				outputs=model.get_layer(layer_name).output)
			
			coded = intermediate_layer_model.predict(testX,  batch_size=self.batchSize)
			#n = 10
			
			
			for k in range(0,2):
				plt.figure(figsize=(20, 8))
				i = np.random.randint(coded.shape[0])
				ax = plt.subplot(1, 3, 1)
				plt.imshow(testX[i].reshape(5,8).T)
				plt.title("Original")
				#plt.gray()
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
				
				ax = plt.subplot(1, 3, 2)
				plt.imshow(coded[i].reshape(3,2).T)
				plt.title("Coded")
				#plt.gray()
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
				
				ax = plt.subplot(1, 3, 3)
				plt.imshow(ydecoded[i].reshape(5,8).T)
				#plt.gray()
				plt.title("Reconstruced")
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
				
				
				plt.suptitle('MAE %f' % maes[i], fontsize=16)
				plt.show()
			
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
	
	
	def __conve1DModelHyperasScore(self,inputFeatures,outputFeatures,timesteps):
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		dropPerc = 0.5
		codeSize = 12
		c = Conv1D(48, 3,activation='relu',name="E1")(inputs)
		if 'more' == 'more':
			if  'drop'  == 'drop':
				c = Dropout(dropPerc)(c)
			c = Conv1D(64, 5,activation='relu',name="E2")(c)
		if  'less' == 'more':
			if  'noDrop'  == 'drop':
				c = Dropout(dropPerc)(c)
			c = Conv1D(128, 3,activation='relu',name="E3")(c)
		
		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Dense(64,activation='relu',name="D1")(enc)
		if  'drop'  == 'drop':
			c = Dropout(dropPerc)(c)
		c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(c)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		autoencoderModel = Model(inputs=inputs, outputs=out)
		#print(autoencoderModel.summary())
		return autoencoderModel
	
	def __conv1DModelHyperasClassic(self,inputFeatures,outputFeatures,timesteps):
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		dropPerc = 0.5
		codeSize = 7
		c = Conv1D(256, 7,activation='relu',name="E1")(inputs)
		
		if 'more' == 'more':
			if 'drop'  == 'drop':
				c = Dropout(dropPerc)(c)
			c = Conv1D(64, 3,activation='relu',name="E2")(c)
		if 'less' == 'more':
			if 'drop'  == 'drop':
				c = Dropout(dropPerc)(c)
			c = Conv1D(256, 5,activation='relu',name="E3")(c)
		
		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Dense(128,activation='relu',name="D1")(enc)
		if  'noDrop'  == 'drop':
			c = Dropout(dropPerc)(c)
		c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(c)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		autoencoderModel = Model(inputs=inputs, outputs=out)
		#print(autoencoderModel.summary())
		return autoencoderModel
	
	
	def __conv2DModelHyperasClassic(self,inputFeatures,outputFeatures,timesteps):
		dropPerc = 0.5
		strideSize = 2
		codeSize = 9

		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = Reshape((5,4,2),name="R2E")(inputs)
		c = Conv2D(128,strideSize,activation='relu',name="E1")(c)
		
		if   'less' == 'more':
			if 'noDrop' == 'drop':
				c = Dropout(dropPerc)(c)
			c = Conv2D(16,strideSize,activation='relu',name="E2")(c)

		if 'drop' == 'drop':
			c = Dropout(dropPerc)(c)
		c = Conv2D(64,strideSize,activation='relu',name="E3")(c)
		
		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Reshape((1,1,codeSize),name="R2D")(enc)
		
		c = Conv2DTranspose(128,strideSize,activation='relu',name="D1")(c)
		if   'less' == 'more':
			if 'drop'  == 'drop':
				c = Dropout(dropPerc)(c)
			c = Conv2DTranspose(256,strideSize,activation='relu',name="D2")(c)
		
		preDecFlat = Flatten(name="F2")(c) 
		c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(preDecFlat)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		autoencoderModel = Model(inputs=inputs, outputs=out)
		return autoencoderModel

	def __conv2DHyperasScore(self,inputFeatures,outputFeatures,timesteps):

		dropPerc = 0.5
		strideSize = 2
		codeSize = 12

		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = Reshape((5,4,2),name="R2E")(inputs)
		c = Conv2D(48,strideSize,activation='relu',name="E1")(c)
		
		if   'less'== 'more':
			if  'noDrop' == 'drop':
				c = Dropout(dropPerc)(c)
			c = Conv2D(16,strideSize,activation='relu',name="E2")(c)

		if 'noDrop' == 'drop':
			c = Dropout(dropPerc)(c)
		c = Conv2D(16,strideSize,activation='relu',name="E3")(c)
		
		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Reshape((1,1,codeSize),name="R2D")(enc)
		
		c = Conv2DTranspose(64,strideSize,activation='relu',name="D1")(c)

		if  'more'  == 'more':
			if 'drop'  == 'drop':
				c = Dropout(dropPerc)(c)
			Conv2DTranspose(48,strideSize,activation='relu',name="D2")(c)
		
		preDecFlat = Flatten(name="F2")(c) 
		c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(preDecFlat)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
			

		autoencoderModel = Model(inputs=inputs, outputs=out)
		return autoencoderModel
	


	def __denseModelHC(self,inputFeatures,outputFeatures,timesteps):
	
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		# START HyperParameters
		dropPerc = 0.5
		codeSize = 9
		codeMultiplier =3
		
		# END HyperParameters
		d = Dense(48,activation='relu',name="E1")(inputs)
		
		if 'more' == 'more':
			if 'noDrop' == 'drop':
				d = Dropout(dropPerc)(d)
			d = Dense(128,activation='relu',name="E2")(d)
		if 'more' == 'more':
			if 'noDrop' == 'drop':
				d = Dropout(dropPerc)(d)
			d = Dense(128,activation='relu',name="E3")(d)
		
		if 'noDrop' == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense(64,activation='relu',name="E4")(d)
		
		### s - encoding
		d = Flatten(name="F1")(d) 
		enc = Dense(codeSize,activation='relu',name="ENC")(d)
		### e - encoding
		
		
		d = Dense(codeSize*codeMultiplier,activation='relu',name="D1")(enc)
		d = Reshape((codeSize, codeMultiplier),name="R")(d)
		
		if 'more' == 'more':
			if 'noDrop' == 'drop':
				d = Dropout(dropPerc)(d)
			d = Dense(64,activation='relu',name="D2")(d)
		
		if 'more' == 'more':
			if 'drop' == 'drop':
				d = Dropout(dropPerc)(d)
			d = Dense(16,activation='relu',name="D3")(d)
		
		if 'drop' == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense(16,activation='relu',name="D4")(d)
		
		d = Flatten(name="F2")(d)
		d = Dense(outputFeatures*timesteps,activation='linear',name="DEC")(d)
		out = Reshape((timesteps, outputFeatures),name="OUT")(d)
	
		autoencoderModel = Model(inputs=inputs, outputs=out)
		#print(autoencoderModel.summary())
		return autoencoderModel
	
	def __denseModelHyperas(self,inputFeatures,outputFeatures,timesteps):
	
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		# START HyperParameters
		dropPerc = 0.5
		codeSize = 9
		codeMultiplier = 3
		
		# END HyperParameters
		d = Dense(48,activation='relu',name="E1")(inputs)
		
		if 'more' == 'more':
			if 'noDrop' == 'drop':
				d = Dropout(dropPerc)(d)
			d = Dense(48,activation='relu',name="E2")(d)
		if 'less' == 'more':
			if 'drop' == 'drop':
				d = Dropout(dropPerc)(d)
			d = Dense(48,activation='relu',name="E3")(d)
		
		if 'drop' == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense(32,activation='relu',name="E4")(d)
		
		### s - encoding
		d = Flatten(name="F1")(d) 
		enc = Dense(codeSize,activation='relu',name="ENC")(d)
		### e - encoding
		
		
		d = Dense(codeSize*codeMultiplier,activation='relu',name="D1")(enc)
		d = Reshape((codeSize, codeMultiplier),name="R")(d)
		
		if  'less' == 'more':
			if 'drop' == 'drop':
				d = Dropout(dropPerc)(d)
			d = Dense(16,activation='relu',name="D2")(d)
		
		if  'less' == 'more':
			if 'noDrop' == 'drop':
				d = Dropout(dropPerc)(d)
			d = Dense(64,activation='relu',name="D3")(d)
		
		if  'noDrop' == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense(16,activation='relu',name="D4")(d)
		
		d = Flatten(name="F2")(d)
		d = Dense(outputFeatures*timesteps,activation='linear',name="DEC")(d)
		out = Reshape((timesteps, outputFeatures),name="OUT")(d)
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		#print(autoencoderModel.summary())
		return autoencoderModel
	
	
		
	def __denseModel(self,inputFeatures,outputFeatures,timesteps):
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		start = 128
		d = Dense(start,activation='relu',name="E1")(inputs)
		d = Dense(int(start / 2),activation='relu',name="E2")(d)
		d = Dense(int(start / 4),activation='relu',name="E3")(d)
		d = Dense(int(start / 8),activation='relu',name="E4")(d)
		d = Dense(int(start / 16),activation='relu',name="E5")(d)
		
		d = Flatten(name="F1")(d) 
		enc = Dense(16,activation='relu',name="ENC")(d)
		
		d = Dense(start*timesteps,activation='relu',name="D1")(enc)
		d = Reshape((timesteps, start),name="R1")(d)
		
		d = Dense(int(start / 16),activation='relu',name="D2")(d)
		d = Dense(int(start / 8),activation='relu',name="D")(d)
		d = Dense(int(start / 4),activation='relu',name="D4")(d)
		d = Dense(int(start / 2),activation='relu',name="D5")(d)
		
		
		d = Flatten(name="F2")(d)
		d = Dense(outputFeatures*timesteps,activation='linear',name="D6")(d)
		out = Reshape((timesteps, outputFeatures),name="OUT")(d)
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		return autoencoderModel

	def	batchCompatible(self,batch_size,data):
		return self.__batchCompatible(batch_size,data)
		
	
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
	