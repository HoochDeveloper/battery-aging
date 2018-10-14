#Standard
import uuid,time,os,logging, numpy as np, matplotlib.pyplot as plt
from logging import handlers as loghds
#Project module
from Demetra import EpisodedTimeSeries
#Kers
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Flatten, Reshape, LSTM, Lambda
from keras.layers import Conv1D
from keras.layers import Conv2DTranspose, Conv2D, Dropout
from keras.models import load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import tensorflow as tf
#Sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.constraints import max_norm

from keras.losses import mse, binary_crossentropy

import keras.backend as K

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


codeDimension = 13

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

def vae_loss(z_mean,z_log_var):

	def loss(y_true, y_pred):

		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss *= -0.5
		kl_loss = K.mean(kl_loss)

		reconstruction_loss = K.mean(huber_loss(K.flatten(y_true), K.flatten(y_pred)))
		kl_loss *= 0.1
		vaeLoss = K.mean(reconstruction_loss + kl_loss)
		return vaeLoss
	
	return loss
	
def sample_z(args):
	mu, log_sigma = args
	eps = K.random_normal(shape=(codeDimension,),mean=0.,stddev=1.)
	return mu + K.exp(log_sigma / 2) * eps

class Minerva():
		
	logFolder = "./logs"
	modelName = "FullyConnected_4_"
	modelExt = ".h5"
	batchSize = 64
	epochs = 600
	ets = None
	eps1   = 5
	eps2   = 5
	alpha1 = 5
	alpha2 = 5
	
	def getModel(self,inputFeatures,outputFeatures,timesteps):
		#return self.VAE(inputFeatures,outputFeatures,timesteps)
		return self.Conv2DQR(inputFeatures,outputFeatures,timesteps)
		#return self.VAE2D(inputFeatures,outputFeatures,timesteps)
		#return self.VAE2D_OPT(inputFeatures,outputFeatures,timesteps)
		
		
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

				
	def loadModel(self,path4save,inputFeatures,outputFeatures,timesteps):
		vae, encoder, decoder = self.getModel(inputFeatures,outputFeatures,timesteps)
		vae.load_weights(path4save,by_name=True)
		if(encoder is not None):
			encoder.load_weights(path4save,by_name=True)
		if(decoder is not None):
			decoder.load_weights(path4save,by_name=True)
		return vae, encoder, decoder
	
	def VAE2D_OPT(self,inputFeatures,outputFeatures,timesteps,summary = False):
		
		activation = 'linear'
		
		codeSize = codeDimension
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		er = Reshape((4, 5, 2),name="ER")
		eh1 = Conv2D(512,2, activation=activation,name = "EH1")
		eh2 = Conv2D(128,2, activation=activation,name = "EH2")
		eh3 = Conv2D(128,2, activation=activation,name = "EH3")
		fe = Flatten(name="FE")
		mean = Dense(codeSize, activation='linear',name = "MU")
		logsg =  Dense(codeSize, activation='linear', name = "LOG_SIGMA")

		mu = mean(fe(eh3(eh2(eh1(er(inputs))))))
		log_sigma = logsg(fe(eh3(eh2(eh1(er(inputs))))))
		encoder = Model(inputs,[mu, log_sigma])
		if(summary):
			print("Encoder")
			encoder.summary()
		
		# Sample z ~ Q(z|X)
		z = Lambda(sample_z,name="CODE")([mu, log_sigma])
		# P(X|z) -- decoder
		#latent_inputs = Input(shape=(1,2,codeSize,), name='z_sampling')
		latent_inputs = Input(shape=(codeSize,), name='z_sampling')
		dr = Reshape((1, 1, codeSize),name="DR")
		dh1 = Conv2DTranspose(256,2, activation=activation,name = "DH1")
		dh2 = Conv2DTranspose(96,2,  activation=activation,name = "DH2")
		dh3 = Conv2DTranspose(512,2, activation=activation,name = "DH3")
		fd = Flatten(name = "FD")
		decoded = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")
		
		decoderOut = Reshape((timesteps, outputFeatures),name="OUT")
		
		decOut = decoderOut( decoded(fd(dh3( dh2 ( dh1(dr(latent_inputs))) ))))
		decoder = Model(latent_inputs,decOut)
		if(summary):
			print("Decoder")
			decoder.summary()

		trainDecOut = decoderOut( decoded(fd(dh3(dh2(dh1(dr(z)))))))
		vae = Model(inputs, trainDecOut)
		if(summary):
			print("VAE")
			vae.summary()
		
		opt = optimizers.Adam(lr=0.0002) 

		vae.compile(loss = huber_loss,optimizer=opt,metrics=['mae'])
		return vae, encoder, decoder
	
	
	
	
	def VAEFC(self,inputFeatures,outputFeatures,timesteps,summary = False):
		
		codeSize = codeDimension
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		
		eh1 = Dense(512, activation='relu',name = "EH1")
		eh2 = Dense(128, activation='relu',name = "EH2")
		
		fe = Flatten(name="FE")
		mean = Dense(codeSize, activation='linear',name = "MU")
		logsg =  Dense(codeSize, activation='linear', name = "LOG_SIGMA")

		mu = mean(fe(eh3(eh2(eh1((inputs))))))
		log_sigma = logsg(fe(eh3(eh2(eh1((inputs))))))
		encoder = Model(inputs,[mu, log_sigma])
		if(summary):
			print("Encoder")
			encoder.summary()
		
		# Sample z ~ Q(z|X)
		z = Lambda(sample_z,name="CODE")([mu, log_sigma])
		
		latent_inputs = Input(shape=(codeSize,), name='z_sampling')
		
		dh1 = Dense(256, activation='relu',name = "DH1")
		dh2 = Dense(48, activation='relu',name = "DH2")
		dh3 = Dense(512, activation='relu',name = "DH2")
		decoded = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")
		
		decoderOut = Reshape((timesteps, outputFeatures),name="OUT")
		
		decOut = decoderOut( decoded( dh3(dh2 ( dh1((latent_inputs))) )))
		decoder = Model(latent_inputs,decOut)
		if(summary):
			print("Decoder")
			decoder.summary()

		trainDecOut = decoderOut( decoded(dh3(dh2(dh1((z))))) )
		vae = Model(inputs, trainDecOut)
		if(summary):
			print("VAE")
			vae.summary()
		
		opt = optimizers.Adam(lr=0.0001) 

		vae.compile(loss = huber_loss,optimizer=opt,metrics=['mae'])
		return vae, encoder, decoder
	
	
	def VAE1D(self,inputFeatures,outputFeatures,timesteps,summary = False):
		
		codeSize = codeDimension
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		
		eh1 = Conv1D(128,7, activation='relu',name = "EH1")
		eh2 = Conv1D(256,3, activation='relu',name = "EH2")
		eh3 = Conv1D(256,3, activation='relu',name = "EH3")
		fe = Flatten(name="FE")
		mean = Dense(codeSize, activation='linear',name = "MU")
		logsg =  Dense(codeSize, activation='linear', name = "LOG_SIGMA")

		mu = mean(fe(eh3(eh2(eh1((inputs))))))
		log_sigma = logsg(fe(eh3(eh2(eh1((inputs))))))
		encoder = Model(inputs,[mu, log_sigma])
		if(summary):
			print("Encoder")
			encoder.summary()
		
		# Sample z ~ Q(z|X)
		z = Lambda(sample_z,name="CODE")([mu, log_sigma])
		
		latent_inputs = Input(shape=(codeSize,), name='z_sampling')
		
		dh1 = Dense(64, activation='relu',name = "DH1")
		dh2 = Dense(256, activation='relu',name = "DH2")
		decoded = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")
		
		decoderOut = Reshape((timesteps, outputFeatures),name="OUT")
		
		decOut = decoderOut( decoded( dh2 ( dh1((latent_inputs))) ))
		decoder = Model(latent_inputs,decOut)
		if(summary):
			print("Decoder")
			decoder.summary()

		trainDecOut = decoderOut( decoded(dh2(dh1((z))))) 
		vae = Model(inputs, trainDecOut)
		if(summary):
			print("VAE")
			vae.summary()
		
		opt = optimizers.Adam(lr=0.0001) 

		vae.compile(loss = huber_loss,optimizer=opt,metrics=['mae'])
		return vae, encoder, decoder
	
	
	
	def Conv2DQR(self,inputFeatures,outputFeatures,timesteps):
		
		strideSize = 2
		codeSize = codeDimension
		lr = 0.0001
		outputActivation = 'linear'
		hiddenActication = 'tanh'
	
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = Reshape((4,5,2),name="R2E")(inputs)
		c = Conv2D(128,strideSize,activation=hiddenActication,name="E1")(c)
		c = Conv2D(512,strideSize,activation=hiddenActication,name="E2")(c)

		
		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Reshape((1,1,codeSize),name="R2D")(enc)

		c = Conv2DTranspose(512,strideSize,activation=hiddenActication,name="D1")(c)
		
		preDecFlat = Flatten(name="F2")(c) 
		c = Dense(timesteps*outputFeatures,activation=outputActivation,name="DECODED")(preDecFlat)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		autoencoderModel = Model(inputs=inputs, outputs=out)
		opt = optimizers.Adam(lr=lr) 
		autoencoderModel.compile(loss=huber_loss, optimizer=opt,metrics=['mae'])
		return autoencoderModel, None, None
	
	def conv1DQR(self,inputFeatures,outputFeatures,timesteps):
		
		dropPerc = 0.5
		norm = 5
		codeSize = codeDimension
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = Conv1D(256,5,activation='relu',name="E1")(inputs)
		c = Dropout(dropPerc)(c)
		c = Conv1D(64, 4,kernel_constraint=max_norm(norm),activation='relu',name="E2")(c)
		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Dense(256,activation='relu',name="D1")(enc)
		c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(c)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		autoencoderModel = Model(inputs=inputs, outputs=out)
		opt = optimizers.Adam(lr=0.00005) 
		autoencoderModel.compile(loss=huber_loss, optimizer=opt,metrics=['mae'])
		return autoencoderModel, None, None
	
	def dense(self,inputFeatures,outputFeatures,timesteps):
		dropPerc = 0.5
		norm = 2.
		codeSize = codeDimension
		codeMultiplier = 4
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		d = Dense(32,activation='relu',name="E1")(inputs)
		d = Dense(16,activation='relu',name="E2")(d)
		d = Dense(64,activation='relu',name="E4")(d)
		
		### s - encoding
		d = Flatten(name="F1")(d) 
		enc = Dense(codeSize,activation='relu',name="ENC")(d)
		### e - encoding
		
		d = Dense(codeSize*codeMultiplier,activation='relu',name="D1")(enc)
		d = Reshape((codeSize, codeMultiplier),name="R")(d)
		d = Dense(256,activation='relu',name="D3")(d)
		d = Dense(96,activation='relu',name="D4")(d)
		
		d = Flatten(name="F2")(d)
		d = Dense(outputFeatures*timesteps,activation='linear',name="DEC")(d)
		out = Reshape((timesteps, outputFeatures),name="OUT")(d)
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		opt = optimizers.Adam(lr=0.00005) 
		autoencoderModel.compile(loss=huber_loss, optimizer=opt,metrics=['mae'])
		return autoencoderModel, None, None
	
	def getMaes(self,testX,ydecoded):
		maes = np.zeros(ydecoded.shape[0], dtype='float32')
		for sampleCount in range(0,ydecoded.shape[0]):
			maes[sampleCount] = mean_absolute_error(testX[sampleCount],ydecoded[sampleCount])
		return maes

	def trainlModelOnArray(self,x_train, y_train, x_valid, y_valid,name4model,encodedSize = 8):
		
		tt = time.clock()
		logger.debug("trainlModelOnArray - start")
		
		x_train = self.__batchCompatible(self.batchSize,x_train)
		y_train = self.__batchCompatible(self.batchSize,y_train)
		x_valid = self.__batchCompatible(self.batchSize,x_valid)
		y_valid = self.__batchCompatible(self.batchSize,y_valid)
		
		logger.debug("Training model %s with train %s and valid %s" % (name4model,x_train.shape,x_valid.shape))

		inputFeatures  = x_train.shape[2]
		outputFeatures = y_train.shape[2]
		timesteps =  x_train.shape[1]
		
		model,_,_ = self.getModel(inputFeatures,outputFeatures,timesteps)

		#print(model.summary())
		
		path4save = os.path.join( self.ets.rootResultFolder , name4model+self.modelExt )
		checkpoint = ModelCheckpoint(path4save, monitor='val_loss', verbose=0,
			save_best_only=True, mode='min',save_weights_only=True)
		
		#early = EarlyStopping(monitor='val_mean_absolute_error',
		#	min_delta=0.0001, patience=100, verbose=1, mode='min')
				
		history  = model.fit(x_train, x_train,
			verbose = 0,
			batch_size=self.batchSize,
			epochs=self.epochs,
			validation_data=(x_valid,x_valid)
			,callbacks=[checkpoint]
			#,callbacks=[checkpoint,early]
		)
		elapsed = (time.clock() - tt)
		
		historySaveFile = name4model+"_history"
		self.ets.saveZip(self.ets.rootResultFolder,historySaveFile,history.history)
		logger.info("Training completed. Elapsed %f second(s)." %  (elapsed))
		
		model,encoder,decoder = self.loadModel(path4save,2,2,20)
		
		valid_decoded = None
		if(encoder is not None):
			m,s = encoder.predict(x_valid)
			np.random.seed(42)
			samples = []
			for i in range(0,m.shape[0]):
				eps = np.random.normal(0, 1, codeDimension)
				z = m[i] + np.exp(s[i] / 2) * eps
				samples.append(z)
			
			samples = np.asarray(samples)
			valid_decoded = decoder.predict(samples)
		else:
			valid_decoded = model.predict(x_valid)
			
		valMae = self.getMaes(x_valid,valid_decoded)
		logger.info("Training completed. Valid MAE %f " %  (valMae.mean()) )
		
	def evaluateModelOnArray(self,testX,testY,model2load,plotMode,scaler=None,showImages=True,num2show=5,phase="Test",showScatter = False):
		
		path4save = os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt)
		testX = self.__batchCompatible(self.batchSize,testX)
		model , encoder, decoder = self.loadModel(path4save,2,2,20)
		ydecoded = None

		if(encoder is not None):
			m,s = encoder.predict(testX)
			np.random.seed(42)
			samples = []
			for i in range(0,m.shape[0]):
				eps = np.random.normal(0, 1, codeDimension)
				z = m[i] + np.exp(s[i] / 2) * eps
				samples.append(z)

			samples = np.asarray(samples)
			ydecoded = decoder.predict(samples)
		else:
			ydecoded = model.predict(testX)
		
		maes = self.getMaes(testX,ydecoded)
		logger.info("Test MAE %f " %  (maes.mean()))
		return maes
	
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