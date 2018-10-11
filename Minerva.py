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


codeDimension = 3


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
	
	
		#kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		#kl_loss = K.sum(kl_loss, axis=-1)
		#kl_loss *= -0.5
		reconstruction_loss = K.mean(mse(y_true, y_pred))
		#temp = K.mean(kl_loss)
		vaeLoss = reconstruction_loss #+ temp
		return vaeLoss
		
		
	return loss
	

def sample_z(args):
	mu, log_sigma = args
	eps = K.random_normal(shape=(1,2,codeDimension,),mean=0.,stddev=1.)
	return mu + K.exp(log_sigma / 2) * eps

	
	
class Minerva():
		
	logFolder = "./logs"
	modelName = "FullyConnected_4_"
	modelExt = ".h5"
	batchSize = 100
	epochs = 350
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

				
			
	
	def VAE(self,inputFeatures,outputFeatures,timesteps):
		
		summary = False
		codeSize = codeDimension

		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		r = Reshape((4,5,2),name="RE")
		
		#encoderHidden1 = Dense(512, activation='relu',name = "EH1")
		#encoderHidden2 = Dense(256, activation='relu',name = "EH2")
		eh1 = Conv2D(256,2, activation='relu',name = "EH1")
		eh2 = Conv2D(128,2, activation='relu',name = "EH2")
		eh3 = Conv2D(64,2, activation='relu',name = "EH3")
		#flatE = Flatten(name="FE")
		mean = Dense(codeSize, activation='linear',name = "MU")
		logsg =  Dense(codeSize, activation='linear', name = "LOG_SIGMA")
		#mu = mean(flatE(encoderHidden2(encoderHidden1(r(inputs)))))
		#log_sigma = logsg(flatE(encoderHidden2(encoderHidden1(r(inputs)))))
		mu = mean((eh3(eh2(eh1(r(inputs))))))
		log_sigma = logsg((eh3(eh2(eh1(r(inputs))))))
		encoder = Model(inputs,[mu, log_sigma])
		if(summary):
			print("Encoder")
			encoder.summary()
		
		# Sample z ~ Q(z|X)
		z = Lambda(sample_z,name="CODE")([mu, log_sigma])
		
	
		# P(X|z) -- decoder
		latent_inputs = Input(shape=(1,2,codeSize,), name='z_sampling')
		decoder_hidden1 = Dense(1024, activation='relu',name = "DH1")
		decoder_hidden2 = Dense(512, activation='relu',name = "DH2")
		decoded = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")
		fd = Flatten(name = "FD")
		decoderOut = Reshape((timesteps, outputFeatures),name="OUT")
		
		decOut = decoderOut( decoded(fd( decoder_hidden2 ( decoder_hidden1(latent_inputs) ) )))
		decoder = Model(latent_inputs,decOut)
		if(summary):
			print("Decoder")
			decoder.summary()

		trainDecOut = decoderOut( decoded( fd(decoder_hidden2(decoder_hidden1(z))))) 
		vae = Model(inputs, trainDecOut)
		if(summary):
			print("VAE")
			vae.summary()
		
		adam = optimizers.Adam() #lr=0.0005
		# Overall VAE model, for reconstruction and training
		#vae.compile(loss=vae_loss(mu,log_sigma), optimizer=adam,metrics=['mae'])
		vae.compile(loss = huber_loss,optimizer=adam,metrics=['mae'])
		return vae, encoder, decoder


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
		#CHOOSE MODEL
		model,_,_ = self.VAE(inputFeatures,outputFeatures,timesteps)
		#__ch2sp(inputFeatures,outputFeatures,timesteps)
		#CHOOSE MODEL
		
		
		if(False):
			model.compile(loss=huber_loss, optimizer=adam,metrics=['mae'])
		
		#print(model.summary())
		
		path4save = os.path.join( self.ets.rootResultFolder , name4model+self.modelExt )
		checkpoint = ModelCheckpoint(path4save, monitor='val_loss', verbose=0,
			save_best_only=True, mode='min',save_weights_only=True)
		
		validY = y_valid
		trainY = y_train
		
		history  = model.fit(x_train, x_train,
			verbose = 1,
			batch_size=self.batchSize,
			epochs=self.epochs,
			validation_data=(x_valid,x_valid)
			,callbacks=[checkpoint]
		)
		elapsed = (time.clock() - tt)
		
		historySaveFile = name4model+"_history"
		self.ets.saveZip(self.ets.rootResultFolder,historySaveFile,history.history)
		
		# loading the best model for evaluation
		#customLoss = {'huber_loss': huber_loss}
		#model = load_model(path4save,custom_objects=customLoss)
		
		
		vae, encoder, decoder = self.VAE(inputFeatures,outputFeatures,timesteps)
		vae.load_weights(path4save,by_name=True)
		encoder.load_weights(path4save,by_name=True)
		decoder.load_weights(path4save,by_name=True)
		
		m,s = encoder.predict(x_valid)
		
		samples = []
		for i in range(0,m.shape[0]):
			eps = np.random.normal(0, 1, codeDimension)
			z = m[i] + np.exp(s[i] / 2) * eps
			samples.append(z)
        
		samples = np.asarray(samples)
        
		ydecoded = decoder.predict(samples)
		
		#ydecoded = vae.predict(x_valid)
		
		maes = np.zeros(ydecoded.shape[0], dtype='float32')
		for sampleCount in range(0,ydecoded.shape[0]):
			maes[sampleCount] = mean_absolute_error(x_valid[sampleCount],ydecoded[sampleCount])
		
		print(np.mean(maes))
		logger.info("Training completed. Elapsed %f second(s)." %  (elapsed))
		
		#trainMae, trainHuber = model.evaluate( x=x_train, y=y_train, batch_size=self.batchSize, verbose=0)
		#valMae, valHuber = model.evaluate( x=x_valid, y=y_valid, batch_size=self.batchSize, verbose=0)
		#logger.info("Training completed. Elapsed %f second(s). Train HL %f - MAE %f  Valid HL %f - MAE %f " %  (elapsed,trainMae,trainHuber,valMae,valHuber) )
		#logger.debug("trainlModelOnArray - end - %f" % (time.clock() - tt) )
	
	
	
	
	def reconstructionProbability(self,model2load,testX):
		path4save = os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt)
		_, encoder, decoder = self.VAE(2,2,20)
		encoder.load_weights(path4save,by_name=True)
		decoder.load_weights(path4save,by_name=True)
		testX = self.__batchCompatible(self.batchSize,testX)
		m,s = encoder.predict(testX)
		
		#print(testX.shape)
		probs = np.zeros(testX.shape[0], dtype='float32')
		for i in range(0,m.shape[0]):
			samples = []
			for j in range(0,100):
				# sample 50
				eps = np.random.normal(0, 1, codeDimension)
				z = m[i] + np.exp(s[i] / 2) * eps
				samples.append(z)
			xhat = decoder.predict(np.asarray(samples))
			
			maes = np.zeros(xhat.shape[0], dtype='float32')
			for sampleCount in range(0,xhat.shape[0]):
				maes[sampleCount] = mean_squared_error(testX[sampleCount],xhat[sampleCount])
			
			recProb = maes.mean()
			
			probs[i] = recProb
		
		print(probs.mean())
		return probs
		
			
	def evaluateModelOnArray(self,testX,testY,model2load,plotMode,scaler=None,showImages=True,num2show=5,phase="Test",showScatter = False):
		
		path4save = os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt)
		_, encoder, decoder = self.VAE(2,2,20)
		
		encoder.load_weights(path4save,by_name=True)
		decoder.load_weights(path4save,by_name=True)
		
		testX = self.__batchCompatible(self.batchSize,testX)
		m,s = encoder.predict(testX)
		
		
		samples = []
		for i in range(0,m.shape[0]):
			eps = np.random.normal(0, 1, codeDimension)
			z = m[i] + np.exp(s[i] / 2) * eps
			samples.append(z)

		samples = np.asarray(samples)
		ydecoded = decoder.predict(samples)
		
		maes = np.zeros(ydecoded.shape[0], dtype='float32')
		for sampleCount in range(0,ydecoded.shape[0]):
			maes[sampleCount] = mean_absolute_error(testX[sampleCount],ydecoded[sampleCount])
		print(np.mean(maes))


		return maes
	
	
	def printModelSummary(self,name4model):
		path4save = os.path.join( self.ets.rootResultFolder , name4model+self.modelExt )
		customLoss = {'huber_loss': huber_loss}
		model = load_model(path4save,custom_objects=customLoss)
		from keras.utils import plot_model
		plot_model(model,show_shapes = True,to_file="%s.png" % name4model)
		print(model.summary())
	
	def getEncoded(self,model2load,data):
	
		customLoss = {'huber_loss': huber_loss}
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt)
			,custom_objects=customLoss)
	
		layer_name = 'ENC'
		encoded = Model(inputs=model.input,
										 outputs=model.get_layer(layer_name).output)
		code = encoded.predict(data)
		print (code.shape)
		return code
	
	def getModel(self,model2load):
		customLoss = {'huber_loss': huber_loss}
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt)
			,custom_objects=customLoss)
		return model
			
	def getMaes(self,model,testX,testY):
		testX = self.__batchCompatible(self.batchSize,testX)
		testY = self.__batchCompatible(self.batchSize,testY)
		ydecoded = model.predict(testX,  batch_size=self.batchSize)
		maes = np.zeros(ydecoded.shape[0], dtype='float32')
		for sampleCount in range(0,ydecoded.shape[0]):
			maes[sampleCount] = mean_absolute_error(testY[sampleCount],ydecoded[sampleCount])
		
		return maes
	
	
	def Conv2D_QR3(self,inputFeatures,outputFeatures,timesteps):
		dropPerc = 0.5
		strideSize = 2
		codeSize = 12
		norm = 4.
		outFun = 'tanh'

		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = Reshape((4,5,2),name="R2E")(inputs)
		c = Conv2D(48,strideSize,activation='relu',name="E1")(c)
		c = Conv2D(48,strideSize,activation='relu',name="E3")(c)

		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Reshape((1,1,codeSize),name="R2D")(enc)

		c = Conv2DTranspose(128,strideSize,activation='relu',name="D1")(c)
		c = Dropout(dropPerc)(c)
		c = Conv2DTranspose(128,strideSize,kernel_constraint=max_norm(norm),activation='relu',name="D3")(c)
		preDecFlat = Flatten(name="F2")(c) 
		c = Dense(timesteps*outputFeatures,activation=outFun,name="DECODED")(preDecFlat)
		
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		autoencoderModel = Model(inputs=inputs, outputs=out)
		return autoencoderModel
	
	def Conv2D_QR2(self,inputFeatures,outputFeatures,timesteps):
		dropPerc = 0.5
		strideSize = 2
		codeSize = 12
		norm = 4.

		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = Reshape((4,5,2),name="R2E")(inputs)
		c = Conv2D(256,strideSize,activation='relu',name="E1")(c)
		c = Conv2D(48,strideSize,activation='relu',name="E3")(c)

		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Reshape((1,1,codeSize),name="R2D")(enc)

		c = Conv2DTranspose(256,strideSize,activation='relu',name="D1")(c)

		c = Dropout(dropPerc)(c) 
		c = Conv2DTranspose(48,strideSize,kernel_constraint=max_norm(norm),activation='relu',name="D2")(c)

		preDecFlat = Flatten(name="F2")(c) 
		c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(preDecFlat)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		autoencoderModel = Model(inputs=inputs, outputs=out)
		return autoencoderModel
	
	def Conv2D_QR(self,inputFeatures,outputFeatures,timesteps):
		dropPerc = 0.5
		strideSize = 2
		codeSize = 11
		norm = 5.

		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = Reshape((4,5,2),name="R2E")(inputs)
		c = Conv2D(128,strideSize,activation='relu',name="E1")(c)
		
	
		c = Dropout(dropPerc)(c)
		c = Conv2D(48,strideSize,kernel_constraint=max_norm(norm),activation='relu',name="E2")(c)

		
		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Reshape((1,1,codeSize),name="R2D")(enc)

		c = Conv2DTranspose(512,strideSize,activation='relu',name="D1")(c)
		c = Dropout(dropPerc)(c) 
		c = Conv2DTranspose(64,strideSize,kernel_constraint=max_norm(norm),activation='relu',name="D2")(c)
		c = Conv2DTranspose(32,strideSize,activation='relu',name="D3")(c)
		
		preDecFlat = Flatten(name="F2")(c) 
		c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(preDecFlat)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		autoencoderModel = Model(inputs=inputs, outputs=out)
		return autoencoderModel
	
	def __ch2sp(self,inputFeatures,outputFeatures,timesteps):
		
		dropPerc = 0.5
		strideSize = 2
		codeSize =11
		norm = 5.

		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		
		c = Reshape((4,5,2),name="R2E")(inputs)
		c = Conv2D(128,strideSize,activation='relu',name="E1")(c)
		c = Dropout(dropPerc)(c)
		c = Conv2D(48,strideSize,kernel_constraint=max_norm(norm),activation='relu',name="E2")(c)

		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Reshape((1,1,codeSize),name="R2D")(enc)
	
		c = Conv2DTranspose(512,strideSize,activation='relu',name="D1")(c)		
		c = Dropout(dropPerc)(c) 
		c = Conv2DTranspose(64,strideSize,kernel_constraint=max_norm(norm),activation='relu',name="D2")(c)	
		c = Conv2DTranspose(32,strideSize,activation='relu',name="D3")(c)
		
		preDecFlat = Flatten(name="F2")(c) 
		c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(preDecFlat)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
			
		autoencoderModel = Model(inputs=inputs, outputs=out)
		return autoencoderModel
		
		
	
		
	def __ch1sp(self,inputFeatures,outputFeatures,timesteps):
		
		dropPerc = 0.5
		codeSize = 11
		norm = 3.
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = Conv1D(64,9,activation='relu',name="E1")(inputs)
		c = Dropout(dropPerc)(c)
		c = Conv1D(32,5,kernel_constraint=max_norm(norm),activation='relu',name="E3")(c)
		
		preEncodeFlat = Flatten(name="F1")(c) 
		enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
		c = Dense(128,activation='relu',name="D1")(enc)
		c = Dense(512,activation='relu',name="D2")(c)
		c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(c)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		
		autoencoderModel = Model(inputs=inputs, outputs=out)
		return autoencoderModel
	
	def conv1DQR(self,inputFeatures,outputFeatures,timesteps):
		
		dropPerc = 0.5
		norm = 5
		codeSize = 10
		
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
		return autoencoderModel
	
	
	def dense(self,inputFeatures,outputFeatures,timesteps):
		dropPerc = 0.5
		norm = 2.
		codeSize = 9
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
	