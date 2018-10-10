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
from sklearn.metrics import mean_absolute_error
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


'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''
def __huber_loss(y_true, y_pred, clip_delta=1.0):
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
	
#eps = K.random_normal(shape=(20, 12), mean=0., stddev=1.)
	
def sample_z(args):
	mu, log_sigma = args
	eps = K.random_normal(shape=(1,5),mean=0.,stddev=1.)
	return mu + K.exp(log_sigma / 2) * eps


def huber_loss(z_mean,z_log_var):

	def loss(y_true, y_pred, clip_delta=1.0):
	
		error = y_true - y_pred	
		cond  = tf.keras.backend.abs(error) < clip_delta
		squared_loss = 0.5 * tf.keras.backend.square(error) 
		linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta) 
		huberLoss = tf.where(cond, squared_loss, linear_loss) 
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss *= -0.5
			
		#print(kl_loss.get_shape())
		#print(huberLoss.get_shape())
		
		tmp1 =  tf.add( huberLoss[:,:,0] , kl_loss)
		tmp2 = 	tf.add( huberLoss[:,:,1] , kl_loss)
		
		err = tf.add(tmp1,tmp2)

		
		return err
		
	
	
	
	return loss
	

	
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

				
	def VAE4(self,inputFeatures,outputFeatures,timesteps):
		
		intermediate_dim = 256
		latent_dim = 5
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		
		
		
		x = Dense(intermediate_dim, activation='relu')(inputs)
		z_mean = Dense(latent_dim, name='z_mean')(x)
		z_log_var = Dense(latent_dim, name='z_log_var')(x)

		# use reparameterization trick to push the sampling out as input
		# note that "output_shape" isn't necessary with the TensorFlow backend
		z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

		# instantiate encoder model
		encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
		print("ENCODER")
		encoder.summary()
		
		# build decoder model
		latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
		x = Dense(intermediate_dim, activation='relu')(latent_inputs)
		y = Dense(outputFeatures*timesteps, activation='sigmoid')(x)
		#outputs = Reshape((timesteps, outputFeatures),name="OUT")(y)
		
		
		# instantiate decoder model
		decoder = Model(latent_inputs, y, name='decoder')
		print("Decoder")
		decoder.summary()
		
		adam = optimizers.Adam(lr=0.0005)		
		out = decoder(encoder(inputs)[2])
		
		
		outputs = Reshape((timesteps, outputFeatures),name="OUT")(out)
		vae = Model(inputs, outputs, name='vae_mlp')
		print("VAE")
		vae.summary();
		#vae.add_loss(vae_loss)
		#huber_loss(z_mean,z_log_var)
		
		vae.compile(loss=huber_loss, optimizer=adam,metrics=['mae'])
		
		return vae,encoder,decoder
		
				
	
	def VAE(self,inputFeatures,outputFeatures,timesteps):
		dropPerc = 0.5
		strideSize = 2
		codeSize = 5
		norm = 4.
		outFun = 'tanh'

		n_z = codeSize
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		h_q = Dense(256, activation='relu')(inputs)
		mu = Dense(n_z, activation='linear')(h_q)
		log_sigma = Dense(n_z, activation='linear')(h_q)
		
		# Sample z ~ Q(z|X)
		z = Lambda(sample_z)([mu, log_sigma])
		
		
		
		
		# P(X|z) -- decoder
		decoder_hidden = Dense(256, activation='relu')
		decoder_out = Dense(128, activation='relu')

		h_p = decoder_hidden(z)
		outputs = decoder_out(h_p)
		
		
		preDecFlat = Flatten(name="F2")(outputs) 
		c = Dense(timesteps*outputFeatures,activation=outFun,name="DECODED")(preDecFlat)
		
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		adam = optimizers.Adam(lr=0.0005)
		# Overall VAE model, for reconstruction and training
		vae = Model(inputs, out)
		
		
		decoder = None
		latent_inputs = Input(shape=(5,), name='z_sampling')
		
		
		decOut = decoder_out(decoder_hidden(latent_inputs))
		decoder = Model(latent_inputs,decOut)
		
		encoder = Model(inputs,[mu, log_sigma])
		
		vae.compile(loss=huber_loss(mu,log_sigma), optimizer=adam,metrics=['mae'])
		return vae, encoder, decoder


	def VAE3(self,inputFeatures,outputFeatures,timesteps):
		codeSize = 10

		outFun = 'linear'

		n_z = codeSize
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		h_q = Dense(256, activation='relu')(inputs)
		mu = Dense(n_z, activation='linear')(h_q)
		log_sigma = Dense(n_z, activation='linear')(h_q)
		
		# Sample z ~ Q(z|X)
		z = Lambda(sample_z)([mu, log_sigma])
		
		# P(X|z) -- decoder
		decoder_hidden = Dense(128, activation='relu')
		decoder_out = Dense(256, activation='relu')

		h_p = decoder_hidden(z)
		outputs = decoder_out(h_p)
		
		preDecFlat = Flatten(name="F2")(outputs) 
		c = Dense(timesteps*outputFeatures,activation=outFun,name="DECODED")(preDecFlat)
		
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		
		# Overall VAE model, for reconstruction and training
		vae = Model(inputs, out)
		return vae
		
	def VAE2(self,inputFeatures,outputFeatures,timesteps):
		dropPerc = 0.5
		strideSize = 2
		codeSize = 7
		
		inputs = Input(shape=(timesteps,inputFeatures),name="IN")
		c = Reshape((4,5,2),name="R2E")(inputs)
		c = Conv2D(128,strideSize,activation='relu',name="E1")(c)
		c = Conv2D(48,strideSize,activation='relu',name="E2")(c)

		mu =  Conv2D(codeSize,strideSize,activation='linear',name="MU")(c)
		log_sigma = Conv2D(codeSize,strideSize,activation='linear',name="SIGMA")(c)
		
		# Sample z ~ Q(z|X)
		z = Lambda(sample_z,name="Z")([mu, log_sigma])
		
		# P(X|z) -- decoder
		decoder_hidden = Conv2DTranspose(48,strideSize,activation='relu',name="D1")
		decoder_out = Conv2DTranspose(128,strideSize,activation='relu',name="D2")

		h_p = decoder_hidden(z)
		outputs = decoder_out(h_p)
		
		#preDecFlat = Flatten(name="F2")(outputs) 
		#c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(preDecFlat)
		
		
		c = Conv2DTranspose(outputFeatures,strideSize,activation='linear',name="D3")(outputs)
		out = Reshape((timesteps, outputFeatures),name="OUT")(c)
		
		# Overall VAE model, for reconstruction and training
		vae = Model(inputs, out)
		return vae
		
				
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
		
		
		#samples = x_train.shape[0]
		#timesteps = x_train.shape[1]
		#features = x_train.shape[2]
		#x_train = x_train.reshape(samples,timesteps*features)
		#
		#samples = x_valid.shape[0]
		#timesteps = x_valid.shape[1]
		#features = x_valid.shape[2]
		#x_valid = x_valid.reshape(samples,timesteps*features)
		
		history  = model.fit(x_train, x_train,
			verbose = 0,
			batch_size=self.batchSize,
			#epochs=self.epochs,
			epochs=5,
			validation_data=(x_valid,x_valid)
			,callbacks=[checkpoint]
		)
		elapsed = (time.clock() - tt)
		
		historySaveFile = name4model+"_history"
		self.ets.saveZip(self.ets.rootResultFolder,historySaveFile,history.history)
		
		# loading the best model for evaluation
		#customLoss = {'huber_loss': huber_loss}
		#model = load_model(path4save,custom_objects=customLoss)
		
		
		model, encoder, decoder = self.VAE(inputFeatures,outputFeatures,timesteps)
		model.load_weights(path4save)
		
		
		
		trainMae, trainHuber = model.evaluate( x=x_train, y=y_train, batch_size=self.batchSize, verbose=0)
		valMae, valHuber = model.evaluate( x=x_valid, y=y_valid, batch_size=self.batchSize, verbose=0)
		logger.info("Training completed. Elapsed %f second(s). Train HL %f - MAE %f  Valid HL %f - MAE %f " %  (elapsed,trainMae,trainHuber,valMae,valHuber) )
		
		logger.debug("trainlModelOnArray - end - %f" % (time.clock() - tt) )
	
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
	
	def evaluateModelOnArray(self,testX,testY,model2load,plotMode,scaler=None,showImages=True,num2show=5,phase="Test",showScatter = False):
		
		#customLoss = {'huber_loss': huber_loss}
		#model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt)
		#	,custom_objects=customLoss)
		
		model,enc,_ = self.VAE(2,2,20)
		
		model.load_weights(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt))
		
		
		[m,s] = enc.predict(testX,  batch_size=self.batchSize)
		
		
		
		logger.debug("There are %s parameters in model" % model.count_params()) 
		
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
	