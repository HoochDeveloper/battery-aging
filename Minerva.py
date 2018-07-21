#Standard Imports
import uuid,time,os,logging, numpy as np, sys, pandas as pd , matplotlib.pyplot as plt

from logging import handlers as loghds

#Project module import
from Demetra import EpisodedTimeSeries


#KERAS
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D,AveragePooling1D
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
	
	
	def decode4month(self,monthIndex,plotMode,showImages=False,xscaler=None,yscaler=None):
		logger.info("Model trained on month 0, autoencoding for month %d" % monthIndex)
		name4model = "Month_%s_%d_%d_%d_%d" % ( self.modelName,self.eps1,self.eps2,self.alpha1,self.alpha2 )
		
		model = load_model(os.path.join( self.ets.rootResultFolder ,name4model+self.modelExt))
		
		batteries = self.ets.loadBlowDataSet(monthIndexes=[monthIndex]) # blows
		self.dropDatasetLabel(batteries)
		x,y = self.__datasetAs3DArray(batteries,xscaler,yscaler)
		#self.evaluateModelOnArray(x, y,name4model,plotMode,yscaler,showImages)
		x = self.__batchCompatible(self.batchSize,x)
		y = self.__batchCompatible(self.batchSize,y)
		decoded = model.predict(x,batch_size=self.batchSize)
		
		from sklearn.metrics import mean_absolute_error
		downScaledMAE = np.zeros(y.shape[0])
		for i in range(y.shape[0]):
			mae = mean_absolute_error(y[i,:,0], decoded[i,:,0])
			downScaledMAE[i] = mae
		
		wholeDownScaleMae = mean_absolute_error(y[:,:,0], decoded[:,:,0])
		print("Full MAE downscale %f" % wholeDownScaleMae)
		
		inscaleDecoded = self.__skScaleBack(decoded,yscaler)
		inscaleReal = self.__skScaleBack(y,yscaler)
		inscaledMAE = np.zeros(y.shape[0])
		for i in range(y.shape[0]):
			mae = mean_absolute_error(inscaleReal[i,:,0], inscaleDecoded[i,:,0])
			inscaledMAE[i] = mae
		
		wholeInscaleMae = mean_absolute_error(inscaleReal[:,:,0], inscaleDecoded[:,:,0])
		print("Full MAE inscale %f" % wholeInscaleMae)
		
		anomaltTh = 0.03
		anomaliesIdx = np.where( downScaledMAE > anomaltTh )   #np.where( downScaledMAE > anomaltTh ) 
		anomalies = downScaledMAE[ anomaliesIdx ]
		
		
		print("Anomalies %d" % anomalies.shape[0])
		toShowReal = y[ anomaliesIdx]
		toShowDecoded = decoded[anomaliesIdx]
		toShowReal = self.__skScaleBack(toShowReal,yscaler)
		toShowDecoded = self.__skScaleBack(toShowDecoded,yscaler)
		self.__plotRealVsDecodedVsResistance(toShowReal,toShowDecoded,30,plotMode=plotMode,random=True,title="Anomlaies_%d" % monthIndex)
		
		regularIdx =  np.where( downScaledMAE <= anomaltTh ) 
		regulars = downScaledMAE[ regularIdx ]
		print("Regulars %d" % regulars.shape[0])
		toShowReal = y[ regularIdx]
		toShowDecoded = decoded[regularIdx]
		toShowReal = self.__skScaleBack(toShowReal,yscaler)
		toShowDecoded = self.__skScaleBack(toShowDecoded,yscaler)
		self.__plotRealVsDecodedVsResistance(toShowReal,toShowDecoded,30,plotMode=plotMode,random=True,title="Regular_%d" % monthIndex)
	
	
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
		self.trainlModelOnArray(xtrain, ytrain, xvalid, yvalid,name4model)

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
			self.trainlModelOnArray(trainX, trainY, validX, validY,name4model)
			self.evaluateModelOnArray(testX, testY,name4model,plotMode,yscaler,False)
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
			self.evaluateModelOnArray(testX, testY,name4model,plotMode,yscaler,showImages)
			validPerc = 0.1
			trainX = x[train_index]
			trainY = y[train_index]
			trainX, validX, trainY, validY = train_test_split( trainX, trainY, test_size=validPerc, random_state=42)
			self.evaluateModelOnArray(trainX, trainY,name4model,plotMode,yscaler,showImages,phase="Train")
			self.evaluateModelOnArray(validX, validY,name4model,plotMode,yscaler,showImages,phase="Valid")
			
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
		
		wholeDownScaleMae = mean_absolute_error(y[:,:,0], decoded[:,:,0])
		print("Full MAE downscale %f" % wholeDownScaleMae)
		
		inscaleDecoded = self.__skScaleBack(decoded,yscaler)
		inscaleReal = self.__skScaleBack(y,yscaler)
		inscaledMAE = np.zeros(y.shape[0])
		for i in range(y.shape[0]):
			mae = mean_absolute_error(inscaleReal[i,:,0], inscaleDecoded[i,:,0])
			inscaledMAE[i] = mae
		
		wholeInscaleMae = mean_absolute_error(inscaleReal[:,:,0], inscaleDecoded[:,:,0])
		print("Full MAE inscale %f" % wholeInscaleMae)
		
		anomaltTh = 0.03
		anomaliesIdx = np.where( downScaledMAE > anomaltTh )   #np.where( downScaledMAE > anomaltTh ) 
		anomalies = downScaledMAE[ anomaliesIdx ]
		
		
		print("Anomalies %d" % anomalies.shape[0])
		toShowReal = y[ anomaliesIdx]
		toShowDecoded = decoded[anomaliesIdx]
		toShowReal = self.__skScaleBack(toShowReal,yscaler)
		toShowDecoded = self.__skScaleBack(toShowDecoded,yscaler)
		self.__plotRealVsDecodedVsResistance(toShowReal,toShowDecoded,30,plotMode=plotMode,random=True,title="Anomlaies")
		
		
		#########
		regularIdx =  np.where( downScaledMAE <= anomaltTh ) 
		regulars = downScaledMAE[ regularIdx ]
		print("Regulars %d" % regulars.shape[0])
		toShowReal = y[ regularIdx]
		toShowDecoded = decoded[regularIdx]
		toShowReal = self.__skScaleBack(toShowReal,yscaler)
		toShowDecoded = self.__skScaleBack(toShowDecoded,yscaler)
		self.__plotRealVsDecodedVsResistance(toShowReal,toShowDecoded,30,plotMode=plotMode,random=True,title="Regular")
		
	
	
	def __plotRealVsDecodedVsResistance(self,toShowReal,toShowDecoded,num2shw,plotMode="server",random=False,title=None):
		my_dpi = 120
		max2show = min(num2shw,toShowReal.shape[0])
		for r in range(max2show):
			if(random):
				i = np.random.randint(toShowReal.shape[0])
			else:
				i = r
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
			
			plt.subplot(3, 2, 5)
			plt.plot(resistance,color="orange",label="Real V / A")
			plt.legend()
			
			plt.subplot(3, 2, 6)
			bins = np.arange(-5,5,.5)
			plt.hist(resistance, bins=bins,weights=weights,color="orange",label="Distr. V / A")
			plt.xticks(bins)
			plt.xticks(rotation=90)
			plt.legend()
			self.ets.plotMode(plotMode,title)
	
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
			verbose = 1,
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
		_ , valMae, pvalMae, valLch, valLch = model.evaluate( x=x_valid, y=[y_valid,y_valid], batch_size=self.batchSize, verbose=0)
		logger.info("Valid MAE %f - LCH %f" % (valMae,valLch))
		logger.info("Valid Probe MAE %f - LCH %f" % (pvalMae,pvalLch))
		logger.debug("trainlModelOnArray - end - %f" % (time.clock() - tt) )
	
	
	
	def evaluateModelOnArray(self,testX,testY,model2load,plotMode,scaler=None,showImages=True,num2show=10,phase="Test"):
		
		model = load_model(os.path.join( self.ets.rootResultFolder ,model2load+self.modelExt))
		
		testX = self.__batchCompatible(self.batchSize,testX)
		testY = self.__batchCompatible(self.batchSize,testY)
		
		logger.debug("Validating model %s with test %s" % (model2load,testX.shape))
		
		tt = time.clock()
		_ , mae, pMae, lch, pLch = model.evaluate( x=testX, y=[testY,testY], batch_size=self.batchSize, verbose=0)
		logger.info("%s MAE %f - LCH %f Elapsed %f" % (phase,mae,lch,(time.clock() - tt)))
		logger.info("%s Probe MAE %f - LCH %f Elapsed %f" % (phase,pMae,pLch,(time.clock() - tt)))
		
		logger.debug("Autoencoding")
		tt = time.clock()
		ydecoded,pdecoded = model.predict(testX,  batch_size=self.batchSize)
		logger.debug("Elapsed %f" % (time.clock() - tt))
		if(scaler is not None):
			ydecoded = self.__skScaleBack(ydecoded,scaler)
			testY = self.__skScaleBack(testY,scaler)
			scaledMAE = self.__skMAE(testY,ydecoded)
			logger.info("%s scaled MAE %f" % (phase,scaledMAE))
			
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
		dprobe = Dense(32,activation='relu',name="DPROBE")(r1)
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
	