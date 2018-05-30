#!/usr/bin/env python

"""
Module for handling data loading
@author: Michele Salvatore Rillo
@email:  michelesalvatore.rillo@gmail.com
@git: HoochDeveloper
Requires: 
	pandas 	(https://pandas.pydata.org/)
	numpy	(http://www.numpy.org/)
"""
#Imports
import uuid,time,os,logging, six.moves.cPickle as pickle, gzip, pandas as pd, numpy as np , matplotlib.pyplot as plt, glob
from datetime import datetime

from logging import handlers as loghds

#Module logging
#logging.basicConfig(filename='./logs/Demetra.log',level=logging.INFO)
logger = logging.getLogger("Demetra")
logger.setLevel(logging.INFO)
#formatter = logging.Formatter('[%(asctime)s %(name)s %(funcName)s %(levelname)s] %(message)s')
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

class EpisodedTimeSeries():
	"""
	Give access to episode in timeseries
	"""
	
	# custom header and types, dataset specific
	""" List of column names as in file """
	dataHeader = ([ "TSTAMP", "THING", "CONF","SPEED","S_IBATCB1_CB1","S_IBATCB2_CB2",
	"S_IOUTBUR1_BUR1","S_IOUTBUR2_BUR2","S_ITOTCB1_CB1","S_ITOTCB2_CB2",
	"S_TBATCB1_CB1","S_TBATCB2_CB2","S_VBATCB1_CB1","S_VBATCB2_CB2","S_VINCB1_CB1","S_VINCB2_CB2",
	"S_CORRBATT_FLG1","S_TENSBATT_FLG1" ])
	
	""" Dictonary for column data types """
	dataTypes = ({ "TSTAMP" : str,"THING" : str,"CONF" : np.float32, "SPEED" : np.float32,
	"S_IBATCB1_CB1" : np.float32,"S_IBATCB2_CB2" : np.float32, "S_IOUTBUR1_BUR1" : np.float32,
	"S_IOUTBUR2_BUR2" : np.float32,"S_ITOTCB2_CB2" : np.float32,"S_TBATCB1_CB1" : np.float32,
	"S_ITOTCB1_CB1" : np.float32,"S_TBATCB2_CB2" : np.float32,"S_VBATCB1_CB1" : np.float32,
	"S_VBATCB2_CB2" : np.float32,"S_VINCB1_CB1" : np.float32,"S_VINCB2_CB2" : np.float32,
	"S_CORRBATT_FLG1" : np.float32,"S_TENSBATT_FLG1" : np.float32 })

	dropX = [dataHeader[0],dataHeader[1]] # columns to drop for X
	keepY = [dataHeader[16],dataHeader[17]] # columns to keep for Y
	
	# Attributes
	timeIndex = None
	nameIndex = None 
	
	currentIndex = None
	voltageIndex = None
	
	root = "."
	rootResultFolder = os.path.join(root,"results")
	episodeImageFolder =  os.path.join(rootResultFolder,"images")
	
	
	espisodeFolder = "episodes"
	espisodePath = None
	
	normalizerFolder = "normalizer"
	normalizerPath = None
	normalizerFile = "norm.pkl"
	
	chargePrefix = "charge_"
	dischargePrefix = "discharge_"
	
	trainsetFolder = None 
	trainsetFile = None
	testsetFolder = None 
	testsetFile = None
	timeSteps = 30
	normalize=True
	
	
		
	#Constructor
	def __init__(self,timeSteps,normalize=True):
		""" 
		Create, if not exists, the result path for storing the episoded dataset
		"""
		logFolder = "./logs"
		# creates log folder
		if not os.path.exists(logFolder):
			os.makedirs(logFolder)
		logPath = logFolder + "/Demetra.log"
		hdlr = loghds.TimedRotatingFileHandler(logPath,
                                       when="H",
                                       interval=1,
                                       backupCount=5)
		
		
		hdlr.setFormatter(formatter)
		logger.addHandler(hdlr)
		
		
		if not os.path.exists(self.rootResultFolder):
			os.makedirs(self.rootResultFolder)
		
		self.normalize = normalize
		self.timeSteps = timeSteps
		
		self.normalizerPath = os.path.join(self.rootResultFolder, self.normalizerFolder)
		
		self.trainsetFolder = os.path.join(self.rootResultFolder, "trainset")
		self.testsetFolder = os.path.join(self.rootResultFolder, "testset")
		
		self.espisodeFolder = "episodes_%s"  % self.timeSteps
			
		self.espisodePath = os.path.join(self.rootResultFolder,self.espisodeFolder)
		
		if not os.path.exists(self.espisodePath):
			os.makedirs(self.espisodePath)
		if not os.path.exists(self.trainsetFolder):
			os.makedirs(self.trainsetFolder)
		if not os.path.exists(self.testsetFolder):
			os.makedirs(self.testsetFolder)
		if not os.path.exists(self.normalizerPath):
			os.makedirs(self.normalizerPath)
		
		self.timeIndex = self.dataHeader[0]
		self.nameIndex = self.dataHeader[1]
		# used for determining when an episode start in charge or discharge
		self.currentIndex = self.dataHeader[16]
		self.voltageIndex = self.dataHeader[17]
		if(self.normalize):
			self.trainsetFile = "normalized_train_%s" % self.timeSteps
			self.testsetFile = "normalized_test_%s" % self.timeSteps
		else:
			self.trainsetFile = "raw_train_%s" % self.timeSteps
			self.testsetFile = "raw_test_%s" % self.timeSteps
		
		logger.debug("Indexes: Current %s Volt %s " % (self.currentIndex,self.voltageIndex))
		
	
	# public methods
	def showEpisodes(self,type=None,normalizer = None,limit=None):
		total = 0
		for f in os.listdir(self.espisodePath):
			if(type == "D" and not str(f).startswith(self.dischargePrefix)):
				continue
			elif(type == "C" and not str(f).startswith(self.chargePrefix)):
				continue
			episodes = self.loadZip(self.espisodePath,f)
			total += len(episodes)
			logger.info("There are %d episodes for %s" % (len(episodes),f))
			max2show = len(episodes)
			if(limit is not None):
				max2show = min(limit,len(episodes))
			for e in range(max2show):
				if(normalizer is not None):
					
					for i in range(1,normalizer.shape[0]):
						col = i + 2
						episodes[e].iloc[:,[col]] =  self.__normalization(episodes[e].iloc[:,[col]],normalizer[i,0],normalizer[i,1],-1,1)
				self.plot(episodes[e])
		logger.info("Total %d" % total)
	
	
	
	def loadTrainset(self,type=None):
		
		train2load = self.trainsetFile
		if(type == "C"):
			train2load = self.chargePrefix + self.trainsetFile
		elif(type == "D"):
			train2load = self.dischargePrefix + self.trainsetFile
		
		load = self.loadZip(self.trainsetFolder,train2load)
		x_train = load[0]
		y_train = load[1]
		x_valid = load[2]
		y_valid = load[3]
		return x_train, y_train, x_valid, y_valid
		
	def loadTestset(self,type=None):
	
		test2load = self.testsetFile
		if(type == "C"):
			test2load = self.chargePrefix + self.testsetFile
		elif(type == "D"):
			test2load = self.dischargePrefix + self.testsetFile
	
		load = self.loadZip(self.testsetFolder,test2load)
		x_test = load[0]
		y_test = load[1]
		return x_test, y_test
	
	def buildEpisodedDataset(self,dataFolder,force=False):
		""" 
		dataFolder: folder thet contains the raw dataset, every file in folder will be treated as indipendent thing
		force: if True entire results will be created even if already exists
		"""
		tt = time.clock()
		logger.info("buildEpisodedDataset - start")
		
		existingEpisodes = len(os.listdir(self.espisodePath))
		
		if(not force and existingEpisodes > 0):
			logger.info( "Datafolder has already %d episodes. Force[%s]" % (existingEpisodes,force) )
		else:
			self.__buildEpisodedDataframesFromFolder(dataFolder,force,dsg=True,chg=True)
		
		if(self.normalize):
			normalizer = self.loadNormalizer()
			if(normalizer is None):
				logger.info("Building normalizer")
				allEpisodes = []
				for f in os.listdir(self.espisodePath):
					episodeList = self.loadZip(self.espisodePath,f)
					allEpisodes += episodeList
				normalizer = self.__getNormalizer(pd.concat(allEpisodes))
				self.saveZip(self.normalizerPath,self.normalizerFile,normalizer)
			else:
				logger.info("Normalizer already exists")
		
		logger.info("buildEpisodedDataset - end - Elapsed: %f" % (time.clock() - tt))
	
	def buildDischargeSet(self,valid=0.2,test=0.2,force=False):
		"""
		Load all the discharge episodes and build train / test of charge episodes only
		"""
		if( not force 
			and os.path.exists(os.path.join(self.trainsetFolder,self.dischargePrefix+self.trainsetFile)) 
			and os.path.exists(os.path.join(self.testsetFolder,self.dischargePrefix+self.testsetFile))
		):
			logger.info("Discharge Train / Test set already exists, nothing to do.")
			return
		
		logger.info("buildDischargeSet - start")
		start = time.clock()
		Xall = []
		Yall = []
		logger.info("Dropping unused columns")
		for f in os.listdir(self.espisodePath):
			if (not str(f).startswith(self.dischargePrefix) ):
				logger.debug("File %s discarded, not a discharge episode." % f)
				continue;
			episodeList = self.loadZip(self.espisodePath,f)
			for e in range(len(episodeList)):
				x = episodeList[e].drop(columns=self.dropX)
				y = episodeList[e][self.keepY]
				Xall.append(x)
				Yall.append(y)
		
		if(len(Xall) == 0):
			logger.warning("No episodes file found, nothing will be done.")
			return
		
		x_test,y_test,x_train, y_train, x_valid, y_valid = self.__trainTestSplit(Xall,Yall,valid,test)
		
		self.saveZip(self.testsetFolder,self.dischargePrefix+self.testsetFile,[x_test,y_test])
		self.saveZip(self.trainsetFolder,self.dischargePrefix+self.trainsetFile,[x_train, y_train, x_valid, y_valid])
		
		logger.info("buildDischargeSet - end - Elapsed: %f" % (time.clock() - start))
	
	
	
	def buildChargeSet(self,valid=0.2,test=0.2,force=False):
		"""
		Load all the charge episodes and build train / test of charge episodes only
		"""
		if( not force 
			and os.path.exists(os.path.join(self.trainsetFolder,self.chargePrefix+self.trainsetFile)) 
			and os.path.exists(os.path.join(self.testsetFolder,self.chargePrefix+self.testsetFile))
		):
			logger.info("Charge Train / Test set already exists, nothing to do.")
			return
		
		logger.info("buildChargeSet - start")
		start = time.clock()
		Xall = []
		Yall = []
		logger.info("Dropping unused columns")
		for f in os.listdir(self.espisodePath):
			if (not str(f).startswith(self.chargePrefix) ):
				logger.debug("File %s discarded, not a charge episode." % f)
				continue;
			episodeList = self.loadZip(self.espisodePath,f)
			
			for e in range(len(episodeList)):	
				x = episodeList[e].drop(columns=self.dropX)
				y = episodeList[e][self.keepY]
				Xall.append(x)
				Yall.append(y)
		
		if(len(Xall) == 0):
			logger.warning("No episodes file found, nothing will be done.")
			return
		
		x_test,y_test,x_train, y_train, x_valid, y_valid = self.__trainTestSplit(Xall,Yall,valid,test)
		
		self.saveZip(self.testsetFolder,self.chargePrefix+self.testsetFile,[x_test,y_test])
		self.saveZip(self.trainsetFolder,self.chargePrefix+self.trainsetFile,[x_train, y_train, x_valid, y_valid])
		
		logger.info("buildChargeSet - end - Elapsed: %f" % (time.clock() - start))
		
	
	
	def buildLearnSet(self,valid=0.2,test=0.2,force=False):
		"""
		Load the dataframe episodes from espisodePath
		Build train, valid and test set
		"""
		if( not force 
			and os.path.exists(os.path.join(self.trainsetFolder, self.trainsetFile)) 
			and os.path.exists(os.path.join(self.testsetFolder, self.testsetFile))
		):
			logger.info("Train / test set already exists, nothing to do.")
			return
	
		logger.info("buildLearnSet - start")
		start = time.clock()
		Xall = []
		Yall = []
		logger.info("Dropping unused columns")
		for f in os.listdir(self.espisodePath):
			episodeList = self.loadZip(self.espisodePath,f)
			for e in range(len(episodeList)):
				x = episodeList[e].drop(columns=self.dropX)
				y = episodeList[e][self.keepY]
				Xall.append(x)
				Yall.append(y)
		
		if(len(Xall) == 0):
			logger.warning("No episodes file found, nothing will be done.")
			return
		
		
		x_test,y_test,x_train, y_train, x_valid, y_valid = self.__trainTestSplit(Xall,Yall,valid,test)
		
		self.saveZip(self.testsetFolder,self.testsetFile,[x_test,y_test])
		self.saveZip(self.trainsetFolder,self.trainsetFile,[x_train, y_train, x_valid, y_valid])
		
		logger.info("buildLearnSet - end - Elapsed: %f" % (time.clock() - start))
	
	def __trainTestSplit(self,Xall,Yall,valid,test):
		logger.debug("Shuffling dataset")
		shuffledX = np.zeros([len(Xall),Xall[0].shape[0],Xall[0].shape[1]])
		shuffledY = np.zeros([len(Yall),Yall[0].shape[0],Yall[0].shape[1]])
		np.random.seed(42)
		shuffled = np.random.permutation(len(Xall))
		for i in range(len(Xall)):
			shuffledX[i] = Xall[shuffled[i]]
			shuffledY[i] = Yall[shuffled[i]]

		if(self.normalize == True):
			logger.debug("Normalized data will be used")
			normalizer = self.loadNormalizer()
			logger.debug(normalizer.shape)
			shuffledX = self.__normalize(shuffledX,normalizer)
			ynormalizer = normalizer[-2:,:] # y has only the last two columns, improve me
			shuffledY = self.__normalize(shuffledY,ynormalizer)	
		else:
			logger.debug("Raw data will be used")
		
		# TEST set
		testIndex = int( len(Xall) * (1 - test) )
		x = shuffledX[:testIndex]
		y = shuffledY[:testIndex]
		x_test = shuffledX[testIndex:]
		y_test = shuffledY[testIndex:]
		logger.info("Test set shape")
		logger.info(x_test.shape)
		
		validIndex = int( x.shape[0] * (1 - valid) )
		#TRAIN set
		x_train = x[:validIndex]
		y_train = y[:validIndex]
		x_valid = x[validIndex:]
		y_valid = y[validIndex:]
		logger.info("Train set shape")
		logger.info(x_train.shape)
		logger.info("Valid set shape")
		logger.info(x_valid.shape)
		
		return x_test,y_test,x_train, y_train, x_valid, y_valid
	
	
	def plot(self,data,savePath=None):
		#column index of the sequence time index
		dateIndex = self.dataHeader.index(self.timeIndex)
		nameIndex = self.dataHeader.index(self.nameIndex)
		# values to plot
		values = data.values
		# getting YYYY-mm-dd for the plot title
		date =  values[:, dateIndex][0].strftime("%Y-%m-%d")
		batteryName =  values[:, nameIndex][0]
		#time series for all data that we want to plot
		# plot each column except TSTAMP and THING(wich is constant for the same battery)
		toPlot = range(2,18)
		i = 1
		plt.figure()
		plt.suptitle("Data for battery %s in day %s" % (batteryName,date), fontsize=16)
		for col in toPlot:
			plt.subplot(len(toPlot), 1, i)
			plt.plot(values[:, col])
			plt.title(data.columns[col], y=0.5, loc='right')
			i += 1
		# For x tick label we just want to use HH:MM:SS as xlabels
		timeLabel = [ d.strftime("%H:%M:%S") for d in values[:, dateIndex] ]
		# integer range, needed for setting xlabel as HH:MM:SS
		xs = range(len(timeLabel))
		# setting HH:MM:SS as xlabel
		frequency = int(len(timeLabel) / 4)
		plt.xticks(xs[::frequency], timeLabel[::frequency])
		plt.xticks(rotation=45)
		if(savePath is None):
			plt.show()
		else:
			batteryImagePath = os.path.join(savePath,batteryName)
			if not os.path.exists(batteryImagePath):
				os.makedirs(batteryImagePath)
			unique_filename = str(uuid.uuid4())
			plt.savefig(os.path.join(batteryImagePath,unique_filename), bbox_inches='tight')
			plt.close()
		
	# private methods
	def __normalize(self,data,normalizer,minrange=-1,maxrange=1):
		for i in range(data.shape[2]):
			minvalue = normalizer[i,0]
			maxvalue = normalizer[i,1]
			data[:,:,i] = self.__normalization(data[:,:,i],minvalue,maxvalue,minrange,maxrange)
			#data[:,:,i] = ((data[:,:,i] - minvalue) / (maxvalue - minvalue)) * (maxrange - minrange) + minrange
		return data
	
	def __normalization(self,data,minvalue,maxvalue,minrange,maxrange):
		norm = ((data - minvalue) / (maxvalue - minvalue)) * (maxrange - minrange) + minrange
		return norm
	
	def __getNormalizer(self,data):
		tt = time.clock()
		logger.debug("Computing normalizer")
		normalizer = np.zeros([data.shape[1]-2,2])
		for i in range(2,data.shape[1]):
			normIdx = i - 2
			logger.debug("Normalizing %s" % self.dataHeader[i])
			normalizer[normIdx,0] = data[self.dataHeader[i]].min()	
			normalizer[normIdx,1] = data[self.dataHeader[i]].max()	
			logger.debug("Min %f Max %f" % (normalizer[normIdx,0],normalizer[normIdx,1]))
		logger.debug("Computed normalizer. Elapsed %f" %  (time.clock() - tt))
		return normalizer
		
	def loadNormalizer(self):
		normalizer = self.loadZip(self.normalizerPath,self.normalizerFile)
		return normalizer
	
	def __buildEpisodedDataframesFromFolder(self,dataFolder,force=False,dsg=True,chg=True):
		""" 
		Read all files in folder as episoded dataframe 
		Every item in the return list is a different thing
		Every inner item is a list of episode for the current thing
		
		result[thing][episode] = dataframe for episode in thing
		
		"""
		tt = time.clock()
		logger.info("Reading data from folder %s" %  dataFolder)
		if( not os.path.isdir(dataFolder)):
			logger.warning("%s is not a valid folder, nothing will be done" % dataFolder )
			return None
		totalFiles = len(os.listdir(dataFolder))
		count = 0
		for file in os.listdir(dataFolder):
			count = count + 1
			logger.info("File %d of %d" % (count,totalFiles))
			if(os.path.isfile(os.path.join(dataFolder,file))):
				loaded = self.__readFileAsDataframe(os.path.join(dataFolder,file))
				if(loaded is not None and loaded.shape[0] > 0):
					batteryName = loaded[self.nameIndex].values[0]
					logger.info("Checking episodes for battery %s" % batteryName)
					savePath = os.path.join(self.espisodePath,batteryName)
					alreadyExistent = len(glob.glob(self.espisodePath + "/*" + batteryName))
					logger.debug("Already existent episodes for %s are %d" % (batteryName,alreadyExistent))
					if(force or alreadyExistent == 0 ):
						dischargeEpisodes , chargeEpisodes = self.__fastEpisodeInDataframe(loaded,self.timeSteps,dsg=dsg,chg=chg)
						if(len(chargeEpisodes) > 0):
							self.saveZip(self.espisodePath,self.chargePrefix+batteryName,chargeEpisodes)
						if(len(dischargeEpisodes) > 0):
							self.saveZip(self.espisodePath,self.dischargePrefix+batteryName,dischargeEpisodes)
						if(len(chargeEpisodes) == 0 and len(dischargeEpisodes) == 0):
							logger.warning("No episodes in file")
					else:
						logger.info("Episodes for battery %s already exists" % batteryName)
		logger.info("Folder read complete. Elapsed %f" %  (time.clock() - tt))
		
	def __readFileAsDataframe(self,file):
		""" 
		Load data with pandas from the specified csv file
		Parameters: 
			file: csv file to read. Must be compliant with the specified dataHeader
		Output:
			pandas dataframe, if an error occurs, return None
		"""
		tt = time.clock()
		logger.debug("Reading data from %s" %  file)
		try:
			data = pd.read_csv(file, compression='gzip', header=None,error_bad_lines=True,sep=',', 
				names=self.dataHeader,
				dtype=self.dataTypes,
				parse_dates=[self.timeIndex],
				date_parser = pd.core.tools.datetimes.to_datetime)
			
			logger.debug("Data read complete. Elapsed %f second(s)" %  (time.clock() - tt))
			logger.debug("Dropping NA")
			data.dropna(inplace=True)
			logger.debug("Indexing")
			data.set_index(self.timeIndex,inplace=True,drop=False)
			logger.debug("Sorting")
			data.sort_index(inplace=True)
			
		except Exception as e:
			print(e)
			logger.error("Can't read file %s" % file)
			data = None
		return data
		
		
	def __fastEpisodeInDataframe(self,df,episodeLength,dsg=True,chg=True):
		logger.debug("__fastEpisodeInDataframe start.")
		tt = time.clock()
		discharges = []
		charges = []
		
		dischargeCount = 0
		chargeCount = 0
		
		zerosCharge = 0
		zerosDischarge = 0
		
		misscompliantCharge = 0
		misscompliantDischarge = 0
		
		misscontextCharge = 0
		misscontextDischarge = 0
		
		groups = [g[1] for g in df.groupby([df.index.year, df.index.month, df.index.day])]
		for dataframe in groups:
			
			if(dsg):
				dischargeEpisodes, zeroCurDiscard, compliantCurDiscard, contextCurDiscard = self.__seekDischargeEpisode(dataframe,episodeLength)
				if(len(dischargeEpisodes) > 0):
					dischargeCount += len(dischargeEpisodes)
					discharges.extend(dischargeEpisodes)
				zerosDischarge += zeroCurDiscard
				misscompliantDischarge += compliantCurDiscard
				misscontextDischarge += contextCurDiscard
			
			if(chg):
				chargeEpisodes, zeroCurDiscard, compliantCurDiscard, contextCurDiscard = self.__seekChargeEpisode(dataframe,episodeLength)
				if(len(chargeEpisodes) > 0):
					chargeCount += len(chargeEpisodes)
					charges.extend(chargeEpisodes)
				
				zerosCharge += zeroCurDiscard
				misscompliantCharge += compliantCurDiscard
				misscontextCharge += contextCurDiscard
		
		
		logger.info("Discharges: %d Zeros: %d Not Compliant: %d Context: %d" % (dischargeCount,zerosDischarge,misscompliantDischarge,misscontextDischarge))
		logger.info("Charges: %d Zeros: %d Not Compliant: %d Context: %d" % (chargeCount,zerosCharge,misscompliantCharge,misscontextCharge))
		
		logger.debug("__fastEpisodeInDataframe end. Elapsed %f" %  (time.clock() - tt))
		return discharges , charges
		
	def __seekDischargeEpisode(self,dataframe,episodeLength):
		
		maxZerosInEpisode = int(episodeLength * 0.75) # must be greater than context len
		contextLength = int(episodeLength / 4)
		
		dischargeEpisodes = []
		zeroDiscarded = 0
		notCompliant = 0
		contextDiscarded = 0
		
		thr = -5
		# select all time steps with -1 < i < 1 and v >= threshold
		
		dischargeStart =  ( 
			dataframe[
			(dataframe[self.currentIndex] <= thr)
			&
			(dataframe[self.voltageIndex] >= 20)
			].index
		)
		
		# since a lot of indexes are consecutive (zone of constant voltage and 0 current ) all consecutive starting indexed are dropped
		# time gap between consecutive time step 
		if(dischargeStart.shape[0] == 0):
			return dischargeEpisodes, zeroDiscarded, notCompliant, contextDiscarded
		
		# handle discharge episodes
		# with future - past the episoded starts from the last holding the condition
		
		
		past = np.roll(dischargeStart,1) # shift the episode one second ahead
		present = np.roll(dischargeStart,0) # convert in numpy array
		diff =  present - past # compute difference between starting episodes
		
		################
		##future = np.roll(dischargeStart,-1) # shift the episode one second ahead
		##present = np.roll(dischargeStart,0) # convert in numpy array
		##diff =  future - present # compute difference between starting episodes
		###############
		
		diff = (diff * 10**-9).astype(int) # convert nanosecond in second
		# keep only start episodes with a gap greater than 1 seconds in order to discard consecutive starting episodes
		dischargeStart = dischargeStart[ (diff >  1 ) ]
		
		logger.debug("Discharge removed %d " % ( len(present) - len(dischargeStart)  ))
		ctxTollerance = thr*2
		for ts in dischargeStart:
			#get integer indexing for time step index
			startRow = dataframe.index.get_loc(ts)
			
			context = dataframe.iloc[startRow-contextLength:startRow,:]
			#checkDischargeContext = context[ (context[self.currentIndex] >= -ctxTollerance) & (context[self.currentIndex] <= ctxTollerance) ].shape[0]
			checkDischargeContext = context[ (context[self.currentIndex] >= ctxTollerance)].shape[0]		
			#if discharge precondition are not meet then skip
			if(checkDischargeContext != contextLength):
				contextDiscarded += 1
				continue

			#include previous context on episode
			startIndex = startRow-contextLength
			endIndex = startIndex+episodeLength
			episodeCandidate = dataframe.iloc[startIndex:endIndex,:]
		
			zeroCurrentCount = episodeCandidate[ episodeCandidate[self.currentIndex] == 0 ].shape[0]
		
			# if there are too many zeros, then the episode is discarded
			if(zeroCurrentCount > maxZerosInEpisode):
				zeroDiscarded +=1
				continue

			dischargeCount = episodeCandidate[ episodeCandidate[self.currentIndex] <= thr ].shape[0]
			if(dischargeCount == (episodeLength-zeroCurrentCount)):
				# this is a discharge episodes
				if(episodeCandidate.shape[0] != episodeLength):
					logger.error("Error episode length discharge")
					logger.error(episodeCandidate.shape)
					notCompliant +=1
					continue
				dischargeEpisodes.append(episodeCandidate)
			else:
				notCompliant +=1
		
		return dischargeEpisodes, zeroDiscarded, notCompliant, contextDiscarded
		
	###################################
	
	def __seekChargeEpisode(self,dataframe,episodeLength):
		
		maxZerosInEpisode = int(episodeLength / 3) # must be greater than context len
		contextLength = int(episodeLength / 4)
		
		chargeEpisodes = []
		zeroDiscarded = 0
		notCompliant = 0
		contextDiscarded = 0
		
		# select all time steps with -1 < i < 1 and v >= threshold
		chargeStart =  ( 
			dataframe[
			(dataframe[self.currentIndex] > 0 ) 
			#&
			#(dataframe[self.voltageIndex] >= 30)
			].index
		)
		
		# since a lot of indexes are consecutive (zone of constant voltage and 0 current ) all consecutive starting indexed are dropped
		# time gap between consecutive time step 
		if(chargeStart.shape[0] == 0):
			return chargeEpisodes, zeroDiscarded, notCompliant, contextDiscarded
		
		# handle discharge episodes
		
		# with present - past the episodes starts from the first ts holding the condition
		
		past = np.roll(chargeStart,1) # shift the episode one second ahead
		present = np.roll(chargeStart,0) # convert in numpy array
		diff =  present - past # compute difference between starting episodes
		diff = (diff * 10**-9).astype(int) # convert nanosecond in second

		# keep only start episodes with a gap greater than 1 seconds in order to discard consecutive starting episodes
		chargeStart = chargeStart[ (diff >  1 ) ]
		
		logger.debug("Charge removed %d " % ( len(present) - len(chargeStart)  ))

		for ts in chargeStart:
			#get integer indexing for time step index
			startRow = dataframe.index.get_loc(ts)
			
			context = dataframe.iloc[startRow-contextLength:startRow,:]
			checkChargeContext = context[ (context[self.currentIndex] <= 0) ].shape[0]
								
			#if discharge precondition are not meet then skip
			if(checkChargeContext != contextLength):
				contextDiscarded += 1
				continue

			#include previous context on episode
			startIndex = startRow-contextLength
			endIndex = startIndex+episodeLength
			episodeCandidate = dataframe.iloc[startIndex:endIndex,:]
				
			zeroCurrentCount = episodeCandidate[ episodeCandidate[self.currentIndex] == 0 ].shape[0]
			
			# if there are too many zeros, then the episode is discarded
			if(zeroCurrentCount > maxZerosInEpisode):
				zeroDiscarded +=1
				continue
			
			chargeCount = episodeCandidate[ episodeCandidate[self.currentIndex] > 0 ].shape[0]
			if(chargeCount == (episodeLength-zeroCurrentCount-contextLength)):
				# this is a discharge episodes
				if(episodeCandidate.shape[0] != episodeLength):
					logger.error("Error episode length charge")
					logger.error(episodeCandidate.shape)
					notCompliant +=1
					continue
				chargeEpisodes.append(episodeCandidate)
			else:
				notCompliant +=1
		
		return chargeEpisodes, zeroDiscarded, notCompliant, contextDiscarded
	
	
	
	###################################
	
	def loadBlowDataSet(self):
		blows = []
		for f in os.listdir(self.espisodePath):
			episodes = self.loadZip(self.espisodePath,f)
			blows += self.seekEpisodesBlow(episodes)
		logger.info("All blows %d" % len(blows))
		return blows
		
	def loadUniformedDataSet(self):
		episodes = self.loadZip(self.espisodePath,"swab_E464004")
		return episodes
		
	def buildUniformedDataSet(self,dataFolder,force=False):
		""" 
		dataFolder: folder thet contains the raw dataset, every file in folder will be treated as indipendent thing
		force: if True entire results will be created even if already exists
		"""
		tt = time.clock()
		logger.info("buildUniformedDataSet - start")
		
		existingEpisodes = len(os.listdir(self.espisodePath))
		
		if(not force and existingEpisodes > 0):
			logger.info( "Datafolder has already %d episodes." % (existingEpisodes) )
		else:
			logger.info("Building episodes. Force: %s" , force)
			self.__buildUniformedDataSetFromFolder(dataFolder,force)
		
		if(self.normalize):
			normalizer = self.loadNormalizer()
			if(normalizer is None):
				logger.info("Building normalizer")
				allEpisodes = []
				for f in os.listdir(self.espisodePath):
					episodeList = self.loadZip(self.espisodePath,f)
					allEpisodes += episodeList
				normalizer = self.__getNormalizer(pd.concat(allEpisodes))
				self.saveZip(self.normalizerPath,self.normalizerFile,normalizer)
			else:
				logger.info("Normalizer already exists")
		
		logger.info("buildUniformedDataSet - end - Elapsed: %f" % (time.clock() - tt))
	
	
	def __buildUniformedDataSetFromFolder(self,dataFolder,force=False):
		""" 
		Read all files in folder as episoded dataframe 
		Every item in the return list is a different thing
		Every inner item is a list of episode for the current thing
		
		result[thing][episode] = dataframe for episode in thing
		"""
		tt = time.clock()
		logger.debug("__buildUniformedDataSetFromFolder - begin")
		logger.info("Reading data from folder %s" %  dataFolder)
		if( not os.path.isdir(dataFolder)):
			logger.warning("%s is not a valid folder, nothing will be done" % dataFolder )
			return None
		totalFiles = len(os.listdir(dataFolder))
		count = 0
		for file in os.listdir(dataFolder):
			count = count + 1
			logger.info("File %d of %d" % (count,totalFiles))
			if(os.path.isfile(os.path.join(dataFolder,file))):
				loaded = self.__readFileAsDataframe(os.path.join(dataFolder,file))
				if(loaded is not None and loaded.shape[0] > 0):
					batteryName = loaded[self.nameIndex].values[0]
					logger.info("Checking episodes for battery %s" % batteryName)
					savePath = os.path.join(self.espisodePath,batteryName)
					alreadyExistent = len(glob.glob(self.espisodePath + "/*" + batteryName))
					logger.debug("Already existent episodes for %s are %d" % (batteryName,alreadyExistent))
					if(force or alreadyExistent == 0 ):
						#here call fast episodes updated
						episodes = self.__seekSwabEpisodes(loaded)
						logger.info("Battery %s has %d episodes" % (batteryName,len(episodes)))
						self.saveZip(self.espisodePath,"swab_"+batteryName,episodes)
					else:
						logger.info("Episodes for battery %s already exists" % batteryName)
				else:
					logger.warning("File %s is invalid as dataframe" % file)
			else:
				logger.debug("Not a file: %s " % file)
		logger.debug("__buildUniformedDataSetFromFolder - end")
		logger.info("Folder read complete. Elapsed %f" %  (time.clock() - tt))
	
	
	def seekEpisodesBlow(self,episodes):
		"""
		episodes: list of dataframe
		"""
		logger.debug("__seekEpisodesBlow - start")
		tt = time.clock()
		dischargeThreshold = -10
		chargeThreshold = 10
		blowInterval = 2
		blows = []
		for episode in episodes:
			firstBlow = None
			lastBlow = None
			
			# select all time-step where the battery is in discharge
			dischargeIndex =  ( 
				episode[
				(episode[self.currentIndex] <= dischargeThreshold)
				].index
			)
			if(dischargeIndex.shape[0] == 0):
				logger.warning("Something wrong. No Discharge")
				continue
			# select all time-step where the battery is in charge
			chargeIndex =  ( 
				episode[
				(episode[self.currentIndex] >= chargeThreshold)
				].index
			)
			if(chargeIndex.shape[0] == 0):
				logger.warning("Something wrong. No charge")
				continue
			
			
			#get the first index in charge
			firstBlow = dischargeIndex[0]
			
			
			#get the first index in charge
			lastBlow = chargeIndex[0]
			
		
			logger.debug("First blow: %s - Last blow: %s" % (firstBlow,lastBlow))
			#self.plot(episode)
			
			dischargeBlowIdx = episode.index.get_loc(firstBlow)
			dischargeBlowCtx = episode.iloc[dischargeBlowIdx-blowInterval:dischargeBlowIdx+blowInterval,:]
			
			#self.plot(dischargeBlowCtx)
			
			chargeBlowIdx = episode.index.get_loc(lastBlow)
			chargeBlowCtx = episode.iloc[chargeBlowIdx-blowInterval:chargeBlowIdx+blowInterval,:]
	
			#self.plot(chargeBlowCtx)
			
			chargeMean = chargeBlowCtx.mean()
			dischargeMean = dischargeBlowCtx.mean()
			
			blows.append([dischargeMean,chargeMean])
	
		logger.info("Current blows %d" % len(blows))
		logger.debug("__seekEpisodesBlow - end. Elapsed %f" %  (time.clock() - tt))
		return blows
	
	def __seekSwabEpisodes(self,df):
		"""
		df is a full dataframe of a battery
		"""
	
		logger.debug("__seekSwabEpisodes - start")
		tt = time.clock()
		
		episodes = []
		contextDiscarded = 0
		debugMax = 0
		# first of all group by day
		groups = [g[1] for g in df.groupby([df.index.year, df.index.month, df.index.day])]
		for dataframe in groups:
			# for every day seek episodes thtat starts and ends with the Swab condition
			
			minimumDischargeDuration = 10
			dischargeThreshold = -10
			swabThreshold = 5
			swabLength = 5
			
			
			# select all timestemps where the battery is in discharge
			dischargeIndex =  ( 
				dataframe[
				(dataframe[self.currentIndex] <= dischargeThreshold)
				].index
			)
			
			
			if(dischargeIndex.shape[0] == 0):
				return episodes
			
			
			
			past = np.roll(dischargeIndex,1) # shift the episode one second behind
			present = np.roll(dischargeIndex,0) # convert in numpy array
			diff = present - past # compute difference indexes
			diff = (diff * 10**-9).astype(int) # convert nanosecond in second

			# keep only index with a gap greater than 1 seconds in order to keep only the first index for discharge
			dischargeStart = dischargeIndex[ (diff >  1 ) ]
			
			logger.debug("Removed consecutive %d " % ( len(present) - len(dischargeStart)  ))

			for ts in dischargeStart:
				#get integer indexing for time step index
				startRow = dataframe.index.get_loc(ts)
				context = dataframe.iloc[startRow-swabLength:startRow,:]
				
				swabContext = context[ 
					(context[self.currentIndex] >= -swabThreshold ) 
					&
					(context[self.currentIndex] <= swabThreshold)

				].shape[0]
									
				#if swab is lesser than swabLength, then discard
				if(swabContext != swabLength):
					contextDiscarded += 1
					continue
				
				# avoid noise
				dischargeContext =  dataframe.iloc[startRow:startRow+minimumDischargeDuration,:]
				dischargeCount = dischargeContext[ 
					(dischargeContext[self.currentIndex] <= dischargeThreshold)

				].shape[0]
				if(dischargeCount != minimumDischargeDuration):
					continue
				# end noise avoidance
				
				#include previous context on episode
				startIndex = startRow-swabLength
				#seek next swab
				seekStartIndex = startRow + minimumDischargeDuration # the first minimumDischargeDuration are for sure in discharge. no need to check swab here
				endIndex = -1
				swabFound = False
				stepCount = 0 # counter in seek
				maxSearch = 3600 #maximun step between one swab and an other
				while not swabFound and stepCount < maxSearch:
					stepCount = stepCount + 1
					startInterval = seekStartIndex + stepCount
					endIntetval = startInterval + swabLength
					interval = dataframe.iloc[startInterval:endIntetval,:]
					swabCount = interval[
						(interval[self.currentIndex] >= -swabThreshold ) 
						&
						(interval[self.currentIndex] <= swabThreshold)
					].shape[0]
					if(swabCount == swabLength):
						swabFound = True
						endIndex = endIntetval
				
				logger.debug("Swabfound: %s count: %d" % (swabFound ,stepCount ))
				
				if(endIndex != -1):
					s = dataframe.index.values[startIndex]
					e = dataframe.index.values[endIndex]
					diff = ((e-s) * 10**-9).astype(int)
					if(diff > maxSearch):
						logger.warning("Inconsistent episode %s - %s" % (s,e))
					else:
						episode = dataframe.iloc[startIndex:endIndex,:]
						episodes.append(episode)
				
		logger.debug("__seekSwabEpisodes - end. Elapsed %f" %  (time.clock() - tt))
		return episodes # TODO return something!
	
	
	
	
	
	#######################################
	def saveZip(self,folder,fileName,data):
		saveFile = os.path.join(folder,fileName)
		logger.debug("Saving %s" % saveFile)
		fp = gzip.open(saveFile,'wb')
		pickle.dump(data,fp,protocol=-1)
		fp.close()
		logger.debug("Saved %s" % saveFile)
	
	def loadZip(self,folder,fileName):
		toLoad = os.path.join(folder,fileName)
		logger.debug("Loading zip %s" % toLoad)
		out = None
		if( os.path.exists(toLoad) ):
			fp = gzip.open(toLoad,'rb') # This assumes that primes.data is already packed with gzip
			out = pickle.load(fp)
			fp.close()
			logger.debug("Loaded zip %s" % fileName)
		else:
			logger.warning("File %s does not exists" % toLoad)
		return out
		
# df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))		