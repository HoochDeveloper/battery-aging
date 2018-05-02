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
import uuid,time,os,logging, six.moves.cPickle as pickle, gzip, pandas as pd, numpy as np #, matplotlib.pyplot as plt
from datetime import datetime

#Module logging
logger = logging.getLogger("Demetra")
logger.setLevel(logging.DEBUG)
#formatter = logging.Formatter('[%(asctime)s %(name)s %(funcName)s %(levelname)s] %(message)s')
formatter = logging.Formatter('[%(name)s][%(levelname)s] %(message)s')
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
	dataTypes = ({ "TSTAMP" : str,"THING" : str,"CONF" : np.int16, "SPEED" : np.float32,
	"S_IBATCB1_CB1" : np.int16,"S_IBATCB2_CB2" : np.int16, "S_IOUTBUR1_BUR1" : np.uint16,
	"S_IOUTBUR2_BUR2" : np.uint16,"S_ITOTCB2_CB2" : np.uint16,"S_TBATCB1_CB1" : np.int16,
	"S_ITOTCB1_CB1" : np.uint16,"S_TBATCB2_CB2" : np.int16,"S_VBATCB1_CB1" : np.uint16,
	"S_VBATCB2_CB2" : np.uint16,"S_VINCB1_CB1" : np.uint16,"S_VINCB2_CB2" : np.uint16,
	"S_CORRBATT_FLG1" : np.int16,"S_TENSBATT_FLG1" : np.float32 })

	dropX = [dataHeader[0],dataHeader[1]] # columns to drop for X
	keepY = [dataHeader[16],dataHeader[17]] # columns to keep for Y
	
	# Attributes
	timeIndex = None
	nameIndex = None 
	
	currentIndex = None
	voltageIndex = None
	
	rootResultFolder = "."
	espisodeFolder = "episodes"
	espisodePath = None
	
	trainsetFolder = None 
	trainsetFile = None
	testsetFolder = None 
	testsetFile = None
	timeSteps = 30
	normalize=True
	
	episodeImageFolder =  os.path.join(rootResultFolder,"images")
	
	#Constructor
	def __init__(self,timeSteps,normalize=True):
		""" 
		Create, if not exists, the result path for storing the episoded dataset
		"""
		self.normalize = normalize
		self.timeSteps = timeSteps
		
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
	def showEpisodes(self):
		for f in os.listdir(self.espisodePath):
			episodes = self.loadZip(self.espisodePath,f)
			for e in range(len(episodes)):
				self.plot(episodes[e])

	
	def loadTrainset(self):
		load = self.loadZip(self.trainsetFolder,self.trainsetFile)
		x_train = load[0]
		y_train = load[1]
		x_valid = load[2]
		y_valid = load[3]
		normalizer = None
		if(self.normalize):
			normalizer = load[4]
		return x_train, y_train, x_valid, y_valid, normalizer
		
	def loadTestset(self):
		load = self.loadZip(self.testsetFolder,self.testsetFile)
		x_test = load[0]
		y_test = load[1]
		normalizer = None
		if(self.normalize):
			normalizer = load[2]
		return x_test, y_test, normalizer
	
	def buildEpisodedDataset(self,dataFolder,force=False):
		""" 
		dataFolder: folder thet contains the raw dataset, every file in folder will be treated as indipendent thing
		force: if True entire results will be created even if already exists
		"""
		tt = time.clock()
		logger.info("buildEpisodedDataset - start")
		
		existingEpisodes = len(os.listdir(self.espisodePath))
		
		if(not force and existingEpisodes > 0):
			logger.info( "Datafolder has already %d espiodes. Force[%s]" % (existingEpisodes,force) )
			return
		
		self.__buildEpisodedDataframesFromFolder(dataFolder,force)
		logger.info("buildEpisodedDataset - end - Elapsed: %f" % (time.clock() - tt))
	
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
		
		logger.debug("Shuffling dataset")
		shuffledX = np.zeros([len(Xall),Xall[0].shape[0],Xall[0].shape[1]])
		shuffledY = np.zeros([len(Yall),Yall[0].shape[0],Yall[0].shape[1]])
		np.random.seed(42)
		shuffled = np.random.permutation(len(Xall))
		for i in range(len(Xall)):
			shuffledX[i] = Xall[shuffled[i]]
			shuffledY[i] = Yall[shuffled[i]]
		
		normalizer = None
		if(self.normalize == True):
			logger.debug("Normalized data will be used")
			shuffledX, normalizer = self.__normalize(shuffledX)
			shuffledY, _ = self.__normalize(shuffledY)	
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
		
		self.saveZip(self.testsetFolder,self.testsetFile,[x_test,y_test,normalizer])
		
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
		
		self.saveZip(self.trainsetFolder,self.trainsetFile,[x_train, y_train, x_valid, y_valid,normalizer])
		
		logger.info("buildLearnSet - end - Elapsed: %f" % (time.clock() - start))
	
	
	def plot(self,data,startIdx=None,endIdx=None,savePath=None):
		#column index of the sequence time index
		dateIndex = self.dataHeader.index(self.timeIndex)
		nameIndex = self.dataHeader.index(self.nameIndex)
		# values to plot
		if(startIdx is None and endIdx is None):
			values = data.values
		else:
			values = data.loc[startIdx:endIdx,:].values
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
			#saveImageName = "%s_%s.png" % (startIdx,endIdx)
			unique_filename = str(uuid.uuid4())
			plt.savefig(os.path.join(batteryImagePath,unique_filename), bbox_inches='tight')
			plt.close()
		
	# private methods
	def __normalize(self,data,minrange=-1,maxrange=1):
		normalizer = np.zeros([data.shape[2],2])
		for i in range(data.shape[2]):
			maxvalue = data[:,:,i].max()
			minvalue = data[:,:,i].min()
			data[:,:,i] = 	((data[:,:,i] - minvalue) / (maxvalue - minvalue)) * (maxrange - minrange) + minrange
			normalizer[i,0] = minvalue		
			normalizer[i,1] = maxvalue		
		return data, normalizer
	
	def __buildEpisodedDataframesFromFolder(self,dataFolder,force=False):
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

		for file in os.listdir(dataFolder):
			if(os.path.isfile(os.path.join(dataFolder,file))):
				loaded = self.__readFileAsDataframe(os.path.join(dataFolder,file))
				if(loaded is not None):
					batteryName = loaded[self.nameIndex].values[0]
					logger.info("Checking episodes for battery %s" % batteryName)
					savePath = os.path.join(self.espisodePath,batteryName)
					if(force or (not os.path.exists(savePath))):
						
						fileEpisodes = self.__fastEpisodeInDataframe(loaded,self.timeSteps)
						if(len(fileEpisodes) > 0):
							self.saveZip(self.espisodePath,batteryName,fileEpisodes)
						else:
							logger.info("No episodes in file")
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
			data.set_index(self.timeIndex,inplace=True,drop=False)
			data.sort_index(inplace=True)
			logger.debug("Data read complete. Elapsed %f second(s)" %  (time.clock() - tt))
		except:
			logger.error("Can't read file %s" % file)
			data = None
		return data
		
		
	def __fastEpisodeInDataframe(self,dataframe,episodeLength):
		tt = time.clock()
		chargeEpisodes = []
		dischargeEpisodes = []
		voltThreshold = 30 
		
		# list of all eligible episodes index (time stamp)
		# select all time steps with i = 0 and v >= threshold
		startinTimeStep =  ( 
			dataframe[(dataframe[self.currentIndex] == 0 ) 
			& 
			(dataframe[self.voltageIndex] >= voltThreshold)].index
		)
		# since a lot of indexes are consecutive (zone of constant voltage and 0 current ) all consecutive starting indexed are dropped
		logger.debug("Dropping plateau")
		# time gap between consecutive time step 
		diff = startinTimeStep - np.roll(startinTimeStep,1)
		# keep only start episodes with a gap greater than 1.5 seconds
		startinTimeStep = startinTimeStep[ (diff.second > 1.5 ) ]
		allCandidates = len(startinTimeStep)
		logger.debug("Found %d candidates" % allCandidates)
		for ts in startinTimeStep:
			startRow = dataframe.index.get_loc(ts)
			episodeCandidate = dataframe.iloc[startRow:startRow+episodeLength,:]
			dischargeCount = episodeCandidate[ episodeCandidate[self.currentIndex] <= 0 ].shape[0]
			if(dischargeCount == (episodeLength)):
				# this is a discharge episodes
				dischargeEpisodes.append(episodeCandidate)
			else:
				chargeCount = episodeCandidate[ episodeCandidate[self.currentIndex] >= 0 ].shape[0]
				if(chargeCount == (episodeLength)):
					# this is a charge episodes
					chargeEpisodes.append(episodeCandidate)

		dischargeFraction = len(dischargeEpisodes) / allCandidates
		chargeFraction = len(chargeEpisodes) / allCandidates
		
		logger.info("Episodes created. Discharge: %s - %f - Charge: %s - %f. Elapsed %f second(s)" %  
						(len(dischargeEpisodes),dischargeFraction, len(chargeEpisodes),chargeFraction, (time.clock() - tt)))
		
		return dischargeEpisodes + chargeEpisodes
		
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