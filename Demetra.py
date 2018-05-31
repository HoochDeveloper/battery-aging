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
logger = logging.getLogger("Demetra")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
consoleHandler.setLevel(logging.INFO)
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
	logFolder = os.path.join(root,"logs")
	rootResultFolder = os.path.join(root,"results")
	episodeImageFolder =  os.path.join(rootResultFolder,"images")
	espisodeFolder = "episodes"
	espisodePath = None
	episodeBlowPath = None
	normalize = False
	normalizerFolder = "normalizer"
	normalizerPath = None
	normalizerFile = "norm.pkl"
	
		
	#Constructor
	def __init__(self,normalize=True):
		""" 
		Create, if not exists, the result path for storing the episoded dataset
		"""
		
		# creates log folder
		if not os.path.exists(self.logFolder):
			os.makedirs(self.logFolder)
		logPath = self.logFolder + "/Demetra.log"
		rotateHandelr = loghds.TimedRotatingFileHandler(logPath,when="H",interval=6,backupCount=5)
		rotateHandelr.setFormatter(formatter)
		rotateHandelr.setLevel(logging.DEBUG)
		logger.addHandler(rotateHandelr)
		
		if not os.path.exists(self.rootResultFolder):
			os.makedirs(self.rootResultFolder)
		self.normalize = normalize
		self.normalizerPath = os.path.join(self.rootResultFolder, self.normalizerFolder)
		
			
		self.espisodePath = os.path.join(self.rootResultFolder,self.espisodeFolder)
		
		self.episodeBlowPath = os.path.join(self.rootResultFolder,self.espisodeFolder+"_blow")
		
		if not os.path.exists(self.espisodePath):
			os.makedirs(self.espisodePath)
		if not os.path.exists(self.normalizerPath):
			os.makedirs(self.normalizerPath)
		if not os.path.exists(self.episodeImageFolder):
			os.makedirs(self.episodeImageFolder)
		if not os.path.exists(self.episodeBlowPath):
			os.makedirs(self.episodeBlowPath)
		
		self.timeIndex = self.dataHeader[0]
		self.nameIndex = self.dataHeader[1]
		# used for determining when an episode start in charge or discharge
		self.currentIndex = self.dataHeader[16]
		self.voltageIndex = self.dataHeader[17]

		
		logger.debug("Indexes: Current %s Volt %s " % (self.currentIndex,self.voltageIndex))
		
	
	# public methods
	def loadEpisodes(self):
		"""
		Load from files previously created episodes
		"""
		tt = time.clock()
		logger.debug("loadEpisodes - start")
		episodes = []
		for f in os.listdir(self.espisodePath):
			batteryEpisodes = self.__loadZip(self.espisodePath,f)
			episodes += batteryEpisodes
		logger.debug("Loaded %d episodes" % len(episodes))
		logger.debug("loadEpisodes - end - %f" % (time.clock() - tt) )
		return episodes
		
	def loadBlowEpisodes(self):
		"""
		Load from files previously created episodes
		"""
		tt = time.clock()
		logger.debug("loadBlowEpisodes - start")
		episodes = []
		for f in os.listdir(self.episodeBlowPath):
			batteryBlowEpisodes = self.__loadZip(self.episodeBlowPath,f)
			episodes += batteryBlowEpisodes
		logger.debug("Loaded %d episodes blow" % len(episodes))
		logger.debug("loadBlowEpisodes - end - %f" % (time.clock() - tt) )
		return episodes
	
	
	def showEpisodes(self,normalizer = None,limit=2,mode="server"):
		"""
		Show previously created episodes
		normalizer: if provided scale up the data
		limit: max image to show, may be set to None
		mode: if server image will be saved on disk, show otherwise
		"""
		total = 0
		for f in os.listdir(self.espisodePath):
			episodes = self.__loadZip(self.espisodePath,f)
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
				self.plot(episodes[e],mode=mode)
		logger.info("Total %d" % total)

	def loadNormalizer(self):
		"""
		load a previosly created normalizer
		"""
		normalizer = self.__loadZip(self.normalizerPath,self.normalizerFile)
		return normalizer
	
	def plot(self,data,mode="server",name=None):
		"""
		Plot data as is
		mode: if server image will be saved on disk, show otherwise
		name: in server mode if specified save the image with the provided name
		"""
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
		if(mode != "server"):
			plt.show()
		else:
			if(name is None):
				name = batteryName +"_"+str(uuid.uuid4())
			else:
				name = batteryName +"_"+name 
			plt.savefig(os.path.join(self.episodeImageFolder,name), bbox_inches='tight')
			plt.close()
			
	def buildUniformedDataSet(self,dataFolder,force=False):
		""" 
		dataFolder: folder thet contains the raw dataset, every file in folder will be treated as indipendent thing
		force: if True entire results will be created even if already exists
		"""
		tt = time.clock()
		logger.debug("buildUniformedDataSet - start")
		existingEpisodes = len(os.listdir(self.espisodePath))
		if(not force and existingEpisodes > 0):
			logger.info( "Datafolder has already %d episodes." % (existingEpisodes) )
		else:
			logger.info("Building episodes. Force: %s" , force)
			self.__buildUniformedDataSetFromFolder(dataFolder,force)
		
		#if(self.normalize):
		#	normalizer = self.loadNormalizer()
		#	if(normalizer is None):
		#		logger.debug("Building normalizer")
		#		allEpisodes = []
		#		for f in os.listdir(self.espisodePath):
		#			episodeList = self.__loadZip(self.espisodePath,f)
		#			allEpisodes += episodeList
		#		normalizer = self.__getNormalizer(pd.concat(allEpisodes))
		#		self.__saveZip(self.normalizerPath,self.normalizerFile,normalizer)
		#	else:
		#		logger.info("Normalizer already exists")
		
		logger.debug("buildUniformedDataSet - end - %f" % (time.clock() - tt))
	
	def buildBlowDataset(self):
		tt = time.clock()
		logger.debug("buildBlowDataset - start")
		for f in os.listdir(self.espisodePath):
			batteryEpisodes = self.__loadZip(self.espisodePath,f)
			batteryBlows = self.__seekEpisodesBlow(batteryEpisodes)
			if(len(batteryBlows) > 0):
				batteryName = str(f)
				self.__saveZip(self.episodeBlowPath,batteryName,batteryBlows)
		logger.debug("buildBlowDataset - end - %f" % (time.clock() - tt))
	
	# private methods
	def __normalize(self,data,normalizer,minrange=-1,maxrange=1):
		"""
		normalize data in -1,1 with the provided normalizer
		normalizer: list of tupler, first element in the min value for that column; second is the max
		"""
		for i in range(data.shape[2]):
			minvalue = normalizer[i,0]
			maxvalue = normalizer[i,1]
			data[:,:,i] = self.__normalization(data[:,:,i],minvalue,maxvalue,minrange,maxrange)
		return data
	
	def __normalization(self,data,minvalue,maxvalue,minrange,maxrange):
		"""
		apply normalization on given value
		"""
		norm = ((data - minvalue) / (maxvalue - minvalue)) * (maxrange - minrange) + minrange
		return norm
	
	def __getNormalizer(self,data):
		"""
		Compute a normalizer for the data
		"""
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
				
	def __readFileAsDataframe(self,file):
		""" 
		Load data with pandas from the specified csv file
		Parameters: 
			file: csv file to read. Must be compliant with the specified dataHeader
		Output:
			pandas dataframe, if an error occurs, return None
		"""
		tt = time.clock()
		logger.debug("__readFileAsDataframe - start")
		logger.debug("Reading data from %s" %  file)
		try:
			ft = time.clock()
			data = pd.read_csv(file, compression='gzip', header=None,error_bad_lines=True,sep=',', 
				names=self.dataHeader,
				dtype=self.dataTypes,
				parse_dates=[self.timeIndex],
				date_parser = pd.core.tools.datetimes.to_datetime)
			
			logger.debug("Data read complete. Elapsed %f second(s)" %  (time.clock() - ft))
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
		logger.debug("__readFileAsDataframe - end - %f" % (time.clock() - tt))
		return data

	def __buildUniformedDataSetFromFolder(self,dataFolder,force=False):
		""" 
		Read all files in folder as episoded dataframe 
		Every item in the return list is a different thing
		Every inner item is a list of episode for the current thing
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
					savePath = os.path.join(self.espisodePath,batteryName)
					alreadyExistent = len(glob.glob(self.espisodePath + "/*" + batteryName))
					logger.debug("Already existent episodes for %s are %d" % (batteryName,alreadyExistent))
					if(force or alreadyExistent == 0 ):
						episodes = self.__seekSwabEpisodes(loaded)
						self.__saveZip(self.espisodePath,batteryName,episodes)
					else:
						logger.info("Episodes for battery %s already exists" % batteryName)
				else:
					logger.warning("File %s is invalid as dataframe" % file)
			else:
				logger.debug("Not a file: %s " % file)
		logger.debug("__buildUniformedDataSetFromFolder - end - %f" %  (time.clock() - tt))
	
	def __seekSwabEpisodes(self,df):
		"""
		Build list of espisodes starting and ending in swab status
		df: Dataframe of a battery
		"""
		logger.debug("__seekSwabEpisodes - start")
		tt = time.clock()
		# parameter - start
		maxSearch = 7200 #maximun step between one swab and an other
		minimumDischargeDuration = 10 # minimun seconds of discharge after swab, lesser will be discarde as noisy episode
		dischargeThreshold = -10 # current must be lower of this to consider the battery in discharge
		swabThreshold = 5 # current between -th and +th will be valid swab
		swabLength = 3  # timesteps of swab to be considered a valid begin and end of a swab episode
		# parameter - end
		
		episodes = []
		contextDiscarded = 0
		noiseDiscarded = 0
		maxSearchDiscarded = 0
		inconsistent = 0
		# first of all group by day
		groups = [g[1] for g in df.groupby([df.index.year, df.index.month, df.index.day])]
		for dataframe in groups:
			
			maxIdx = dataframe.shape[0]
			# for every day seek episodes thtat starts and ends with the Swab condition
			
			# select all timestemps where the battery is in discharge
			dischargeIndex =  ( 
				dataframe[
				(dataframe[self.currentIndex] <= dischargeThreshold)
				].index
			)
			if(dischargeIndex.shape[0] == 0):
				continue
			
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
					noiseDiscarded += 1
					continue
				# end noise avoidance
				
				#include previous context on episode
				startIndex = startRow-swabLength
				#seek next swab
				seekStartIndex = startRow + minimumDischargeDuration # the first minimumDischargeDuration are for sure in discharge. no need to check swab here
				endIndex = -1
				terminate = False
				stepCount = 0 # counter in seek
				
				while not terminate and stepCount < maxSearch:
					stepCount = stepCount + 1
					startInterval = seekStartIndex + stepCount
					endIntetval = startInterval + swabLength
					#if(endIntetval > maxIdx or startInterval > maxIdx):
					#	terminate = True
					interval = dataframe.iloc[startInterval:endIntetval,:]
					swabCount = interval[
						(interval[self.currentIndex] >= -swabThreshold ) 
						&
						(interval[self.currentIndex] <= swabThreshold)
					].shape[0]
					if(swabCount == swabLength):
						terminate = True
						endIndex = endIntetval
				logger.debug("Swabfound: %s count: %d" % (terminate ,stepCount ))
				
				if(endIndex != -1):
					s = dataframe.index.values[startIndex]
					e = dataframe.index.values[endIndex]
					diff = ((e-s) * 10**-9).astype(int)
					# this is necessary because the are missing intervale between the data
					# e.g. t_0 = 7 o'clock t_1 = 8 o'clock
					# so the episoder is not consistent
					if(diff > maxSearch):
						logger.debug("Inconsistent episode %s - %s" % (s,e))
						inconsistent += 1
					else:
						episode = dataframe.iloc[startIndex:endIndex,:]
						episodes.append(episode)
				else:
					s = dataframe.index.values[startIndex]
					idxe = min(maxIdx-1,startIndex+maxSearch)
					e = dataframe.index.values[idxe]
					diff = ((e-s) * 10**-9).astype(int)
					if(diff > maxSearch):
						inconsistent += 1
					else:
						maxSearchDiscarded += 1
					
		logger.info("------------------------------------------------------")
		logger.info("Valid episodes: %d" % len(episodes))
		logger.info("Maxsearch discard: %d" % maxSearchDiscarded)
		logger.info("Inconsistent discard: %d" % inconsistent)
		logger.info("Noisy discard: %d" % noiseDiscarded)
		logger.info("Context discard %d" % contextDiscarded)
		logger.info("------------------------------------------------------")
		logger.debug("__seekSwabEpisodes - end - %f" %  (time.clock() - tt))
		return episodes 
	
	
	def __seekEpisodesBlow(self,episodes,blowInterval = 3):
		"""
		episodes: list of dataframe
		return a list of tuples of dataframe.
		The first element in the tuple is the discharge blow dataframe
		The secondo element in the tuple is the charge blow dataframe
		"""
		logger.debug("__seekEpisodesBlow - start")
		tt = time.clock()
		dischargeThreshold = -10
		chargeThreshold = 10
		
		blowsEpisodes = []
		count = 0
		for episode in episodes:
			count +=1
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
			
			
			
			chargeBlowIdx = episode.index.get_loc(lastBlow)
			chargeBlowCtx = episode.iloc[chargeBlowIdx-blowInterval:chargeBlowIdx+blowInterval,:]
			
			if(chargeBlowCtx.shape[0] > 0 and dischargeBlowCtx.shape[0] > 0):
				#self.plot(dischargeBlowCtx,name="D"+str(count))
				#self.plot(chargeBlowCtx,name="C"+str(count))
				blowsEpisodes.append([dischargeBlowCtx,chargeBlowCtx])
		
	
		logger.info("Found %d blows" % len(blowsEpisodes))
		logger.debug("__seekEpisodesBlow - end - %f" %  (time.clock() - tt))
		return blowsEpisodes
	
	
	def __saveZip(self,folder,fileName,data):
		saveFile = os.path.join(folder,fileName)
		logger.debug("Saving %s" % saveFile)
		fp = gzip.open(saveFile,'wb')
		pickle.dump(data,fp,protocol=-1)
		fp.close()
		logger.debug("Saved %s" % saveFile)
	
	def __loadZip(self,folder,fileName):
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