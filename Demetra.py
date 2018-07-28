#!/usr/bin/env python

"""
Module for handling data loading
@author: Michele Salvatore Rillo
@email:  michelesalvatore.rillo@gmail.com
@git: HoochDeveloper
Requires: 
	pandas 	(https://pandas.pydata.org/)
	numpy	(http://www.numpy.org/)
	seaborn (https://seaborn.pydata.org/)
"""
#Standard
import uuid,time,os,logging, six.moves.cPickle as pickle, gzip, pandas as pd, numpy as np , matplotlib.pyplot as plt
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
	
	SEP = "---------------------------------------------------------"
	
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
	
	syntheticImport = [dataHeader[1],dataHeader[16],dataHeader[17]] # columns for synthetic import
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
	espisodePath = os.path.join(rootResultFolder,"episodes")
	
	synthetcBlowPath = "synthetic_%d_%d_%d_%d"
	
	
	chargeThreshold = 10 # specify current threshold for the battery to be considered in charge state
	dischargeThreshold = -10 # specify current threshold for the battery to be considered in discharge state
	swabThreshold = 5 # current between -swabThreshold and +swabThreshold will considered in SWAB state
	
	eps1 = 10
	eps2 = 10
	alpha1 = 15
	alpha2 = 15
	
	#Constructor
	def __init__(self,eps1,eps2,alpha1,alpha2):
		""" 
		Create, if not exists, the result path for storing the episoded dataset
		"""
		self.eps1   = eps1
		self.eps2   = eps2
		self.alpha1 = alpha1
		self.alpha2 = alpha2
		
		self.espisodePath = os.path.join(self.rootResultFolder,"episodes_%d_%d_%d_%d" % (self.eps1,self.eps2,self.alpha1,self.alpha2))
		self.synthetcBlowPath = os.path.join(self.rootResultFolder,self.synthetcBlowPath % (self.eps1,self.eps2,self.alpha1,self.alpha2))
		
		# creates log folder
		if not os.path.exists(self.logFolder):
			os.makedirs(self.logFolder)
		if not os.path.exists(self.rootResultFolder):
			os.makedirs(self.rootResultFolder)
		if not os.path.exists(self.espisodePath):
			os.makedirs(self.espisodePath)
		
		logPath = os.path.join(self.logFolder,"Demetra.log")
		rotateHandelr = loghds.TimedRotatingFileHandler(logPath,when="H",interval=6,backupCount=5)
		rotateHandelr.setFormatter(formatter)
		rotateHandelr.setLevel(logging.DEBUG)
		logger.addHandler(rotateHandelr)
		
		self.timeIndex = self.dataHeader[0]
		self.nameIndex = self.dataHeader[1]
		# used for determining when an episode start in charge or discharge
		self.currentIndex = self.dataHeader[16]
		self.voltageIndex = self.dataHeader[17]
		
		logger.debug("Indexes: Current %s Volt %s " % (self.currentIndex,self.voltageIndex))
		
	
	# public methods
	def buildDataSet(self,dataFolder,mode="swab2swab",force=False):
		""" 
		dataFolder: folder that contains the raw dataset, every file in folder will be treated as independent thing;
		force: if True entire results will be created even if already exists;
		mode: specify episode to extract:  swab2swab or swabCleanDischarge
		
		return None, the dataset will be saved in one file per battery, every file contains a list with the following structure
		[monthIndex][episodeInMonthIndex] = dataframe
		"""
		tt = time.clock()
		logger.debug("buildDataSet - start")
		self.__buildDataSetFromFolder(dataFolder,mode,force,self.eps1,self.eps2,self.alpha1,self.alpha2)
		logger.debug("buildDataSet - end - %f" % (time.clock() - tt))
	
	def loadBatteryAsSingleEpisode(self,batteryName):
		tt = time.clock()
		logger.debug("loadBatteryEpisode - start")
		batteryEpisodes = self.__loadZip(self.espisodePath,batteryName+".gz")
		logger.debug("loadBatteryEpisode - end - %f" % (time.clock() - tt) )
		
		return batteryEpisodes
	
	def loadDataSet(self):
		"""
		Load from the files created with buildDataSet
		return: list of dataframes for all the batteries with the following structure
		[batteryIndex][monthIndex][episodeInMonthIndex] = dataframe
		
		Example:
		batteriesEpisodes = self.loadDataSet()
		batteriesEpisodes[0][1][3] #access the dataframe for the first battery, second month, fourth episode
		
		"""
		tt = time.clock()
		logger.debug("loadDataSet - start")
		episodes = []
		if(len(os.listdir(self.espisodePath)) == 0):
			logger.warning("No episodes found, call buildUniformedDataSet first!")
		else:
			for f in os.listdir(self.espisodePath):
				batteryEpisodes = self.__loadZip(self.espisodePath,f)	
				episodes.append(batteryEpisodes)
			logger.debug("Loaded %d episodes" % len(episodes))
		logger.debug("loadDataSet - end - %f" % (time.clock() - tt) )
		return episodes
	
	def seekEpisodesBlows(self,batteryEpisodes,monthIndexes=[],join=True):
		tt = time.clock()
		logger.debug("seekEpisodesBlows - start")
		batteryBlows = self.__seekEpisodesBlow(batteryEpisodes,monthIndexes,join,self.eps1,self.alpha1,self.alpha2)
		logger.debug("seekEpisodesBlows - end - %f" % (time.clock() - tt))
		return batteryBlows 
	
	def loadSyntheticBlowDataSet(self,soc,monthIndexes=[],join=True):
		tt = time.clock()
		logger.debug("loadSyntheticBlowDataSet - start")
		loadPath = self.synthetcBlowPath + "_%d" % soc
		syntheticBatteries = [] # three level list, [battery][month][episode]
		for f in os.listdir(loadPath):
			syntheticBlows = self.__loadZip(loadPath,f)
			syntheticBatteries.append(syntheticBlows)	
		logger.debug("loadSyntheticBlowDataSet - end - %f" % (time.clock() - tt))
		return syntheticBatteries 
	
	def loadBlowDataSet(self,monthIndexes=[],join=True):
		"""
		Load data from the files created with buildDataSet.
		For every episode in battery seek the blow and create a dataframe with discharge blow and charge blow
		monthIndexes: if specified build blows dataset only for the specified months
		join: if True the result is just a dataframe paired discharge and charge blow, otherwise a tuple [dischargeBlow,chargeBlow]
		
		return: list of dataframes for all the batteries with the following structure
		[batteryIndex][monthIndex][episodeInMonthIndex] = [dischargeBlowDataframe,chargeBlowDataframe]
		
		Example:
		e = self.loadBlowDataSet(monthIndex=3)
		print(len(e)) # number of batteries, e[0] gives the first battery 
		print(len(e[0])) #number of data months for battery 0, e[0][0] gives the first month for first battery 
		print(len(e[0][0])) #number of episodes in first month of first battery, e[0][0][0] gives the 1st episode of 1st month of 1st battery
		if(join == False)
			print(e[0][0][0][0].shape) #discharge blow
			print(e[0][0][0][1].shape) #charge blow
			self.plotDataFrame(e[0][0][0][0],mode="GUI",name=None) # plot discharge blow
			self.plotDataFrame(e[0][0][0][1],mode="GUI",name=None) # plot charge blow
		else:
			print(e[0][0][0].shape) #join blow
			self.plotDataFrame(e[0][0][0],mode="GUI",name=None) # plot join blow
		"""
		tt = time.clock()
		logger.debug("loadBlowDataSet - start")
		loadPath = self.espisodePath
		episodes = [] # three level list, [battery][month][episode]
		for f in os.listdir(loadPath):
			batteryEpisodes = self.__loadZip(loadPath,f)
			batteryBlows = self.__seekEpisodesBlow(batteryEpisodes,monthIndexes,join,self.eps1,self.alpha1,self.alpha2)
			episodes.append(batteryBlows)	
		logger.debug("loadBlowDataSet - end - %f" % (time.clock() - tt))
		return episodes 
			
	def dataSetSummary(self,batteries):
		logger.info("Data from %d different batteries" % len(batteries))
		logger.info("Every battery has %d month of data" % len(batteries[0]))
		monthlyEpisodes = np.zeros(len(batteries[0]))
		batt = 0
		for battery in batteries:
			logger.info(self.SEP)
			logger.info("Episode length summary for battery %d" % batt)
			month = 0
			min = np.inf
			max = 0
			totalEpisodeInBattery = 0
			for episodeInMonth in battery:
				totalEpisodeInBattery += len(episodeInMonth)
			
			distribution = np.zeros(totalEpisodeInBattery)
			distroIdx = 0
			for episodeInMonth in battery:
				monthlyEpisodes[month] += len(episodeInMonth)
				monthTotalTimeSteps = 0
				minMonth = np.Inf
				maxMonth = 0
				for episode in episodeInMonth:
					monthTotalTimeSteps += episode.shape[0]
					distribution[distroIdx] = episode.shape[0]
					distroIdx += 1
					if(episode.shape[0] < min):
						min = episode.shape[0]
					if(episode.shape[0] > max):
						max = episode.shape[0]
					if(episode.shape[0] < minMonth):
						minMonth = episode.shape[0]
					if(episode.shape[0] > maxMonth):
						maxMonth = episode.shape[0]

				meanEpisodeTimeSteps = 0
				if(len(episodeInMonth) > 0):
					meanEpisodeTimeSteps = float(monthTotalTimeSteps) / float(len(episodeInMonth))
				logger.info("Month: %d Mean: %f Min: %g Max %g" %  (month,meanEpisodeTimeSteps,minMonth,maxMonth))
				month += 1
			logger.info("Min: %g Max: %g" % (min,max))
			batt +=1
		
		logger.info(self.SEP)
		for month in range(len(batteries[0])):
			logger.info("There are %d episodes in month %d" % (monthlyEpisodes[month], month))
		
	def showEpisodes(self,monthIndex=0,limit=1,mode="server"):
		"""
		Load data from the files created with buildDataSet.
		monthIndex: show only episode for the provided month index
		limit: max image to show, may be set to None
		mode: if server image will be saved on disk, show otherwise
		
		Example:
		self.showEpisodes(monthIndex=2,mode="GUI")
		"""
		folder = self.espisodePath
		total = 0
		for f in os.listdir(folder):
			batteryEpisodes = self.__loadZip(folder,f)
			total += len(batteryEpisodes[monthIndex])
			logger.info("There are %d episodes for %s" % (len(batteryEpisodes[monthIndex]),f))
			max2show = len(batteryEpisodes[monthIndex])
			if(limit is not None):
				max2show = min(limit,len(batteryEpisodes[monthIndex]))
			for e in range(max2show):
				self.plotDataFrame(batteryEpisodes[monthIndex][e],mode=mode)
		logger.info("Total %d" % total)

	
	
	def plotDataFrame(self,data,mode="server",name=None):
		"""
		Plot the input dataframe as is
		mode: in server mode, images will be saved on disk, shown otherwise
		name: in server mode if specified save the image with the provided name
		"""
		if(mode == "server"):
			plt.switch_backend('agg')
			if not os.path.exists(self.episodeImageFolder):
				os.makedirs(self.episodeImageFolder)
		
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
		
		imgTitle = ""
		if(name is None):
			imgTitle = batteryName +"_"+str(uuid.uuid4())
		else:
			imgTitle = batteryName +"_"+name 
		self.plotMode(mode,imgTitle)
		
	def plotMode(self,mode,imgTitle=None,autoClose=False):
		if(mode != "server"):
			blk = True
			if(autoClose == True):
				blk = False
			plt.show(block=blk)
			if(autoClose == True):
				plt.close()
		else:
			if(imgTitle is None):
				name = str(uuid.uuid4())
			else:
				name = imgTitle +"_"+str(uuid.uuid4())
			plt.savefig(os.path.join(self.episodeImageFolder,name), bbox_inches='tight')
			plt.close()
		
	# private methods
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

	def __buildDataSetFromFolder(self,dataFolder,mode,force,eps1,eps2,alpha1,alpha2):
		""" 
		Read all files in folder and save as episode dataframe 
		Return: None
		For every battery creates a file containing a list of dataframes
		"""
		tt = time.clock()
		logger.debug("__buildDataSetFromFolder - begin")
		logger.info("Reading data from folder %s in mode %s" %  (dataFolder,mode))
		if( not os.path.isdir(dataFolder)):
			logger.warning("%s is not a valid folder, nothing will be done" % dataFolder )
			return None
		totalFiles = len(os.listdir(dataFolder))
		count = 0
		for file in os.listdir(dataFolder):
			count = count + 1
			if(os.path.isfile(os.path.join(dataFolder,file))):
				fileName = str(file)
				savePath = os.path.join(self.espisodePath,fileName)
				if(force or not os.path.isfile(savePath)):
					logger.info("Processing file %d of %d" % (count,totalFiles))
					loaded = self.__readFileAsDataframe(os.path.join(dataFolder,fileName))
					if(loaded is not None and loaded.shape[0] > 0):
						self.__seekEpisodes(loaded,fileName,mode,eps1,eps2,alpha1,alpha2)
					else:
						logger.warning("File %s is invalid as dataframe" % fileName)
				else:
					logger.debug("Episodes for battery %s already exists" % fileName)
			else:
				logger.debug("Not a file: %s " % file)
		logger.debug("__buildDataSetFromFolder - end - %f" %  (time.clock() - tt))
	
	def __seekEpisodes(self,df,fileName,mode,eps1,eps2,alpha1,alpha2):
		"""
		Build list of episodes starting and ending in swab status
		df: Dataframe of a battery
		Return: list of dataframe, every dataframe is an episode starting in swab and ending in swab. Episodes may have different time length
		"""
		logger.debug("__seekEpisodes - start")
		tt = time.clock()
		episodes = [] #this list will have an entry for every month of data in battery
		valid = 0
		contextDiscarded = 0
		noiseDiscarded = 0
		inconsistent = 0
		incomplete = 0
		
		# group battery data by month
		groups = [g[1] for g in df.groupby([df.index.year, df.index.month])]
		
		for grp in groups:
			month = grp.index.month[0]
			monthEpisodes,monthSwabDiscarded,monthNoiseDiscarded,monthInconsistent,monthIncomplete = self.__seekInGroup(grp,mode,eps1,eps2,alpha1,alpha2)
			logger.debug("%s: found %d episodes in month %d" % (fileName,len(monthEpisodes),month))
			episodes.append(monthEpisodes)
			valid            += len(monthEpisodes)
			contextDiscarded += monthSwabDiscarded
			noiseDiscarded   += monthNoiseDiscarded
			inconsistent     += monthInconsistent
			incomplete       += monthIncomplete
		
		logger.info("--------------------------")
		logger.info("Valid:        %d" 	% valid)
		logger.info("Inconsistent: %d" 	% inconsistent)
		logger.info("Incomplete    %d" 	% incomplete)
		logger.info("Noisy:        %d" 	% noiseDiscarded)
		logger.info("Context:      %d" 	% contextDiscarded)
		logger.info("--------------------------")

		
		self.__saveZip(self.espisodePath,fileName,episodes)
		
		
		logger.debug("__seekEpisodes - end - %f" %  (time.clock() - tt))
	
	def __seekInGroup(self,dataframe,mode,eps1,eps2,alpha1,alpha2):
		"""
		mode: 	swabCleanDischarge - search clean discharges, starting in swab
				swab2swab - search continuous swab to swab episodes
		"""
		contextDiscarded = 0
		noiseDiscarded = 0
		inconsistent = 0
		incomplete = 0
		groupEpisodes = []
		
		# for every day seek episodes thtat starts and ends with the Swab condition
		
		# select all time-temps where the battery is in discharge
		dischargeIndex =  ( 
			dataframe[
			(dataframe[self.currentIndex] <= self.dischargeThreshold)
			].index
		)
		if(dischargeIndex.shape[0] == 0):
			return groupEpisodes,contextDiscarded,noiseDiscarded,inconsistent,incomplete
		
		past = np.roll(dischargeIndex,1) # shift the episode one second behind
		present = np.roll(dischargeIndex,0) # convert in numpy array
		diff = present - past # compute difference indexes
		diff = (diff * 10**-9).astype(int) # convert nanosecond in second

		# keep only index with a gap greater than 1 seconds in order to keep only the first index for discharge
		dischargeStart = dischargeIndex[ (diff >  1 ) ]
		logger.debug("Removed consecutive %d " % ( len(present) - len(dischargeStart)  ))

		for i in range(1,len(dischargeStart)):
			#get integer indexing for time step index
			
			nextTs = dischargeStart[i]
			ts = dischargeStart[i-1]
			startRow = dataframe.index.get_loc(ts)
			nextRow = dataframe.index.get_loc(nextTs) # if during search we hit this index the episode should be discarde?
			
			rowsInEpisode = nextRow - startRow # this is the maximun number of row in episode
			
			context = dataframe.iloc[startRow-eps1:startRow,:]
			
			swabContext = context[ 
				(context[self.currentIndex] >= -self.swabThreshold ) 
				&
				(context[self.currentIndex] <= self.swabThreshold)

			].shape[0]
								
			#if swab is lesser than eps1, then discard
			if(swabContext != eps1):
				contextDiscarded += 1
				continue
			
			# avoid noise
			dischargeContext =  dataframe.iloc[startRow:startRow+alpha1,:]
			dischargeCount = dischargeContext[ 
				(dischargeContext[self.currentIndex] <= self.dischargeThreshold)

			].shape[0]
			if(dischargeCount != alpha1):
				noiseDiscarded += 1
				continue
			# end noise avoidance
			
			#seek valid episode
			endIndex = -1
			if(mode=="swab2swab"):
				seekStartIndex = startRow + alpha1 # the first alpha1 are for sure in discharge. no need to check swab here
				endIndex = self.__seekSwabEnd(seekStartIndex,dataframe,eps2,alpha2,nextRow)
			elif(mode=="swabCleanDischarge"):
				endIndex = self.__seekCleanDischarge(startRow,dataframe,nextRow,self.dischargeThreshold)
			else:
				# default is swab2swab
				logger.warning("%s is not a valid mode, swab2swab will be used" % mode)
				seekStartIndex = startRow + alpha1 # the first alpha1 are for sure in discharge. no need to check swab here
				endIndex = self.__seekSwabEnd(seekStartIndex,dataframe,eps2,alpha2,nextRow)
			
			#include swab context in episode
			startIndex = startRow-eps1
			
			if(endIndex != -1):
				s = dataframe.index.values[startIndex]
				e = dataframe.index.values[endIndex]
				diff = ((e-s) * 10**-9).astype(int)
				# this is necessary because the are missing intervale between the data
				# e.g. t_0 = 7 o'clock t_1 = 8 o'clock
				# so the episoder is not consistent
				if(diff > rowsInEpisode):
					logger.debug("Inconsistent episode %s - %s" % (s,e))
					inconsistent += 1
				else:
					episode = dataframe.iloc[startIndex:endIndex,:]
					groupEpisodes.append(episode)
			else:
				incomplete += 1
		
		return groupEpisodes,contextDiscarded,noiseDiscarded,inconsistent,incomplete
	
	def __seekSwabEnd(self,seekStartIndex,dataframe,eps2,alpha2,nextRow):
		"""
		Starting from startIndex, look in dataframe for the next swab state untill nextRow
		"""
	
		endIndex = -1
		terminate = False
		stepCount = 0 # counter in seek
		chargeAlpah2 = False
		while not terminate and (seekStartIndex + stepCount) < nextRow:
			stepCount = stepCount + 1
			startInterval = seekStartIndex + stepCount
			endIntetval = startInterval + eps2
			
			#search inside the episode for charge greater than alpha2
			if not chargeAlpah2:
				episodeInterval = dataframe.iloc[seekStartIndex:startInterval,:]
				chargeCount = episodeInterval[
					(episodeInterval[self.currentIndex] >= self.chargeThreshold)
				].shape[0]
				if( chargeCount >= alpha2 ):
					chargeAlpah2 = True
			
			interval = dataframe.iloc[startInterval:endIntetval,:]
			swabCount = interval[
				(interval[self.currentIndex] >= -self.swabThreshold ) 
				&
				(interval[self.currentIndex] <= self.swabThreshold)
			].shape[0]
			
			if(swabCount == eps2 and chargeAlpah2):
				terminate = True
				endIndex = endIntetval
		
		logger.debug("Swabfound: %s count: %d" % (terminate ,stepCount ))
		return endIndex
	
	def __seekCleanDischarge(self,seekStartIndex,dataframe,nextRow):
		endIndex = -1
		terminate = False
		stepCount = 0 # counter in seek
		startInterval = seekStartIndex
		while not terminate and (seekStartIndex + stepCount) < nextRow:
			stepCount = stepCount + 1
			
			endIntetval = startInterval + stepCount
			
			interval = dataframe.iloc[startInterval:endIntetval,:]
			endDischargeCount = interval[
				(interval[self.currentIndex] > self.dischargeThreshold)
			].shape[0]
			if(endDischargeCount > 0):
				terminate = True
				endIndex = endIntetval-1
			else:
				endIndex = endIntetval
		
		logger.debug("Clean Discharge: %s duration: %d" % (terminate ,stepCount ))
		return endIndex
	
	
	
	
	def __seekEpisodesBlow(self,episodes,monthIndexes,join,eps1,alpha1,alpha2):
		"""
		episodes: list of list of dataframe, outer index is the month, inner index is the episode
		monthIndexes: indexes of the month(s) of interest, if empty all available month will be returned
		join: if True the result is just a dataframe paired discharge and charge blow, otherwise a tuple [dischargeBlow,chargeBlow]
		eps1: starting swab duration (sec)
		alpha1: minimum discharge duration (sec) after swab
		alpha2: minimum charge duration (sec) after discharge
		
		return a list of tuples of dataframe.
		The first element in the tuple is the discharge blow dataframe
		The second element in the tuple is the charge blow dataframe
		"""
		logger.debug("__seekEpisodesBlow - start")
		tt = time.clock()
		blowsEpisodes = []
		if(len(monthIndexes) == 0):
			for month in episodes:
				monthlyBlow = []
				for episode in month:
					b = self.__getBlow(episode,join,eps1,alpha1,alpha2)
					if(b is not None):
						monthlyBlow.append(b)
				blowsEpisodes.append(monthlyBlow)
		else:
			for m in monthIndexes:
				month = episodes[m]
				monthlyBlow = []
				for episode in month:
					b = self.__getBlow(episode,join,eps1,alpha1,alpha2)
					if(b is not None):
						monthlyBlow.append(b)
				blowsEpisodes.append(monthlyBlow)
			
		logger.debug("__seekEpisodesBlow - end - %f" %  (time.clock() - tt))
		return blowsEpisodes
	
	
	def __getBlow(self,episode,join,eps1,alpha1,alpha2):
		"""
		episode: to search blow in
		blowInterval: how much timestep to get before and after the blow
		join: if True the result is just a dataframe paired discharge and charge blow, otherwise a tuple [dischargeBlow,chargeBlow]
		eps1: starting swab duration (sec)
		alpha1: minimum discharge duration (sec) after swab
		alpha2: minimum charge duration (sec) after discharge
		"""
		firstBlow = None
		lastBlow = None
		
		# select all time-step where the battery is in discharge
		dischargeIndex =  ( 
			episode[
			(episode[self.currentIndex] <= self.dischargeThreshold)
			].index
		)
		if(dischargeIndex.shape[0] == 0):
			logger.debug("Something wrong. No Discharge")
			return None
		# select all time-step where the battery is in charge
		chargeIndex =  ( 
			episode[
			(episode[self.currentIndex] >= self.chargeThreshold)
			].index
		)
		if(chargeIndex.shape[0] == 0):
			logger.debug("Something wrong. No charge")
			return None
		
		
		#get the first index in discharge
		firstBlow = dischargeIndex[0]
		
		
		#get the first index in charge
		lastBlow = chargeIndex[0]
		
	
		#logger.debug("First blow: %s - Last blow: %s" % (firstBlow,lastBlow))
		#self.plotDataFrame(episode)
		
		dischargeBlowIdx = episode.index.get_loc(firstBlow)
		dischargeBlowCtx = episode.iloc[dischargeBlowIdx-eps1:dischargeBlowIdx+alpha1,:]
		
		chargeBlowIdx = episode.index.get_loc(lastBlow)
		chargeBlowCtx = episode.iloc[chargeBlowIdx-alpha1:chargeBlowIdx+alpha2,:]
		
		if(chargeBlowCtx.shape[0] > 0 and dischargeBlowCtx.shape[0] > 0):
			if join:
				return pd.concat([dischargeBlowCtx,chargeBlowCtx])
			else:
				return [dischargeBlowCtx,chargeBlowCtx]
		else:
			return None
		
	def saveZip(self,folder,fileName,data):
		return self.__saveZip(folder,fileName,data)
	
	def loadZip(self,folder,fileName):
		return self.__loadZip(folder,fileName)
	
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
	