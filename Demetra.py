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
import uuid
import time,os,logging, matplotlib.pyplot as plt, six.moves.cPickle as pickle, gzip
from datetime import datetime
import pandas as pd
import numpy as np
from math import sqrt,ceil,trunc
from sklearn.preprocessing import MinMaxScaler

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
	groupIndex = None 
	
	currentIndex = None
	voltageIndex = None
	
	
	rootResultFolder = "."
	resultFolder = "episodes"
	resultPath = None
	
	timeSteps = 30
	
	episodeImageFolder =  os.path.join(rootResultFolder,"images")
	
	#Constructor
	def __init__(self):
		""" 
		Create, if not exists, the result path for storing the episoded dataset
		"""
		self.resultPath = os.path.join(self.rootResultFolder,self.resultFolder)
		if not os.path.exists(self.resultPath):
			os.makedirs(self.resultPath)
		self.timeIndex = self.dataHeader[0]
		self.groupIndex = self.dataHeader[1]
		# used for determing when an episode start, charge and discharge
		self.currentIndex = self.dataHeader[16]
		self.voltageIndex = self.dataHeader[17]
	
	def loadTrainSet(self):
		load = self.loadZip("scaled","XYT_XYV")
		
		x_train = load[0]
		y_train = load[1]
		x_valid = load[2]
		y_valid = load[3]
		
		return x_train, y_train, x_valid, y_valid
	
	def scaleTrainSet(self,validPerc=0.2):
		"""
		build x_train, y_train, x_valid, y_valid
		"""
		
		Xall = []
		Yall = []
		logger.info("Build lists X,Y")
		for f in os.listdir(self.resultPath):
			episodeList = self.loadZip(self.resultPath,f)
			for e in range(len(episodeList)):
				x = episodeList[e].drop(columns=self.dropX)
				y = episodeList[e][self.keepY]
				Xall.append(x)
				Yall.append(y)
		
		
		logger.info("Scaling")
		minRange=-1
		maxRange=1
		scaler = self.getScaleDict(Xall)
		for i in range(len(Xall)):
			logger.info("Scaling X %d of %d" % (i+1,len(Xall)))
			for k in scaler.keys():
				min_value = scaler[k][0]
				max_value = scaler[k][1]
				Xall[i][k] = (Xall[i][k] - min_value) / (max_value - min_value)
				Xall[i][k] =  Xall[i][k] * (maxRange - minRange) + minRange 
			logger.info("Scaling Y %d of %d" % (i+1,len(Xall)))
			for k in self.keepY:
				min_value = scaler[k][0]
				max_value = scaler[k][1]
				Yall[i][k] = (Yall[i][k] - min_value) / (max_value - min_value)
				Yall[i][k] =  Yall[i][k] * (maxRange - minRange) + minRange 
				
		
		logger.info("Build shuffled")
		shuffledX = np.zeros([len(Xall),Xall[0].shape[0],Xall[0].shape[1]])
		shuffledY = np.zeros([len(Yall),Yall[0].shape[0],Yall[0].shape[1]])
		logger.info(shuffledX.shape)
		logger.info(shuffledY.shape)
		
		np.random.seed(42)
		shuffled = np.random.permutation(len(Xall))
		for i in range(len(Xall)):
			shuffledX[i] = Xall[shuffled[i]]
			shuffledY[i] = Yall[shuffled[i]]
		
		validStartIdx = int( len(Xall) * (1 - validPerc) )
		x_train = shuffledX[:validStartIdx]
		y_train = shuffledY[:validStartIdx]
		x_valid = shuffledX[validStartIdx:]
		y_valid = shuffledY[validStartIdx:]
		
		self.saveZip("scaled","XYT_XYV",[x_train, y_train, x_valid, y_valid])
		
		return x_train, y_train, x_valid, y_valid
	
	
	# public methods
	def timeSeries2relevantEpisodes(self,dataFolder,force=False):
		""" 
		dataFolder: folder thet contains the raw dataset, every file in folder will be treated as indipendent thing
		force: if True entire results will be created even if already exists
		"""
		tt = time.clock()
		logger.info("timeSeries2relevantEpisodes - start")
		self.__readFolderAsEpisodedDataframes(dataFolder,force)
		logger.info("timeSeries2relevantEpisodes - end - Elapsed: %f" % (time.clock() - tt))
			
	def plot(self,data,startIdx=None,endIdx=None,savePath=None):
		#column index of the sequence time index
		dateIndex = self.dataHeader.index(self.timeIndex)
		groupIndex = self.dataHeader.index(self.groupIndex)
		# values to plot
		if(startIdx is None and endIdx is None):
			values = data.values
		else:
			values = data.loc[startIdx:endIdx,:].values
		# getting YYYY-mm-dd for the plot title
		date =  values[:, dateIndex][0].strftime("%Y-%m-%d")
		batteryName =  values[:, groupIndex][0]
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
	
	
	
	def __readFolderAsEpisodedDataframes(self,dataFolder,force=False):
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
					batteryName = loaded[self.groupIndex].values[0]
					logger.info("Checking episodes for battery %s" % batteryName)
					savePath = os.path.join(self.resultPath,batteryName)
					if(force or (not os.path.exists(savePath))):
						fileEpisodes = self.__findEpisodeInDataframe(loaded,self.timeSteps)
						logger.info("Episodes are %d" % len(fileEpisodes))
						if(len(fileEpisodes) > 0):
							self.saveZip(self.resultPath,batteryName,fileEpisodes)
							
						else:
							logger.info("No episodes in file")
					else:
						logger.info("Episodes for battery %s already exists" % batteryName)
		logger.info("Folder read complete. Elapsed %f" %  (time.clock() - tt))
		# TODO load all dataframe from file?
		
	def __readFileAsDataframe(self,file):
		""" 
		Load data with pandas from the specified csv file
		Parameters: 
			file: csv file to read. Must be compliant with the specified dataHeader
		Output:
			pandas dataframe, if an error occurs, return None
		"""
		tt = time.clock()
		logger.info("Reading data from %s" %  file)
		try:
			data = pd.read_csv(file, compression='gzip', header=None,error_bad_lines=True,sep=',', 
				names=self.dataHeader,
				dtype=self.dataTypes,
				parse_dates=[self.timeIndex],
				date_parser = pd.core.tools.datetimes.to_datetime)
			data.set_index(self.timeIndex,inplace=True,drop=False)
			data.sort_index(inplace=True)
			logger.info("Data read complete. Elapsed %f second(s)" %  (time.clock() - tt))
		except:
			logger.error("Can't read file %s" % file)
			data = None
		return data
		
	def __findEpisodeInDataframe(self,dataframe,episodeLength=30):
		"""
		Creates episoed of episodeLength sec length where the series starts at zero current and same voltage.
		The series musth be 60sec complete of pure charge or discharge.
		"""
		tt = time.clock()
		episodes = []
		maxVoltage = 30 #int(np.max(dataframe[self.voltageIndex].values))
		logger.debug("Max integer voltage %s" % maxVoltage)
		episodeStart = False
		
		dischargeCount = 0
		chargeCount = 0
		startIndex = None
		endIndex = None
		for index, row in dataframe.iterrows():

			i = row[self.currentIndex]
			v = int(row[self.voltageIndex])
			
			# CHECK DISCHARGE
			if( (i < 0 and episodeStart) # start discharging
				or
				(i < 0 and dischargeCount > 0) # continue discharging
			): 
				dischargeCount += 1
				if(dischargeCount == episodeLength):
					#logger.debug("Complete discharge at index %s to %s" % (startIndex , index))
					#self.plot(dataframe,startIndex,index,self.episodeImageFolder)
					episodes.append(dataframe.loc[startIndex:index,:])
					startIndex = None
					dischargeCount = 0
					
			else:
				dischargeCount = 0
			# CHECK CHARGE
			if( (i > 0 and episodeStart) # start discharging
				or
				(i > 0 and chargeCount > 0) # continue discharging
			): 
				chargeCount += 1
				if(chargeCount == episodeLength):
					#logger.debug("Complete charge at index %s to %s" % (startIndex , index))
					#self.plot(dataframe,startIndex,index,self.episodeImageFolder)
					episodes.append(dataframe.loc[startIndex:index,:])
					startIndex = None
					chargeCount = 0
				
			else:
				chargeCount = 0
			
			# CHECK EPISODE START
			if( v == maxVoltage and i == 0 ):
				#logger.debug("Starting episode at index %s" % index)
				startIndex = index
				episodeStart = True
			else:
				episodeStart = False
		logger.info("Episodes created. Elapsed %f second(s)" %  (time.clock() - tt))
		return episodes
		
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
		
	def getScaleDict(self,list,minRange=-1,maxRange=1):
		""" 
		scale the specified columns in the min - max range 
		Parameters:
			data: dataframe wich columns will be scaled
		Output:
			dataframe with specified columns scaled
		"""
		data = pd.concat(list)
		scaleDict = {}
		tt = time.clock()
		for i in data.columns:
			max_value = data[i].max()
			min_value = data[i].min()
			scaleDict[i] = [min_value,max_value]
		return scaleDict
		

		
	
		
	
		