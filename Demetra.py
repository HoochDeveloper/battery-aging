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
import time,os,logging, matplotlib.pyplot as plt, six.moves.cPickle as pickle, gzip
from datetime import datetime
import pandas as pd
import numpy as np
from math import sqrt,ceil,trunc
from random import randint, shuffle
from sklearn.preprocessing import MinMaxScaler

#Module logging
logger = logging.getLogger("Demetra")
logger.setLevel(logging.DEBUG)
#formatter = logging.Formatter('[%(asctime)s %(name)s %(funcName)s %(levelname)s] %(message)s')
formatter = logging.Formatter('[%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

class TimeSeriesDataset():
	
	# Attributes
	timeIndex = None
	groupIndex = None 
	timesteps = 3600
	
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

	dataColumns = dataHeader[2:] # column with valid data
	
	relevant = dataHeader[4:10] # relevant columns
	tsp = None
	rootFolder = "."
	resultFolder = "out"
	testPerc = 0.2
	outFolder = None
	
	#Constructor
	def __init__(self):
		""" 
		Init the TimeSeriesLoader, set the timeindex and the groupindex
		"""
		self.outFolder = os.path.join(self.rootFolder,self.resultFolder)
		if not os.path.exists(self.outFolder):
			os.makedirs(self.outFolder)
		self.timeIndex = self.dataHeader[0]
		self.groupIndex = self.dataHeader[1]
		self.tsp = TimeSeriesPreprocessing(self.outFolder)
		
	
	# public methods
	def supervisedData4KerasLSTM(self,dataFolder,force=False):
		""" 
		Format the data to be used by Keras LSTM in batch fashion
		"""
		tt = time.clock()
		loaded = None
		#if(not force):
		#	loaded = self.__loadDataFrame(dataFile)		
		if( loaded is None ):
			logger.info("Building supervised episodes")
			#self.__supervisedEpisodes(dataFolder)
			logger.info("Builded supervised episodes %f" % (time.clock() - tt))
			
	def plot(self,data):
		#column index of the sequence time index
		dateIndex = self.dataHeader.index(self.timeIndex)
		groupIndex = self.dataHeader.index(self.groupIndex)
		# values to plot
		values = data.values
		# getting YYYY-mm-dd for the plot title
		date =  values[:, dateIndex][0].strftime("%Y-%m-%d")
		batteryName =  values[:, groupIndex][0]
		#time series for all data that we want to plot
		# plot each column except TSTAMP and THING(wich is constant for the same battery)
		toPlot = range(2,18)
		i = 1
		plt.figure()
		plt.suptitle("Data for battery %s in day %s " % (batteryName,date), fontsize=16)
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
		frequency = int(len(timeLabel) / 40)
		plt.xticks(xs[::frequency], timeLabel[::frequency])
		plt.xticks(rotation=90)
		plt.show()
	
	# private methods
	
	def __supervisedEpisodes(self,dataFolder):
		""" 
		Load the whole dataset form specified dataFolder
		grouped in episodes
		Save dataset to specifid file
		list of [battery][day][hour]
		"""
		#df = None # remove me
		#df = self.__readFolder(dataFolder)
		# drop not relevant values form dataframe
		#df = self.tsp.dropIrrelevant(df,self.relevant)
		# scale all dataset
		#df = self.tsp.scale(df,self.dataColumns)
		# subset dataset by battery batteries, outputlfiles
		#self.tsp.groupAndSort(df,self.groupIndex,self.timeIndex,True)
		#
		
		#things = len(os.listdir(self.tsp.outFolder)
		#testSize = ceil(totalSize) * self.testPerc); 
		#trainSize = totalSize - testSize
		#logger.info("Train: %d - Test: %d"  % (trainSize,testSize))
		
		padding = range(self.timesteps)
		for f in  os.listdir(self.outFolder):
			currentBattery = self.tsp.loadZip(f)
			days = []
			dailyDf = self.tsp.groupByDayTimeIndex(currentBattery,self.timeIndex)
			for j in range(len(dailyDf)):
				hourDfList = self.tsp.groupByHourTimeIndex(dailyDf[j],self.timeIndex)
				for h in range(len(hourDfList)):
					hourDf = hourDfList[h]
					hourDf.drop(columns=["TSTAMP", "THING"],inplace=True)
					
					if(hourDf.shape[0] != self.timesteps):
						logger.debug("Rows are different than %s. Padding hour %s in dat %s" % (self.timesteps,h,j))
						hourDf.reset_index(drop=True,inplace=True)
						hourDf = hourDf.reindex(index=padding)
						# after reindex is necessary to put in NA a out of range value
						hourDf.fillna(-2,inplace=True)
						#df[self.timeIndex].fillna(pd.to_datetime('1900-01-01T00:00:00'),inplace=True)
					
					X = hourDf.values
					hourDf.drop(columns=["CONF","SPEED"],inplace=True)
					Y = hourDf.values
					days.append([X,Y])
			x = np.array( [ [ [x for x in X ] for X in day[0] ] for day in days  ] )
			logger.debug(x.shape)
			y = np.array( [ [ [y for y in Y ] for Y in day[1] ] for day in days  ] )
			logger.debug(y.shape)
			self.tsp.saveZip(f,[x,y])
	
	def __readFolder(self,dataFolder):
		""" Read all files in folder as one pandas dataframe """
		tt = time.clock()
		logger.info("Reading data from folder %s" %  dataFolder)
		if( not os.path.isdir(dataFolder)):
			logger.warning("%s is not a valid folder, nothing will be done")
			return None
		folderDataframe = []
		for file in os.listdir(dataFolder):
			if(os.path.isfile(os.path.join(dataFolder,file))):
				loaded = self.__readFile(os.path.join(dataFolder,file))
				if(loaded is not None):
					folderDataframe.append(loaded)
		logger.info("Folder read complete. Elapsed %f second(s)" %  (time.clock() - tt))
		return pd.concat( folderDataframe )
		
	def __readFile(self,file):
		""" 
		Load data with pandas from the specified dataSource file
		Parameters: 
			file: file to read dataset from
		Output:
			pandas dataframe
		"""
		tt = time.clock()
		logger.info("Reading data from %s" %  file)
		try:
			data = pd.read_csv(file, compression='gzip', header=None,error_bad_lines=True,sep=',', 
				names=self.dataHeader,
				dtype=self.dataTypes,
				parse_dates=[self.timeIndex],
				date_parser = pd.core.tools.datetimes.to_datetime)
			logger.info("Data read complete. Elapsed %f second(s)" %  (time.clock() - tt))
		except:
			logger.error("Can't read file %s" % file)
			data = None
		return data
		
	def __saveDataFrame(self,data,saveFile=None):
		""" Save dataframe to a gzip file """
		if(saveFile):
			tt = time.clock()
			logger.info("Saving dataframe to %s" % saveFile)
			fp = gzip.open(saveFile,'wb')
			pickle.dump(data,fp,protocol=-1)
			fp.close()
			logger.info("Dataframe saved to %s. Elapsed %f second(s)" % (saveFile, (time.clock() - tt)))
		else:
			logger.debug("No save file specified, nothing to do.")
			
	def __loadDataFrame(self,dataFile=None):
		""" Load a previous saved dataframe from gzip file """
		out = None
		if(dataFile and os.path.isfile(dataFile)):
			tt = time.clock()
			logger.info("Loading data from %s" % dataFile)
			fp = gzip.open(dataFile,'rb') # This assumes that primes.data is already packed with gzip
			out=pickle.load(fp)
			fp.close()
			logger.info("Data loaded from %s. Elapsed %f second(s)" % ( dataFile, (time.clock() - tt) ) ) 
		else:
			logger.info("No data file specified, nothing to do.")
		return out

class TimeSeriesPreprocessing():
		
	outFolder = None
	
	def __init__(self,outFolder):
		self.outFolder = outFolder
		
	def dropIrrelevant(self,data,relevant,threshold=0.001):
		"""
		Drop from data all rows that have a value lesser than threshold
		in relevant columns
		"""
		logger.info("Dropping row with %s lesser than %f" % (relevant,threshold) )
		tt = time.clock()
		data = data.loc[(data[relevant] >= threshold).any(axis=1)] # axis 1 tells to check in rows
		logger.info("Dropping row completed in %f second(s)" % (time.clock() - tt))
		return data

		
	def scale(self,data,normalization,minRange=-1,maxRange=1):
		""" 
		scale the specified columns in the min - max range 
		Parameters:
			data: dataframe wich columns will be scaled
			normalization: array of columns to scale
		Output:
			dataframe with specified columns scaled
		"""
		logger.info("Data normalization in (%f,%f)" % (minRange,maxRange) )
		tt = time.clock()
		for i in normalization:
			logger.debug("Normalizing %s" % i)
			max_value = data[i].max()
			min_value = data[i].min()
			data[i] = (data[i] - min_value) / (max_value - min_value)
			data[i] = data[i] * (maxRange - minRange) + minRange 
		logger.info("Normalization completed in %f second(s)" % (time.clock() - tt))
		return data
	
	def groupAndSort(self,data,groupIndex=None,sortIndex=None,asc=True):
		""" 
		In:
		data: dataframe to group and sort
		groupIndex: column name to group by
		sortIndex: column name to sort by
		asc: perform sorting ascending, otherwise descending
		Out:
		list of dataframe grouped by groupIndex and sorted by sortIndex
		"""
		#tt = time.clock()
		#if(groupIndex):
		#	logger.debug("Grouping by %s", groupIndex)
		#	count = 1
		#	for group in data.groupby(groupIndex,sort=False,squeeze=True):
		#		self.saveZip("THING_"+str(count)+".gzip",group[1])
		#		count += 1
		#	logger.debug("End save")
		#else:
		#	logger.debug("No group option specified. Nothing will be done.")
		#	self.saveZip("wholeDataset.gzip",data)
		#	
		#logger.debug("There are %s things in dataset. Elapes %s second(s)" % (count,(time.clock() - tt) ))
		
		if(sortIndex):
			tt = time.clock()
			for file in os.listdir(self.outFolder):
				df2sort = self.loadZip(file)
				df2sort.sort_values(by=sortIndex,ascending=asc,inplace=True)
				self.saveZip(file,df2sort)
			logger.debug("Data sort complete. Elapsed %f second(s)" %  (time.clock() - tt))	
		else:
			logger.debug("No sort option specified. Nothing will be done.")
		
		# prepare the dataframe
		
	def groupByDayTimeIndex(self,data,timeIndex):
		""" 
		In:
		data: dataframe to group
		timeIndex: name of column to group by day 
		Out:
		Create a list of dataframe grouped by day for the specified time index.
		"""
		#logger.debug("Grouping data by day on column %s" % timeIndex)
		#tt = time.clock()
		dayData = [ group[1] for group in data.groupby([data[timeIndex].dt.date]) ]
		#logger.debug("There are %s day(s) in current group. Elapes %s second(s)" % (len(dayData),(time.clock() - tt) ))
		return dayData
		
	def groupByHourTimeIndex(self,data,timeIndex):
		""" 
		Create a list of dataframe grouped by hour for the specified timeIndex.
		"""
		#logger.debug("Grouping data by hour on column %s" % timeIndex)
		#startHour = data[timeIndex].dt.hour.min()
		#endHour = data[timeIndex].dt.hour.max()
		#logger.debug("Hour starts form %s and span to %s" % (startHour,endHour) )
		#tt = time.clock()
		hourData = [ group[1] for group in data.groupby([data[timeIndex].dt.hour]) ]
		#logger.debug("There are %s hour(s) in current data. Elapes %s second(s)" % (len(hourData),(time.clock() - tt) ))
		return hourData
		
	def saveZip(self,fileName,data):
		saveFile = os.path.join(self.outFolder,fileName)
		logger.debug("Saving %s" % saveFile)
		fp = gzip.open(saveFile,'wb')
		pickle.dump(data,fp,protocol=-1)
		fp.close()
		logger.debug("Saved %s" % saveFile)
		
	def loadZip(self,fileName):
		logger.debug("Loading zip %s" % fileName)
		toLoad = os.path.join(self.outFolder,fileName)
		fp = gzip.open(toLoad,'rb') # This assumes that primes.data is already packed with gzip
		out=pickle.load(fp)
		fp.close()
		logger.debug("Loaded zip %s" % fileName)
		return out
		