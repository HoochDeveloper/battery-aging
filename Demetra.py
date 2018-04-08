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
	
	#Constructor
	def __init__(self):
		""" 
		Init the TimeSeriesLoader
		"""
		self.timeIndex = self.dataHeader[0]
		self.groupIndex = self.dataHeader[1]
	
	# public methods
	def dataFormatKerasLSTM(self,dataFolder,dataFile=None,force=False,testPerc=0.2):
		tt = time.clock()
		loaded = None
		if(not force):
			loaded = self.__loadDataFrame(dataFile)
		
		trainX = []
		trainY = []
		validX = []
		validY = []
		testX  = []
		testY  = []
		
		if( loaded is None ):
			train = [] #0 is X, 1 is Y
			valid = []
			test = []
			logger.info("Building a new dataframe")
			dataList = self.__loadEpisodedDataset(dataFolder)
			testSize = ceil(len(dataList) * testPerc); 
			trainSize = len(dataList) - testSize
			logger.info("Train: %d - Test: %d"  % (trainSize,testSize))
			padding = range(3600)
			for battery in range(0,len(dataList)):
				dailyEpisodes = []
				logger.info("Building battery %d" % battery)
				for day in range(0,len(dataList[battery])): 
					for hour in range(0,len(dataList[battery][day])):
						df = dataList[battery][day][hour]
						# dropping timestamp and thing
						df = df.drop(columns=["TSTAMP", "THING"])
						df.reset_index(drop=True,inplace=True)
						df = df.reindex(index=padding)
						#df[self.timeIndex].fillna(pd.to_datetime('1900-01-01T00:00:00'),inplace=True)
						df.fillna(0,inplace=True)
						X = df.values#.reshape(df.shape[0],df.shape[1]) 
						# dropping conf and speed from prediction
						df = df.drop(columns=["CONF","SPEED"])
						Y = df.values#.reshape(df.shape[0],df.shape[1])
						dailyEpisodes.append([X,Y])
						#end hour
					#end day
				# add the whole day data to train or test set
				logger.debug("Days %d" % len(dailyEpisodes))
				if(battery < trainSize):
					# adding sample to train data
					#split train and valid
					trainIdx =  ceil(len(dailyEpisodes) * 0.8)
					train.append(dailyEpisodes[:trainIdx])
					valid.append(dailyEpisodes[trainIdx:])
					
				else:
					# adding sample to test data
					logger.debug("test %d" % len(dailyEpisodes))
					test.append(dailyEpisodes)
			#end for battery
			logger.info("Building train: %d" % len(train))
			for i in range(len(train)):
				x = np.array( [ [ [x for x in X ] for X in day[0] ] for day in train[i]  ] )
				y = np.array( [ [ [y for y in Y ] for Y in day[1] ] for day in train[i]  ] )
				logger.info(x.shape)
				logger.info(y.shape)
				trainX.append(x)
				trainY.append(y)
				x = np.array( [ [ [x for x in X ] for X in day[0] ] for day in valid[i]  ] )
				y = np.array( [ [ [y for y in Y ] for Y in day[1] ] for day in valid[i]  ] )
				validX.append(x)
				validY.append(y)
				logger.info(x.shape)
				logger.info(y.shape)
			
			logger.info("Building test: %d" % len(test) )
			for i in range(len(test)):
				x = np.array( [ [ [x for x in X ] for X in day[0] ] for day in test[i]  ] )
				y = np.array( [ [ [y for y in Y ] for Y in day[1] ] for day in test[i]  ] )
				testX.append(x)
				testY.append(y)
				logger.info(x.shape)
				logger.info(y.shape)
			self.__saveDataFrame([trainX, trainY, validX, validY, testX, testY],dataFile)
		else:
			trainX = loaded[0]
			trainY = loaded[1]
			validX = loaded[2]
			validY = loaded[3]
			testX  = loaded[4]
			testY  = loaded[5]

		return trainX, trainY, validX, validY, testX, testY
		

		
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
	
	def __loadEpisodedDataset(self,dataFolder):
		""" 
		Load the whole dataset form specified dataFolder
		grouped in episodes
		Save dataset to specifid file
		list of [battery][day][hour]
		"""
		df = self.__readFolder(dataFolder)
		tsp = TimeSeriesPreprocessing()
		# drop not relevant values form dataframe
		df = tsp.dropIrrelevant(df,self.relevant)
		# scale all dataset
		df = tsp.scale(df,self.dataColumns)
		# subset dataset by battery batteries
		batteriesDf = tsp.groupAndSort(df,self.groupIndex,self.timeIndex,True)
		batteries = []
		for i in range(len(batteriesDf)):
			daily = []
			dailyDf = tsp.groupByDayTimeIndex(batteriesDf[i],self.timeIndex)
			for j in range(len(dailyDf)):
				hourDf = tsp.groupByHourTimeIndex(dailyDf[j],self.timeIndex)
				daily.append(hourDf)
			batteries.append(daily)
		return batteries
	
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
				folderDataframe.append(self.__readFile(os.path.join(dataFolder,file)))
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
		data = pd.read_csv(file, compression='gzip', header=None,error_bad_lines=True,sep=',', 
			names=self.dataHeader,
			dtype=self.dataTypes,
			parse_dates=[self.timeIndex],
			date_parser = pd.core.tools.datetimes.to_datetime)
		logger.info("Data read complete. Elapsed %f second(s)" %  (time.clock() - tt))
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
		tt = time.clock()
		if(groupIndex):
			logger.debug("Grouping by %s", groupIndex)
			grouped = [ group[1] for group in data.groupby(groupIndex) ]
		else:
			logger.debug("No group option specified. Nothing will be done.")
			grouped = [data]
		logger.debug("There are %s groups in dataset. Elapes %s second(s)" % (len(grouped),(time.clock() - tt) ))
		if(sortIndex):
			for idx in range(0,len(grouped)):			
				tt = time.clock()
				logger.info("Sorting data by %s" % sortIndex)
				grouped[idx] = grouped[idx].sort_values(by=sortIndex,ascending=asc)
				logger.info("Data sort complete. Elapsed %f second(s)" %  (time.clock() - tt))
		else:
			logger.debug("No sort option specified. Nothing will be done.")
		return grouped
	
	def groupByDayTimeIndex(self,data,timeIndex):
		""" 
		In:
		data: dataframe to group
		timeIndex: name of column to group by day 
		Out:
		Create a list of dataframe grouped by day for the specified time index.
		"""
		logger.debug("Grouping data by day on column %s" % timeIndex)
		tt = time.clock()
		dayData = [ group[1] for group in data.groupby([data[timeIndex].dt.date]) ]
		logger.debug("There are %s day(s) in current group. Elapes %s second(s)" % (len(dayData),(time.clock() - tt) ))
		return dayData
		
	def groupByHourTimeIndex(self,data,timeIndex):
		""" 
		Create a list of dataframe grouped by hour.
		"""
		logger.debug("Grouping data by hour on column %s" % timeIndex)
		startHour = data[timeIndex].dt.hour.min()
		endHour = data[timeIndex].dt.hour.max()
		logger.debug("Hour starts form %s and span to %s" % (startHour,endHour) )
		tt = time.clock()
		hourData = [ group[1] for group in data.groupby([data[timeIndex].dt.hour]) ]
		logger.debug("There are %s hour(s) in current data. Elapes %s second(s)" % (len(hourData),(time.clock() - tt) ))
		return hourData
		