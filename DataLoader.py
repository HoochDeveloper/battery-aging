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

#Module logging
logger = logging.getLogger("DataLoader")
logger.setLevel(logging.DEBUG)
#formatter = logging.Formatter('[%(asctime)s %(name)s %(funcName)s %(levelname)s] %(message)s')
formatter = logging.Formatter('[%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

class TimeSeriesDataset():
	
	# Attributes
	timeIndex = None # data attribute for time index in data loaded

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

	normalization = dataHeader[2:]
	
	#Constructor
	def __init__(self,timeIdx):
		""" 
		Init the TimeSeriesLoader to work with the specified index as datetime index 
		"""
		self.timeIndex = timeIdx
	
	# public methods
	def load(self,dataFolder,dataFile=None,force=False):
		""" 
		Load the whole dataset form specified dataFolder
		Save dataset to specifid file
		"""
		df = self.__loadDataFrame(dataFile) # try to load existing dataframe
		if( not df or force ): # if no df found or force option, reload data from folder and save in file
			df = self.__readFolder(dataFolder)
			self.__saveDataFrame(df,dataFile)
		return df
		
	def plot(self,data):
		#column index of the sequence time index
		dateIndex = self.dataHeader.index(self.timeIndex)
		# values to plot
		values = data.values
		# getting YYYY-mm-dd for the plot title
		date =  values[:, dateIndex][0].strftime("%Y-%m-%d")
		batteryName =  values[:, 1][0]
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
	def __readFolder(self,dataFolder):
		""" Read all files in folder as pandas dataframe """
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
			file: file to read from
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
		""" Save dataframe to gzip file, if specified """
		if(saveFile):
			logger.info("Saving dataframe to %s" % saveFile)
			fp = gzip.open(saveFile,'wb')
			pickle.dump(data,fp,protocol=-1)
			fp.close()
			logger.info("Dataframe saved to %s" % saveFile)
		else:
			logger.debug("No save file specified, nothing to do.")
			
	def __loadDataFrame(self,dataFile=None):
		""" Load a previous saved dataframe (gzip) """
		out = None
		if(dataFile and os.path.isfile(dataFile)):
			logger.info("Loading data from %s" % dataFile)
			fp = gzip.open(dataFile,'rb') # This assumes that primes.data is already packed with gzip
			out=pickle.load(fp)
			fp.close()
			logger.info("Data loaded from %s" % dataFile)
		else:
			logger.info("No data file specified, nothing to do.")
		return out

class TimeSeriesPreprocessing():
	
	timeIndex = None
	groupIndex = None
	
	def __init__(self,timeIndex,groupIndex):
		self.timeIndex = timeIndex
		self.groupIndex = groupIndex
	
	def normalize(self,data,normalization,minRange=-1,maxRange=1):
		""" 
		Normalize the specified columns in the min - max range 
		Parameters:
			data: dataframe to normalize
			normalization: array of columns to normalize
		Output:
			dataframe with specified columns normalized
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
	
	def groupDataFrame(self,data):
		""" 
		In:
		dat: dataframe to group
		Out:
		list of dataframe grouped by groupIndex.
		"""
		tt = time.clock()
		thingData = [ group[1] for group in data.groupby(self.groupIndex) ]
		logger.debug("There are %s thing(s) in dataset. Elapes %s second(s)" % (len(thingData),(time.clock() - tt) ))
		if(self.timeIndex):
			for idx in range(0,len(thingData)):			
				tt = time.clock()
				logger.info("Sorting data by %s" % self.timeIndex)
				thingData[idx] = thingData[idx].sort_values(by=self.timeIndex,ascending=True)
				logger.info("Data sort complete. Elapsed %f second(s)" %  (time.clock() - tt))
		return thingData
		