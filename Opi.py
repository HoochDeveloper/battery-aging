import time,logging,os, numpy as np
from logging import handlers as loghds

from sklearn import preprocessing

#Module logging
logger = logging.getLogger("Opi")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler) 

class Opi():
	
	idxTS = None
	idxName = None
	data2keep = None
	
	def __init__(self,idxTS,idxName,data2keep,logFolder="./logs"):
		
		if not os.path.exists(logFolder):
			os.makedirs(logFolder)
		
		logFile = logFolder + "/Opi.log"
		hdlr = loghds.TimedRotatingFileHandler(logFile,
                                       when="H",
                                       interval=1,
                                       backupCount=30)
		hdlr.setFormatter(formatter)
		logger.addHandler(hdlr)
		
		self.idxTS   = idxTS
		self.idxName = idxName
		self.data2keep = data2keep
	
	def kfoldByKind(self,batteries,k):
		"""
		Build K fold for the input. It is granted that every episode of a battery are all in the
		same fold.
		
		Output: 
			index: array that contains the battery label and for every episode the starting TS
			data: array with episodes divided in K fold
		"""
		tt = time.clock()
		logger.debug("kfoldByKind - start")
		
		logger.debug("There are %d batteries to distribute in %d fold" % (len(batteries),k))
		batteris4fold = int( len(batteries) / k )
		logger.debug("Batteries for fold: %d" % batteris4fold)
		
		episodesInDataset = 0
		for battery in batteries:
			totalEpisodeInBattery = 0	
			for episodeInMonth in battery:
				totalEpisodeInBattery += len(episodeInMonth)
			episodesInDataset += totalEpisodeInBattery
			batteryName = self.__getBatteryName(battery)
			logger.debug("There are %d episode in battery %s" % (totalEpisodeInBattery,batteryName))
		
		logger.info("There are %d episode in dataset." % (episodesInDataset))
		indexes,datas = self.__foldSplit(batteries,episodesInDataset,k)
		logger.debug("kfoldByKind - end - %f" % (time.clock() - tt))
		return indexes,datas
	
	
	def getScaler(self,foldedData):
		data2dimension = []
		for fold in foldedData:
			for episode in fold:
				for t in range(0,episode.shape[0]):
						data2dimension.append(episode.values[t])
		data2dimension = np.asarray(data2dimension)
		scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
		scaler.fit(data2dimension)
		return scaler
		
	def foldAs3DArray(self,fold,scaler = None):
		"""
		Convert the dataset list structure to numpy 3D array
		batteries: 3 layer list of dataframe [battery][month][episode] = dataframe
		if scaler are specified, data will be transformed
		"""
		tt = time.clock()
		logger.debug("foldAs3DArray - start")
		tmpData = []
		
		#for fold in folds:
		for e in fold:
			x = e.values
			if(scaler is not None):
				x = scaler.transform(x)
			tmpData.append( x )
		
		outData = np.asarray(tmpData)
		logger.debug("foldAs3DArray - end - %f" % (time.clock() - tt) )
		return outData
	
	def leaveOneFoldOut(self,k):	
		train = [[ j for j in range(k) if j != i ] for i in range(k)] 
		test = [[ j for j in range(k) if j == i ] for i in range(k)] 
		return train,test
		#return [[fold for fold in folds if not np.array_equal(fold,folds[i]) ] for i in range(len(folds))]
	
	
	def __foldSplit(self,batteries,episodesInDataset,k):
		
		tt = time.clock()
		logger.debug("__foldSplit - start")
		
		maxEpisodesForFold = int( episodesInDataset / k )
		logger.debug("Max episodes for fold %d" % maxEpisodesForFold)
		currentFold = 0
		
		foldIndex = []
		foldIndex.append([])
		foldData  = []
		foldData.append([])
		
		np.random.seed(42)
		permutedIdx = np.random.permutation(len(batteries))
		assigned = 0
		for idx in permutedIdx:
			# iteration over batteries
			battery = batteries[idx]
			
			batteryName = self.__getBatteryName(battery)
			
			totalEpisodeInBattery = 0
			batteryIndex = []
			batteryData  = []
			for episodeInMonth in battery:
				# iteration over months in battery
				totalEpisodeInBattery += len(episodeInMonth)
				for episode in episodeInMonth:
					# iteration over episodes in month
					startTS = episode.values[:, self.idxTS][0]
					indexRecord = (batteryName,startTS)
					batteryIndex.append(indexRecord)
					batteryData.append(episode[self.data2keep])
			
			# check how many episodes are in the fold, it there are more than max, then switch fold
			episodeInFold = len(foldIndex[currentFold])
			if((episodeInFold + totalEpisodeInBattery) > maxEpisodesForFold and currentFold < (k - 1)):
				assigned += episodeInFold
				logger.info("End of fold %d, dimension %d" % (currentFold,len(foldIndex[currentFold])))
				currentFold += 1
				foldIndex.append([])
				foldData.append([])
			
			foldIndex[currentFold] += batteryIndex
			foldData[currentFold] += batteryData
			episodeInFold = len(foldIndex[currentFold])
		
		logger.info("Last fold has %d episode" % (episodesInDataset - assigned))
			
		logger.debug("__foldSplit - end - %f" % (time.clock() - tt))
		return foldIndex, foldData

	def __getBatteryName(self,battery):
		batteryName = None
		for episodeInMonth in battery:
			if(len(episodeInMonth) > 0):
				batteryName = episodeInMonth[0].values[:, self.idxName][0]
		return batteryName
		
		
		