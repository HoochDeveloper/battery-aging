from Demetra import EpisodedTimeSeries
import pandas as pd, numpy as np, os, sys

class Mercurio():
	
	eps1=5
	eps2=5
	alpha1=5
	alpha2=5
	ets = None
		
	def __init__(self):
		self.ets = EpisodedTimeSeries(self.eps1,self.eps2,self.alpha1,self.alpha2)

	def exportForSynthetic(self):
		"""
		Creates the swab2swab dataset (if not exists)
		For every battery build a folder
		In every folder there is one csv file for every espisodes
			BattertName -> 1_1.csv, 1_2.csv, .... , 4_59.csv 
		"""

		root4save = os.path.join(".","exportEpisodes")
		if not os.path.exists(root4save):
			os.makedirs(root4save)
			
		
		## Episode creation for real data- start
		mode = "swab2swab"
		self.ets.buildDataSet(os.path.join(".","dataset"),mode=mode,force=False) # creates dataset if does not exists
		## Episode creation for real data - end

		idxName = self.ets.dataHeader.index(self.ets.nameIndex)
		batteries = self.ets.loadDataSet()
		
		for battery in batteries:
			
			batteryName = self.getBatteryName(battery,idxName)
			print("Processing %s" %  batteryName)
			batteryFolder = os.path.join(root4save,batteryName)
			if not os.path.exists(batteryFolder):
				os.makedirs(batteryFolder)
			month_count = 0
			for month in battery:
				month_count += 1
				episode_count = 0
				for episode in month:
					episode_count += 1
					ep = episode[self.ets.keepY]
					fileName = "%d_%d.csv" % (month_count,episode_count)
					ep.to_csv( os.path.join(batteryFolder,fileName), index=False)
	

	def importSynthetic(self):
		
		"""
		Import the synthetic data generated from the file in csv 
		with the MatExport.py
		
		All file will be in synthetic_eps1_eps2_alpha1_alpha2_SOC
		every file is a pandas dataframe zipped
		One file for battery
		"""

		root4load = os.path.join(".","synthetic_data")
		root4saveNoSOC = os.path.join(".",self.ets.synthetcBlowPath)
		
		for batteryFoldeSOC in os.listdir(root4load):
			
			batteryName,soc = self.getBatteryNameAndSOCFromFile(batteryFoldeSOC);
			print("Importing synthetic data for %s @ soc %s" % (batteryName,soc))
			saveFolder = root4saveNoSOC + "_%s" % soc
			if not os.path.exists(saveFolder):
				os.makedirs(saveFolder)
			
			
			socLoadFolder = os.path.join(root4load,batteryFoldeSOC)

			syntheticBatteryEpisode = []
			battery = self.ets.loadBatteryAsSingleEpisode(batteryName)
			monthCount = 0
			for month in battery:
				syntheticMonthEpisode = []
				monthCount += 1
				episodeCount = 0
				for episode in month:
					episodeCount += 1
					dfReal = episode[self.ets.syntheticImport]
					episode2load = os.path.join(socLoadFolder,"%d_%d.csv" % (monthCount,episodeCount))
					dfSynthetic = pd.read_csv(episode2load,sep=',', 
						names=([ self.ets.dataHeader[17]]),
						dtype=({ self.ets.dataHeader[17] : np.float32}))
					tempDf = dfReal.copy()
					tempDf.loc[:,self.ets.dataHeader[17]] = dfSynthetic[self.ets.dataHeader[17]].values
					syntheticMonthEpisode.append(tempDf)
				if(len(syntheticMonthEpisode) > 0):
					allSyntheticMonth = pd.concat(syntheticMonthEpisode)
					syntheticBatteryEpisode.append(allSyntheticMonth)
			
			syntheticSingleEpisode = pd.concat(syntheticBatteryEpisode)
			# starting from the corresponding real blow, 
			# creates the relative synthetic blows
			realBlows = self.ets.seekEpisodesBlows(battery)
			synthetic_months = []
			for month in realBlows:
				synthetic_blows = []
				for blow in month:
					hybridBlow = syntheticSingleEpisode.ix[ blow.index ]
					if(hybridBlow.shape[0] != 20):
						print("Warning missing index for battery %s" % batteryName)
						print(hybridBlow.shape)
					else:
						synthetic_blows.append(hybridBlow)
				synthetic_months.append(synthetic_blows)
			self.ets.saveZip(saveFolder,batteryName+".gz",synthetic_months)
			
	def getBatteryName(self,battery,idxName):
		batteryName = None
		for episodeInMonth in battery:
			if(len(episodeInMonth) > 0):
				batteryName = episodeInMonth[0].values[:, idxName][0]
		return batteryName

	def getBatteryNameAndSOCFromFile(self,fileName):
		fileName = os.path.splitext(fileName)[0]
		batteryName = fileName.split("_")[0][1:]
		soc = fileName.split("_")[1]
		return batteryName,soc
		
def main():
	if(len(sys.argv) != 2):
		print("Expected one argument: import / export")
		return
	action = sys.argv[1]
	mercurio = Mercurio()
	if(action == "import"):
		print("Mercurio has come back with synthetic data!")
		mercurio.importSynthetic()
	elif(action == "export"):
		print("Mercurio is going to synthetize data!")
		mercurio.exportForSynthetic()
	else:
		print("Mercurio does not want to perform %s!" % action)
		
main()