from Demetra import EpisodedTimeSeries
import pandas as pd
import numpy as np
import os

def main():
	
	"""
	Import the synthetic data generated from the file in csv 
	with the MatExport.py
	
	All file will be in synthetic_eps1_eps2_alpha1_alpha2_SOC
	every file is a pandas dataframe zipped
	One file for battery
	"""
	
	eps1=5
	eps2=5
	alpha1=5
	alpha2=5
	
	ets = EpisodedTimeSeries(eps1,eps2,alpha1,alpha2)
	
	root4load = os.path.join(".","synthetic_data")
	root4saveNoSOC = os.path.join(".",ets.synthetcBlowPath)
	

	for batteryFoldeSOC in os.listdir(root4load):
		
		batteryName,soc = getBatteryNameAndSOCFromFile(batteryFoldeSOC);
		print("Importing synthetic data for %s @ soc %s" % (batteryName,soc))
		saveFolder = root4saveNoSOC + "_%s" % soc
		if not os.path.exists(saveFolder):
			os.makedirs(saveFolder)
		
		
		socLoadFolder = os.path.join(root4load,batteryFoldeSOC)

		syntheticBatteryEpisode = []
		battery = ets.loadBatteryAsSingleEpisode(batteryName)
		monthCount = 0
		for month in battery:
			syntheticMonthEpisode = []
			monthCount += 1
			episodeCount = 0
			for episode in month:
				episodeCount += 1
				dfReal = episode[ets.syntheticImport]
				episode2load = os.path.join(socLoadFolder,"%d_%d.csv" % (monthCount,episodeCount))
				dfSynthetic = pd.read_csv(episode2load,sep=',', 
					names=([ ets.dataHeader[17]]),
					dtype=({ ets.dataHeader[17] : np.float32}))
				tempDf = dfReal.copy()
				tempDf.loc[:,ets.dataHeader[17]] = dfSynthetic[ets.dataHeader[17]].values
				syntheticMonthEpisode.append(tempDf)	
			allSyntheticMonth = pd.concat(syntheticMonthEpisode)
			syntheticBatteryEpisode.append(allSyntheticMonth)
		
		syntheticSingleEpisode = pd.concat(syntheticBatteryEpisode)
		# starting from the corresponding real blow, 
		# creates the relative synthetic blows
		realBlows = ets.seekEpisodesBlows(battery)
		synthetic_months = []
		for month in realBlows:
			synthetic_blows = []
			for blow in month:
				#print(blow.head(10))
				hybridBlow = syntheticSingleEpisode.ix[ blow.index ]
				if(hybridBlow.shape[0] != 20):
					print("Warning missing index for battery %s" % batteryName)
					print(hybridBlow.shape)
				else:
					synthetic_blows.append(hybridBlow)
				
				#print(hybridBlow.head(10))
					
			synthetic_months.append(synthetic_blows)
		ets.saveZip(saveFolder,batteryName+".gz",synthetic_months)
		
def getBatteryNameAndSOCFromFile(fileName):
	fileName = os.path.splitext(fileName)[0]
	batteryName = fileName.split("_")[0][1:]
	soc = fileName.split("_")[1]
	return batteryName,soc

main()