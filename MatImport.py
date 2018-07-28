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
	root4saveNoSOC = os.path.join(ets.synthetcBlowPath)
	
	socCol = "SOC_BAT"
	for file in os.listdir(root4load):
		
		batteryName,soc = getBatteryNameAndSOCFromFile(file);
		
		saveFolder = root4saveNoSOC + "_%s" % soc
		if not os.path.exists(saveFolder):
			os.makedirs(saveFolder)
		
		file2load = os.path.join(root4load,file)
		dfSynthetic = pd.read_csv(file2load,sep=',', 
				names=([ ets.dataHeader[17], socCol]),
				dtype=({ ets.dataHeader[17] : np.float32, socCol : np.float32})
		)

		allBatteryEpisode = []
		battery = ets.loadBatteryAsSingleEpisode(batteryName)
		for month in battery:
			for episode in month:
				ep = episode[ets.syntheticImport]
				allBatteryEpisode.append(ep)
		
		dfHybrid = pd.concat(allBatteryEpisode)
		dfHybrid[ets.dataHeader[17]] = dfSynthetic[ets.dataHeader[17]].values
		dfHybrid[socCol] = dfSynthetic[socCol].values
		
		# starting from the corresponding real blow, 
		# creates the relative synthetic blows
		realBlows = ets.seekEpisodesBlows(battery)
		synthetic_months = []
		for month in realBlows:
			synthetic_blows = []
			for blow in month:
				hybridBlow = dfHybrid.ix[ blow.index ]
				if(hybridBlow.shape[0] != 20):
					print("Warning missing index for battery %s" % batteryName)
					print(hybridBlow.shape)
				else:
					synthetic_blows.append(hybridBlow)
			synthetic_months.append(synthetic_blows)
		ets.saveZip(saveFolder,batteryName+".gz",synthetic_months)
		
def getBatteryNameAndSOCFromFile(fileName):
	fileName = os.path.splitext(fileName)[0]
	batteryName = fileName.split("_")[0][1:]
	soc = fileName.split("_")[1]
	return batteryName,soc

main()