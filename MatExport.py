from Demetra import EpisodedTimeSeries
import pandas as pd
import os

def main():
	"""
	Creates the swab2swab dataset (if not exists)
	For every battery build a folder
	In every folder there is one csv file for every espisodes
		BattertName -> 1_1.csv, 1_2.csv, .... , 4_59.csv 
	"""

	root4save = os.path.join(".","exportEpisodes")
	if not os.path.exists(root4save):
		os.makedirs(root4save)
		
	ets = EpisodedTimeSeries(5,5,5,5)
	
	## Episode creation for real data- start
	#mode = "swab2swab"
	#ets.buildDataSet(os.path.join(".","dataset"),mode=mode,force=False) # creates dataset if does not exists
	## Episode creation for real data - end

	idxName = ets.dataHeader.index(ets.nameIndex)
	batteries = ets.loadDataSet()
	
	for battery in batteries:
		
		batteryName = getBatteryName(battery,idxName)
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
				ep = episode[ets.keepY]
				fileName = "%d_%d.csv" % (month_count,episode_count)
				ep.to_csv( os.path.join(batteryFolder,fileName), index=False)
		
			
def getBatteryName(battery,idxName):
		batteryName = None
		for episodeInMonth in battery:
			if(len(episodeInMonth) > 0):
				batteryName = episodeInMonth[0].values[:, idxName][0]
		return batteryName
			
main()