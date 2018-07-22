from Demetra import EpisodedTimeSeries
import pandas as pd
import os

def main():
	
	root4save = os.path.join(".","csv")
	ets = EpisodedTimeSeries(5,5,5,5)
	idxName = ets.dataHeader.index(ets.nameIndex)
	batteries = ets.loadDataSet()
	
	for battery in batteries:
		batteryName = getBatteryName(battery,idxName)
		allEpisode = []
		for month in battery:
			for episode in month:
				ep = episode[ets.keepY]
				allEpisode.append(ep)
		fileName = "%s.csv" % (batteryName)
		df = pd.concat(allEpisode)
		df.to_csv( os.path.join(root4save,fileName), index=False)
		print("Saved %s" % fileName)
			
def getBatteryName(battery,idxName):
		batteryName = None
		for episodeInMonth in battery:
			if(len(episodeInMonth) > 0):
				batteryName = episodeInMonth[0].values[:, idxName][0]
		return batteryName
			
main()