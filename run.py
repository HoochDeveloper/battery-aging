import uuid,time,os,logging, numpy as np, sys, pandas as pd , matplotlib.pyplot as plt
from logging import handlers as loghds

#Project module import
from Minerva import Minerva
from Opi import Opi

from sklearn.model_selection import train_test_split

#Module logging
logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler) 

def main():
	K = 5
	plotMode="GUI"
	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)
	nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
	tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
	opi = Opi(tsIndex,nameIndex,minerva.ets.keepY)	
	print("SOC 100")
	evaluate(minerva,opi,K,100)
	print("SOC 95")
	evaluate(minerva,opi,K,95)
	print("SOC 90")
	evaluate(minerva,opi,K,90)
	print("SOC 85")
	evaluate(minerva,opi,K,85)
	print("SOC 80")
	evaluate(minerva,opi,K,80)
	
def evaluate(minerva,opi,K,soc):
	batteries = minerva.ets.loadSyntheticBlowDataSet(soc)
	
	plotMode="GUI"
	k_idx,k_data = opi.kfoldByKind(batteries,K)
	scaler = opi.getScaler(k_data)
	folds4learn = []
	for i in range(len(k_data)):
		fold = k_data[i]
		foldAs3d = opi.foldAs3DArray(fold,scaler)
		folds4learn.append(foldAs3d)

	trainIdx,testIdx = opi.leaveOneFoldOut(K)
	count = 0
	for train_index, test_index in zip(trainIdx,testIdx): 
		count += 1
		train = [folds4learn[i] for i in train_index]
		train = np.concatenate(train)
		test =  [folds4learn[i] for i in test_index]
		test =  np.concatenate(test)
		train,valid,_,_ = train_test_split( train, train, test_size=0.2, random_state=42)
		name4model = "Synthetic_100_InceptionModel_K_%d" % (count)
		minerva.evaluateModelOnArray(test, test,name4model,plotMode,scaler,True)
	
def train(minerva,opi,K):
	
	## Episode creation - start
	#mode = "swab2swab" #"swabCleanDischarge"
	#minerva.ets.buildDataSet(os.path.join(".","dataset"),mode=mode,force=False) # creates dataset if does not exists
	## Episode creation - end
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	soc = 100
	batteries = minerva.ets.loadSyntheticBlowDataSet(soc)
	
	k_idx,k_data = opi.kfoldByKind(batteries,K)
	scaler = opi.getScaler(k_data)
	folds4learn = []
	for i in range(len(k_data)):
		fold = k_data[i]
		foldAs3d = opi.foldAs3DArray(fold,scaler)
		folds4learn.append(foldAs3d)

	trainIdx,testIdx = opi.leaveOneFoldOut(K)
	count = 0
	for train_index, test_index in zip(trainIdx,testIdx): 
		count += 1
		train = [folds4learn[i] for i in train_index]
		train = np.concatenate(train)
		test =  [folds4learn[i] for i in test_index]
		test =  np.concatenate(test)
		train,valid,_,_ = train_test_split( train, train, test_size=0.2, random_state=42)
		name4model = "Synthetic_%d_InceptionModel_K_%d" % (soc,count)
		minerva.trainlModelOnArray(train, train, valid, valid,name4model,encodedSize = 8)
		minerva.evaluateModelOnArray(test, test,name4model,plotMode,scaler,False)

main()