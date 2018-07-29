import uuid,time,os,logging, numpy as np, sys, pandas as pd , matplotlib.pyplot as plt
from logging import handlers as loghds

#Project module import
from Minerva import Minerva
from Astrea import Astrea

from sklearn.model_selection import train_test_split

#Module logging
logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler) 


plotMode="GUI"
modelNameTemplate = "Enc_%d_Synthetic_%d_ConvModel_K_%d"

modelNameTemplate = "Enc_%d_Test_%d_ConvModel_K_%d"


def main():
	K = 5
	encSize = 8
	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)
	
	nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
	tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
	astrea = Astrea(tsIndex,nameIndex,minerva.ets.keepY)	
	
	#if(True):
	if(False):
		train(minerva,astrea,K,encSize)
	else:
		print(100)
		evaluate(minerva,astrea,K,100,encSize)
		print(95)
		evaluate(minerva,astrea,K,95,encSize)
		print(90)
		evaluate(minerva,astrea,K,90,encSize)
		print(85)
		evaluate(minerva,astrea,K,85,encSize)
		print(80)
		evaluate(minerva,astrea,K,80,encSize)
	
	

def evaluate(minerva,astrea,K,soc,encSize):
	batteries = minerva.ets.loadSyntheticBlowDataSet(soc)
	
	k_idx,k_data = astrea.kfoldByKind(batteries,K)
	scaler = astrea.getScaler(k_data)
	folds4learn = []
	for i in range(len(k_data)):
		fold = k_data[i]
		foldAs3d = astrea.foldAs3DArray(fold,scaler)
		folds4learn.append(foldAs3d)

	trainIdx,testIdx = astrea.leaveOneFoldOut(K)
	count = 0
	for train_index, test_index in zip(trainIdx,testIdx): 
		count += 1
		train = [folds4learn[i] for i in train_index]
		train = np.concatenate(train)
		test =  [folds4learn[i] for i in test_index]
		test =  np.concatenate(test)
		train,valid,_,_ = train_test_split( train, train, test_size=0.2, random_state=42)
		name4model = modelNameTemplate % (encSize,100,count)
		minerva.evaluateModelOnArray(test, test,name4model,plotMode,scaler,False)
	
def train(minerva,astrea,K,encSize):
	
	train_soc = 100
	batteries = minerva.ets.loadSyntheticBlowDataSet(train_soc)
	k_idx,k_data = astrea.kfoldByKind(batteries,K)
	scaler = astrea.getScaler(k_data)
	folds4learn = []
	for i in range(len(k_data)):
		fold = k_data[i]
		foldAs3d = astrea.foldAs3DArray(fold,scaler)
		folds4learn.append(foldAs3d)

	trainIdx,testIdx = astrea.leaveOneFoldOut(K)
	count = 0
	for train_index, test_index in zip(trainIdx,testIdx): 
		count += 1
		train = [folds4learn[i] for i in train_index]
		train = np.concatenate(train)
		test =  [folds4learn[i] for i in test_index]
		test =  np.concatenate(test)
		train,valid,_,_ = train_test_split( train, train, test_size=0.2, random_state=42)
		name4model = modelNameTemplate % (encSize,train_soc,count)
		minerva.trainlModelOnArray(train, train, valid, valid,name4model,encodedSize = encSize)
		minerva.evaluateModelOnArray(test, test,name4model,plotMode,scaler,False)


main()