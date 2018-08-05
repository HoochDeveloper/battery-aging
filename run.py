import uuid,time,os,logging, numpy as np, sys, pandas as pd , matplotlib.pyplot as plt
from logging import handlers as loghds

#Project module import
from Minerva import Minerva
from Astrea import Astrea

from sklearn.model_selection import train_test_split
from scipy.stats import norm

import random
from scipy import stats

#Module logging
logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler) 


plotMode="GUI"
modelNameTemplate = "Enc_%d_Synthetic_%d_Dense_K_%d"

def execute(mustTrain,encSize = 8,K = 5):

	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)	
	nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
	tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
	astrea = Astrea(tsIndex,nameIndex,minerva.ets.keepY)	
	
	#minAgeChargeScale = 50
	#maxAgeChargeScale = 105
	#step = 5
	#dataRange(minerva,astrea,K,minAgeChargeScale,maxAgeChargeScale,step)
	#return
	
	if(mustTrain):
		train(minerva,astrea,K,encSize)
	batteries = minerva.ets.loadSyntheticBlowDataSet(100)
	k_idx,k_data = astrea.kfoldByKind(batteries,K)
	scaler = astrea.getScaler(k_data)
	evaluate(minerva,astrea,K,encSize,scaler,[50,60,70,80,90,100],show=False,showScatter=False)
	
def evaluate(minerva,astrea,K,encSize,scaler,ageScales,show=False,showScatter=False,boxPlot=True):

	trainIdx,testIdx = astrea.leaveOneFoldOut(K)
	count = 0
	for _, test_index in zip(trainIdx,testIdx): 
		count += 1
		print("Fold %d" % count)
		name4model = modelNameTemplate % (encSize,100,count)
		maes = []
		labels = []
	
		for ageScale in ageScales:
			batteries = minerva.ets.loadSyntheticBlowDataSet(ageScale)
			_,k_data = astrea.kfoldByKind(batteries,K)
			folds4learn = []
			for i in test_index:
				fold = k_data[i]
				foldAs3d = astrea.foldAs3DArray(fold,scaler)
				folds4learn.append(foldAs3d)
			
			test =  np.concatenate(folds4learn)
			
			mae = minerva.evaluateModelOnArray(test, test,name4model,plotMode,scaler,False)
			maes.append(mae)
			
			labels.append("Age %d" % ageScale)
		
		if(boxPlot):
			tit = "MAE %s" % name4model 
			errorBoxPlot(maes,labels,tit)
			
		
def errorBoxPlot(errorList,labels,title):
	fig = plt.figure()
	plt.boxplot(errorList,0,'')
	plt.xticks(range(1,len(labels)+1),labels)
	plt.title(title)
	plt.grid()
	plt.savefig(title, bbox_inches='tight')
	#plt.show()
		
		
def train(minerva,astrea,K,encSize):
	
	train_ageScale = 100
	batteries = minerva.ets.loadSyntheticBlowDataSet(train_ageScale)
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
		name4model = modelNameTemplate % (encSize,train_ageScale,count)
		minerva.trainlModelOnArray(train, train, valid, valid,name4model,encodedSize = encSize)
		minerva.evaluateModelOnArray(test, test,name4model,plotMode,scaler,False)

		
def dataRange(minerva,astrea,K,min,max,step):
	for ageScale in range(min,max,step):
		batteries = minerva.ets.loadSyntheticBlowDataSet(ageScale)
		k_idx,k_data = astrea.kfoldByKind(batteries,K)
		scaler = astrea.getScaler(k_data)
		print(ageScale)
		print(scaler.data_min_)
		print(scaler.data_max_)
		
def main():
	if(len(sys.argv) != 2):
		print("Expected train / evaluate")
		return
	action = sys.argv[1]
	if(action == "train"):
		execute(True,16)
	elif(action=="evaluate"):
		execute(False,16)
	else:
		print("Can't perform %s" % action)
		
main()