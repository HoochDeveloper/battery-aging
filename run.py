import uuid,time,os,logging, numpy as np, sys, matplotlib.pyplot as plt
from logging import handlers as loghds
from sklearn.model_selection import train_test_split

#Project module import
from Astrea import Astrea
from Demetra import EpisodedTimeSeries

#Module logging
logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler) 


plotMode="GUI"
modelNameTemplate = "Enc_%d_Synthetic_%d_%s_K_%d"

maeFolder = os.path.join(".","evaluation")

def execute(mustTrain,encSize = 8,K = 5,type="Dense"):
	from Minerva import Minerva
	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)	
	nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
	tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
	astrea = Astrea(tsIndex,nameIndex,minerva.ets.keepY)	
	
	if(mustTrain):
		train(minerva,astrea,K,encSize,type=type)
	
	batteries = minerva.ets.loadSyntheticBlowDataSet(100)
	k_idx,k_data = astrea.kfoldByKind(batteries,K)
	scaler = astrea.getScaler(k_data)
	evaluate(minerva,astrea,K,encSize,scaler,range(50,105,5),show=False,showScatter=False,type=type)
	
def loadEvaluation(encSize,K=5,type="Dense"):
	
	ets = EpisodedTimeSeries(5,5,5,5)
	nameIndex = ets.dataHeader.index(ets.nameIndex)
	tsIndex = ets.dataHeader.index(ets.timeIndex)
	astrea = Astrea(tsIndex,nameIndex,ets.keepY)
	trainIdx,testIdx = astrea.leaveOneFoldOut(K)
	count = 0
	for _, test_index in zip(trainIdx,testIdx): 
		count += 1
		print("Load evaluation for fold %d" % count)
		name4model = modelNameTemplate % (encSize,100,type,count)
		[maes,labels] = ets.loadZip(maeFolder,name4model+".out",)
		tit = "MAE %s" % name4model 
		errorBoxPlot(maes,labels,tit,False)
	
def evaluate(minerva,astrea,K,encSize,scaler,ageScales,type="Dense",show=False,showScatter=False,boxPlot=False):

	if not os.path.exists(maeFolder):
		os.makedirs(maeFolder)

	trainIdx,testIdx = astrea.leaveOneFoldOut(K)
	count = 0
	for _, test_index in zip(trainIdx,testIdx): 
		count += 1
		print("Fold %d" % count)
		name4model = modelNameTemplate % (encSize,100,type,count)
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
		
		minerva.ets.saveZip(maeFolder,name4model+".out",[maes,labels])
		
		if(boxPlot):
			tit = "MAE %s" % name4model 
			errorBoxPlot(maes,labels,tit)
			
		
def errorBoxPlot(errorList,labels,title,save=True):
	fig = plt.figure()
	plt.boxplot(errorList,whis=[0, 99],sym='')
	plt.xticks(range(1,len(labels)+1),labels)
	plt.title(title)
	plt.grid()
	if(save):
		plt.savefig(title, bbox_inches='tight')
		plt.close()
	else:
		plt.show()
		
		
def train(minerva,astrea,K,encSize,type="Dense"):
	
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
		name4model = modelNameTemplate % (encSize,train_ageScale,type,count)
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
	if(len(sys.argv) != 4):
		print("Expected train / evaluate")
		return
	encSize = int(sys.argv[2])
	type = sys.argv[3]
	print("Encoded size %s type %s" % (encSize,type))
	action = sys.argv[1]
	if(action == "train"):
		execute(True,encSize,type=type)
	elif(action=="evaluate"):
		execute(False,encSize,type=type)
	elif(action=="show_evaluation"):
		
		loadEvaluation(encSize,type=type)
	else:
		print("Can't perform %s" % action)
		
main()