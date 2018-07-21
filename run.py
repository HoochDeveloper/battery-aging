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
	## Episode creation - start
	#mode = "swab2swab" #"swabCleanDischarge"
	#minerva.ets.buildDataSet(os.path.join(".","dataset"),mode=mode,force=False) # creates dataset if does not exists
	## Episode creation - end
	
	
	nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
	tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
	
	opi = Opi(tsIndex,nameIndex,minerva.ets.keepY)	
		
	batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
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
		
		name4model = "TestModel_K_%d" % count
		minerva.trainlModelOnArray(train, train, valid, valid,name4model,encodedSize = 4)
		print(name4model)
		minerva.evaluateModelOnArray(test, test,name4model,plotMode,scaler,True)
	
def __old():
	from keras.utils import plot_model
	#model2load = "Month_" + minerva.modelName + "_5_5_5_5"
	#model = load_model(os.path.join( minerva.ets.rootResultFolder ,model2load+minerva.modelExt))
	#plot_model(model, to_file='model.png',show_shapes=True, show_layer_names=True)
	
	######################### 
	# show the histogram of resistance distribution month by month for every battery
	#logger.info("Battery resistance distribution - start")
	#minerva.ets.resistanceDistribution(batteries,join=True,mode=plotMode)
	#logger.info("Battery resistance distribution - end")
	########################
	#logger.info("Autoencoder trained on month 0 - start")
	### Train the model on first month data for all batteris
	#minerva.train4month(0,forceTrain=False)
	### Month by month prediction
	#scaleDataset = True
	#xscaler,yscaler = None, None
	#if(scaleDataset):
	#	logger.info("Loading dataset")
	#	allDataset = minerva.ets.loadDataSet()
	#	minerva.dropDatasetLabel(allDataset)
	#	logger.info("Compute scaler")
	#	xscaler,yscaler = minerva.getXYscaler(allDataset)
	#	logger.info("Scaler loaded")
	#### predict for every other months
	#minerva.decode4month(1,plotMode,showImages=True,xscaler=xscaler,yscaler=yscaler)
	#minerva.decode4month(2,plotMode,showImages=True,xscaler=xscaler,yscaler=yscaler)
	#minerva.decode4month(3,plotMode,showImages=True,xscaler=xscaler,yscaler=yscaler)
	#logger.info("Autoencoder trained on month 0 - end")
	########################
	## Train on all batteries and all months
	########################
	#logger.info("Autoencoder trained all months - start")
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	#minerva.crossTrain(batteries,forceTrain=False) #  cross train the model
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	#minerva.crossValidate(batteries,plotMode=plotMode,showImages=True) 	# cross validate the model
	#logger.info("Autoencoder trained all months - end")
	
	#######################
	## Anomaly detection
	#######################
	#logger.info("Loading the dataset")
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	#logger.info("Anomlay detection - start")
	#model2load = "Fold_2_" + minerva.modelName + "_5_5_5_5"
	#minerva.anomalyDetect(batteries,model2load,scaleDataset=True,plotMode=plotMode)
	#logger.info("Anomlay detection - end")
	
	
	##Show encoded plot
	#model2load = "Fold_1_" + minerva.modelName + "_5_5_5_5"
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	#encodedSize =8
	#minerva.plotEncoded(batteries,model2load,scaleDataset=True,plotMode=plotMode,encodedSize=encodedSize)
	
	
	#batteries = minerva.ets.loadBlowDataSet(join=True) # load the dataset
	#model2load = "Fold_1_" + minerva.modelName + "_5_5_5_5"
	#minerva.decodeAndShow(batteries,model2load,scaleDataset=True,plotMode=plotMode)
	
	
	#print("Month 1")
	#batteries = minerva.ets.loadBlowDataSet(monthIndexes=[1])
	#minerva.decodeAndShow(batteries,model2load,scaleDataset=True,plotMode=plotMode)
	#print("Month 2")
	#batteries = minerva.ets.loadBlowDataSet(monthIndexes=[2])
	#minerva.decodeAndShow(batteries,model2load,scaleDataset=True,plotMode=plotMode)
	#print("Month 3")
	#batteries = minerva.ets.loadBlowDataSet(monthIndexes=[3])
	#minerva.decodeAndShow(batteries,model2load,scaleDataset=True,plotMode=plotMode)


main()