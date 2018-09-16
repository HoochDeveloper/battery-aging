import uuid,time,os,logging, numpy as np, sys, matplotlib.pyplot as plt
from logging import handlers as loghds


#Project module import
from Astrea import Astrea
from Demetra import EpisodedTimeSeries
from sklearn.manifold import TSNE
import pandas as pd
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

def mapTable(encSize,type,modelNumber):
	
	dfTemplate = "map_model_%s_th_%d"
	mapFolder = os.path.join(".","maps")
	name4model = modelNameTemplate % (encSize,100,type,modelNumber)
	thresholdPercentile = 90
	dataFrameName = dfTemplate % (name4model,thresholdPercentile)
	fullPath = os.path.join(mapFolder,dataFrameName)
	dataSet = None
	force = False
	if not os.path.exists(fullPath) or  force:

		if not os.path.exists(mapFolder):
			os.makedirs(mapFolder)

		from Minerva import Minerva
		minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)	

		nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
		tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
		astrea = Astrea(tsIndex,nameIndex,minerva.ets.keepY)	
	
		[maes,labels] = minerva.ets.loadZip(maeFolder,name4model+".out",)
		
		fullHealthError = maes[0]
		tmp = np.percentile(fullHealthError,[thresholdPercentile])
		thresholdValue = tmp[0]
		#print(thresholdValue)
		model = minerva.getModel(name4model)
		
		ageScale = 100
		ageTh = 85
		K = 8
		ageStep = 5
		batteries = minerva.ets.loadSyntheticBlowDataSet(ageScale)
		_,k_data = astrea.kfoldByKind(batteries,K)
		scaler = astrea.getScaler(k_data)
		dataSet = pd.DataFrame({'MAE' : [],'TP' : [], "TN" : [], "FP":[], "FN":[]})
		
		#fpc = 0
		#fnc = 0
		totalPositive = 0
		totalNegative = 0
		for b in range(0,K):
			testX = astrea.foldAs3DArray(k_data[b],scaler)
			maes = minerva.getMaes(model,testX,testX)
			tp = None
			tn = None
			fp = None
			if(ageScale >= ageTh):
				# no true positive
				totalNegative += len(maes)
				fp = np.where(maes >= thresholdValue, 1,0)
				tn = np.where(maes < thresholdValue, 1,0)
				tp = np.where(True, 0,0)
				fn = np.where(True, 0,0)
			else:
				totalPositive += len(maes)
				tp = np.where(maes >= thresholdValue, 1,0)
				fn = np.where(maes < thresholdValue, 1, 0)
				tn = np.where(True, 0,0)
				fp = np.where(True, 0,0)
				
			
			df = pd.DataFrame({'MAE':maes,'TP':tp, 'TN':tn, 'FP':fp, 'FN':fn})
			dataSet = dataSet.append(df)
			
			ageScale -= ageStep
			batteries = minerva.ets.loadSyntheticBlowDataSet(ageScale)
			_,k_data = astrea.kfoldByKind(batteries,K)
			
		
		dataSet.sort_values(by="MAE",ascending=False,inplace=True)
		
		dataSet["P_RC"] = dataSet["TP"].cumsum() / totalPositive
		dataSet["P_PR"] = dataSet["TP"].cumsum() / (dataSet["TP"].cumsum() + dataSet["FP"].cumsum())
		dataSet["N_RC"] = dataSet["TN"].cumsum() / totalNegative
		dataSet["N_PR"] = dataSet["TN"].cumsum() / (dataSet["TN"].cumsum() + dataSet["FN"].cumsum())
	
		dataSet.to_pickle(fullPath)
	else:
		dataSet = pd.read_pickle(fullPath)
	
	p_pr_mean = dataSet["P_PR"].mean()
	n_pr_mean = dataSet["N_PR"].mean()
	map = np.mean([n_pr_mean,p_pr_mean])
	print("p_pr_mean %f n_pr_mean %f map %f" % (p_pr_mean,n_pr_mean,map))

	if(True):
		plt.plot( dataSet["N_RC"],dataSet["N_PR"],label="TN")
		plt.plot( dataSet["P_RC"],dataSet["P_PR"],label="TP")
		plt.legend()
		plt.show()
	
	#print(dataSet.head(20))
	#print(dataSet.tail(20))
	
	
	
def execute(mustTrain,encSize = 8,K = 3,type="Dense"):
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
	evaluate(minerva,astrea,K,encSize,scaler,range(100,60,-5),show=False,showScatter=False,type=type)
	
def loadEvaluation(encSize,K=3,type="Dense"):
	
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
	
	
	mae2Save = [None] * K
	lab2Save = [None] * K
	for c in range(0,K):
		foldMaes = [None] * len(ageScales)
		foldLabels = [None] * len(ageScales)
		mae2Save[c] = foldMaes
		lab2Save[c] = foldLabels
	
	for a in range(0,len(ageScales)):
		
		ageScale = ageScales[a]
		
		batteries = minerva.ets.loadSyntheticBlowDataSet(ageScale)
		_,k_data = astrea.kfoldByKind(batteries,K)
		
		count = 0
		for _, test_index in zip(trainIdx,testIdx): 
			
			print("Fold %d Age: %d" % (count+1,ageScale))
			name4model = modelNameTemplate % (encSize,100,type,count+1)
			maes = []
			labels = []
			folds4learn = []
			for i in test_index:
				fold = k_data[i]
				foldAs3d = astrea.foldAs3DArray(fold,scaler)
				folds4learn.append(foldAs3d)
		
			test =  np.concatenate(folds4learn)
			mae = minerva.evaluateModelOnArray(test, test,name4model,plotMode,scaler,False)
			mae2Save[count][a] = mae
			lab2Save[count][a] = "Q@%d" % ageScale
			count += 1
			#maes.append(mae)
			#labels.append("Age %d" % ageScale)
		
	for c in range(0,K):
		name4model = modelNameTemplate % (encSize,100,type,c+1)
		maes = mae2Save[c]
		labels = lab2Save[c]
		minerva.ets.saveZip(maeFolder,name4model+".out",[maes,labels])

def codeProjection(encSize,type,K):
	
	ageScales = [100,70]
	from mpl_toolkits.mplot3d import Axes3D
	from sklearn.decomposition import PCA
	from Minerva import Minerva
	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)	
	nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
	tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
	astrea = Astrea(tsIndex,nameIndex,minerva.ets.keepY)
	
	trainIdx,testIdx = astrea.leaveOneFoldOut(K)
	count = 0
	for _, test_index in zip(trainIdx,testIdx): 
		count += 1
		print("Fold %d" % count)
		name4model = modelNameTemplate % (encSize,100,type,count)
		maes = []
		labels = []
		codes = []
		for ageScale in ageScales:
			batteries = minerva.ets.loadSyntheticBlowDataSet(ageScale)
			_,k_data = astrea.kfoldByKind(batteries,K)
			scaler = astrea.getScaler(k_data)
			folds4learn = []
			for i in test_index:
				fold = k_data[i]
				foldAs3d = astrea.foldAs3DArray(fold,scaler)
				folds4learn.append(foldAs3d)
			test =  np.concatenate(folds4learn)
			code = minerva.getEncoded(name4model,test)
			
			
			
			tsne = TSNE(n_components=2, n_iter=300)
			pr = tsne.fit_transform(code)
			codes.append(pr)
			#pca = PCA(n_components=2)
			#pc = pca.fit_transform(code)
			#codes.append(pc)
			
			
		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection='3d')

		
		for code in codes:
			plt.scatter(code[:,0],code[:,1])
		plt.show()
	


	

	
	
def errorBoxPlot(errors,labels,title,save=True):
	
	#for c in range(0,len(errors)):
	#	err = errors[c]
	#	prc = np.percentile(err,[25,50,75])
	#	print("%f	%f	%f" % ( prc[0],prc[1],prc[2] ))
	
	
	#meanAvgPrecision(errors)
	
	fp = 0
	
	lastPerc = 90
	print("Metrics with threshold @ %d" % lastPerc)
	percFull = np.percentile(errors[0],[lastPerc])
	fullTh = percFull[0]
	#idxFull = np.digitize(errors[0],percFull)
	#uf, cf = np.unique(idxFull,return_counts = True)
	errAtAge = errors[0]
	errAtAge = np.where(errAtAge >= fullTh)
	fp += errAtAge[0].shape[0]
	
	ageThIdx = 4
	
	for error in range(1,ageThIdx):
		errAtAge = errors[error]
		errAtAge = np.where(errAtAge >= fullTh)
		fp += errAtAge[0].shape[0]
		
	tp = 0
	fn = 0
	for error in range(ageThIdx,8):
		errAtAge = errors[error]
		
		falseNegative = np.where(errAtAge < fullTh)
		fn += falseNegative[0].shape[0]
		truePositive = np.where(errAtAge >= fullTh)
		tp += truePositive[0].shape[0]
		
	recall = tp / (tp+fn)
	precision = tp / (tp + fp)
	fscore = 2 * precision * recall / (precision + recall)
	
	print("Fscore: %f Precision: %f Recall: %f" % (fscore,precision,recall))
	
	if(False):
		fig = plt.figure()
		plt.boxplot(errors,sym='',whis=[3, lastPerc]) #
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
	
	
	degraded = []
	for i in range(70,100,5):
		
		corrupted = minerva.ets.loadSyntheticBlowDataSet(i)
		degraded.append(corrupted)
	
	
	#degradationPerc = [.03,.05,.10,.15,.20]
	#degradationPerc = [.02,.04,.06,.08,.10]
	#degradationPerc = [.02,.04,.06,.08,.15]
	degradationPerc = [.01,.02,.03,.04,.05]
	
	#k_idx,k_data = astrea.kfoldByKind(batteries,K)
	print("Degraded")
	k_idx,k_data = astrea.kFoldWithDegradetion(batteries,degraded,degradationPerc,K)
	
	scaler = astrea.getScaler(k_data)
	folds4learn = []
	
	for i in range(len(k_data)):
		fold = k_data[i]
		foldAs3d = astrea.foldAs3DArray(fold,scaler)
		folds4learn.append(foldAs3d)
		

	trainIdx,testIdx = astrea.leaveOneFoldOut(K)	
	count = 0
	for train_index, test_index in zip(trainIdx,testIdx): 
		# TRAIN #VALID
		trainStr = ""
		trainX = []
		validX = None
		validY = None
		for i in range(0,len(train_index)):
			if i != (count % len(train_index)):
				trainX.append(folds4learn[train_index[i]])
				trainStr += " TR " + str(train_index[i])
			else:
				validX = folds4learn[train_index[i]]
				validY = validX
				trainStr += " VL " + str(train_index[i])
		
		trainX = np.concatenate(trainX)
		trainY = trainX

		#TEST
		trainStr += " TS " + str(test_index[0])
		testX = [folds4learn[i] for i in test_index]
		testX =  np.concatenate(testX)
		testY =  testX
		print(trainStr)
		count += 1
		
		name4model = modelNameTemplate % (encSize,train_ageScale,type,count)
		minerva.trainlModelOnArray(trainX, trainY, validX, validY,
			name4model,encodedSize = encSize)
		minerva.evaluateModelOnArray(testX, testY,name4model,plotMode,scaler,False)

		
def dataRange(minerva,astrea,K,min,max,step):
	for ageScale in range(min,max,step):
		batteries = minerva.ets.loadSyntheticBlowDataSet(ageScale)
		k_idx,k_data = astrea.kfoldByKind(batteries,K)
		scaler = astrea.getScaler(k_data)
		print(ageScale)
		print(scaler.data_min_)
		print(scaler.data_max_)
		
		
def learningCurve(encSize,type,K):
	ets = EpisodedTimeSeries(5,5,5,5)
	for k in range(1,K+1):
		hfile = modelNameTemplate % (encSize,100,type,k)
		history = ets.loadZip(ets.rootResultFolder,hfile+ "_history")
		
		print("Val Loss")
		for s in range(0,400,50):
			print(history['val_loss'][s])
		print("Train Loss")
		for s in range(0,400,500):
			print(history['loss'][s])
		print("-------------------------------")
		
		plt.plot(history['loss'])
		plt.plot(history['val_loss'])
		plt.title("Learning Curve for %s" % hfile)
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()

def showModel(encSize,type):
	from Minerva import Minerva
	name4model = modelNameTemplate % (encSize,100,type,1)
	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)	
	minerva.printModelSummary(name4model)
		
def main():
	if(len(sys.argv) != 4):
		print("Expected train / evaluate")
		return
	K = 5
	encSize = int(sys.argv[2])
	type = sys.argv[3]
	print("Encoded size %s type %s" % (encSize,type))
	action = sys.argv[1]
	if(action == "train"):
		execute(True,encSize,type=type,K = K)
	elif(action=="evaluate"):
		execute(False,encSize,type=type, K = K)
	elif(action=="map"):
		for i in range(1,6):
			mapTable(encSize,type,i)
	elif(action=="show_evaluation"):
		loadEvaluation(encSize,type=type, K = K)
	elif(action=="learning_curve"):
		learningCurve(encSize,type,K)
	elif(action == "show"):
		showModel(encSize,type)
	elif(action == "proj"):
		codeProjection(encSize,type,K)
	else:
		print("Can't perform %s" % action)


		
main()