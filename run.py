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

def mapTable(encSize,type,modelNumber,thresholdPercentile):
	
	print("Percentile: %d Model: %d" % (thresholdPercentile,modelNumber))
	
	dfTemplate = "map_model_%s_th_%d"
	mapFolder = os.path.join(".","maps")
	name4model = modelNameTemplate % (encSize,100,type,modelNumber)
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
		
		tmp = dataSet.loc[ (dataSet["TP"] == 1) | (dataSet["FP"] == 1) ]
		
		rcl = tmp["TP"].cumsum() / totalPositive
		prc = tmp["TP"].cumsum() / (tmp["TP"].cumsum() + tmp["FP"].cumsum())
		posDataset = pd.DataFrame({'RCL':rcl,'PRC':prc})
		

		dataSet.sort_values(by="MAE",ascending=True,inplace=True)
		tmp = dataSet.loc[ (dataSet["TN"] == 1) | (dataSet["FN"] == 1) ]
		rcl = tmp["TN"].cumsum() / totalNegative
		prc = tmp["TN"].cumsum() / (tmp["TN"].cumsum() + tmp["FN"].cumsum())
		negDataset = pd.DataFrame({'RCL':rcl,'PRC':prc})
		
		
		posDataset.to_pickle(fullPath+"_pos")
		negDataset.to_pickle(fullPath+"_neg")
		dataSet.to_pickle(fullPath)
	else:
		dataSet = pd.read_pickle(fullPath)
		posDataset = pd.read_pickle(fullPath+"_pos")
		negDataset = pd.read_pickle(fullPath+"_neg")
		
	if(True):
		plt.plot( posDataset["RCL"],posDataset["PRC"],label="POS")
		plt.grid()
		plt.legend()
		plt.show()	
		plt.plot( negDataset["RCL"],negDataset["PRC"],label="NEG")
		plt.grid()
		plt.legend()
		plt.show()
		
	#print(dataSet.shape)
	#print("FN ",dataSet.loc[ (dataSet["FN"] == 1) ].shape[0])
	#print("TP ",dataSet.loc[ (dataSet["TP"] == 1) ].shape[0])
	#print("FP ",dataSet.loc[ (dataSet["FP"] == 1) ].shape[0])
	#print("TN ",dataSet.loc[ (dataSet["TN"] == 1) ].shape[0])
	
	posAp = posDataset["PRC"].mean()
	negAp = negDataset["PRC"].mean()
	
	map = np.mean([posAp,negAp])
	print("PosAp: %f NegAp: %f MAP: %f" % (posAp,negAp,map))
	
	precision = dataSet["TP"].sum() / (dataSet["TP"].sum() + dataSet["FP"].sum())
	recall = dataSet["TP"].sum() / (dataSet["TP"].sum() + dataSet["FN"].sum())
	fscore = 2 * (precision*recall) / (precision+recall)
	print("Positive Precision: %f Recall: %f F: %f" % (precision,recall,fscore))
	
	precision = dataSet["TN"].sum() / (dataSet["TN"].sum() + dataSet["FN"].sum())
	recall = dataSet["TN"].sum() / (dataSet["TN"].sum() + dataSet["FP"].sum())
	fscore = 2 * (precision*recall) / (precision+recall)
	print("Negative Precision: %f Recall: %f F: %f" % (precision,recall,fscore))
		
	
	#if(True):
	#	dataSet = pd.read_pickle(fullPath+"_pos")
	#else:
	#	dataSet = pd.read_pickle(fullPath+"_neg")
	#
	#print(dataSet.head(10))
	#print(dataSet.tail(10))
	
	
	
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
	evaluate(minerva,astrea,K,encSize,scaler,range(100,45,-5),show=False,showScatter=False,type=type)
	
def loadEvaluation(encSize,K=3,type="Dense"):
	mustPlot = False
	ets = EpisodedTimeSeries(5,5,5,5)
	nameIndex = ets.dataHeader.index(ets.nameIndex)
	tsIndex = ets.dataHeader.index(ets.timeIndex)
	astrea = Astrea(tsIndex,nameIndex,ets.keepY)
	trainIdx,testIdx = astrea.leaveOneFoldOut(K)
	count = 0
	for _, test_index in zip(trainIdx,testIdx): 
		count += 1
		fig, ax = plt.subplots()
		print("Load evaluation for fold %d" % count)
		name4model = modelNameTemplate % (12,100,"C2QR",count)
		[maes,lab] = ets.loadZip(maeFolder,name4model+".out")
		
		labels = []
		for i in range(0,len(lab)):
			mul = 100 - (5*i)
			q = int(340 * mul / 100)
			labels.append("%d" % q)
		
		x,y,n = __evaluation(maes,labels,name4model)
		if(mustPlot):
			#plt.plot(x, y, label="Conv2D")
			ax.scatter(x, y, label="Conv2D")
			for i, txt in enumerate(n):
				ax.annotate(txt, (x[i], y[i]))
		
		##name4model = modelNameTemplate % (11,100,"C1",count)
		##[maes,lab] = ets.loadZip(maeFolder,name4model+".out")
		##x,y,n = __evaluation(maes,labels,name4model)
		##if(mustPlot):
		##	plt.plot(x, y, label="Conv1D")
		
		#ax.scatter(x, y, label="Conv1D")
		#for i, txt in enumerate(n):
		#	ax.annotate(txt, (x[i], y[i]))
		
		#fig.suptitle('RvP @ different threshold')
		
		##name4model = modelNameTemplate % (9,100,"D",count)
		##[maes,lab] = ets.loadZip(maeFolder,name4model+".out")
		##x,y,n = __evaluation(maes,labels,name4model)
		##if(mustPlot):
		##	plt.plot(x, y, label="FC")
		
		if(mustPlot):
			
			plt.xlabel('Recall')
			plt.ylabel('Precision')
			plt.legend()
			plt.grid()
			plt.show()
	

def precisionRecallOnRandPopulation(errors,lowTH,upTH,population):
	
	percFull = np.percentile(errors[0],[lowTH])
	fullTh = percFull[0]
	
	upperFull = np.percentile(errors[0],[upTH+5])
	upperTh = upperFull[0]
	
	
	healthly = []
	degraded = []
	maes = []
	mTP = []
	mTN = []
	mFP = []
	mFN = []
	
	unknown = 0

	np.random.seed(42)
	prob = np.random.rand(len(errors[0]))
	for i in range(0,len(prob)):
		if(prob[i] >= population[0]):
			if(errors[4][i] < fullTh or errors[4][i] > upperTh):
				degraded.append(errors[4][i])
				maes.append(errors[4][i])
				if(errors[4][i] >= fullTh):
					mTP.append(1); mFP.append(0); mTN.append(0); mFN.append(0); 
				else:
					mTP.append(0); mFP.append(0); mTN.append(0); mFN.append(1); 
			else:
				unknown += 1
		elif(prob[i] >= population[1]):
			if(errors[3][i] < fullTh or errors[3][i] > upperTh):
				degraded.append(errors[3][i])
				maes.append(errors[3][i])
				if(errors[3][i] >= fullTh):
					mTP.append(1); mFP.append(0); mTN.append(0); mFN.append(0); 
				else:
					mTP.append(0); mFP.append(0); mTN.append(0); mFN.append(1); 
			else:
				unknown += 1
		elif(prob[i] >= population[2]): # 90
			if(errors[2][i] < fullTh or errors[2][i] > upperTh):
				healthly.append(errors[2][i])
				maes.append(errors[2][i])
				if(errors[2][i] >= fullTh):
					mTP.append(0); mFP.append(1); mTN.append(0); mFN.append(0); 
				else:
					mTP.append(0); mFP.append(0); mTN.append(1); mFN.append(0); 
			else:
				unknown += 1
		elif(prob[i] >= population[3]): # 95
			if(errors[1][i] < fullTh or errors[1][i] > upperTh):
				healthly.append(errors[1][i])
				maes.append(errors[1][i])
				if(errors[1][i] >= fullTh):
					mTP.append(0); mFP.append(1); mTN.append(0); mFN.append(0); 
				else:
					mTP.append(0); mFP.append(0); mTN.append(1); mFN.append(0); 
			else:
				unknown += 1
		else: # 100
			if(errors[0][i] < fullTh or errors[0][i] > upperTh):
				healthly.append(errors[0][i])
				maes.append(errors[1][i])
				if(errors[0][i] >= fullTh):
					mTP.append(0); mFP.append(1); mTN.append(0); mFN.append(0); 
				else:
					mTP.append(0); mFP.append(0); mTN.append(1); mFN.append(0); 
			else:
				unknown += 1
	
	#print(unknown)
	#print(len(prob)-unknown)
	
	falseNegative = np.where(degraded < fullTh)
	fn = falseNegative[0].shape[0]
	truePositive = np.where(degraded >= fullTh)
	tp = truePositive[0].shape[0]
	
	falsePositive = np.where(healthly >= fullTh)
	fp = falsePositive[0].shape[0]
	
	trueNegative = np.where(healthly < fullTh)
	tn = falsePositive[0].shape[0]
	
	recall = tp / (tp+fn)
	precision = tp / (tp + fp)
	fscore = 2 * precision * recall / (precision + recall)
	print("Fscore: %f Precision: %f Recall: %f" % (fscore,precision,recall))	
	
	if(True):
		### MAP
		dataSet = pd.DataFrame({'MAE':maes,'TP':mTP, 'TN':mTN, 'FP':mFP, 'FN':mFN})
		dataSet.sort_values(by="MAE",ascending=False,inplace=True)
			
		tmp = dataSet.loc[ (dataSet["TP"] == 1) | (dataSet["FP"] == 1) ]
		
		rcl = tmp["TP"].cumsum() / (tp+fn)
		prc = tmp["TP"].cumsum() / (tmp["TP"].cumsum() + tmp["FP"].cumsum())
		posDataset = pd.DataFrame({'RCL':rcl,'PRC':prc})
		#print(posDataset.head(20))
		plt.plot( posDataset["RCL"],posDataset["PRC"],label="POS")
		plt.grid()
		plt.legend()
		plt.show()
		### end MAP
	
	
	
	
	if(False):
		#ageThIdx = 3 # 90
		boxes = [np.asarray(healthly), np.asarray(degraded)]
		plt.boxplot(boxes,sym='',whis=[100-lowTH, lowTH]) #
		plt.axhline(y=fullTh, color='blue', linestyle='-')
		plt.axhline(y=upperTh, color='blue', linestyle='-')
		#plt.axvline(x=(ageThIdx+0.5),color='gray',)
		#plt.axvline(x=(lastAge+0.5),color='gray',)
		plt.xticks(range(1,3),["Healthly","Degraded"])
		plt.title("%d %d" % (lowTH,upTH) )
		plt.grid()
		plt.xlabel('Status')
		plt.ylabel('MAE')
		plt.show()
	
	
	return precision,recall

	
def __evaluation(maes,labels,name4model):
	
	tit = "MAE %s" % name4model 
	
	x = []
	y = []
	n = []
	
	a = np.zeros((20,3))
	
	i = 0
	print(name4model)
	population = [0.90,0.60,0.50,0.40]
	for perc in range(80,82):
		#precision,recall = errorBoxPlot(maes,labels,tit,lastPerc=perc,save=False)
		precision,recall = precisionRecallOnRandPopulation(maes,perc,perc+4,population)
		x.append(recall)
		y.append(precision)
		n.append(perc)
		a[i,0] = "{:10.3f}".format(perc) 
		a[i,1] = "{:10.3f}".format(precision)
		a[i,2] = "{:10.3f}".format(recall)
		i+=1
	if(False):
		print (" \\\\\n".join([" & ".join(map(str,line)) for line in a]) )
		
	return x,y,n
		
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
	

def errorBoxPlot(errors,labels,title,lastPerc=90,save=True):
	
	#for c in range(0,len(errors)):
	#	err = errors[c]
	#	prc = np.percentile(err,[25,50,75])
	#	print("%f	%f	%f" % ( prc[0],prc[1],prc[2] ))
		
	fp = 0
	#print("Metrics with threshold @ %d" % lastPerc)
	percFull = np.percentile(errors[0],[lastPerc])
	fullTh = percFull[0]
	#idxFull = np.digitize(errors[0],percFull)
	#uf, cf = np.unique(idxFull,return_counts = True)
	errAtAge = errors[0]
	errAtAge = np.where(errAtAge >= fullTh)
	fp += errAtAge[0].shape[0]
	
	ageThIdx = 3
	lastAge = 5 #len(errors)
	for error in range(1,ageThIdx):
		errAtAge = errors[error]
		errAtAge = np.where(errAtAge >= fullTh)
		fp += errAtAge[0].shape[0]
		
	tp = 0
	fn = 0
	for error in range(ageThIdx,lastAge):
		errAtAge = errors[error]
		
		falseNegative = np.where(errAtAge < fullTh)
		fn += falseNegative[0].shape[0]
		truePositive = np.where(errAtAge >= fullTh)
		tp += truePositive[0].shape[0]
		
	recall = tp / (tp+fn)
	precision = tp / (tp + fp)
	fscore = 2 * precision * recall / (precision + recall)
	
	
	print("Fscore: %f Precision: %f Recall: %f" % (fscore,precision,recall))
	if(True):
		print("TP %d FN %d FP %d" % (tp,fn,fp))
		
		#fig = plt.figure()
		plt.boxplot(errors,sym='',whis=[100-lastPerc, lastPerc]) #
		plt.axhline(y=fullTh, color='blue', linestyle='-')
		plt.axvline(x=(ageThIdx+0.5),color='gray',)
		plt.axvline(x=(lastAge+0.5),color='gray',)
		plt.xticks(range(1,len(labels)+1),labels)
		plt.title(title)
		plt.grid()
		plt.xlabel('Q')
		plt.ylabel('MAE')
		if(save):
			plt.savefig(title, bbox_inches='tight')
			plt.close()
		else:
			plt.show()
	
	return precision,recall
		
		
def train(minerva,astrea,K,encSize,type="Dense"):
	
	train_ageScale = 100
	batteries = minerva.ets.loadSyntheticBlowDataSet(train_ageScale)
	k_idx,k_data = astrea.kfoldByKind(batteries,K)
	
	if(False):
		degraded = []
		for i in range(50,110,10):
			
			corrupted = minerva.ets.loadSyntheticBlowDataSet(i)
			degraded.append(corrupted)
		
		#degradationPerc = [.02,.05,.10,.15,.20]
		#degradationPerc = [.02,.04,.06,.08,.15]
		degradationPerc = [.02,.04,.06,.08,.10]
		#degradationPerc = [.01,.02,.03,.04,.05]
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
		#for tau in range(75,100,5):
		for tau in range(90,100,2):
			#for i in range(1,6):
			i = 4
			mapTable(encSize,type,i,tau)
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