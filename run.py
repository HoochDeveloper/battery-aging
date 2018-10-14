import uuid,time,os,logging, numpy as np, sys, matplotlib.pyplot as plt
from logging import handlers as loghds

from sklearn.metrics import auc


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
	force = True
	if not os.path.exists(fullPath) or  force:

		if not os.path.exists(mapFolder):
			os.makedirs(mapFolder)

		from Minerva import Minerva
		minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)	

		nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
		tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
		astrea = Astrea(tsIndex,nameIndex,minerva.ets.keepY)	
	
		[loadmaes,labels] = minerva.ets.loadZip(maeFolder,name4model+".out",)
		
		fullHealthError = loadmaes[0]
		tmp = np.percentile(fullHealthError,[thresholdPercentile])
		thresholdValue = tmp[0]
		
		dataSet = pd.DataFrame({'MAE' : [],'TP' : [], "TN" : [], "FP":[], "FN":[]})
		
		np.random.seed(42)
		prob = np.random.rand(len(loadmaes[0]))
		
		tp = []
		tn = []
		fp = []
		fn = []
		mapMaes = []
		p = 0
		n = 0
		for i in range(0,len(prob)):
			
			if prob[i] >= .95:
				b = 4 # 80 SOH
			elif prob[i] >= .85:
				b = 3 # 85 SOH
			elif prob[i] >= .75:
				b = 2 # 90 SOH
			elif prob[i] >= .35:
				b = 1 # 95 SOH
			else:
				b = 0 # 100 SOH
			
			mae = loadmaes[b][i]
			mapMaes.append(mae)
			if(b <= 1):
				n += 1
				if(mae >= thresholdValue):
					fp.append(1); tp.append(0); fn.append(0); tn.append(0)
				else:
					fp.append(0); tp.append(0); fn.append(0); tn.append(1)
			else:
				p +=1
				if(mae >= thresholdValue):
					fp.append(0); tp.append(1); fn.append(0); tn.append(0)
				else:
					fp.append(0); tp.append(0); fn.append(1); tn.append(0)
				
			
		df = pd.DataFrame({'MAE':mapMaes,'TP':tp, 'TN':tn, 'FP':fp, 'FN':fn})
		dataSet = dataSet.append(df)
		
		dataSet.sort_values(by="MAE",ascending=False,inplace=True)
		
		intervention = 150
		right = dataSet.head(intervention)["TP"].sum()
		
		print(right / intervention)
		#print(dataSet.head(100))
		
		tmp = dataSet.loc[ (dataSet["TP"] == 1) | (dataSet["FP"] == 1) ]
		
		rcl = tmp["TP"].cumsum() / p
		prc = tmp["TP"].cumsum() / (tmp["TP"].cumsum() + tmp["FP"].cumsum())
		posDataset = pd.DataFrame({'RCL':rcl,'PRC':prc})
		

		dataSet.sort_values(by="MAE",ascending=True,inplace=True)
		tmp = dataSet.loc[ (dataSet["TN"] == 1) | (dataSet["FN"] == 1) ]
		rcl = tmp["TN"].cumsum() / n
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
	

	
def execute(mustTrain,encSize = 8,K = 3,type="Dense"):
	from Minerva import Minerva
	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)	

	nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
	tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
	astrea = Astrea(tsIndex,nameIndex,minerva.ets.keepY)	
	
	K = 6
	batteries = minerva.ets.loadSyntheticBlowDataSet(100)
	_,k_data = astrea.kfoldByKind(batteries,K)
	scaler = astrea.getScaler(k_data)

	folds4learn = []
	for i in range(len(k_data)-1):
		fold = k_data[i]
		foldAs3d = astrea.foldAs3DArray(fold,scaler)
		folds4learn.append(foldAs3d)
	
	if(mustTrain):
		train(minerva,astrea,K,encSize,folds4learn,type=type)
	
	#testFold = k_data[-1]
	evaluate(minerva,astrea,K,encSize,scaler,range(100,75,-5),folds4learn,show=False,showScatter=False,type=type)
	
def loadEvaluation(encSize,K=3,type="Dense"):
	mustPlot = True

	name4model = None
	# search the only best model saved
	for count in range(1,K+1):
		tmp = modelNameTemplate % (encSize,100,type,count)
		if(os.path.exists(os.path.join(maeFolder,tmp+".out"))):
			name4model = tmp
			break
	
	fig, ax = plt.subplots()
	ets = EpisodedTimeSeries(5,5,5,5)
	[maes,lab] = ets.loadZip(maeFolder,name4model+".out")
	labels = []
	for i in range(0,len(lab)):
		mul = 100 - (5*i)
		q = int(340 * mul / 100)
		labels.append("%d" % q)
	
	x,y,n = __evaluation(maes,labels,name4model,evalBox= not mustPlot)

	if(mustPlot):
		print("AUROC %f" % auc(np.asarray(x),np.asarray(y)))
		ax.scatter(x, y, label=type)
		for i, txt in enumerate(n):
			ax.annotate(txt, (x[i], y[i]))
		
		ax.plot((0, 1), (0, 1))
		
		plt.xlabel('FPR')
		plt.ylabel('TPR')
		plt.legend()
		plt.grid()
		plt.show()

def __evaluation(maes,labels,name4model, evalBox=False):
	
	tit = "MAE %s" % name4model 
	
	x = []
	y = []
	n = []
	
	a = np.zeros((100,3))
	
	i = 0
	print(name4model)
	
	#population = [0.90,0.80,0.70,0.25]
	population = [0.95,0.85,0.45,0.15]
	#for perc in range(86,87):
	
	bestScore = 0
	bestTh = 0
	bestPrc = 0
	bestRecall = 0
	
	if(evalBox == False):
		ran = range(10,99)
	else:
		#ran = range(85,86)
		ran = range(95,96)
	
	for perc in ran:
		
		if(evalBox == False):
			precision,recall,fprate = precisionRecallOnRandPopulation(maes,perc,population)
			x.append(fprate)
			y.append(recall)
		else:
			plot=True
			precision,recall = errorBoxPlot(maes,labels,tit,lastPerc=perc,save=False,plot = plot)
			x.append(recall)
			y.append(precision)
		
		fscore = 2 * precision * recall / (precision + recall)
		if(fscore > bestScore):
			bestScore = fscore
			bestTh = perc
			bestPrc = precision
			bestRecall = recall
		
		n.append(perc)
		
		a[i,0] = "{:10.3f}".format(perc) 
		a[i,1] = "{:10.3f}".format(precision)
		a[i,2] = "{:10.3f}".format(recall)
		i+=1
	if(False):
		print (" \\\\\n".join([" & ".join(map(str,line)) for line in a]) )
	
	print ("Best fscore is %f at threshold %d. Prec: %f Recall: %f" 
		% (bestScore, bestTh, bestPrc, bestRecall))
	return x,y,n
			

def precisionRecallOnRandPopulation(errors,lowTH,population):
	
	#percFull = np.percentile(errors[0],[25,75])
	#iqr = percFull[1] - percFull[0]
	#fullTh = percFull[1] + 1.5*iqr #
	
	percFull = np.percentile(errors[0],[lowTH])
	fullTh = percFull[0]
	

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
			#if(errors[4][i] < fullTh or errors[4][i] > upperTh):
			degraded.append(errors[4][i])
			maes.append(errors[4][i])
			if(errors[4][i] >= fullTh):
				mTP.append(1); mFP.append(0); mTN.append(0); mFN.append(0); 
			else:
				mTP.append(0); mFP.append(0); mTN.append(0); mFN.append(1); 
			#else:
			#	unknown += 1
		elif(prob[i] >= population[1]):
			#if(errors[3][i] < fullTh or errors[3][i] > upperTh):
			degraded.append(errors[3][i])
			maes.append(errors[3][i])
			if(errors[3][i] >= fullTh):
				mTP.append(1); mFP.append(0); mTN.append(0); mFN.append(0); 
			else:
				mTP.append(0); mFP.append(0); mTN.append(0); mFN.append(1); 
			#else:
			#	unknown += 1
		elif(prob[i] >= population[2]): # 90
			x = 0
			#healthly.append(errors[2][i])
			#maes.append(errors[2][i])
			#if(errors[2][i] >= fullTh):
			#	mTP.append(0); mFP.append(1); mTN.append(0); mFN.append(0); 
			#else:
			#	mTP.append(0); mFP.append(0); mTN.append(1); mFN.append(0);
			
			
			degraded.append(errors[2][i])
			maes.append(errors[2][i])
			if(errors[2][i] >= fullTh):
				mTP.append(1); mFP.append(0); mTN.append(0); mFN.append(0); 
			else:
				mTP.append(0); mFP.append(0); mTN.append(0); mFN.append(1); 

		elif(prob[i] >= population[3]): # 95
			#if(errors[1][i] < fullTh or errors[1][i] > upperTh):
			healthly.append(errors[1][i])
			maes.append(errors[1][i])
			if(errors[1][i] >= fullTh):
				mTP.append(0); mFP.append(1); mTN.append(0); mFN.append(0); 
			else:
				mTP.append(0); mFP.append(0); mTN.append(1); mFN.append(0); 
			#else:
			#	unknown += 1
		else: # 100
			#if(errors[0][i] < fullTh or errors[0][i] > upperTh):
			healthly.append(errors[0][i])
			maes.append(errors[1][i])
			if(errors[0][i] >= fullTh):
				mTP.append(0); mFP.append(1); mTN.append(0); mFN.append(0); 
			else:
				mTP.append(0); mFP.append(0); mTN.append(1); mFN.append(0); 
			#else:
			#	unknown += 1
	
	#print(unknown)
	#print(len(prob)-unknown)
	
	falseNegative = np.where(degraded < fullTh)
	fn = falseNegative[0].shape[0]
	truePositive = np.where(degraded >= fullTh)
	tp = truePositive[0].shape[0]
	
	falsePositive = np.where(healthly >= fullTh)
	fp = falsePositive[0].shape[0]
	
	trueNegative = np.where(healthly < fullTh)
	tn = trueNegative[0].shape[0]
	
	recall = tp / (tp+fn )
	fpRate = fp / (fp + tn )
	
	precision = tp / (tp + fp)
	fscore = 2 * precision * recall / (precision + recall)
	
	
	
	if(False):
		print("TP %d FP %d TN %d FN %d" % (tp,fp,tn,fn))
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
		print("Fscore: %f Precision: %f Recall: %f" % (fscore,precision,recall))	
		boxes = [np.asarray(healthly), np.asarray(degraded)]
		plt.boxplot(boxes,sym='',whis=[100-lowTH, lowTH]) #
		plt.axhline(y=fullTh, color='blue', linestyle='-')
		#plt.axhline(y=upperTh, color='blue', linestyle='-')
		#plt.axvline(x=(ageThIdx+0.5),color='gray',)
		#plt.axvline(x=(lastAge+0.5),color='gray',)
		plt.xticks(range(1,3),["Healthly","Degraded"])
		plt.title("%d" % (lowTH) )
		plt.grid()
		plt.xlabel('Status')
		plt.ylabel('MAE')
		plt.show()

	return precision,recall,fpRate
	
def evaluate(minerva,astrea,K,encSize,scaler,ageScales,folds4learn,type="Dense",show=False,showScatter=False,boxPlot=False):

	if not os.path.exists(maeFolder):
		os.makedirs(maeFolder)
	
	#selecting best model to evaluate on test set
	
	print("Selecting best model on train folds")
	bestMae = float('Inf')
	bestModel = None
	
	ageScale = 100
	batteries = minerva.ets.loadSyntheticBlowDataSet(ageScale)
	_,k_data = astrea.kfoldByKind(batteries,K)
	
	trainIdx,valIdx = astrea.leaveOneFoldOut(K-1)	
	count = 0
	for train_index, valid_index in zip(trainIdx,valIdx): 
		# TRAIN #VALID
		validX = [folds4learn[i] for i in valid_index]
		validX =  np.concatenate(validX)		
		count += 1
		name4model = modelNameTemplate % (encSize,100,type,count)
		mae = minerva.evaluateModelOnArray(validX, validX,name4model,plotMode,scaler,False)
		score = np.mean(mae)
		if(score < bestMae):
			bestMae = score
			bestModel = name4model
		
	print("Best model is %s with mae %f" % ( bestModel, bestMae) )

	mae2Save = [None] * (1)
	lab2Save = [None] * (1)
	for c in range(0,1):
		foldMaes = [None] * len(ageScales)
		foldLabels = [None] * len(ageScales)
		mae2Save[c] = foldMaes
		lab2Save[c] = foldLabels
	
	count = 0
	for a in range(0,len(ageScales)):
		
		ageScale = ageScales[a]
		batteries = minerva.ets.loadSyntheticBlowDataSet(ageScale)
		_,k_data = astrea.kfoldByKind(batteries,K)
		
		folds4test = []
		fold = k_data[-1]
		foldAs3d = astrea.foldAs3DArray(fold,scaler)
		folds4test.append(foldAs3d)
		test =  np.concatenate(folds4test)
		
		print("Model %s Age: %d" % (bestModel,ageScale))	
		mae = minerva.evaluateModelOnArray(test, test,bestModel,plotMode,scaler,False)
		mae2Save[count][a] = mae
		lab2Save[count][a] = "SOH %d" % ageScale
	# end evaluation on all ages
	maes = mae2Save[0]
	labels = lab2Save[0]
	minerva.ets.saveZip(maeFolder,bestModel+".out",[maes,labels])

def errorBoxPlot(errors,labels,title,lastPerc=90,save=True,plot=False):
	
		
	fp = 0
	#percFull = np.percentile(errors[0],[25,75])
	#iqd = (percFull[1] - percFull[0]) / 3
	#fullTh =  percFull[1] + iqd
	
	percFull = np.percentile(errors[0],[lastPerc])
	fullTh =  percFull[0]
	
	errAtAge = errors[0]
	errAtAge = np.where(errAtAge >= fullTh)
	fp += errAtAge[0].shape[0]
	
	ageThIdx = 2
	lastAge = 5 #len(errors)
	for error in range(1,ageThIdx):
		errAtAge = errors[error]
		errAtAge = np.where(errAtAge >= fullTh)
		fp += errAtAge[0].shape[0]
		
	tp = 0
	fn = 0
	#for error in range(ageThIdx+1,lastAge):
	for error in range(ageThIdx,lastAge):
		errAtAge = errors[error]
		
		falseNegative = np.where(errAtAge < fullTh)
		fn += falseNegative[0].shape[0]
		truePositive = np.where(errAtAge >= fullTh)
		tp += truePositive[0].shape[0]
		
	recall = tp / (tp+fn)
	precision = tp / (tp + fp)
	
	fscore = 2 * precision * recall / (precision + recall)
	
	
	
	if(plot):
		print("Fscore: %f Precision: %f Recall: %f" % (fscore,precision,recall))
		print("TP %d FN %d FP %d" % (tp,fn,fp))
		
		#fig = plt.figure()
		plt.boxplot(errors,sym='',whis=[100-lastPerc, lastPerc]) #
		plt.axhline(y=fullTh, color='blue', linestyle='-')
		plt.axvline(x=(ageThIdx+0.5),color='gray',)
		#plt.axvline(x=(ageThIdx+1.5),color='gray',)
		#plt.axvline(x=(lastAge+0.5),color='gray',)
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

def train(minerva,astrea,K,encSize,folds4learn,type="Dense"):
	trainIdx,valIdx = astrea.leaveOneFoldOut(K-1)	
	count = 0
	for train_index, valid_index in zip(trainIdx,valIdx): 
		# TRAIN #VALID
		trainStr = ""
		trainX = [folds4learn[i] for i in train_index]
		trainX = np.concatenate(trainX)
		validX = [folds4learn[i] for i in valid_index]
		validX =  np.concatenate(validX)		
		count += 1
		
		name4model = modelNameTemplate % (encSize,100,type,count)
		minerva.trainlModelOnArray(trainX, trainX, validX, validX,
			name4model,encodedSize = encSize)
		
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
	elif(action=="show_evaluation"):
		loadEvaluation(encSize,type=type, K = K)
		#mapTable(encSize,type,1,98)
	elif(action=="learning_curve"):
		learningCurve(encSize,type,K)
	else:
		print("Can't perform %s" % action)


main()