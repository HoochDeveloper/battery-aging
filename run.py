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
	evaluate(minerva,astrea,K,encSize,scaler,range(100,75,-5),show=False,showScatter=False,type=type)
	
def loadEvaluation(encSize,K=3,type="Dense"):
	mustPlot = True
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
		name4model = modelNameTemplate % (encSize,100,type,count)
		[maes,lab] = ets.loadZip(maeFolder,name4model+".out")
		
		labels = []
		for i in range(0,len(lab)):
			mul = 100 - (5*i)
			q = int(340 * mul / 100)
			labels.append("%d" % q)
		
		x,y,n = __evaluation(maes,labels,name4model)
		if(mustPlot):
			#plt.plot(x, y, label="Conv2D")
			ax.scatter(x, y, label=type)
			for i, txt in enumerate(n):
				ax.annotate(txt, (x[i], y[i]))
	
			plt.xlabel('FPR')
			plt.ylabel('TPR')
			plt.legend()
			plt.grid()
			plt.show()

def __evaluation(maes,labels,name4model):
	
	tit = "MAE %s" % name4model 
	
	x = []
	y = []
	n = []
	
	a = np.zeros((50,3))
	
	i = 0
	print(name4model)
	
	#population = [0.95,0.80,0.60,0.35]
	population = [0.95,0.60,0.59,0.35]
	for perc in range(80,99):
		#precision,recall,fprate = precisionRecallOnRandPopulation(maes,perc,population)
		#x.append(fprate)
		#y.append(recall)
		
		precision,recall = errorBoxPlot(maes,labels,tit,lastPerc=perc,save=False)
		
		fscore = 2 * precision * recall / (precision + recall)
		
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
			
		
	for c in range(0,K):
		name4model = modelNameTemplate % (encSize,100,type,c+1)
		maes = mae2Save[c]
		labels = lab2Save[c]
		minerva.ets.saveZip(maeFolder,name4model+".out",[maes,labels])

def errorBoxPlot(errors,labels,title,lastPerc=90,save=True,plot=False):
	
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
	
	ageThIdx = 2
	lastAge = 5 #len(errors)
	for error in range(1,ageThIdx):
		errAtAge = errors[error]
		errAtAge = np.where(errAtAge >= fullTh)
		fp += errAtAge[0].shape[0]
		
	tp = 0
	fn = 0
	for error in range(ageThIdx+1,lastAge):
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
		plt.axvline(x=(ageThIdx+1.5),color='gray',)
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
	elif(action=="learning_curve"):
		learningCurve(encSize,type,K)
	else:
		print("Can't perform %s" % action)


main()