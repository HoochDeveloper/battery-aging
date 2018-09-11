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

## TODO minerva.getEncoded


def codeProjection(encSize,type,K):
	
	ageScales = [100,50]
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
			
			pca = PCA(n_components=3)
			pc = pca.fit_transform(code)
			codes.append(pc)
			#print(pc.shape)
			#plt.scatter(pc[:,0],pc[:,1])
			#plt.show()
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		
		for code in codes:
			ax.scatter(code[:3,0],code[:3,1],code[:3,2])
		plt.show()
			
def execute(mustTrain,encSize = 8,K = 3,type="Dense"):
	from Minerva import Minerva
	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode=plotMode)	
	nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
	tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
	astrea = Astrea(tsIndex,nameIndex,minerva.ets.keepY)	
	
	if(mustTrain):
		train(minerva,astrea,K,encSize,type=type,denoise=False)
	
	batteries = minerva.ets.loadSyntheticBlowDataSet(100)
	k_idx,k_data = astrea.kfoldByKind(batteries,K)
	scaler = astrea.getScaler(k_data)
	evaluate(minerva,astrea,K,encSize,scaler,range(100,75,-5),show=False,showScatter=False,type=type)
	
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
	
	for c in range(0,len(errorList)):
		err = errorList[c]
		prc = np.percentile(err,[25,50,75])
		print("%f	%f	%f" % ( prc[0],prc[1],prc[2] ))
	
	fig = plt.figure()
	plt.boxplot(errorList,sym='',whis=[10, 90]) #
	plt.xticks(range(1,len(labels)+1),labels)
	plt.title(title)
	plt.grid()
	if(save):
		plt.savefig(title, bbox_inches='tight')
		plt.close()
	else:
		plt.show()
		
		
def train(minerva,astrea,K,encSize,type="Dense",denoise=False):
	
	train_ageScale = 100
	batteries = minerva.ets.loadSyntheticBlowDataSet(train_ageScale)
	batteriesDenoise = None
	if(denoise):
		batteriesDenoise = minerva.ets.loadSyntheticBlowDataSet(100)
	#batteries = minerva.ets.loadSyntheticMixedAgeBlowDataSet()
	
	k_idx,k_data = astrea.kfoldByKind(batteries,K)
	d_data = None
	if(batteriesDenoise is not None):
		_,d_data = astrea.kfoldByKind(batteriesDenoise,K)
	scaler = astrea.getScaler(k_data)
	folds4learn = []
	foldsDenoise = []
	for i in range(len(k_data)):
		fold = k_data[i]
		foldAs3d = astrea.foldAs3DArray(fold,scaler)
		folds4learn.append(foldAs3d)
		if(d_data is not None):
			fold = d_data[i]
			foldAs3d = astrea.foldAs3DArray(fold,scaler)
			#print(foldAs3d.shape)
			noise = np.random.normal(0,0.02,foldAs3d.shape)
			
			foldAs3d += noise
			foldsDenoise.append(foldAs3d)

	trainIdx,testIdx = astrea.leaveOneFoldOut(K)
	count = 0
	for train_index, test_index in zip(trainIdx,testIdx): 
		count += 1
		# TRAIN X
		trainX = None
		if len(foldsDenoise) > 0:
			trainX = [foldsDenoise[i] for i in train_index]
		else:
			trainX = [folds4learn[i] for i in train_index]
		trainX = np.concatenate(trainX)
		# TRAIN  Y
		trainY = [folds4learn[i] for i in train_index]
		trainY = np.concatenate(trainY)
		#TEST X
		testX = None
		if len(foldsDenoise) > 0:
			testX = [foldsDenoise[i] for i in test_index]
		else:
			testX = [folds4learn[i] for i in test_index]
		testX =  np.concatenate(testX)
		
		# TEST Y
		testY =  [folds4learn[i] for i in test_index]
		testY =  np.concatenate(testY)
		
		trainX,validX,trainY,validY = train_test_split( trainX, trainY, test_size=0.2,
			random_state=42)
		
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
		for s in range(0,500,100):
			print(history['val_loss'][s])
		print("Train Loss")
		for s in range(0,500,100):
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
	K = 4
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
	elif(action == "show"):
		showModel(encSize,type)
	elif(action == "proj"):
		codeProjection(encSize,type,K)
	else:
		print("Can't perform %s" % action)
		
main()