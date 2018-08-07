from __future__ import print_function
import tensorflow as tf
from hyperas.distributions import choice


from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
import numpy as np

	
def batchCompatible(batch_size,data):
	"""
	Transform data shape 0 in a multiple of batch_size
	"""
	exceed = data.shape[0] % batch_size
	if(exceed > 0):
		data = data[:-exceed]
	return data

	
def data():
	from Astrea import Astrea
	from Minerva import Minerva
	from sklearn.model_selection import train_test_split
	
	K = 3
	minerva = Minerva(eps1=5,eps2=5,alpha1=5,alpha2=5,plotMode="GUI")	
	nameIndex = minerva.ets.dataHeader.index(minerva.ets.nameIndex)
	tsIndex = minerva.ets.dataHeader.index(minerva.ets.timeIndex)
	astrea = Astrea(tsIndex,nameIndex,minerva.ets.keepY)
	
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
	
	X_train = None
	Y_train = None
	X_test =  None
	Y_test =  None
	
	for train_index, test_index in zip(trainIdx,testIdx): 
		count += 1
		train = [folds4learn[i] for i in train_index]
		train = np.concatenate(train)
		#test =  [folds4learn[i] for i in test_index]
		#test =  np.concatenate(test)
		
		train,valid,_,_ = train_test_split( train, train, test_size=0.2, random_state=42)
		
		X_train = minerva.batchCompatible(100,train)
		Y_train = minerva.batchCompatible(100,train)
		X_test =  minerva.batchCompatible(100,valid)
		Y_test =  minerva.batchCompatible(100,valid)
		break
	
	return X_train, Y_train, X_test, Y_test
	
def model(X_train, Y_train, X_test, Y_test):
	
	from keras.models import Model
	from keras.layers import Dense, Input, Flatten, Reshape, Dropout
	from keras import optimizers
	from Minerva import huber_loss
	
	timesteps = 20
	inputFeatures = 2
	outputFeatures = 2

	inputs = Input(shape=(timesteps,inputFeatures),name="IN")
	
	d = Dense({{choice([64,128,256])}},activation='relu',name="D1")(inputs)
	
	if {{choice(['drop', 'noDrop'])}} == 'drop':
		d = Dropout(.2)(d)

	d = Dense({{choice([32,64,128])}},activation='relu',name="D2")(d)

	
	d = Flatten(name="F1")(d) 
	enc = Dense({{choice([16,8,4])}},activation='relu',name="ENC")(d)
	
	d = Dense(2*timesteps,activation='relu',name="D6")(enc)
	d = Reshape((timesteps, 2),name="R1")(d)
	
	d = Dense({{choice([32,64,128])}},activation='relu',name="D7")(d)
	
	d = Flatten(name="F2")(d)
	d = Dense(outputFeatures*timesteps,activation='linear',name="D11")(d)
	out = Reshape((timesteps, outputFeatures),name="OUT")(d)
	
	model = Model(inputs=inputs, outputs=out)
	
	
	adam = optimizers.Adam()		
	model.compile(loss=huber_loss, optimizer=adam,metrics=['mae'])
	
	model.fit(X_train, Y_train,
		verbose = 0,
		batch_size=100,
		epochs=100,
		validation_data=(X_test, Y_test)
		
	)
	HL, MAE = model.evaluate(X_test, Y_test, verbose=2)
	print("HL: %f MAE: %f" % (HL, MAE))
	return {'loss': HL, 'status': STATUS_OK, 'model': model}


def main():
	best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
	
	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))
	print(best_run)
	
	
main()	