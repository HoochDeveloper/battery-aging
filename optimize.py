from __future__ import print_function
import tensorflow as tf
from hyperas.distributions import choice, uniform


from hyperopt import Trials, STATUS_OK, tpe, rand
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
	
	K = 2
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
		
		train,valid,_,_ = train_test_split( train, train, test_size=0.2, random_state=42)
		
		X_train = minerva.batchCompatible(100,train)
		Y_train = minerva.batchCompatible(100,train)
		X_test =  minerva.batchCompatible(100,valid)
		Y_test =  minerva.batchCompatible(100,valid)
		break
	
	return X_train, Y_train, X_test, Y_test
	
def denseMoel(X_train, Y_train, X_test, Y_test):
	from keras.models import load_model
	from keras.callbacks import ModelCheckpoint
	from keras.models import Model
	from keras.layers import Dense, Input, Flatten, Reshape, Dropout
	from keras import optimizers
	from Minerva import huber_loss
	from Demetra import EpisodedTimeSeries
	
	ets = EpisodedTimeSeries(5,5,5,5)
	timesteps = 20
	inputFeatures = 2
	outputFeatures = 2
	inputs = Input(shape=(timesteps,inputFeatures),name="IN")
	
	# START HyperParameters
	
	
	dropPerc = 0.5
	codeSize = {{choice([5,7,9,12])}}
	depth = {{choice(['more', 'less'])}}
	drop = {{choice(['drop', 'noDrop'])}}
	
	# END HyperParameters
	d = Dense({{choice([16,32,48,64,96,128])}},activation='relu',name="E1")(inputs)
	if depth == 'more':
		if drop == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense({{choice([16,32,48,64,96,128])}},activation='relu',name="E2")(d)
		if drop == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense({{choice([16,32,48,64,96,128])}},activation='relu',name="E3")(d)
	
	
	
	### s - encoding
	d = Flatten(name="F1")(d) 
	enc = Dense(codeSize,activation='relu',name="ENC")(d)
	### e - encoding
	
	d = Dense(codeSize*timesteps,activation='relu',name="D1")(enc)
	d = Reshape((timesteps, codeSize),name="R1")(d)
	
	if depth == 'more':
		if drop == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense({{choice([16,32,48,64,96,128])}},activation='relu',name="D2")(d)
	
		if drop == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense({{choice([16,32,48,64,96,128])}},activation='relu',name="D3")(d)
		
	
	d = Flatten(name="F2")(d)
	d = Dense(outputFeatures*timesteps,activation='linear',name="D6")(d)
	out = Reshape((timesteps, outputFeatures),name="OUT")(d)
	model = Model(inputs=inputs, outputs=out)
	
	adam = optimizers.Adam(lr={{choice([0.001,0.0005,0.002])}})		
	model.compile(loss=huber_loss, optimizer=adam,metrics=['mae'])
	path4save = "./optimizedModel.h5"
	checkpoint = ModelCheckpoint(path4save, monitor='val_loss', verbose=0,
			save_best_only=True, mode='min')
	
	model.fit(X_train, Y_train,
		verbose = 0,
		batch_size={{choice([50,100,150,200])}},
		epochs=150,
		validation_data=(X_test, Y_test)
		,callbacks=[checkpoint]
	)
	# loading the best model
	customLoss = {'huber_loss': huber_loss}
	model = load_model(path4save
			,custom_objects=customLoss)

	HL, MAE = model.evaluate(X_test, Y_test, verbose=0)
	print("HL: %f MAE: %f" % (HL, MAE))
	return {'loss': HL, 'status': STATUS_OK, 'model': model}

def conv2DModel(X_train, Y_train, X_test, Y_test):
	
	from keras.models import load_model
	from keras.callbacks import ModelCheckpoint
	from keras.models import Model
	from keras.layers import Dense, Input, Flatten, Reshape, Dropout
	from keras import optimizers
	from Minerva import huber_loss
	from keras.layers import Conv2DTranspose, Conv2D
	from Demetra import EpisodedTimeSeries
	#from keras.backend import constant as cnt
	
	ets = EpisodedTimeSeries(5,5,5,5)
	
	inputFeatures = 2
	outputFeatures = 2
	timesteps = 20
	

	dropPerc = 0.5
	strideSize = 2
	codeSize = {{choice([5,7,9,12])}}
	depth = {{choice(['more', 'less'])}}
	drop = {{choice(['drop', 'noDrop'])}}
	
	inputs = Input(shape=(timesteps,inputFeatures),name="IN")
	c = Reshape((5,4,2),name="R2E")(inputs)
	c = Conv2D({{choice([16,32,48,64,96,128])}},strideSize,activation='relu',name="E1")(c)
	
	if depth == 'more':
		if drop  == 'drop':
			c = Dropout(dropPerc)(c)
		c = Conv2D({{choice([16,32,48,64,96,128])}},strideSize,activation='relu',name="E2")(c)
		if drop  == 'drop':
			c = Dropout(dropPerc)(c)
		c = Conv2D({{choice([16,32,48,64,96,128])}},strideSize,activation='relu',name="E3")(c)
	

	preEncodeFlat = Flatten(name="F1")(c) 
	enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
	c = Reshape((1,1,codeSize),name="R2D")(enc)
	c = Conv2DTranspose({{choice([16,32,48,64,96,128])}},strideSize,activation='relu',name="D1")(c)

	if depth == 'more':
		if drop  == 'drop':
			c = Dropout(dropPerc)(c)
		Conv2DTranspose({{choice([16,32,48,64,96,128])}},strideSize,activation='relu',name="D2")(c)
		if drop  == 'drop':
			c = Dropout(dropPerc)(c)
		Conv2DTranspose({{choice([16,32,48,64,96,128])}},strideSize,activation='relu',name="D2")(c)
	
	preDecFlat = Flatten(name="F2")(c) 
	c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(preDecFlat)
	out = Reshape((timesteps, outputFeatures),name="OUT")(c)
	model = Model(inputs=inputs, outputs=out)
	adam = optimizers.Adam(
		lr={{choice([0.001,0.0005,0.002])}}
		,decay={{choice([0.00,0.01,0.005])}}
	)	
	model.compile(loss=huber_loss, optimizer=adam,metrics=['mae'])
	
	path4save = "./optimizedModel.h5"
	checkpoint = ModelCheckpoint(path4save, monitor='val_loss', verbose=0,
			save_best_only=True, mode='min')
	
	model.fit(X_train, Y_train,
		verbose = 0,
		batch_size={{choice([50,100,150,200])}},
		epochs=150,
		validation_data=(X_test, Y_test)
		,callbacks=[checkpoint]
	)
	
	customLoss = {'huber_loss': huber_loss}
	model = load_model(path4save
			,custom_objects=customLoss)
	
	HL , MAE= model.evaluate(X_test, Y_test, verbose=2)
	print("HL: %f MAE: %f" % (HL, MAE))
	return {'loss': HL, 'status': STATUS_OK, 'model': model}

	
def main():
	best_run, best_model = optim.minimize(
										  model = denseMoel,
                                          data=data,
                                          algo=tpe.suggest,
										  #algo=rand.suggest,
                                          max_evals=30,
                                          trials=Trials())
	
	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))
	print(best_run)
		
main()	