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
	
	
	learnRate =0.001
	codeSize =6
	
	dropPerc = 0.5
	
	e1Size = {{choice([16,32,64,128,256])}}
	e2Size = {{choice([16,32,64,128,256])}}
	e3Size = {{choice([16,32,64,128,256])}}
	
	e2More = {{choice(['more','less'])}}
	e2Drop = {{choice(['drop', 'noDrop'])}}
	e3More = {{choice(['more','less'])}}
	e3Drop = {{choice(['drop', 'noDrop'])}}
	
	
	d1Size = {{choice([16,32,64,128,256])}}
	d2Size = {{choice([16,32,64,128,256])}}
	d3Size = {{choice([16,32,64,128,256])}}
	
	d2More = {{choice(['more','less'])}}
	d2Drop = {{choice(['drop', 'noDrop'])}}
	d3More = {{choice(['more','less'])}}
	d3Drop = {{choice(['drop', 'noDrop'])}}
	
	
	# END HyperParameters
	d = Dense(e1Size,activation='relu',name="E1")(inputs)
	if e2More == 'more':
		if e2Drop == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense(e2Size,activation='relu',name="E2")(d)
	
	if e3More == 'more':
		if e3Drop == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense(e3Size,activation='relu',name="E3")(d)
	
	
	
	### s - encoding
	d = Flatten(name="F1")(d) 
	enc = Dense(codeSize,activation='relu',name="ENC")(d)
	### e - encoding
	
	d = Dense(d1Size*timesteps,activation='relu',name="D1")(enc)
	d = Reshape((timesteps, d1Size),name="R1")(d)
	
	if d2More == 'more':
		if d2Drop == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense(d2Size,activation='relu',name="D2")(d)
	
	if d3More == 'more':
		if d3Drop == 'drop':
			d = Dropout(dropPerc)(d)
		d = Dense(d3Size,activation='relu',name="D3")(d)
		
	
	d = Flatten(name="F2")(d)
	d = Dense(outputFeatures*timesteps,activation='linear',name="D6")(d)
	out = Reshape((timesteps, outputFeatures),name="OUT")(d)
	model = Model(inputs=inputs, outputs=out)
	
	adam = optimizers.Adam(lr=learnRate)		
	model.compile(loss=huber_loss, optimizer=adam,metrics=['mae'])
	path4save = "./optimizedModel.h5"
	checkpoint = ModelCheckpoint(path4save, monitor='val_loss', verbose=0,
			save_best_only=True, mode='min')
	
	model.fit(X_train, Y_train,
		verbose = 0,
		batch_size=100,
		epochs=175,
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
	
	postDec = 5
	dropPerc = 0.5
	strideSize = 2
	codeSize = 8
	
	filt1 = {{choice([16,32,48,64,96,128,192,256])}}
	
	filt2 = {{choice([16,32,48,64,96,128,192,256])}}
	more2 =  {{choice(['more', 'less'])}}
	drop2 = {{choice(['drop', 'noDrop'])}} 
	
	filt3 = {{choice([16,32,48,64,96,128,192,256])}}
	more3 =  {{choice(['more', 'less'])}}
	drop3 = {{choice(['drop', 'noDrop'])}} 
	
	filt4 = {{choice([16,32,48,64,96,128,192,256])}}
	
	filt5 = {{choice([16,32,48,64,96,128,192,256])}}
	more5 =  {{choice(['more', 'less'])}}
	drop5 = {{choice(['drop', 'noDrop'])}} 
	
	filt6 = {{choice([16,32,48,64,96,128,192,256])}}
	more6 =  {{choice(['more', 'less'])}}
	drop6 = {{choice(['drop', 'noDrop'])}} 
	
	inputs = Input(shape=(timesteps,inputFeatures),name="IN")
	c = Reshape((5,4,2),name="R2E")(inputs)
	c = Conv2D(filt1,strideSize,activation='relu',name="E1")(c)
	
	if more2 == 'more':
		if drop2 == 'drop':
			c = Dropout(dropPerc)(c)
		c = Conv2D(filt2,strideSize,activation='relu',name="E2")(c)
	
	if more3 == 'more':
		if drop3 == 'drop':
			c = Dropout(dropPerc)(c)
		c = Conv2D(filt3,strideSize,activation='relu',name="E3")(c)
	
	preEncodeFlat = Flatten(name="F1")(c) 
	enc = Dense(codeSize,activation='relu',name="ENC")(preEncodeFlat)
	c = Reshape((1,1,codeSize),name="R2D")(enc)
	c = Conv2DTranspose(filt4,strideSize,activation='relu',name="D1")(c)

	if more5 == 'more':
		if drop5 == 'drop':
			c = Dropout(dropPerc)(c)
		Conv2DTranspose(filt5,strideSize,activation='relu',name="D2")(c)
	
	if more6 == 'more':
		if drop6 == 'drop':
			c = Dropout(dropPerc)(c)
		Conv2DTranspose(filt6,strideSize,activation='relu',name="D3")(c)
	
	
	preDecFlat = Flatten(name="F2")(c) 
	c = Dense(timesteps*outputFeatures,activation='linear',name="DECODED")(preDecFlat)
	out = Reshape((timesteps, outputFeatures),name="OUT")(c)
	model = Model(inputs=inputs, outputs=out)
	adam = optimizers.Adam()		
	model.compile(loss=huber_loss, optimizer=adam,metrics=['mae'])
	
	path4save = "./optimizedModel.h5"
	checkpoint = ModelCheckpoint(path4save, monitor='val_loss', verbose=0,
			save_best_only=True, mode='min')
	
	model.fit(X_train, Y_train,
		verbose = 0,
		batch_size=100,
		epochs=100,
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
										  model = conv2DModel,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())
	
	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))
	print(best_run)
		
main()	