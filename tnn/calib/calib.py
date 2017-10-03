# -*- coding: utf-8 -*- 
import importlib

from tnn.network import Network
from tnn.data_handler.load_raw_data import *


def split(data,rate=0.7):
	index = int (len(data) * rate)
	train = data[:index]
	test  = data[index:]
	return train, test

def calib (configfile):
	config = importlib.import_module("tnn.calib."+configfile)
	config = config.params

	netw = Network(
		numNodes = config["network"]["nodes"],
		numLayers= len(config["network"]["nodes"]),
		numFeatures= config["network"]["num_inputs"],
		numLabels=config["network"]["bins"],
		activationFuncs=config["network"]["activationFuncs"],
	)

	x,y,profit = example_data() # а вот сюда, Савелий, пойдет функция генерации инпутов
	#assert (len(x) == len(y) and len(y) == len(profit))

	xtrain,xtest = split(x)
	ytrain,ytest = split(y)
	ptrain,ptest = split(profit)
	netw.learn(xtrain, ytrain, profit=ptrain, xTest=xtest, yTest=ytest, profitTest=ptest,
		   learningRate=config["learningRate"], 
		   numEpochs=config["numEpochs"], 	
		   optimizer=config["optimizer"],
        	   summaryDir=config["summaryDir"], 
		   trainTestRegression=config["trainTestRegression"])		
	

if __name__=="__main__":
	calib("config")
