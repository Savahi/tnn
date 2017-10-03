import importlib

from tnn.network import Network
from data_handler.load_raw_data import *


def split(data,rate=0.7):
	index = int (len(data) * rate)
	train = data[:index]
	test  = data[index:]
	return train, test

def calib (configfile)
	config = importlib.import_module("calib."+configfile)
	config = config.config

	netw = Network(
		num_nodes = config["network"]["nodes"],
		num_layers= len(config["network"]["nodes"]),
		num_features= config["network"]["num_inputs"],
		numLabels=config["network"]["bins"],
	)

	x,y,profit = [],[],[] # а вот сюда, Савелий, пойдет функция генерации инпутов
	assert (len(x) == len(y) and len(y) == len(profit))

	xtrain,xtest = split(x)
	ytrain,ytest = split(y)
	ptrain,ptest = None, None #split(profit)

	netw.learn(xtrain, ytrain, profit=ptrain, xTest=xtest, yTest=ytest, profitTest=ptest,
		   learningRate=config["learningRate"], 
		   numEpochs=config["numEpochs"], 	
		   activationFuncs=config["activationFuncs"]
		   optimizer=config["optimizer"],
        	   summaryDir=config["summaryDir"], 
		   printRate=config["printRate"],
		   trainTestRegression=config["trainTestRegression"])		
	
