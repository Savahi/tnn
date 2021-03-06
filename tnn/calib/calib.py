# -*- coding: utf-8 -*- 
import importlib

from tnn.io import prepareData
from tnn.calib import graph
from tnn.network import Network
from tnn.data_handler.load_raw_data import *

def calib (configfile):
	config = importlib.import_module("tnn.calib."+configfile)
	config = config.params

	nn = Network(
		numNodes = config["network"]["nodes"],
		numLayers= len(config["network"]["nodes"]),
		numFeatures= config["network"]["num_inputs"],
		numLabels=config["network"]["bins"],
		activationFuncs=config["network"]["activationFuncs"],
	)
	f = open(config["raw_file"], "r")
	trainData, testData = prepareData(fileWithRates=config["raw_file"],detachTest=20, calcData=config["calcData"])
	if trainData is None:
		print "Failed to prepare data.\nExiting..."
		return

	"""nn.learn(trainData['inputs'], trainData['labels'], trainData['profit'], testData['inputs'], testData['labels'], testData['profit'],
		   learningRate=config["learningRate"], 
		   numEpochs=config["numEpochs"], 	
		   optimizer=config["optimizer"],
        	   summaryDir=config["summaryDir"])"""

	nn.learn( trainData['inputs'], trainData['labels'], trainData['profit'], testData['inputs'], testData['labels'], testData['profit'], 
		numEpochs=config["numEpochs"], balancer=0.0, learningRate=config["learningRate"], prognoseProb=None,
		optimizer=config["optimizer"], tradingLabel=None, flipOverTrading=True, learnIndicators=True, saveRate=20 )	

	"""import matplotlib.pyplot as plt
	titleText = "fl=%s, lr=%g, bl=%g, opt=%s, ep=%d fl=%d" % (config["raw_file"], config["learningRate"], 0, config["optimizer"], config["numEpochs"], 0)
	plt.figure(  )
	plt.subplot(221)
	plt.scatter( nn.costTrain, nn.costTest, marker = '+', color = 'blue' )
	plt.title( titleText + "\n\ncost-function: train vs test")
	plt.grid()
	plt.subplot(222)
	plt.scatter( nn.accuracyTrain, nn.accuracyTest, marker = '+', color = 'blue' )
	plt.title("accuracy: train vs test")
	plt.grid()
	plt.subplot(223)
	plt.scatter( nn.tradeAccuracyTrain, nn.tradeAccuracyTest, marker = '+', color = 'blue' )
	plt.title("trade accuracy: train vs test")
	plt.grid()
	plt.subplot(224)
	plt.scatter( nn.balanceTrain, nn.balanceTest, marker = '+', color = 'blue' )
	plt.title("balance: train vs test")
	plt.grid()
		
	plt.gcf().set_size_inches( 16, 8 )
	#plt.savefig( nn.learnDir + ".png", bbox_inches='tight' )

	print ("Showing the plot...")
	plt.show()"""

	"""def cb(plt):
		plt.show()
		plt.savefig( nn.learnDir + ".png", bbox_inches='tight' )"""
	graph.plot_curves(nn)	


if __name__=="__main__":
	calib("config")
