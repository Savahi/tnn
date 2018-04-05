from tnn.io import prepareData, loadNetwork
import numpy as np
import sys
from tnn.calib import graph
from tnn.network import Network
#from tnn.boosting import aggregate
from tnn.data_handler.RTS_Ret_Vol_Mom_Sto_RSI_SMA_6 import calc_data

def simple_sum(threshold = 2):
	def f(decisions):
		d = sum(decisions)
		if d >= threshold:
			return 1
		elif d <= -threshold:
			return -1
		else:
			return 0
	return f

def processData(fileWithRates, calcData):
	trainData, testData = prepareData(fileWithRates=fileWithRates, calcData=calcData)
	#data = {key:np.array(list(trainData[key])+list(testData[key])) for key in trainData.keys()}
	data = np.array(list(trainData['inputs'])+list(testData['inputs']))
	profits= np.array(list(trainData['profit'])+list(testData['profit']))
	return {'inputs':data, 'profits':profits}


def trade_single(NNfile, fileWithRates,calcData=None):
	nn = loadNetwork(NNfile)
	if nn is None:
		print "Failed to load network, exiting"
		return
	
	trainData, testData = prepareData(fileWithRates=fileWithRates, calcData=calcData)
	data = np.array(list(trainData['inputs'])+list(testData['inputs']))
	profits= np.array(list(trainData['profit'])+list(testData['profit']))
	print ("Trading...")
	decisions = nn.calcOutputs(data)
	pnl = [0]
	for decision, profit in zip(decisions, profits):		
		if max(decision) in (decision[0], decision[1]): # short
			pnl.append(pnl[-1]-profit)
		elif max(decision) in (decision[-1], decision[-2]): # long
			pnl.append(pnl[-1]+profit)
		else:
			pnl.append(0)
	pnl = pnl[1:]
	return {"decisions":decisions, "pnl":pnl, "profits":profits}

# old trade aggregate, without flipover trading
"""def trade_aggregate(NNfiles, fileWithRates,calcDatas,  aggregateDecisions=None):
	#ggregateDecisions = aggregateDecisions or simple_sum
	results = [trade_single(x,fileWithRates,y) for x,y in zip(NNfiles,calcDatas)]
	decisions=[x["decisions"] for x in results]
	print(decisions)
	pnls = [x["pnl"] for x in results]
	profits=results[0]["profits"]
	decisionPoints = zip(*decisions)
	totalDecisions = [aggregateDecisions(x) for x in decisionPoints]
	pnl = [0]
	for decision, profit in zip(totalDecisions, profits):		
		if max(decision) == (decision[0]): # short
			pnl.append(pnl[-1]-profit)
		elif max(decision) == (decision[-1]): # long
			pnl.append(pnl[-1]+profit)
		else:
			pnl.append(0)
	pnl = pnl[1:]
	return {"decisions":decisions, "pnl":pnl, "profits":profits}"""

# with flipover trading
def trade_aggregate(NNfiles, fileWithRates,calcDatas,  aggregateDecisions=None):

	def flatten_decisions(decisions):
		res = []
		for decision in decisions:
			if max(decision) == (decision[0]):
				res.append(-1)
			elif max(decision) == decision[-1]:
				res.append(+1)
			else:
				res.append(0)
		return res

	def flipover(decisions):
		res = [0] #extra 0 to be removed
		for i in range(len(decisions)):
			point = decisions[i]
			if point is 0:
				res.append(res[-1])
			else:
				res.append(point)
		return res[1:]
				

	results = [trade_single(x,fileWithRates,y) for x,y in zip(NNfiles,calcDatas)]
	decisions=[x["decisions"] for x in results]
	flat_decisions = [ flatten_decisions(x) for x in decisions]
	flip_decisions = [ flipover(x) for x in flat_decisions]
 	pnls = [x["pnl"] for x in results]
	profits=results[0]["profits"]
	decisionPoints = zip(*flip_decisions)
	totalDecisions = [aggregateDecisions(x) for x in decisionPoints]
	flipTotalDecisions = flipover(totalDecisions)
	pnl = [0]
	for decision, profit in zip(flipTotalDecisions, profits):		
		if decision == -1: # short
			pnl.append(pnl[-1]-profit)
		elif decision == +1: # long
			pnl.append(pnl[-1]+profit)
		else:
			pnl.append(0)
	pnl = pnl[1:]
	return {"decisions":decisions, "pnl":pnl, "profits":profits}

def show_graphs(result):
	import matplotlib.pyplot as plt

	# Train
	plt.figure ()
	plt.plot(result["pnl"])
	plt.show()

if __name__=="__main__":
	from tnn.calib.trade_config import params

	NNFiles = params["networks"]
	fileWithRates=params["fileWithRates"]
	calcDatas = params["calcDatas"]
	aggregateLogic = params["aggregateLogic"]
	result=None
	if len(NNFiles) is 1:
		result = trade_single(NNFiles[0], fileWithRates, calcDatas[0])
	else:
		result = trade_aggregate(NNFiles, fileWithRates, calcDatas, aggregateLogic)

	show_graphs(result)
