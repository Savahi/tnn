from tnn.io import prepareData, loadNetwork
import numpy as np
import sys
from tnn.calib import graph
from tnn.network import Network
from tnn.boosting import aggregate
from tnn.data_handler.RTS_Ret_Vol_Mom_Sto_RSI_SMA_6 import calc_data


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
	#data = {key:np.array(list(trainData[key])+list(testData[key])) for key in trainData.keys()}
	data = np.array(list(trainData['inputs'])+list(testData['inputs']))
	profits= np.array(list(trainData['profit'])+list(testData['profit']))
	pnl = [0] # this zero is a mempty, to be removed
	for input, profit, _ in zip(data,profits,range(500)): #TODO: remove limitation
		decision = nn.calcOutput(input)[0]
		#I'll put in laxer trading decision rules		
		#if max(decision) is decision[0]:
		if decision[0]>0.2:
			# short
			pnl.append(pnl[-1]-profit)
		#elif max(decision) is decision[-1]:
		elif decision[-1]>0.2:
			# long
			pnl.append(pnl[-1]+profit)
		else:
			# stay
			pnl.append(pnl[-1])
		
		sys.stdout.write ("\rTrading {}% done".format (int (100.0 * len(pnl) / 500)) )
		sys.stdout.flush()
	pnl = pnl[1:]
	print(pnl, len(pnl))

def trade_aggregate(NNfiles, calcDatas, fileWithRates):
	NNs = []
	for NNfile in NNfiles:
		nn=loadNetwork(NNfile)
		if nn is None:
			print "Failed to load network, exiting"
			return

		NNs.append(nn)

	get_decisions = aggregate(NNs, calcDatas)

	
### boosting

def decisionLogic_default(decisions):
	return [sum(x) for x in zip(*decisions[0])]

def aggregate(NNs, calcDatas, decisionLogic=decisionLogic_default):

	def get_decisions(fileWithRates=None):
		data = [prepare_data(fileWithRates=fileWithRates, calcData=x) for x in calc_datas]
		
	
	return get_decisions
	

if __name__=="__main__":
	NNfile = "20171017161235/999_c_2.219_a_0.3883_p_8.867e+04"
	fileWithRates = "../tnn-test/RTS_1h_150820_170820.txt"
	calcData = calc_data(
		trans_cost=10,
		indnames=["Return","Volume","Momentum","Stochastic","RSI","SMA"],
		history_tail=5,
	)
	trade_single(NNfile, fileWithRates,calcData)
