from tnn.io import prepare_data

def decisionLogic_default(decisions):
	return [sum(x) for x in zip(*decisions[0])]

def aggregate(NNs, calcDatas, decisionLogic=decisionLogic_default):

	def get_decisions(fileWithRates=None):
		data = [prepare_data(fileWithRates=fileWithRates, calcData=x) for x in calc_datas]
		data = [x[1] for x in data] # leaving only Test in
		decisions = [[NN.calcOutput(x) for x in inputs] for NN,inputs in zip(NNs,data)]
		aggregate_decisions = [decisionLogic(x) for x in decisions]
		return aggregate_decisions
	
	return get_decisions
