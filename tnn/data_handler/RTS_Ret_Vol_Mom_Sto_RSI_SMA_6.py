# -*- coding: utf-8 -*- 
import taft
import numpy as np
import sys

def calc_data(pastRates, futureRates):
	history_tail = 5
	indicator_period = 25 # for some reason, doesn't work on smaller values TODO
	if len(pastRates['cl']) < max( history_tail, indicator_period ):
		return None, None, None

	ahead = 1
	if ahead+1 >= len(futureRates['cl']):
		return None, None, None

	# Inputs
	def history(nback):
		return {x:(y if nback is 0 else y[:-nback]) for x,y in pastRates.items()}

	indicators = {
		"ret": lambda rates: ( rates['cl'][0] - rates['op'][0] ) / rates['op'][0] ,
		"vol": lambda rates: rates['vol'][0] ,
		"mom": lambda rates: rates['cl'][0] - rates['cl'][1] ,
		"sto": lambda rates: taft.stochastic ( hi=rates['hi'], lo=rates['lo'], cl=rates['cl']) ['K'] ,
		"rsi": lambda rates: taft.rsi (rates=rates['cl']) ['rsi'],
		"sma": lambda rates: taft.sma (rates=rates['cl']) ,
	}

	def single(shift):
		rates = history(shift)
		inputs = [indicators[f](rates) for f in sorted(indicators.keys())]
		return inputs

	inputs = []
	for i in range(history_tail):
		inputs += (single(i))
	inputs = np.array(inputs)

	# Outputs

	# Вычисляем "аутпут" - отношение (CLOSE-OPEN) / (HIGH-LOW) на указанном (переменной ahead) отрезке "ближайшего будущего".
	# Каждое значения "аутпута" будет отнесено к одной из трех категорий и представлено в виде one-hot вектора длиной 3.
	# Маленькие значения будут кодироваться [1,0,0], средние - [0,1,0], большие - [0,0,1].  
	bins = 3
	op = futureRates['op'][0]
	cl = futureRates['cl'][ahead]
	hi = np.max( futureRates['hi'][:ahead+1] )
	lo = np.min( futureRates['lo'][:ahead+1] )
	clLessOp = cl - op
	hiLessLo = hi - lo
	if hiLessLo > 0:
		observed = clLessOp / hiLessLo
	else:
		observed = 0.0
	observedBin = int( float(bins) * ( (observed + 1.0) / (2.0 + 1e-10) ) )
	output = np.zeros( shape=[bins], dtype=np.float32 )
	output[observedBin] = 1.0

	profit = clLessOp

	# Print doneness 
	done = len(pastRates['cl'])
	todo = len(futureRates['cl'])
	sys.stdout.write ("\rProcessing inputs {}% done".format (int (100.0 * done / (done+todo))) )
	sys.stdout.flush()
	return inputs, output, profit
