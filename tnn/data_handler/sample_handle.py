# -*- coding: utf-8 -*- 
import datetime as dt
import numpy as np
import taft # Импортируем библиотеку для работы с индикаторами и моделирования сделок
import sys
import re
import shelve


def calcData(rates):
	op = rates['op']
	hi = rates['hi']
	lo = rates['lo']
	cl = rates['cl']
	dtm = rates['dtm']
	vol = rates['vol']
	length = rates['length']

	bins = 3

	hiLessLo = hi - lo
	clLessOp = cl - op
	opLessLo = op - lo
	hiLessOp = hi - op

	nnInputs = []
	nnObserved = []
	nnLabels = []
	nnProfit = []
	for i in range(length-1,0,-1):
		# Inputs
		inputs = getInputs( candles, i )
		if inputs is not None:
			nnInputs.append( inputs )
			nnProfit.append( clLessOp[i-1] )

			if hiLessLo[i] > 0:
				observed = clLessOp[i-1] / hiLessLo[i-1]
			else:
				observed = 0
			observedBin = int( float(bins) * ( (observed + 1.0) / (2.0 + 1e-10) ) )
			labels = np.zeros( shape=[bins], dtype=np.float32 )
			labels[observedBin] = 1.0
			nnObserved.append( observedBin )
			nnLabels.append( labels )
			
	# end of while

	nnInputs = np.array( nnInputs, dtype='float' )
	rows, cols = np.shape( nnInputs )
	nnObserved = np.array( nnObserved, dtype=np.int8 )
	nnLabels = np.array( nnLabels, dtype='float' )
	nnProfit = np.array( nnProfit, dtype='float' )		
	nnMean = np.zeros( shape=[cols], dtype='float' )
	nnStd = np.zeros( shape=[cols], dtype='float' )
	
	if firstTestInPct:
		firstTest = int( rows * firstTest / 100 )

	for i in range(cols):
		status, mean, std = taft.normalize( nnInputs[:,i], normInterval=[0,firstTest] )
		if status is None:
			print "Can't normalize %d column. Exiting..." % (i)
			sys.exit(0)
		nnMean[i] = mean
		nnStd[i] = std

	return {"inputs":nnInputs, "labels":nnLabels, "profits": nnProfit}
	


# Рассчитывает и возвращает массив (list) "инпутов" для заданной точки (candlesNum) внутри массивов котировок (candles)   
# Параметры:
# 	candles - (dict) словарь с котировками, объемами и датами: candles['cl'] - close rates, candles['vol'] - volumes etc
# 	candleNum - (int) индекс (позиция внутри массивов с котировками) для которого рассчитываем "инпуты"
def getInputs( candles, candleNum, bSmooth = False ):

	historyIntervals = [ [0,0], [1,1], [2,2], [3,3], [0,4], [0,5] ]

	inputs = []

	cl0 = candles['cl'][candleNum]

	for i in range(len(historyIntervals)):
		candleStart = candleNum + historyIntervals[i][0]
		candleEnd = candleNum + historyIntervals[i][1]
		if candleEnd >= len(candles['cl']):
			return None

		high = candles['hi'][candleStart:candleEnd+1]
		highestHigh = max( high )
		if bSmooth:		
			meanHigh = np.mean( high )
			stdHigh = np.std( high )
			highestHigh = meanHigh + 3*meanStd
		inputs.append( highestHigh - cl0 )

		low = candles['lo'][candleStart:candleEnd+1]
		lowestLow = min( low )
		if bSmooth:		
			meanLow = np.mean( low )
			stdLow = np.std( low )
			lowestLow = meanLow - 3*meanStd
		inputs.append( cl0 - lowestLow )

	return inputs
# end of getInputRow


