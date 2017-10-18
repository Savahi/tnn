# -*- coding: utf-8 -*- 
import sys
import re
import shelve
import numpy as np
import tensorflow as tf
import tnn.utils as utils
from tnn.network import Network 
from tnn.calcdata import CalcData
from tnn.io import prepareData
from tnn.utils import countLabels

def main():

	# Получаем имя файла с данными, на которых будем обучать и тестировать сеть (fileWithRates)
	fileWithRates = None
	if len( sys.argv ) >= 2:
		fileWithRates = sys.argv[1].strip()
	if fileWithRates == None:
		print "Use: %s <train-data-file> <parameters (optional)>.\nExiting..." % ( __file__ )
		sys.exit(0)

	# Параметры сети и ее оптимизации (и их значения по умолчанию)
	numLayers = 1 # Число hidden-слоев
	numNodes1 = 36 # Число узлов в 1-м hidden-слое (может быть переопределено ниже)
	numNodes2 = 36 # Число узлов в 2-м hidden-слое (если numLayers > 1) (может быть переопределено ниже)
	numNodes3 = 36 # Число узлов в 3-м hidden-слое (если numLayers > 2) (может быть переопределено ниже)
	numNodes4 = 36 # Число узлов в 4-м hidden-слое (если numLayers > 3) (может быть переопределено ниже) 
	learningRate = 0.050 # Self explained
	prognoseProb = None # Пронозная вероятность - решение на сделку принимается, если значение в "торговом" бине > prognoseProb
	numEpochs = 1000 # Self explained
	optimizer = None # Тип оптимизатора
	balancer = 0.0 # Дополнительный вес для последнего бина при вычислении cost-функции. Используется, если balancer > 0.0
	flipOverTrading = False # Если flip-over=yes, при проверке качества модели имитируется торговля "с переворотом позиции".
	summaryDir = None # Если summary=yes, будет создана папка с summary; tensorboard --logdir=<текущие-дата-и-время_summary>
	saveRate = None # Как часто сохранять конфигурацию (веса) сети; имя папки = текущие дата и время

	# Читаем аргументы командной строки, с помощью которых можно переопределить значения, заданные по умолчанию
	for argvNum in range( 2, len(sys.argv) ):
		matchObj = re.match( r'layers *\= *([0-9\.]+)', sys.argv[argvNum], re.I )
		if matchObj:
			numLayers = int( matchObj.group(1) )
		matchObj = re.match( r'nodes1 *\= *([0-9\.]+)', sys.argv[argvNum], re.I )
		if matchObj:
			numNodes1 = int( matchObj.group(1) )
		matchObj = re.match( r'nodes2 *\= *([0-9\.]+)', sys.argv[argvNum], re.I )
		if matchObj:
			numNodes2 = int( matchObj.group(1) )
		matchObj = re.match( r'nodes3 *\= *([0-9\.]+)', sys.argv[argvNum], re.I )
		if matchObj:
			numNodes3 = int( matchObj.group(1) )
		matchObj = re.match( r'nodes4 *\= *([0-9\.]+)', sys.argv[argvNum], re.I )
		if matchObj:
			numNodes4 = int( matchObj.group(1) )
		matchObj = re.match( r'learning-rate *\= *([0-9\.]+)', sys.argv[argvNum], re.I )
		if matchObj:
			learningRate = float( matchObj.group(1) )
		matchObj = re.match( r'prognose-prob *\= *([0-9\.]+)', sys.argv[argvNum], re.I )
		if matchObj:
			prognoseProb = float( matchObj.group(1) )
		matchObj = re.match( r'epochs *\= *([0-9]+)', sys.argv[argvNum], re.I )
		if matchObj:
			numEpochs = int( matchObj.group(1) )
		matchObj = re.match( r'balancer *\= *([0-9\.\-]+)', sys.argv[argvNum], re.I )
		if matchObj:
			balancer = np.float64( matchObj.group(1) )
		matchObj = re.match( r'optimizer *\= *([a-zA-Z0-9\.\_\-]+)', sys.argv[argvNum], re.I )
		if matchObj:
			optimizer = matchObj.group(1)
		matchObj = re.match( r'flip-over *\= *([yY]|[yY][eE][sS])', sys.argv[argvNum], re.I )
		if matchObj:
			flipOverTrading = True
		matchObj = re.match( r'summary *\= *([yY]|[yY][eE][sS])', sys.argv[argvNum], re.I )
		if matchObj:
			summaryDir = ""
		matchObj = re.match( r'save-rate *\= *([0-9]+)', sys.argv[argvNum], re.I )
		if matchObj:
			saveRate = int( matchObj.group(1) )

	#cdt = CalcData( 5, intraDay=True, tradingDays=[1,2,3], tradingTime=[ [13,None],[14,None],[15,None],[16,None],[17,None] ] )
	calcData = CalcData( 5 )

	calcData.addLookBackOp( "rsi", 0, 6 )
	calcData.addLookBackOp( "stochastic", 0, 6 )
	calcData.addLookBackOp( "roc", 0, 6 )
	calcData.addLookBackOp( "sma", 0, 6 )
	calcData.addLookBackOp( "return", 0, 6 )
	calcData.addLookBackOp( "vol", 0, 6 )

	calcData.addLookAheadOp( "return", 1, bounds=[] )

	# Готовим данные для сети
	trainData, testData = prepareData( fileWithRates=fileWithRates, detachTest=20, calcData=calcData )
	if trainData is None:
		print "Failed to prepare data.\nExiting..."
		sys.exit(0)
	print "Labels: " + str( countLabels( trainData['labels'] ) )

	# for i in range( len( trainData['profit'] ) ):
	#	utils.log( str(trainData['labels'][i]) + ":" + str(trainData['profit'][i]) )
	# utils.log( str( testData['profit'] ) )

	numSamples = trainData['numSamples']
	numFeatures = trainData['numFeatures']
	numLabels = trainData['numLabels']

	# Эту строку - argText - потом выведем в заголовок графика
	argText = "file:%s, lrn:%g, bln:%g, opt:%s, epo:%d flp:%d" % \
		(fileWithRates, learningRate, balancer, optimizer, numEpochs, flipOverTrading)
	if numLayers == 1:
		argText += " nds:%d" % (numNodes1)
	if numLayers >= 2:
		argText += " nds1:%d nds2:%d" % ( numNodes1, numNodes2 )
		if numLayers >= 3:
			argText += " nds3:%d" % (numNodes3)
			if numLayers >= 4:
				argText += " nds4:%d" % (numNodes4)
	if prognoseProb is not None:
		argText += " prg:%g" % (prognoseProb)

	numNodes = [ numNodes1 ]
	if numLayers > 1:
		numNodes.append( numNodes2 )
	if numLayers > 2:
		numNodes.append( numNodes3 )
	if numLayers > 3:
		numNodes.append( numNodes4 )

	nn = Network( numLayers, numNodes, numFeatures, numLabels )
	
	# Это на будущее - чтобы потом проинициализировать AdamOptimizer более детально
	if optimizer is not None:
		if optimizer == "Adam":
			optimizer = tf.train.AdamOptimizer( learning_rate = learningRate )

	nn.learn( trainData['inputs'], trainData['labels'], trainData['profit'], testData['inputs'], testData['labels'], testData['profit'], 
		numEpochs=numEpochs, balancer=balancer, autoBalancers=False, learningRate=learningRate, prognoseProb=prognoseProb,
		optimizer=optimizer, tradingLabel=None, flipOverTrading=flipOverTrading, 
		learnIndicators=True, saveRate=saveRate, summaryDir=summaryDir )

	import matplotlib.pyplot as plt
	plt.figure()
	plt.subplot(221)
	plt.scatter( nn.costTrain, nn.costTest, marker = '+', color = 'blue', alpha=0.1  )
	plt.title( "cost-function: train vs test")
	plt.grid()
	plt.subplot(222)
	plt.scatter( nn.accuracyTrain, nn.accuracyTest, marker = '+', color = 'blue', alpha=0.1  )
	plt.title("accuracy: train vs test")
	plt.grid()
	plt.subplot(223)
	plt.scatter( nn.tradeAccuracyTrain, nn.tradeAccuracyTest, marker = '+', color = 'blue', alpha=0.1  )
	plt.title("trade accuracy: train vs test")
	plt.grid()
	plt.subplot(224)
	plt.scatter( nn.balanceTrain, nn.balanceTest, marker = '+', color = 'blue', alpha=0.1 )
	plt.title("balance: train vs test")
	plt.grid()

	plt.suptitle( argText )
		
	plt.gcf().set_size_inches( 16, 8 )
	plt.savefig( nn.learnDir + ".png", bbox_inches='tight' )
	# plt.show()
# end of main

main()
