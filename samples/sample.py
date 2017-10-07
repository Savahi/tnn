
	# ...

	numLayers = 1
	numNodes = 40
	learningRate = 0.050
	numEpochs = 1000

	trainData, testData = prepareData( fileWithRates=fileWithRates, detachTest=20, calcData=None )

	x = trainData['inputs'] # инпуты, нормализованные значения, 2d, shape = numSamples x numFeatures, [ [x0,x1,...,xn], [x0,x1,...,xn],... ]
	numFeatures = trainData['numFeatures']
	y = trainData['labels'] # "правильные" аутпуты (labels), в формате one-hot, 2d, shape = numSamples x numLables, н-р [ [0,0,1], [1,0,0],... , [0,1,0] ]
	numLabels = trainData['numLabels']
	profit = trainData['profit'] # доходность, 1d, length=numSamples
	
	xTest = testData['inputs'] # формат, как у 'x', для тестирования
	yTest = testData['labels'] # формат, как у 'y', для тестирования
	profitTest = testData['profit'] # формат, как у 'profit', для тестирования

	# Создаем сеть
	nn = Network( numLayers, numNodes, numFeatures, numLabels )

	# Запускаем обучение сети
	nn.learn( x, y, profit, xTest, yTest, profitTest, numEpochs=numEpochs, balancer=balancer, trainTestRegression=True )

	# Выводим scatter-графики для отображения регрессионной зависимости между значениями cost-функции, точности и доходности
	# на TRAIN и на ТЕСТ - по мере обучения от эпохи к эпохи. 
	import matplotlib.pyplot as plt
	plt.subplot(311)
	plt.scatter( nn.costRegTrain, nn.costRegTest )
	plt.title("COST-FUNCTION: TRAIN VS TEST")
	plt.subplot(312)
	plt.scatter( nn.accuracyRegTrain, nn.accuracyRegTest )
	plt.title("ACCURACY: TRAIN VS TEST")
	plt.subplot(313)
	plt.scatter( nn.balanceRegTrain, nn.balanceRegTest )
	plt.title("BALANCE: TRAIN VS TEST")
	plt.show()

	# ...
