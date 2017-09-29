
	# ...

	numLayers = 1
	numNodes1 = 36
	numNodes2 = 36
	numNodes3 = 36
	numNodes4 = 36
	learningRate = 0.050
	numEpochs = 1000
	balancer = 0.0	

	x = ... # инпуты, нормализованные значения, 2d, shape = numSamples x numFeatures, [ [x0,x1,...,xn], [x0,x1,...,xn],... ]
	numSamples, numFeatures = np.shape(x)
	y = ... # "правильные" аутпуты (labels), в формате one-hot, 2d, shape = numSamples x numLables, н-р [ [0,0,1], [1,0,0],... , [0,1,0] ]
	_, numLabels = np.shape( y )
	profit = .. # доходность, 1d, length=numSamples
	
	xTest = ... # формат, как у 'x', для тестирования
	yTest = ... # формат, как у 'y', для тестирования
	profitTest = ... # формат, как у 'profit', для тестирования

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
