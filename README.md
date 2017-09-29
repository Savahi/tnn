TNN: Tensorflow Neural Networks for Algorithmic Trading 
==========================================
version 0.0.1

> **Important notice**:
> ...

Functions
----------

### Network ###
~~~
def __init__(self, numLayers=1, numNodes=[10], numFeatures=10, numLabels=2, stdDev=0.03 ):
~~~
    numLayers (integer, default:1) - Число hidden-слоев сети
    numNodes (list of integers, default:[10]) - Число узлов в каждом слое  
    numFeatures (integer, default:10) - Размерность "инпутов" (x, они же "samples", в каждом sample присутствует numFeatures значений)
    numLabels (integer, default:2) - Размерность "аутпутов" (y, они же "labels")
    stdDev (float) - Стандартное отклонение для первичной генерации весов сети, default: 0.03 

	Returns (float) - the value of the indicator, 'None' if failed

[See the sample code here](samples/network.py) 

### learn ###
~~~
def learn( x, y, profit=None, xTest=None, yTest=None, profitTest=None, learningRate=0.05, numEpochs=1000, 
    balancer=0.0, activationFuncs=None, optimizer=None, predictionProb=None, 
    summaryDir="summary", printRate=20 )
~~~
    x (2d numpy array, np.float64) - "инпуты" (samples) для обучения сети, размерность numSamples x numFeatures -> в placeholder self.x
    y (2d numpy array, np.float64) - "аутпуты" (labels) для обучения сети, размерность numSamples x numLabels -> в placeholder self.y
    profit (1d numpy array, np.float64, default- None)- значения прибыли (убытка) по каждому sample, 
        размерность: numSamples (как в x и y)
    xTest (2d numpy array, np.float64, default:None) - "инпуты" (samples) для тестирования сети, размерность: numSamples x numFeatures 
    yTest (2d numpy array, np.float64, default:None) - "аутпуты" (labels) для тестирования сети, размерность: numSamples x numLabels
    profitTest (1d numpy array, np.float64, default: None) - значения прибыли (убытка) по каждому sample, 
        размерность: numSamples (как в xTest и yTest)
    learningRate (float, default:0.05) - 
    numEpochs (int, defaul:1000) -
    balancer (float, default:0.0) - если balancer > 0.0, gри вычислении cost-функции совпадение/несовпадение по последнему 
        бину получит весовой коэффициент (balancer+1.0), в то время как по остальным бинам коэффициент будет 1.0.
    activationFuncs (list, default:None) - функция активации, размерность numLayers+1 (число hidden-слоев + 1). 
        Если "None", то активация hidden-слоев через relu, а output-слоя - через softmax
        Если не "None", то элемент списка может быть:
            1) строкой: "relu", "sigmoid", "softmax"
            2) непосредственно функцией активации
    optimizer (string или func, default:None) - способ оптимизации. Если "None", то используется GradientDescentOptimizer. 
        Если не "None", то способ оптимизации может быть задан
            1) строкой ("GradientDescent", "Adadelta" и т.д.)
            2) напрямую, например: tf.train.GradientDescentOptimizer 
    predictionProb (float, default:None) - пороговое значение.
        При превышении это знаачения "аутпутом" (y) в последнем ("торгующем") бине мы считаем, что сеть дает сигнал а сделку. 
        Если predictionProb=None, по сигнал на сделку дается, если значение "аутпута" в последнем бине больше, 
        чем значения в остальных бинах. 
    summaryDir (string, default:None) - папка, куда tensorflow пишет summary ("отчет"). 
        Если "None", то summary записываться не будет.
        Если summaryDir=="", то имя папки будет сгенерировано автоматически из текущих даты и времени (только числа, без других знаков)
    printRate (int, default:20) - частота, с которой во время обучения выводятся параметры обучения и тестирования сети.
        Если "None", то вывода параметров не будет

	Returns (dict) - { 'adx': the ADX value, 'dx': the DX value, "+DI": the "+DI" value, "-DI": the "-DI value, 
	"+DMsm": the smoothed "+DM" value, "-DMsm": the smoothed "-DM" value, "TRsm": the smoothed 'true-range' value }, 'None' if failed

[See the sample code here](samples/test-adx.py) 

