TNN: Tensorflow Neural Networks for Algorithmic Trading 
==========================================
version 0.0.1

> **Important notice**:
> ...

Functions
----------

### Network ###
The constructor creates a Network object. / Конструктор, создает объект Network.

~~~
def __init__(self, numLayers=1, numNodes=[10], numFeatures=10, numLabels=2, stdDev=0.03 ):
~~~
    numLayers (integer, default:1) - Число hidden-слоев сети
    numNodes (list of integers, default:[10]) - Число узлов в каждом слое
    numFeatures (integer, default:10) - Размерность "инпутов" (x, они же "samples", в каждом sample присутствует numFeatures значений)
    numLabels (integer, default:2) - Размерность "аутпутов" (y, они же "labels")
    stdDev (float) - Стандартное отклонение для первичной генерации весов сети, default: 0.03 

	Returns (float) - the Network object. / Возвращает - объект Network

[See the sample code here / Пример кода см. здесь](samples/sample.py) 

### learn ###
Learns the network. / Обучает сеть
~~~
def learn( x, y, profit=None, xTest=None, yTest=None, profitTest=None, 
    learningRate=0.05, numEpochs=1000, balancer=0.0, activationFuncs=None, optimizer=None, predictionProb=None, 
    summaryDir=None, printRate=20, trainTestRegression=False )
~~~
    x (2d numpy array, np.float64) - "инпуты" (samples) для обучения сети, размерность numSamples x numFeatures -> в placeholder self.x
    y (2d numpy array, np.float64) - "аутпуты" (labels) для обучения сети, размерность numSamples x numLabels -> в placeholder self.y
    profit (1d numpy array, np.float64, default: None) - значения прибыли (убытка) по каждому sample, 
        размерность: numSamples (как в x и y)
    xTest (2d numpy array, np.float64, default:None) - "инпуты" (samples) для тестирования сети, размерность: numSamples x numFeatures 
    yTest (2d numpy array, np.float64, default:None) - "аутпуты" (labels) для тестирования сети, размерность: numSamples x numLabels
    profitTest (1d numpy array, np.float64, default: None) - значения прибыли (убытка) по каждому sample, 
        размерность: numSamples (как в xTest и yTest)
    learningRate (float, default:0.05) - self explained 
    numEpochs (int, defaul:1000) - self explained
    balancer (float, default:0.0) - если balancer > 0.0, то при вычислении cost-функции совпадение/несовпадение по последнему 
        бину получит весовой коэффициент (balancer+1.0), в то время как по остальным бинам коэффициент будет 1.0.
    activationFuncs (list, default:None) - функции активации, размерность numLayers+1 (число hidden-слоев + 1). 
        Если "None", то активация hidden-слоев будет осуществляться через relu, а output-слоя - через softmax
        Если не "None", то элемент списка может быть:
            1) строкой: "relu", "sigmoid", "softmax"
            2) непосредственно функцией активации
    optimizer (string или func, default:None) - способ оптимизации. Если "None", то используется GradientDescentOptimizer. 
        Если не "None", то способ оптимизации может быть задан:
            1) строкой ("GradientDescent", "Adadelta" и т.д.)
            2) напрямую, например: tf.train.GradientDescentOptimizer 
    predictionProb (float, default:None) - пороговое значение оценки вероятности.
        При превышении этого значения "аутпутом" (y) в последнем ("торгующем") бине мы считаем, что сеть дает сигнал на сделку. 
        Если predictionProb==None, по сигнал на сделку дается, если значение "аутпута" в последнем бине больше, 
        чем значения в остальных бинах. 
    summaryDir (string, default:None) - папка, куда tensorflow пишет summary ("отчет"). 
        Если summaryDir==None, отчеты записываться не будут.
        Если summaryDir=="", то имя папки будет сгенерировано автоматически из текущих даты и времени (только числа, без других знаков).
    printRate (int, default:20) - частота, с которой во время обучения на терминал выводятся параметры обучения и тестирования сети
        (значение cost-функции, точность (accuracy), баланс (если задан на входе)).
        Если printRate=="None", то вывода параметров не будет
    trainTestRegression (boolean, default:False) - если задать True, в процессе обучения, для каждой эпохи будут записываться
        пары значений (для train и test данных): 
        - cost-функция на тест vs cost-функция на train
        - точность (accuracy) на тест vs точность (accuracy) на train
        - доходность на тест vs доходность на train.
        По этим парам значений можно будет построить регрессионную зависимость.

	Returns - Nothing. / Возвращает - ничего

[See the sample code here / Пример кода см. здесь](samples/sample.py) 


## data_handler.InputsShape

A container class for a schema used to generate network inputs from raw data


~~~
def __init__(self, indicators, history, num_cand_inds):
~~~
    indicators - List of values to be included into the inputs, in order. Example: ["<OPEN>", "<CLOSE>", "<RSI>"]
    Full list of possible values: "<OPEN>", "<CLOSE>", "<HIGH>", "<LOW>", "<VOL>", "RETURN", "ADX","ATR","BOLLINGER","CCI","EMA","RSI","SMA"
    history - how many candles back are used.
    num_cand_inds - how many candles are used to calculate indicators

~~~
def num_inputs(self):
~~~
    returns the total size of one input. For example, with indicators = ["<OPEN>", "<CLOSE>", "<RSI>"], and history=6, num_inputs would equal 18.

## data_hander.RawData

A class to read data from Finam files and convert it into network-compatible form


~~~
def __init__(self, filename):
~~~
    filename - path to the finam data file
    

~~~
def form_inputs(self, inputs_shape):
~~~
    inputs_shape - a data_handler.InputsShape object
    Returns an array of numpy.arrays, each corresponding to an input, latest to earliest
    
~~~
def normalize_inputs(self, norm=None, std=None):
~~~
    If previously formed inputs, will return the normalized version. If norm and std are not provided, they will be automatically calculated
    
~~~
def dump_to_file(self, filename):
~~~
    Wrties the inputs into a shelve file
    
   
