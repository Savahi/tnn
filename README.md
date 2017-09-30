TNN: Tensorflow Neural Network Framework for Algorithmic Trading 
================================================================
version 0.0.2

> **Important notice**:
> Nothing important yet... :)

Functions
----------

### Network ###
The constructor creates a Network object. / Конструктор, создает объект Network.
~~~
def __init__(self, numLayers=1, numNodes=[10], numFeatures=10, numLabels=2, stdDev=0.03, activationFuncs=None ):
~~~
    numLayers (integer, default:1) - Число hidden-слоев сети
    numNodes (list of integers, default:[10]) - Число узлов в каждом слое
    numFeatures (integer, default:10) - Размерность "инпутов" (x, они же "samples", в каждом sample присутствует numFeatures значений)
    numLabels (integer, default:2) - Размерность "аутпутов" (y, они же "labels")
    stdDev (float) - Стандартное отклонение для первичной генерации весов сети, default: 0.03 
    activationFuncs (list, default:None) - функции активации, размерность numLayers+1 (число hidden-слоев + 1). 
        Если "None", то активация hidden-слоев будет осуществляться через relu, а output-слоя - через softmax
        Если не "None", то элемент списка может быть:
            1) строкой: "relu", "sigmoid", "softmax"
            2) непосредственно функцией активации
    weights (list of 2d numpy arrays, default:None) - список, состоящий из двумерных матриц, 
        хранящих весовые коэффициенты слоев сети. Если weights==None, весовые коэффициенты будут сгенерированы автоматически
    bs (list of 1d numpy arrays, default:None) - список, состоящий из векторов, 
        хранящих весовые bias-коэффициенты (свободные члены) сети. Если bs==None, коэффициенты будут сгенерированы автоматически
	
	Returns (float) - the Network object. / Возвращает - объект Network

[See the sample code here / Пример кода см. здесь](samples/sample.py) 

### learn ###
Learns the network. / Обучает сеть
~~~
def learn( x, y, profit=None, xTest=None, yTest=None, profitTest=None, 
    learningRate=0.05, numEpochs=1000, balancer=0.0, optimizer=None, predictionProb=None, 
    summaryDir=None, printRate=20, trainTestRegression=False, saveRate=None, saveDir=None )
~~~
    x (2d numpy array, np.float64) - "инпуты" (samples) для обучения сети, размерность numSamples x numFeatures -> в placeholder self.x
    y (2d numpy array, np.float64) - "аутпуты" (labels) для обучения сети, размерность numSamples x numLabels -> в placeholder self.y
    profit (1d numpy array, np.float64, default: None) - значения прибыли (убытка) по каждому sample, 
        размерность: numSamples (как у x и y по оси 0)
    xTest (2d numpy array, np.float64, default:None) - "инпуты" (samples) для тестирования сети, размерность: numSamples x numFeatures 
    yTest (2d numpy array, np.float64, default:None) - "аутпуты" (labels) для тестирования сети, размерность: numSamples x numLabels
    profitTest (1d numpy array, np.float64, default: None) - значения прибыли (убытка) по каждому sample, 
        размерность: numSamples (как у xTest и yTest по оси 0)
    learningRate (float, default:0.05) - self explained 
    numEpochs (int, defaul:1000) - self explained
    balancer (float, default:0.0) - если balancer > 0.0, то при вычислении cost-функции совпадение/несовпадение по последнему 
        бину получит весовой коэффициент (balancer+1.0), в то время как по остальным бинам коэффициент будет 1.0.
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
    saveRate (integer, default:None) - задает как часто надо сохранять веса сети в процессе обучения.
        Веса сохраняются в папку saveDir (см. ниже) в файл, имя которого состоит из номера эпохи и
        текущих значений: cost-функции (см. переменную cost в теле функции), 
        точности модели (см. переменную accuracy) и прибыльности (см. переменную profit)  
        Если saveRate==None, веса не сохраняются.
        Сохраненные веса можно прочитать функцией load().
    saveDir (string, default:None) - задает имя папки, в которую будет сохраннен файл с весами сети.
        Если saveDir==None, имя папки будет сгенерировано на основе текущих даты и времени.

	Returns - Nothing. / Возвращает - ничего

[See the sample code here / Пример кода см. здесь](samples/sample.py) 
	
### calcOutput ###
Calculates the output of the Network. / Вычисляет "аутпут" (ответ) сети
~~~
    def calcOutput( self, x )
~~~
    x (1d numpy array, np.float) - "инпут", размерность: numFeatures [x0,x1,...,xn] (это число задается при создании сети - см. конструктор)

	Returns a 1d numpy array of length numLabels (the value previously passed to the constructor).
	Возвращает 1d numpy array размерностью numLables (число numLabels задается при создании сети - см. конструктор) 

[See the sample code here / Пример кода см. здесь](samples/calcOutput.py) 

### loadNetwork ###
Loads network from file 'fileName'. / Загружает сеть из файла 'fileName'.
~~~
def loadNetwork( fileName ):
~~~
    fileName (string) - Файл, в котором хранятся веса и функции активации сети.
        Файл должен был быть предварительно сохранен в процессе обучения сети, для чего
        при вызове функции learn() параметру saveRate должно было быть присвоено значение, отличное от 'None' или '0'
        (см. функцию learn()).
 
	Returns the Network object or 'None' if fails. / Возвращает объект Network в случае успеха и None в случае ошибки.

