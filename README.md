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
    learningRate=0.05, numEpochs=1000, balancer=0.0, optimizer=None, prognoseProb=None, 
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
    optimizer (string или func, default:None) - способ оптимизации. Если optimizer=None, то используется GradientDescentOptimizer. 
        Если optimizer is not None, то способ оптимизации может быть задан:
            1) строкой (возможные варианты: "GradientDescent", "Adadelta", "Adagrad", "Adam", "Ftrl", "RMSProp" и т.д.)
            2) напрямую объектом, например: tf.train.GradientDescentOptimizer(learning_rate=0.01) 
    prognoseProb (float, default:None) - пороговое значение оценки вероятности.
        При превышении этого значения "аутпутом" (y) в последнем ("торгующем") бине мы считаем, что сеть дает сигнал на сделку. 
        Если prognoseProb==None, по сигнал на сделку дается, если значение "аутпута" в последнем бине больше, 
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

### prepareData ###
Prepares data for network training and testing. / Готовит данные для обучения и тестирования сети.
~~~
def prepareData( fileWithRates=None, rates=None, normalize=True, detachTest=20, calcInputs=None, calcLabels=None ):
~~~
    fileWithRates (string) - файл с котировками в формате finam. 
        Если указать fileWithRates==None, котировки можно передать через параметр "rates" (см. ниже).
    rates (dict, default:None) - словарь с массивами котировок.
        Если указан fileWithRates, значение игнорируется. Формат котировок должен быть такой же, какой
        возвращает функция taft.readFinam: 
        rates['op'] (numpy array) - цены open,  
        rates['hi'] (numpy array) - цены high, 
        rates['lo'] (numpy array) - цены low, 
        rates['cl'] (numpy array) - цены close, 
        rates['vol'] (numpy array) - объемы.
        0-й индекс вышеуказанных массивов соответствует последним по времени поступления данным.
    normalize (boolean, default:True) - Значение "True" означает, что данные будут нормализованы.
    detachTest (int, default:20) - Указывает, какой процент последних по времени поступения котировок
        будет преобразован в отдельный блок данных для тестирования сети.
    calcData (function, defalut:None) - Функция, которая берет на входе котировки и выдает "инпуты" (inputs),  
        "аутпуты" (labels) и доходность (profit) для обучения и тестирования сети. 
        Если указать None, все данные будут сгенерированы встроенной функцией.
	Функция должны быть определена так:
	def myCalcData( pastRates, futureRates ): где 
            "pastRates" - словарь с котировками для рассчета "инпутов" (формат см. выше).
                0-й индекс массивов словаря соответствует "текущему моменту" - точке временного ряда, 
                для которой мы считаем "инпуты". 
	        Самая последняя по времени поступления котировка - это rates['cl'][0].
	        По мере увеличения индекса мы двигаемся "назад" во времени. 
            "futureRates" - словарь с котировками для рассчета "аутпутов" (labels) и доходности (формат см. выше); 
                По мере увеличения индекса мы двигаемся "вперед" во времени.
                Ближайшая к нам котировка, таким образом, это futureRates['op'][0] (цена открытия ближайшего периода)
        Функция должна вернуть массив или список "инпутов", а также
        список "аутпутов" в формате "one-hot" (один элемент равен "1", остальные "0") и 
	float переменную, которая хранит доходность сделки в данной точке временного ряда, например:
            return [0.2234, 0.43234,..., 0.9934], [0,0,1], 525.2
        Если аутпуты подсчитать не удается, функция должна вернуть None, None, None.
	[Пример реализации встроенной функции calcData см. здесь](samples/calcData.py)
 
	Функция prepareData возвращает две переменные-словари: trainData и testData. 
        Если detachTest==None, testData будет равен "None".
	Формат обеих переменных следующий:
		data['numSamples'] - число примеров, равное размерности массивов data['inputs'] и data['labels'] по оси 0.
		data['numFeatures'] - число переменных, равное размерности data['inputs'] по оси 1.
		data['numLabels'] - число "бинов" (классов), которые распознает сеть.


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
    
   
