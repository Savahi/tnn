# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import datetime as dt

class Network:

    # Общее число сетей
    numNetworks = 0

    '''
    numLayers - (integer, default:1) Число hidden-слоев сети
    numNodes - (list of integers, default:[10]) Число узлов в каждом слое  
    numFeatures - (integer, default:10) Размерность "инпутов" (x, они же "samples", в каждом sample присутствует numFeatures значений)
    numLabels  - (integer, default:2) Размерность "аутпутов" (y, они же "labels") заданных в формате "one-hot"
    stdDev - (float) Стандартное отклонение для первичной генерации весов сети, default: 0.03 
    '''
    def __init__(self, numNodes=[10], numFeatures=10, numLabels=2, stdDev=0.03 ):
        self.numLayers = len(numNodes) # (integer) Число hidden-слоев
        self.numNodes = numNodes # (list of integers) Число узлов в каждом слое, numNodes[0] - число слоев в первом hidden-слое 
        self.numFeatures = numFeatures # (integer) Число features, т.е. размерность "инпутов" (x0,x1,x2,...,xn).
        self.numLabels = numLabels # One-hot labels

        # placeholder для "инпутов" (размерность: numSamples x numFeatures) - x [ [0.5,0.8,0.7], [0.4,0.7,0.6], ...  ] 
        self.x = tf.placeholder( tf.float64, [ None, numFeatures ] )

        # placeholder для "аутпутов" (они же "labels", размерность numSamples x numLables) - y  [ [0,1], [1,0], ... ]   
        self.y = tf.placeholder( tf.float64, [ None, numLabels ] )

        self.weights = [] # (list of tensors) Веса всех слоев сети 
        self.bs = [] # (list of tensors) Biases всех слоев сети

        for i in range( numLayers ):
            # Веса (w) + bias-столбец (b) для связи "инпутов" x и очередного hidden-слоя.
            if i == 0:
                numNodes0 = numFeatures
            else:
                numNodes0 = numNodes[i-1]
            w = tf.Variable( tf.random_normal( [ numNodes0, numNodes[i] ], stddev=stdDev, dtype=tf.float64 ), name='W'+str(i) )
            b = tf.Variable( tf.random_normal( [ numNodes[i] ], dtype=tf.float64 ), name='b'+str(i) )
            self.weights.append(w)
            self.bs.append(b)

        # Веса (w) + bias column (b) для связи hidden-слоя и output-слоя. 
        w = tf.Variable( tf.random_normal( [ numNodes[numLayers-1], numLabels ], stddev=stdDev, dtype=tf.float64 ), name='W'+str(numLayers) )
        b = tf.Variable( tf.random_normal( [ numLabels ], dtype=tf.float64 ), name='b'+str(numLayers) )

        self.weights.append(w)
        self.bs.append(b)

        self.layerActivationOps = []

        Network.numNetworks += 1
    # end of __init__

    '''
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
    '''
    def learn( self, x, y, profit=None, xTest=None, yTest=None, profitTest=None, learningRate=0.05, numEpochs=1000, 
        balancer=0.0, activationFuncs=None, optimizer=None, predictionProb=None, 
        summaryDir=None, printRate=20, trainTestRegression=False ):

        self.printRate = printRate

        inputMatrix = self.x
        for i in range( self.numLayers ):
            # Вычисление (активация) hidden-слоя
            inputMatrix = tf.add( tf.matmul( inputMatrix, self.weights[i] ), self.bs[i] )
            if activationFuncs is None: # Если функция активации не задана, используем relu
                inputMatrix = tf.nn.relu( inputMatrix )
            else:
                activationFunc = getActivationFunc( activationFuncs, i ) 
                inputMatrix = activationFunc( inputMatrix )

        # Операция для вычисления "выхода" сети
        outputMatrix = tf.add( tf.matmul( inputMatrix, self.weights[self.numLayers] ), self.bs[self.numLayers] )
        if activationFuncs is None: 
            yOp = tf.nn.softmax( outputMatrix )
        else:
            activationFunc = getActivationFunc( activationFuncs, numLayers, outputLayer=True ) 
            yOp = activationFunc( outputMatrix )
    
        # Все значения меньше 1e-10 превращаем в 1e-10, все значения больше 0.99999999 превращаем в 0.99999999: 
        yClippedOp = tf.clip_by_value( yOp, 1e-10, 0.99999999 )

        # Операция возвращает 1-мерный массив длиной numSamples: 
        # '1.0', если для данного набора "инпутов" был предсказан последний бин (т.е. сигнал на сделку) и '0', если нет
        if predictionProb is None:
            predictTradeOp = tf.cast( tf.equal( tf.argmax( yOp,1 ), self.numLabels-1 ), tf.float64 )
        else:
            predictTradeOp = tf.cast( tf.greater( yOp[:,self.numLabels-1], predictionProb ), tf.float64 )

        # Cost-функция.
        if balancer > 0.0:
            balancerOp = tf.zeros( [ tf.shape(self.y)[0], self.numLabels-1 ] )
            balancerOp = tf.concat( [ balancerOp, tf.reshape( predictTradeOp*balancer, [tf.shape(self.y)[0],1] ) ], 1 ) + 1.0
            costOp = -tf.reduce_mean( tf.reduce_sum( self.y * tf.log(yClippedOp) * balancerOp + (1.0 - self.y) * tf.log(1.0 - yClippedOp ) * balancerOp, axis=1 ) )
        else: 
            costOp = -tf.reduce_mean( tf.reduce_sum( self.y * tf.log(yClippedOp) + (1.0 - self.y) * tf.log(1.0 - yClippedOp), axis=1 ) )

        # Оптимизатор. Если None, то используем GradientDescentOptimizer
        if optimizer is None:
            optimiserOp = tf.train.GradientDescentOptimizer( learning_rate=learningRate ).minimize( costOp )
        else: # Если не None
            optimiserOp = getOptimizer( optimizer )( learning_rate=learningRate ).minimize( costOp )

        # Операции для вычисления доходности
        profitBySamples = tf.placeholder( tf.float64, [ None ] )
        profitByTrades = tf.multiply( predictTradeOp, profitBySamples )
        finalBalanceOp = tf.reduce_sum( profitByTrades )  

        # Операции для оценки точности модели. Точность оценивается по числу совпадений предсказаний по ВСЕМ бинам
        correctPredictionOp = tf.equal( tf.argmax(self.y, 1), tf.argmax(yOp, 1) )
        accuracyOp = tf.reduce_mean( tf.cast( correctPredictionOp, tf.float64 ) )

        # Для summary
        if summaryDir is not None:
            accuracySumm = tf.summary.scalar( 'Точность (Train)', accuracyOp )
            costSumm = tf.summary.scalar( 'Cost (Train)', costOp )
            if profit is not None:
                balanceSumm = tf.summary,scalar( 'Profit (Train)', finalBalanceOp )
            if xTest is not None and yTest is not None:
                accuracyTestSumm = tf.summary.scalar( 'Accuracy (Test)', accuracyOp )
                costTestSumm = tf.summary.scalar( 'Cost (Test)', costOp )
                if profitTest is not None:
                    balanceTestSumm = tf.summary.scalar( 'Final Balance (Test)', finalBalanceOp )

            # Если папка для summary = "", имя папки будет состоять из сегодняшней даты и времени (только числа, без других знаков) 
            if summaryDir == "":
                summaryDir = dt.datetime.now().strftime("%Y%m%d%H%M%S")
                # summaryDir = filter( lambda c: c.isdigit(), summaryDir )
            writer = tf.summary.FileWriter( summaryDir )

        # Запускаем сессию
        with tf.Session() as sess:
            # Инициализируем переменные
            sess.run( tf.global_variables_initializer() )

            feedDict = { self.x: x, self.y: y, profitBySamples: profit }
            feedDictTest = { self.x: xTest, self.y: yTest, profitBySamples: profitTest }

            if trainTestRegression:
                self.trainTestRegressionInit( numEpochs )

            for epoch in range(numEpochs):

                epochLog = ""

                sess.run([optimiserOp], feed_dict = feedDict )
                
                cost, accuracy = sess.run( [costOp, accuracyOp], feed_dict = feedDict )
                if profit is not None:
                    finalBalance = sess.run( finalBalanceOp, feed_dict = feedDict )
                else:
                    finalBalance = 0.0

                if summaryDir is not None:
                    writer.add_summary( sess.run( accuracySumm, feed_dict = feedDict ), epoch )
                    writer.add_summary( sess.run( costSumm, feed_dict = feedDict ), epoch )
                    if profit is not None:
                        writer.add_summary( sess.run( balanceSumm, feed_dict = feedDict ), epoch )

                epochLog += "Epoch %d/%d: cost=%.4f accuracy=%.4f $=%f" % ( epoch+1, numEpochs, cost, accuracy, finalBalance )

                if xTest is not None and yTest is not None:
                    costTest, accuracyTest = sess.run( [costOp, accuracyOp], feed_dict = feedDictTest )
                    if profitTest is not None:
                        finalBalanceTest = sess.run( finalBalanceOp, feed_dict = feedDictTest )    
                    else:
                        finalBalanceTest = 0.0

                    if summaryDir is not None:
                        writer.add_summary( sess.run( accuracyTestSumm, feed_dict = feedDictTest ), epoch )
                        writer.add_summary( sess.run( costTestSumm, feed_dict = feedDictTest ), epoch )
                        if profitTest is not None:
                            writer.add_summary( sess.run( balanceTestSumm, feed_dict = feedDictTest ), epoch )

                    epochLog += "  TEST: cost=%.4f accuracy=%.4f $=%f" % (costTest, accuracyTest, finalBalanceTest)
    
                # Добавляем перевод строки
                self.printEpochLog( epoch, epochLog )

                if trainTestRegression:
                    self.trainTestRegressionPut( epoch, cost, costTest, accuracy, accuracyTest, finalBalance, finalBalanceTest )

            print("\nDone!")

            if summaryDir is not None:
                writer.add_graph(sess.graph)
        # end of with tf.Session() as sess
    # end of learn()

    def trainTestRegressionInit( self, numEpochs ):
        self.costRegTrain = np.zeros( shape=[numEpochs], dtype=np.float32)  
        self.costRegTest = np.zeros( shape=[numEpochs], dtype=np.float32 )  
        self.accuracyRegTrain = np.zeros( shape=[numEpochs], dtype=np.float32 )  
        self.accuracyRegTest = np.zeros( shape=[numEpochs], dtype=np.float32 )  
        self.balanceRegTrain = np.zeros( shape=[numEpochs], dtype=np.float32 )  
        self.balanceRegTest = np.zeros( shape=[numEpochs], dtype=np.float32 )  
    # end of def

    def trainTestRegressionPut( self, epoch, costTrain, costTest, accuracyTrain, accuracyTest, balanceTrain, balanceTest ):
        self.costRegTrain[epoch] = costTrain
        self.costRegTest[epoch] = costTest
        self.accuracyRegTrain[epoch] = accuracyTrain
        self.accuracyRegTest[epoch] = accuracyTest
        self.balanceRegTrain[epoch] = balanceTrain
        self.balanceRegTest[epoch] = balanceTest
    # end of def

    def printEpochLog( self, epochNum, epochLog ):
        if self.printRate is not None:
            if epochNum % self.printRate == 0:
                print epochLog
    # end of def

# end of class


def getOptimizer( optimizer ):
    if type( optimizer ) is str: # Оптимизатор задан строкой
        if optimizer == 'GradientDescent': 
            return tf.train.GradientDescentOptimizer
        elif optimizer == 'Adadelta':
            return tf.train.AdadeltaOptimizer
        elif optimizer == 'Adagrad':
            return tf.train.AdagradOptimizer
        elif optimizer == 'AdagradDA':
            return tf.train.AdagradDAOptimizer
        elif optimizer == 'Momentum':
            return tf.train.MomentumOptimizer
        elif optimizer == 'Adam':
            return tf.train.AdamOptimizer
        elif optimizer == 'Ftrl':
            return tf.train.FtrlOptimizer
        elif optimizer == 'ProximalGradientDescent':
            return tf.train.ProximalGradientDescentOptimizer
        elif optimizer == 'ProximalAdagrad':
            return tf.train.ProximalAdagradOptimizer
        elif optimizer == 'RMSProp':
            return tf.train.RMSPropOptimizer
        else: 
            None
    elif callable( optimizer ): # Оптимизатор задан напрямую
        return optimizer
    else:
        return None
# end of def 

def getActivationFunc( activationFuncs, index, outputLayer=False ):
    if index >= len( activationFuncs ):
        if outputLayer == False:
            return tf.nn.relu
        else:
            return tf.nn.softmax 

    activationFunc = activationFuncs[index]
    if type( activationFunc ) is str: # Функция активации задана строкой
        if activationFunc == 'relu':
            return tf.nn.relu
        elif activationFunc == 'softmax':
            return tf.nn.softmax
        elif activationFunc == 'sigmoid':
            return tf.nn.sigmoid
        else:
            return None
    elif callable(activationFunc) : # Функция активации задана напрямую 
        return activationFunc
    else:
        return None
# end of def

