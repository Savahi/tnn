# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import shelve

# Принимает кодовую строку, обозначающую оптимизатор
# Возвращает object - оптимизатор. Если был передан объект-оптимизатор - возвращает его же (сделано для удобства вызова)
def getOptimizer( optimizer, learningRate ):
    if type( optimizer ) is str: # Оптимизатор задан строкой
        if optimizer == 'GradientDescent': 
            return tf.train.GradientDescentOptimizer(learning_rate=learningRate)
        elif optimizer == 'Adadelta':
            return tf.train.AdadeltaOptimizer(learning_rate=learningRate)
        elif optimizer == 'Adagrad':
            return tf.train.AdagradOptimizer(learning_rate=learningRate)
        elif optimizer == 'Adam':
            return tf.train.AdamOptimizer(learning_rate=learningRate)
        elif optimizer == 'Ftrl':
            return tf.train.FtrlOptimizer(learning_rate=learningRate)
        elif optimizer == 'RMSProp':
            return tf.train.RMSPropOptimizer(learning_rate=learningRate)
        else: 
            None
    elif isinstance( optimizer, object ): # Оптимизатор задан напрямую объектом
        return optimizer
    else:
        return None
# end of def 

# Принимает кодовую строку, обозначающую функцию активации
# Возвращает callable-переменную - функцию активации. Если была передана функция - возвращает ее же (сделано для удобства вызова)
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


def log( logStr, fileName="log.txt" ):
    ok = True

    try:
        logFile = open( fileName, "a" )
    except Exception:
        ok = False

    if ok:
        try:
            logFile.write( str(logStr) + "\n" )
        except Exception:
            ok = False
        finally:
            logFile.close()
    return ok
# end of def


# Подсчитывает число labels в выборке и возвращает соответствующий np.array
# Используется для контроля качества данных, подаваемых в сеть для обучения 
def countLabels( labels ):
    rows, cols = np.shape( labels )
    labelsCounter = np.zeros( shape=[cols] )
    for i in range( rows ):
        for j in range( cols ):
            if labels[i][j] == 1:
                labelsCounter[j] += 1
    return labelsCounter
# end of def


# Saves data with a given key in a shelve file.
# Returns True if ok.
# Returns False if fails.
def save( fileName, data, key ):
    ok = True

    try:    
        s = shelve.open( fileName )
    except Exception:
        ok = False

    if ok:
        try:
            s[key] = data
        except Exception:
            ok = False
        finally:
            s.close()

    return ok
# end of def 


# Loads a given key from a shelve file.
# Returns data is success.
# Returns None if fails.
def load( fileName, key ):
    ok = True

    try:
        s = shelve.open( fileName )
    except Exception:
        ok = False
        
    if ok:
        try:
            data = s[key]
            s.close()
        except Exception:
            ok = False
        finally:
            s.close()

    if ok:
        return data
    else:
        return None
# end of def