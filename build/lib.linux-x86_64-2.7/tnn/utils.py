# -*- coding: utf-8 -*- 
import tensorflow as tf

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
