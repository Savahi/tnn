# -*- coding: utf-8 -*- 
import numpy as np
import taft
from network import Network

# Переменная для записи сообщений о ошибках
logMessage = ""

def getLastError():
    global logMessage
    return logMessage
# end of def getLastError()


# Загружает веса сети из файла fileName.
# Возвращает объект Network в случае успеха и None в случае ошибки.  
def loadNetwork( fileName ):
    ok = True 

    try:
        s = shelve.open( fileName, flag='r' )
    except Exception:
        ok = False

    if ok:
        try:
            weights = s['weights']
            bs = s['bs']
            activationFuncs = s['activationFuncs']
        except Exception:
            ok = True
        finally:
            s.close()

    if ok:
        numLayers = len( weights ) - 1
        numFeatures = np.shape(weights[0])[0]
        numNodes = [] 
        for i in range( numLayers ):
            numNodes.append( np.shape( weights[i] )[1] )
        numLabels = np.shape( weights[numLayers] )[1]
        network = Network( numLayers, numNodes, numFeatures, numLabels, activationFuncs=activationFuncs, weights=weights, bs=bs )
        return network

    return None
# end of loadNetwork()


# Готовит данные для обучения и тестирования сети. 
def prepareData( fileWithRates=None, rates=None, normalize=True, detachTest=20, calcData=None ):
    global logMessage
    logMessage = ""
    retError = None, None

    if fileWithRates is not None:
        rates = taft.readFinam( fileWithRates ) # Читаем котировки из файла finam
        if rates is None:
            logMessage += "Failed to read rates from finam file %s.\n" % (fileWithRates)

    if rates is None:
        logMessage += "No rates.\n"
        return retError

    op = rates['op']
    hi = rates['hi']
    lo = rates['lo']
    cl = rates['cl']
    vol = rates['vol']
    length = rates['length']

    if calcData == None:
        calcData = __calcData

    nnInputs = []
    nnLabels = []
    nnProfit = []
    for i in range(length-1,0,-1):
        # Inputs
        pastRates = { 'op': op[i:], 'hi':hi[i:], 'lo':lo[i:], 'cl':cl[i:], 'vol':vol[i:] }
        futureRates = { 'op': op[i-1::-1], 'hi':hi[i-1::-1], 'lo':lo[i-1::-1], 'cl':cl[i-1::-1], 'vol':vol[i-1::-1] }

        inputs, labels, profit = calcData( pastRates, futureRates )
        if inputs is None or labels is None:
            continue
        nnInputs.append( inputs )
        nnLabels.append( labels )
        nnProfit.append( profit )

    nnInputs = np.array( nnInputs, dtype='float' )
    numSamples, numFeatures = np.shape( nnInputs )
    nnLabels = np.array( nnLabels, dtype='float' )
    numLabels = np.shape( nnLabels )[1]
    nnProfit = np.array( nnProfit, dtype='float' )      
    nnMean = np.zeros( shape=[numFeatures], dtype='float' )
    nnStd = np.zeros( shape=[numFeatures], dtype='float' )

    if normalize: # Если нужна нормализация
        normIntervalStart = 0
        if detachTest is not None:
            normIntervalStart = int( float(numSamples) * detachTest / 100.0 )

        for i in range(numFeatures):
            status, mean, std = taft.normalize( nnInputs[:,i], normInterval=[normIntervalStart,numSamples] )
            if status is None:
                logMessage += "Can't normalize %d column\n." % (i)
                return None
            nnMean[i] = mean
            nnStd[i] = std
    else:
        logMessage += "Normalization skipped.\n"
        nnMean = None
        nnStd = None

    if detachTest is None:
        retval1 = { 'inputs': nnInputs, 'labels': nnLabels, 'profit': nnProfit, 
            'numSamples':numSamples, 'numFeatures':numFeatures, 'numLabels':numLabels, 'mean':nnMean, 'std':nnStd }    
        retval2 = None
    else:
        retval1 = { 'inputs': nnInputs[normIntervalStart:], 'labels': nnLabels[normIntervalStart:], 'profit': nnProfit[normIntervalStart:], 
            'numSamples':numSamples-normIntervalStart, 'numFeatures':numFeatures, 'numLabels':numLabels, 'mean':nnMean, 'std':nnStd }    
        retval2 = { 'inputs': nnInputs[:normIntervalStart], 'labels': nnLabels[:normIntervalStart], 'profit': nnProfit[:normIntervalStart], 
            'numSamples':normIntervalStart, 'numFeatures':numFeatures, 'numLabels':numLabels, 'mean':nnMean, 'std':nnStd }    

    return( retval1, retval2 )
# end of def prepareData

__lookBack = None
__lookAhead = None

def setLookBack( lookBack ):
    global __lookBack
    __lookBack = lookBack

def setLookAhead( lookAhead ):
    global __lookAhead
    __lookAhead = lookAhead

def __calcData( pastRates, futureRates ):
    global __lookBack
    global __lookAhead
    retErr = None, None, None

    # Calculating inputs / Вычисляем "инпуты"
    if __lookBack is not None: # Какие временные отрезки "прошлого" мы будем пересчитывать в "инпуты" сети  
        lookBack = __lookBack 
    else:
        lookBack = [ [0,0], [1,1], [2,3], [4,6], [7,10] ] # По умолчанию

    inputs = []

    cl0 = pastRates['cl'][0] # "Сейчас" мы в точке '0', последняя известная цена - цена закрытия
    lenRates = len( pastRates['cl'] ) 

    for i in range( len( lookBack ) ):
        startRate = lookBack[i][0]
        endRate = lookBack[i][1]
        if endRate >= lenRates: # Если нужная нам точка лежит за пределами массива, "инпуты" подсчитать не удастся
            return retErr

        high = pastRates['hi'][startRate:endRate+1] # Массив со значениями HIGH на выбранном интервале прошлого
        highestHigh = np.max( high ) # Ищем максимальный HIGH
        inputs.append( highestHigh - cl0 ) # Добавляем "инпут"

        low = pastRates['lo'][startRate:endRate+1] # Массив со значениями LOW на выбранном интервале прошлого
        lowestLow = np.min( low ) # Ищем минимальный LOW
        inputs.append( cl0 - lowestLow ) # Добавляем "инпут"

    # Calculating labels and profits / Вычисляем "аутпуты" (labels) и ддоходность
    if __lookAhead is None: # На сколько периодов вперед надо смотреть
        ahead = 0 # По умолчанию смотрим вперед на один период, а значит нас интересует цена закрытия 0-го периода
    else:
        ahead = __lookAhead
    if ahead >= len(futureRates): # Если требуется смотреть за пределы массивов с котировками, расчет невозможен.
        return retErr

    # Вычисляем "аутпут" - отношение (CLOSE-OPEN) / (HIGH-LOW) на указанном (переменной ahead) отрезке "ближайшего будущего".
    # Каждое значения "аутпута" будет отнесено к одной из трех категорий и представлено в виде one-hot вектора длиной 3.
    # Маленькие значения будут кодироваться [1,0,0], средние - [0,1,0], большие - [0,0,1].  
    bins = 3
    op = futureRates['op'][0]
    cl = futureRates['cl'][ahead]
    hi = np.max( futureRates['hi'][:ahead+1] )
    lo = np.min( futureRates['lo'][:ahead+1] )
    clLessOp = cl - op
    hiLessLo = hi - lo
    if hiLessLo > 0:
        observed = clLessOp / hiLessLo
    else:
        observed = 0.0
    observedBin = int( float(bins) * ( (observed + 1.0) / (2.0 + 1e-10) ) )
    labels = np.zeros( shape=[bins], dtype=np.float32 )
    labels[observedBin] = 1.0

    profit = clLessOp # Позиция будет открыта "сейчас" (futureRates['op'][0]) и закрыть ее через ahead периодов (futureRates['cl'][ahead]).
                        # Доходность на этом отрезке составит CLOSE-OPEN (этого временного отрезка) 

    return inputs, labels, profit
# end of def __calcData


def saveData( fileName, data, normOnly=False ):
    global logMessage
    bOk = True

    logMessage += "Saving into \"%s\"...\n" % (fileName)

    try:    
        s = shelve.open( fileName )
    except Exception:
        ok = False

    if ok:
        try:
            if not normOnly:
                s['inputs'] = data['inputs']
                s['labels'] = data['labels']
                s['profit'] = data['profit']
            s['mean'] = data['mean']
            s['std'] = data['std']
        except Exception:
            ok = False
        finally:
            s.close()

    return ok
# end of saveData()


def loadData( fileName, normOnly=False ):
    global logMessage
    ok = True

    logMessage += "Reading data from \"%s\"...\n" % (fileName)
    try:
        s = shelve.open( fileName )
    except Exception:
        ok = False
        logMessage += "Can't open file %s.\n" % (fileName)
        
    if ok:
        try:
            if not normOnly:
                data['inputs'] = s['inputs']
                data['labels'] = s['labels']
                data['profit'] = s['profit']
            data['mean'] = s['mean']
            data['std'] = s['std']
            s.close()
        except Exception:
            ok = False
            logMessage += "Error reading the data.\n"
        finally:
            s.close()

    if ok:
        return data
    else:
        return None
# end of loadData()

