# -*- coding: utf-8 -*- 

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
def prepareData( fileWithRates=None, rates=None, normalize=True, detachTest=None, calcInputs=None, calcLabels=None ):
    global logMessage
    logMessage = ""

    if fileWithRates is not None:
        rates = taft.readFinam( fileWithRates ) # Читаем котировки из файла finam
        if rates is None:
            logMessage += "Failed to read rates from finam file %s.\n" % (fileWithRates)

    if rates is None:
        logMessage += "No rates.\n"
        return None

    op = candles['op']
    hi = candles['hi']
    lo = candles['lo']
    cl = candles['cl']
    vol = candles['vol']
    length = candles['length']

    if calcInputs == None:
        calcInputs = __calcInputs

    if calcLabels == None:
        calcLabels = __calcLabels

    nnInputs = []
    nnLabels = []
    nnProfit = []
    for i in range(length-1,0,-1):
        # Inputs
        currentRates = { 'op': op[i:], 'hi':hi[i:], 'lo':lo[i:], 'cl':cl[i:], 'vol':vol[i:] }
        inputs = calcInputs( currentRates )
        if inputs is None:
            continue

        currentRates = { 'op': op[0:i], 'hi':hi[0:i], 'lo':lo[0:i], 'cl':cl[0:i], 'vol':vol[0:i] }
        labels, profit = calcOutputs( currentRates )
        if labels is None:
            continue
        nnInputs.append( inputs )
        nnLabels.append( labels )
        nnProfit.append( profit )

    nnInputs = np.array( nnInputs, dtype='float' )
    numSamples, numFeatures = np.shape( nnInputs )
    nnLabels = np.array( nnLabels, dtype='float' )
    nnProfit = np.array( nnProfit, dtype='float' )      
    nnMean = np.zeros( shape=[cols], dtype='float' )
    nnStd = np.zeros( shape=[cols], dtype='float' )

    # Нормализация нужна?
    if normalize:
        normIntervalStart = 0
        normIntervalEnd = numSamples-1
        if detachTest is not None:
            normIntervalStart = int( float(numSamples) * detachTest / 100.0 )

        for i in range(numFeatures):
            status, mean, std = taft.normalize( nnInputs[:,i], normInterval=[normIntervalStart,normIntervalEnd] )
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
            'numSamples':numSamples, 'numFeatures':numFeatures, 'mean':nnMean, 'std':nnStd }    
        retval2 = None
    else:
        retval1 = { 'inputs': nnInputs[normIntervalStart:], 'labels': nnLabels[normIntervalStart:], 'profit': nnProfit[normIntervalStart:], 
            'numSamples'normIntervalEnd-normIntervalStart+1, 'numFeatures':numFeatures, 'mean':nnMean, 'std':nnStd }    
        retval2 = { 'inputs': nnInputs[:normIntervalStart], 'labels': nnLabels[:normIntervalStart], 'profit': nnProfit[:normIntervalStart], 
            'numSamples'normIntervalStart, 'numFeatures':numFeatures, 'mean':nnMean, 'std':nnStd }    

    return( retval1, retval2 )
# end of def prepareData


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


def __calcInputs( rates ):
    historyIntervals = [ [0,0], [1,1], [2,2], [3,3], [0,4], [0,5] ]

    inputs = []

    cl0 = rates['cl'][0]
    lenRates = len(rates['cl'])

    for i in range(len(historyIntervals)):
        startRate = historyIntervals[i][0]
        endRate = historyIntervals[i][1]
        if endRate >= lenRates:
            return None

        high = rates['hi'][startRate:endRate+1]
        highestHigh = max( high )
        if bSmooth:     
            meanHigh = np.mean( high )
            stdHigh = np.std( high )
            highestHigh = meanHigh + 3*meanStd
        inputs.append( highestHigh - cl0 )

        low = rates['lo'][startRate:endRate+1]
        lowestLow = min( low )
        if bSmooth:     
            meanLow = np.mean( low )
            stdLow = np.std( low )
            lowestLow = meanLow - 3*meanStd
        inputs.append( cl0 - lowestLow )

    return inputs
# end of def __calcInputs


def __calcLabels( rates ):
    now = len(rates) - 1

    clLessOp = rates['cl'][now] - rates['op'][now]
    hiLessLo = rates['hi'][now] - rates['lo'][now]
    if hiLessLo > 0:
        observed = clLessOp / hiLessLo
    else:
        observed = 0.0
    observedBin = int( float(bins) * ( (observed + 1.0) / (2.0 + 1e-10) ) )
    labels = np.zeros( shape=[bins], dtype=np.float32 )
    labels[observedBin] = 1.0

    profit = clLessOp 

    return labels, profit
# end of def __calcOutputs

