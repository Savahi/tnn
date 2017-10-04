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

