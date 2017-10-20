# -*- coding: utf-8 -*- 
from tnn.data_handler.RTS_Ret_Vol_Mom_Sto_RSI_SMA_6 import calc_data
from tnn.calcdata import CalcData

# количество бинов
CD = CalcData( 5 )

# длина истории
history_tail = 5

for i in range(history_tail):
	# задаем индикаторы
	CD.addLookBackOp( "rsi", i, 6 )
	CD.addLookBackOp( "stochastic", i, 6 )
	CD.addLookBackOp( "roc", i, 6 )
	CD.addLookBackOp( "sma", i, 6 )
	CD.addLookBackOp( "return", i, 6 )
	CD.addLookBackOp( "vol", i, 6 )

# что предсказываем
CD.addLookAheadOp( method="return", interval=1 )

params = {
	"network": {
		"nodes": [22,16,10],
		"num_inputs": 30,
		"bins": 5,
		"activationFuncs": ["sigmoid","sigmoid","sigmoid"],
	},

	
	"raw_file":"../tnn-test/RTS_1h_150820_170820.txt",

	"calcData": CD,

	# Параметры обучения
	"learningRate":0.050,
	"numEpochs":1000,
	"optimizer":'Adam',

	# папка, куда tensorflow пишет summary ("отчет"). 
        # Если summaryDir==None, отчеты записываться не будут.
        # Если summaryDir=="", то имя папки будет сгенерировано автоматически из текущих даты и времени (только числа, без других знаков).
        "summaryDir":'',
	
	# если задать True, в процессе обучения, для каждой эпохи будут записываться
        # пары значений (для train и test данных): 
        #- cost-функция на тест vs cost-функция на train
        #- точность (accuracy) на тест vs точность (accuracy) на train
        #- доходность на тест vs доходность на train.
        #По этим парам значений можно будет построить регрессионную зависимость.
	"trainTestRegression":True,
	
	# задает как часто надо сохранять веса сети в процессе обучения.
	"saveRate":20,

	#имя папки, в которую будет сохраннен файл с весами сети.
        #Если saveDir==None, имя папки будет сгенерировано на основе текущих даты и времени.
	"saveDir":None
}
