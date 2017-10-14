# -*- coding: utf-8 -*- 
from tnn.data_handler.RTS_Ret_Vol_Mom_Sto_RSI_SMA_6 import calc_data

params = {
	
	# Либо считать сеть из файла
	#network = { "load": "filename.shelve" },
	
	# Либо создать новую
	"network": {
		"nodes": [36,16,10],
		"num_inputs": 10,
		"bins": 3,
		"activationFuncs": ["sigmoid","sigmoid","sigmoid"],
	},

	
	"raw_file":"RTS_1h_150820_170820.txt",

	# Функции формирования данных; None=поведение по умолчанию
	"calcData": None, #calc_data,

	# Параметры обучения
	"learningRate":0.050,
	"numEpochs":2000,
	"optimizer":'GradientDescent',

	# папка, куда tensorflow пишет summary ("отчет"). 
        # Если summaryDir==None, отчеты записываться не будут.
        # Если summaryDir=="", то имя папки будет сгенерировано автоматически из текущих даты и времени (только числа, без других знаков).
        "summaryDir":'TIMTEST',
	
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
	"saveDir":"TIMTEST"
}
