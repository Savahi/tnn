# -*- coding: utf-8 -*- 
from tnn.data_handler.RTS_Ret_Vol_Mom_Sto_RSI_SMA_6 import calc_data
from tnn.calcdata import CalcData
from tnn.calib.trade import simple_sum

CD = CalcData( 5 )
history_tail = 5

for i in range(history_tail):
	CD.addLookBackOp( "rsi", i, 6 )
	CD.addLookBackOp( "stochastic", i, 6 )
	CD.addLookBackOp( "roc", i, 6 )
	CD.addLookBackOp( "sma", i, 6 )
	CD.addLookBackOp( "return", i, 6 )
	CD.addLookBackOp( "vol", i, 6 )

CD.addLookAheadOp( method="return", interval=1 )


params = {
	"networks": [
		"20171024113300/499_c_2.346_a_0.3188_p_1.279e+05",
		"20171024113300/200_c_2.457_a_0.2657_p_-9.871e+04",
	],

	"calcDatas": [CD,CD],

	"fileWithRates": "../tnn-test/RTS_1h_150820_170820.txt",

	"aggregateLogic": simple_sum(threshold=2),

}


