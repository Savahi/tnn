from tnn.data_handler.RTS_Ret_Vol_Mom_Sto_RSI_SMA_6 import calc_data
from tnn.calcdata import CalcData

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
		"20171017161235/999_c_2.219_a_0.3883_p_8.867e+04",
		"20171017161235/860_c_2.228_a_0.3896_p_8.819e+04",
	],

	"calcDatas": [CD,CD],

	"fileWithRates": "../tnn-test/RTS_1h_150820_170820.txt",

	"aggregateLogic": None,

}


