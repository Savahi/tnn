from tnn.data_handler.RTS_Ret_Vol_Mom_Sto_RSI_SMA_6 import calc_data

params = {
	"networks": [
		"20171017161235/999_c_2.219_a_0.3883_p_8.867e+04",
		"20171017161235/860_c_2.228_a_0.3896_p_8.819e+04",
	],

	"calcDatas": [
		calc_data(
			trans_cost=10,
			indnames=["Return","Volume","Momentum","Stochastic","RSI","SMA"],
			history_tail=5,
		)

	]*2,

	"fileWithRates": "../tnn-test/RTS_1h_150820_170820.txt",

	"aggregateLogic": None,

}


