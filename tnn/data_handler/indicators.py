import pandas as pd
import numpy as np
import taft


n = np.array

def extract_tail(df, period):
	hi=n(df["<HIGH>"])#[:period+5]
	lo=n(df["<LOW>"])#[:period+5]
	cl=n(df["<CLOSE>"])#[:period+5]
	
	# Taft requires arrays to go early-first

	#hi = np.flip(hi,0)
	#lo = np.flip(lo,0)
	#cl = np.flip(cl,0)

	return hi,lo,cl

def ADX(df, period):
	hi,lo,cl = extract_tail(df,period)
	return taft.adx(period=period, hi=hi,lo=lo,cl=cl)['adx']

def AD(df,period):
	hi,lo,cl = extract_tail(df,period)
	vo = n(df["<VOL>"])
	return taft.ad(period=period, hi=hi,lo=lo,cl=cl, vo=vo)

def ATR(df, period):
	hi,lo,cl = extract_tail(df,period)
	return taft.atr(period=period, hi=hi,lo=lo,cl=cl)['atr']

def BOLLINGER(df, period):
	hi,lo,cl = extract_tail(df,period)
	return taft.bollinger(period=period, rates=cl)['middle']

def CCI(df, period):
	hi,lo,cl = extract_tail(df,period)
	return taft.cci(period=period, hi=hi,lo=lo,cl=cl)['cci']

def EMA(df, period):
	hi,lo,cl = extract_tail(df,period)
	return taft.ema(period=period, rates=cl)

#def STOCHASTIC(df, period):
#	hi,lo,cl = extract_tail(df,period)
#	return taft.stochastic(periodhi=hi,lo=lo,cl=cl)['K']

def ROC(df,period):
	hi,lo,cl = extract_tail(df,period)
	return taft.roc(period=period, rates=cl)
	
def RSI(df,period):
	hi,lo,cl = extract_tail(df,period)
	return taft.rsi(period=period, rates=cl)['rsi']

def SMA(df,period):
	hi,lo,cl = extract_tail(df,period)
	return taft.sma(period=period, rates=cl)

indicators = {
	"<OPEN>": 	lambda df, period: df["<OPEN>"].iloc[0],
	"<CLOSE>":	lambda df, period: df["<CLOSE>"].iloc[0],
	"<HIGH>": 	lambda df, period: df["<HIGH>"].iloc[0],
	"<LOW>":  	lambda df, period: df["<LOW>"].iloc[0],
	"<VOL>":  	lambda df, period: df["<VOL>"].iloc[0],
	"<RETURN>":	lambda df, period: ( df["<CLOSE>"].iloc[0] - df["<OPEN>"].iloc[0]) / df["<OPEN>"].iloc[0] ,
	"<ADX>": ADX,"<ATR>":ATR,"<BOLLINGER>":BOLLINGER,"CCI":CCI,"EMA":EMA,
	#"STOCHASTIC":STOCHASTIC,
	#"ROC":ROC,
	"RSI":RSI,"SMA":SMA,

}



def indicator(ind): 
	return indicators[ind]

