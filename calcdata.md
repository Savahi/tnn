TNN: CalcData Class 
================================================================
version 0.0.4
    
> **Important notice**:
> Nothing important yet... :)

## CONTENTS / СОДЕРЖАНИЕ ##
[CacData](#calcdata) - CalcDataConstructor / Конструктор.  
[addLookBackOp](#addlookbackop) - Adding a new input (a method and details of how the input is to be calculated).  
[addLookAheadOp](#addlookbackop) - Calculating of expected output and profit.  
[addLookAheadFunc](#addlookbackfunc) - Calculating of expected output and profit.   
[run](#run) - self explained. 

### CalcData ###
Creates an instance of CalcData class. / Создает экземпляр объекта CalcData.
The CalcData object is used when:
- Creating data for training and testing a network with the [prepareData](README.md#preparedata) function. In this case the CalcData object serves as the 'calcData' parameter of the function.
- When using the network for prognosing. In this case the CalcData object is used for generating inputs for the network.  

Объект используется:  
- при создании данных для обучения и тестирования сети; в этом случае  он передается функции [prepareData](README.md#preparedata). 
- при использовании сети в торговле; в этом случае он используется для генерации входных данных.

```python
	CalcData( numLabels=5, intraDay=False, tradingDays=None, tradingTime=None, precalcData=None ):
```
	numLabels (int, default:5) - The number of labels (classes)
	intraDay (boolean, default:False) - If 'True' 'pastRates' and 'futureRates' are monitored 
		to belong to the same day.
	tradingDays (list of int, default:None) - Day(s) of week allowed for trading, '0'-monday, 4-'friday'
	tradingTime (list of int pairs, default:None ) - Hours and minutes allowed for trading
	precalcData (function, default:None) - A custom function to run before starting to 
		generate data. The precalc function takes two arguments: a CalcData object and rates.
For example: 
```python CalcData( 5, tradingDays=[1,2,3], tradingTime=[ [ 14, None ], [ 15, [0,15] ] )```
stands for 5 labels, tuesday, wednesday and thurthday trading days and trading hours 
starting from 14:00 to 15:15. 


### addLookBackOp ###
Adds another input for the network - an operation to be performed for calculating the input.
```python 
	addLookBackOp( name="sma", shift=0, period=10 )
```	
~~~
	name (string, default:"sma") - The name of an indicator to generate an input for the network. 
		Possible values:
			"high" - the highest value at the [shift:shift+period] interval
			"low" - the lowest value at the [shift:shift+period] interval
			"sma" - the Simple Moving Average value at the at the [shift:shift+period] interval
			"rsi" - the RSI value at the [shift:shift+period] interval
			"stochastic" - the Stochastic value at the [shift:shift+period] interval
			"roc" - the ROC value at the [shift:shift+period] interval
			"vol" - the total volume of trades at the [shift:shift+period] interval
			"return" - close[shift+period] - open[shift]
	shift (int, default:0) - Shift "into the past". 0 stands for the most latest rates. 
	period (int, default:10) - The lenght of an interval used to calculate 
		the indicator given by 'name'.
~~~
Example: ```python calcDataObject.addLookBackOp( "sma", 1, 5 )``` stands for the Simple Moving
Average with period 5 to be calculated for the last but one (1th) price candle. 


### addLookAheadOp ###
Determines how the sample outputs of the network (labels) should be calculated.
```python
	addLookAheadOp( method="ochl", interval=1, amplitude=None, scale=None, noOvernight=False ):
```
~~~
method (string, default: 'cohl') - the method of calculating sample outputs (labels) 
	for the network. The allowed values are:
		'cohl': The ratio <close-less-open> divided by <high-less-low> is calculated within 
			the interval given by the 'interval' parameter (see below);
		'amplitude': Waits for the price to change from the highest to the lowest 
			(or from the lowest to the highest)	value for the given 'amplitude' 
			value (see below). Returns the final price change.
		'return': Calculates the return at the interval given 
				by the 'interval' parameter (see below). 
interval (int, default:1) - Interval to look ahead. The parameter takes effect only 
	if method=='cohl' or method=='return'. 
amplitude (float or function, default:None) - Price amplitude for the 'amplitude' method
scale (list or array of float, default:None) - A scale used to group oberved values 
		(calculated by look-ahead functionality) into classes (labels).
~~~
Example: 
```python 
calcDataObject = CalcData(5)
calcDataObject.addLookAheadOp( "ochl", 2 )
``` 


### addLookAheadFunc ###
Instead of using the addLookAheadOp function a custom function can be used.
```python
	addLookAheadFunc( func )
```
~~~
	func (function) - A custom function to calculate sample outputs (labels) and profits
		used for network training and testing.  
~~~
	Example:
```python
		calcDataObject = CalcData(5)

		def lookAhead( calcDataObject, pastRates, futureRates ):
			# doing calculations
			return [0,1,0,0,0], -100 

		calcDataObject.addLookAheadFunc( lookAhead )
```


### run ###
Calculates inputs and (if required) sample output and profit.
```python
	run( pastRates, futureRates )
```
~~~
	pastRates (dictionary) - Historic data (rates) used to calcuate inputs.
		pastRates['op'] (1d numpy array, float) - open rates where the latest element has index 0.
		pastRates['hi'] (1d numpy array, float) - high rates where the latest element has index 0.
		pastRates['lo'] (1d numpy array, float) - low rates where the latest element has index 0.
		pastRates['cl'] (1d numpy array, float) - close rates where the latest element has index 0.
		pastRates['dtm'] (1d array, datetime) - date and time values. 
		pastRates['vol']  (1d numpy array, float) - volume data.
		All the arrays are synchronized with each other so that for example:
		pastRates['op'][0] is the open price of 0th candles, 
		pastRates['cl'][0] is the close price of the 0th candle,
		pastRates['dtm'][0] is the date and time for the 0th candle and
		pastRates['vol'][0] is the volume for the 0th candle.
	futureRates (dictionary) - Data used to calculate sample output and profit for network
		training and testing. If futureRates is None no calculations would be performed.
		The format of the futureRates variable is the same as for the pastRates variable
		with one exception: 0th index stands for the nearest "future", i.e.
		futureRates['op'][0] is the first rate to come.

	Returns inputs, labels, profit:
		inputs (a list or numpy array, float) - The inputs.
		labels (1d one-hot numpy array or a float) - Expected output of the network.
			If a float (raw observed value) is returned, this value would be converted
			into the one-hot array automatically.
		profit (float) - Expected profit of the potential trade. 
~~~

The function is called implicitly by the prepareData function. As well the function
can be called explicitly when running the network in order to obtain trading signals.

### save ###
Saves the CalcData object into a file for future load by the load() function.
```python
	calcDataObject.save( fileName )
```

### load ###
Loads the CalcData object previously saved by the save() function.
def load( fileName ):

Example:
```python
	calcData = CalcData(5)
	# ...
	calcData.save("calcdata.svd")
	# ...
	import tnn.calcdata
	calcData2 = tnn.calcdata.load('calcdata.svd')
```


