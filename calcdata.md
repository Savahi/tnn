TNN: CalcData Class 
================================================================
version 0.0.4
    
> **Important notice**:
> Nothing important yet... :)

## CONTENTS / СОДЕРЖАНИЕ ##
[CacData](#calcdata) - CalcDataConstructor / Конструктор.  
[addLookBackOp](#addlookbackop) - Adding a new input (a method and details of how the input is to be calculated).  
[addLookAheadOp](#addlookaheadop) - Calculating of expected output and profit.  
[addLookAheadFunc](#addlookaheadfunc) - Calculating of expected output and profit.   
[run](#run) - self explained.  
Related documents: [Network Class library](README.md), [Historic Rates Format](rates.md)  

### CalcData ###
Creates an instance of CalcData class. The CalcData object is used:
- Coupled with the [prepareData](README.md#preparedata) function when generating input data for network training and testing.
In this case the CalcData object serves as the 'calcData' parameter of the [prepareData](README.md#preparedata) function.
- When running the trained network in 'trade mode' for prognosing.
In this case the CalcData object is used for generating inputs for the network.  

Создает экземпляр объекта CalcData. Объект используется:  
- при создании данных для обучения и тестирования сети; в этом случае  он передается функции [prepareData](README.md#preparedata). 
- при использовании сети в торговле; в этом случае он используется для генерации входных данных.

```python
	CalcData( numLabels=5, intraDay=False, tradingDays=None, tradingTime=None, precalcData=None ):
```
	numLabels (int, default:5) - The number of labels (classes)
	intraDay (boolean, default:False) - If 'True' 'pastRates' and 'futureRates' are monitored 
		to belong to the same day.
	tradingDays (list of int, default:None) - Day(s) of week allowed for trading, '0'-monday, 4-'friday'.
		For example: [1,2,3] stands for tuesday, wednesday and thurthday as the days when 
		trading is allowed. 
	tradingTime (list of lists, default:None ) - Hours and minutes allowed for trading. 
		Each item of the list is a pair of elements where the first one stands for the 
		hour(s) and the second one stands for the minute(s). For example:
		[[14,15]] - 14:15
		[[14,[0,30]]] - from 14:00 to 14:30 (hours: 14, minutes: from 0 to 30)
		[[14, None]] - from 14:00 to 14:59 (hours: 14, minutes: any)
		[[[14-15],None]] - from 14:00 to 15:59 (hours: from 14 to 15, minutes: any)
		[[None,0]] - beginning of every hour (hours: any, minutes: 0)
		[[14,None],[15,[0,30]]] - from 14:00 to 15:30
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
	name (string, default:"sma") - The name of the indicator to use for generating an input value for the network. 
		Possible values:
			"high" - the highest value at the [shift:shift+period] interval
			"low" - the lowest value at the [shift:shift+period] interval
			"sma" - the Simple Moving Average value for the [shift:shift+period] interval
			"rsi" - the RSI value for the [shift:shift+period] interval
			"stochastic" - the Stochastic value for the [shift:shift+period] interval
			"roc" - the ROC value for the [shift:shift+period] interval
			"macd" - the MACD signal line value for the [shift:shift+period] interval
			"vol" - the total volume of trades at the [shift:shift+period] interval
			"return" - close[shift+period] - open[shift]
	shift (int, default:0) - Shift "into the past". 0 stands for the most latest rates. 
	period (int, default:10) - The lenght of the interval used to calculate 
		the indicator ('name').
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
	for the network. The 	allowed values are:
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
import tnn.calcdata
calcDataObject = tnn.calcdata.CalcData(5)
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
		import tnn.calcdata
		calcDataObject = tnn.calcdata.CalcData(5)

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
~~~
[See the format of the 'pastRates' variable here.](rates.md)
~~~
	futureRates (dictionary) - Data used to calculate sample output and profit for network
		training and testing. If futureRates is None no calculations would be performed.
		The format of the 'futureRates' variable is the same as for the 'pastRates' one
		with an exception: 0th index stands for the nearest "future", i.e.
		futureRates['op'][0] is the first rate to come.

	Returns inputs, labels, profit:
		inputs (a list or numpy array, float) - The inputs.
		labels (1d one-hot numpy array or a float) - Expected output of the network.
			If a float (raw observed value) is returned, this value would be converted
			into the one-hot array automatically. 
			If futureRates is None the 'labels' variable is None too.
		profit (float) - Expected profit of the potential trade.
			If futureRates is None the 'profit' variable is None too. 
~~~

The function is called implicitly by the prepareData function. As well the function
can be called explicitly when running the network in order to create inputs and 
obtain from the network a trading signal.
Example:
```python
	import tnn.calcdata
	calcData = tnn.calcdata.load('calcdata.svd')
	inputs = calcData.run( rates, None )

	from tnn.io import loadNetwork
	nn = loadNetwork('nn.db')

	from tnn.network import calcOutput
	signal = calcOutput( inputs )
```

### save ###
Saves the CalcData object into a file for future use.
The object can be loaded later by the load() function.
```python
	calcDataObject.save( fileName )
```

### load ###
Loads the CalcData object previously saved by the save() function.
```python
load( fileName )
```
Example:
```python
	calcData = CalcData(5)
	# ...
	calcData.save('calcdata.svd')
	# ...
	import tnn.calcdata
	calcData2 = tnn.calcdata.load('calcdata.svd')
```


