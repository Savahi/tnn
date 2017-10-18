import taft

class CalcData:
	'''
	numLabels (int, default:5) - The number of labels (classes)
	intraDay (boolean, default:False) - If 'True' 'pastRates' and 'futureRates' are iterated within a single day only
	tradingDays (list of int, default:None) - Day(s) of week allowed for trading, '0'-monday, 4-'friday'  
	tradingTime (list of int pairs, default:None ) - Hours and minutes allowed for trading
		Example: [ [14, None], [15, 0], [15,5], [15,10] ] allows trades to open from 14:00 to 15:10
	'''
	def __init__( self, numLabels=5, intraDay=False, tradingDays=None, tradingTime=None ):

		self.intraDay = intraDay
		self.numLabels = numLabels

		self.tradingDays = tradingDays
		self.tradingTime = tradingTime

		self.lookBackOps = [] # { 'name': , 'shift':  , 'period': , etc }
	# end of def

	'''
		method (string, default: 'ochl-ratio') - Possible values:
			'ochl-ratio': The ratio <close-less-open> divided by <high-less-low> is calculated within 
				the interval given by the 'interval' parameter (see below);
			'tpsl-slid': A trade is simulated with sliding take-profit and stop-loss value given by the 
				'tpsl' parameter (see below).
			'return': The return (<close-less-open> value) is calculated for the interval given 
				by the 'interval' parameter (see below). 
		interval (int, default:1) - Interval to look ahead. The parameter takes effect only 
			if method == 'ochl-ratio' or 'return'. 
		tpsl (float or function, default:None) - Take-profit and stop-loss used when simulating a trade
			(for 'tpsl-slid' method only, see above).
			If 'function' calculates the size of take-profit/stop-loss and returns the value. Example: 
				def tpsl( pastRates ): 
					return (pastRates['hi'][0] - pastRates['lo'][0])
		bounds (list of float, default:None) - Bounds to split return (profit) into classes (labels)
			Example: [ -1, 0, 1 ] splits the whole range of values into four groups: less than -1, 
				from -1 to 0, from 0 to 1, more than 1.
			Takes effect only if method=='return' (see above).
		calc (function, default:None) - If calc is not None it should be a function that calculates 
			a one-hot list or numpy array of labels. The other parameters of the 'addLookAheadOp' 
			function have no effect if calc is not None. 
			Example:
				def calc( futureRates ):
					# Doing calculations 
					return [0,1,0,0,0]
	'''
	def addLookAheadOp( method="", interval=1, tpsl=None, bounds=None, noOvernight=False, calc=None ):
		self.lookAheadInterval = interval-1
		self.lookAheadMethod = method
		self.lookAheadNoOvernight = noOvernight
		self.lookAheadTPSL = tpsl
		self.lookAheadBounds = bounds

		self.lookAheadCalc = calc
	# end of def

	'''
	name (string, default:"sma") - The name of an indicator to generate an input value.
		Possible values:
			"high"
			"low"
			"sma"
			"rsi"
			"stochastic"
			"roc"
			"vol"
	shift (int, default:0) - The shift inside the 'past rates' arrays. 
		0th index points to the latest rates.
	period (int, default:10) - The interval used to calculate the indicator, given by 'name'.
	Example:
		calcDataObject.addLookBackOp( "sma", 1, 5 )
	'''
	def addLookBackOp( self, name="sma", shift=0, period=10, **kwargs ):

		lookBackOpDict = { 'name':name, 'shift':shift, 'period':period }
		for key in kwargs:
			lookBackOpDict[key] = kwargs[key]

		if name == 'high' or name == 'low':
			if not 'smoothing' in lookBackOpDict:
				lookBackOpDict['smoothing'] = 1

		self.lookBackOps.append( lookBackOpDict )
	# end of def


	def run( self, pastRates, futureRates ):
		retErr = None, None, None

		if len(pastRates) == 0 or len(futureRates) == 0:
			return retErr

		inputs = []
		labels = None
		profit = None

		cl0 = pastRates['cl'][0] # Последняя известная нам котировка - цена закрытия последней по времени поступления свечи
		op0 = futureRates['op'][0] # Ближайшее "доступное" нам "будущее" - это цена открытия 0-го (по futureRates) периода
		time0 = futureRates['dtm'][0] # Ближайшее "доступное" нам "будущее" - его дата и время

		# Trading days of week are given
		if self.tradingDays is not None:
			found = False
			for i in range( len( self.tradingDays ) ):
				if time0.weekday() == self.tradingDays[i]:
					found = True
					break
			if not found: # Wrong trading day
				return retErr

		# Trading time is given
		if self.tradingTime is not None:
			found = False
			for i in range( len(self.tradingTime) ):
				if time0.hour() == self.tradingTime[i][0]:
					if self.tradingTime[i][1] == None: # Если минуты не заданы
						found = True
						break
					elif time0.minute() == self.tradingTime[i][1]: # Если минуты заданы, нужно сравнение
						found = True
						break
			if not found:
				return retErr

		# ****************************************************************************************************************
		# **** LOOK BACK SECTION ****

		for i in range( len( self.lookBackOps ) ):
			name = self.lookBackOps[i]['name']
			shift = self.lookBackOps[i]['shift']
			period = self.lookBackOps[i]['period']
			smoothing = self.lookBackOps[i]['smoothing']

			if shift + period > (len(pastRates)): # If we have to look beyond the possibe bounds of arrays with rates 
				return retErr

			if self.intraDay: # If intra-day trading 
				if time0.day != pastRates['dtm'][shift+period-1].day: # if two different days
					return retErr

			inp = None # Another input to calculate and append to the inputs[] list

			if name == 'high':
				if smoothing == 1:
					inp = np.max( pastRates['hi'][shift:shift+period] ) - cl0
				else:
		 			upper = np.sort( pastRates['hi'][shift:shift+period] )
		 			inp = np.mean( upper[-smoothing:] ) - cl0
		 	elif name == 'low':
				if smoothing == 1:
					inp = cl0 - np.min( pastRates['lo'][shift:shift+period] )
				else:
		 			lower = np.sort( pastRates['lo'][shift:shift+period] )
		 			inp = cl0 - np.mean( lower[:smoothing] )
		 	elif name == 'sma':
		 		inp = taft.sma( period = period, shift = shift, rates=pastRates['cl'] )
		 	elif name == 'rsi':
		 		inp = taft.rsi( period = period, shift = shift, rates=pastRates['cl'] )
		 	elif name == 'stochastic':
		 		inp = taft.stochastic( periodK = period, shift = shift, hi=pastRates['hi'], lo=pastRates['lo'], cl=pastRates['cl'] )
		 	elif name == 'roc':
		 		inp = taft.roc( period = period, shift = shift, rates=pastRates['cl'] )
		 	elif name == 'vol':
		 		inp = np.sum( pastRates['vol'][shift:shift+period] )

		 	if inp is None:
		 		return retErr

			inputs.append(inp)
		# end of for

		# ****************************************************************************************************************
		# **** LOOK AHEAD SECTION ****
		if self.lookAheadCalc is not None: # A function calculating label and profit has been given
			labels, profit = self.lookAheadCalc( futureRates )
		else:
			if self.lookAheadMethod == 'ochl-ratio': # close-open / high-low ratio method
				if self.lookAheadInterval is None: # Look ahead period should had been given
					return retErr
				if self.lookAheadInterval >= len(futureRates):
					return retErr
				clAhead = futureRates['cl'][lookAhead]
				hiAhead = np.max( futureRates['hi'][:lookAhead+1] )
				loAhead = np.min( futureRates['lo'][:lookAhead+1] )
				diapason = hiAhead - loAhead
				if diapason > 0:
					observed = profit / diapason
				else:
					observed = 0.0
				label = int( float( self.numLabels) * ( (observed + 1.0) / (2.0 + 1e-10) ) )
				profit = clAHead - op0
			elif self.lookAheadMethod == 'return': # A return after a given interval of time method  
				if self.lookAheadInterval >= len(futureRates): # Can't look beyond the bounds
					return retErr
				profit = futureRates['cl'][self.lookAheadInterval] - op0
				label=-1
				if self.lookAheadBounds is None:
					return retErr					
				for b in range( len(self.lookAheadBounds) ):
					if profit < self.lookAheadBounds[b]:
						label=b
						break
				if label == -1:
					label = self.numLabels-1
			elif self.lookAheadMethod == 'slid-tpsl':
				if isinstance( tpsl, float ):
					tpsl = self.lookAheadTPSL
				elif isinstance( tpsl, callable ):
					tpsl = self.lookAheadTPSL( pastRates )
					if tpsl == None:
						return retErr
				else:
					return retErr

				hitUp = 0
				hitDown = 0
				isHit = False
				dayNow = futureRates['dtm'][0].day
				if self.lookAheadInterval is not None:
					lookAhead = self.lookAheadInterval
				else:
					lookAhead = len(futureRates)
				for ahead in range( lookAhead ):
					if ahead > 0 and self.intraDay:
						if futureRates['dtm'][ahead].day != dayNow:
							return retErr
					up = futureRates['hi'][ahead] - op0
					down = op0 - futureRates['lo'][ahead]
					if up > hitUp:
						hitUp = up
						if hitUp + hitDown >= tpsl:
							profit = tpsl - hitDown
							isHit = True
							break
					if down > hitDown:
						hitDown = down
						if hitUp + hitDown >= tpsl:
							profit = hitUp - tpsl
							isHit = True
							break
				if not isHit:
					return retErr
				label = int( float(self.numLabels) * ( 1.0 + profit/tp ) / (2.0 + 1e-10) )
				# end of if 

			labels = np.zeros( shape=[self.numLabels], dtype=np.float32 )
			labels[label] = 1.0
			# end of if

		return inputs, labels, profit
	# end of def run()
# end of Class CalcData
