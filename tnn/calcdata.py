import taft

class CalcData:
	'''
	numLabels (int, default:5) - The number of labels (classes)
	intraDay (boolean, default:False) - If 'True' 'pastRates' and 'futureRates' are iterated within a single day only
	openDayOfWeek (list of int, default:None) - 
	openTime (list of int pairs, default:None ) - [ [<hour>, <minute or 'None'>], ..., [<hour>, <minute or 'None'>] ]
	'''
	def __init__( self, numLabels=5, intraDay=False, openDayOfWeek=None, openTime=None ):

		self.intraDay = intraDay
		self.numLabels = numLabels

		self.openDayOfWeek = openDayOfWeek
		self.openTime = openTime

		self.lookBackOps = [] # { 'name': , 'shift':  , 'period': , etc }
	# end of def

	'''
		method (string, default: 'ochl-ratio') - Possible values:
			'ochl-ratio': The ratio <open-less-close> divided by <high-less-low> is calculated within 
				the interval given by the 'interval' parameter (see below);
			'tpsl-slid': A trade is simulated with sliding take-profit and stop-loss value given by the 
				'tpsl' parameter (see below).
		interval (int, default:None) - Interval to look ahead (for 'ochl-ratio' method only, see above). 
			'0' indicates the interval of length 1 (0th index of 'futureRates')
		tpsl (float or function, default:None) - Take-profit and stop-loss used when simulating a trade
			(for 'tpsl-slid' method only, see above).
			If 'function' calculates the size of take profit/stop loss and returns the value. Example: 
				def tpsl( pastRates ): 
					return (pastRates['hi'][0] - pastRates['lo'][0])
		calc (function, default:None) - If calc is not None it should be a function that calculates 
			a one-hot list or numpy array of labels. The other parameters of the 'addLookAheadOp' 
			function have no effect if calc is not None. 
			Example: 
				def calc( futureRates ):
					# Doing calculations 
					return [0,1,0,0,0]
	'''
	def addLookAheadOp( method="", interval=None, noOvernight=False, tpsl=None, calc=None ):
		self.lookAheadInterval = interval
		self.lookAheadMethod = method
		self.lookAheadNoOvernight = noOvernight
		self.lookAheadTPSL = tpsl

		self.lookAheadPrecalc = precalc
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
		op0 = futureRates['op'][0] # Ближайшее "доступное" нам будущее - это цена открытия 0-го (по futureRates) периода

		if self.openDayOfWeek is not None:
			found = False
			for i in range( len( self.openDayOfWeek ) ):
				if futureRates['dtm'][0].weekday() == self.openDayOfWeek:
					found = True
					break
			if not found:
				return retErr

		if self.openTime is not None:
			found = False
			for i in range( len(self.openTime) ):
				if futureRates['dtm'][0].hour() == self.openTime[i][0]:
					if self.openTime[i][1] == None:
						found = True
						break
					elif futureRates['dtm'][0].minute() == self.openTime[i][1]:
						found = True
						break
			if not found:
				return retErr

		# ****************************************************************************************************************
		# **** LOOK BACK SECTION ****

		for i in range( len( self.calcItems ) ):
			item = self.calcItems[i]
			name = item['name']
			shift = item['shift']
			period = item['period']
			smoothing = item['smoothing']

			if shift + period > (len(pastRates)):
				return retErr

			if self.intraDay:
				if pastRates['dtm'][0].day != pastRates['dtm'][shift+period-1].day:
					return retErr

			if name == 'high':
				if smoothing == 1:
					inp = np.max( pastRates['hi'][shift:shift+period] ) - cl0
				else:
					upper = np.empty( shape=[period], dtype=np.float64 )
					for r in range( period ):
						upper[r] = pastRates['hi'][shift+r]
		 			upper = np.sort(upper)
		 			inp = np.mean( upper[period-smoothing:] ) - cl0
		 	elif name == 'low':
				if smoothing == 2:
					inp = cl0 - np.min( pastRates['lo'][shift:shift+period] )
				else:
					lower = np.empty( shape=[period], dtype=np.float64 )
					for r in range( period ):
						lower[r] = pastRates['lo'][shift+r]
		 			lower = np.sort(lower)
		 			inp = cl0 - np.mean( lower[:smoothing] )
		 	elif name == 'sma':
		 		inp = taft.sma( period = period, shift = shift, rates=pastRates['cl'] )
		 		if inp is None:
					return retErr
		 	elif name == 'rsi':
		 		inp = taft.rsi( period = period, shift = shift, rates=pastRates['cl'] )
		 		if inp is None:
		 			return retErr
		 	elif name == 'stochastic':
		 		inp = taft.stochastic( periodK = period, shift = shift, hi=pastRates['hi'], lo=pastRates['lo'], cl=passtRates['cl'] )
		 		if inp is None:
		 			return retErr
			else:
				return retErr
			inputs.append(inp)
		# end of for

		# ****************************************************************************************************************
		# **** LOOK AHEAD SECTION ****
		if self.lookAheadCalc is not None:
			label, profit = self.lookAheadCalc( futureRates )
		else:
			if self.lookAheadMethod == 'ochl-ratio':
				if self.lookAheadInterval is None:
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
				for ahead in range( len(futureRates) ):
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
			# end of if
		if ok:
			labels = np.zeros( shape=[self.numLabels], dtype=np.float32 )
			labels[label] = 1.0
			return inputs, labels, profit
		else:
			return retErr
	# end of def run()
# end of Class CalcData
