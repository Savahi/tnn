# -*- coding: utf-8 -*- 
import numpy as np
import taft
import utils

class CalcData:
	'''
	numLabels (int, default:5) - The number of labels (classes)
	intraDay (boolean, default:False) - If 'True' 'pastRates' and 'futureRates' are iterated within a single day only
	tradingDays (list of int, default:None) - Day(s) of week allowed for trading, '0'-monday, 4-'friday'  
	tradingTime (list of int pairs, default:None ) - Hours and minutes allowed for trading
		Example: [ [14, None], [15, 0], [15,5], [15,10] ] allows trades to open from 14:00 to 15:10
	'''
	def __init__( self, numLabels=5, intraDay=False, tradingDays=None, tradingTime=None, precalcData=None ):

		self.intraDay = intraDay
		self.numLabels = numLabels

		self.tradingDays = tradingDays
		self.tradingTime = tradingTime

		self.lookBackOps = [] # { 'name': , 'shift':  , 'period': , etc }

		self.precalcData = precalcData
	# end of def

	'''
		method (string, default: 'cohl') - Possible values:
			'cohl-ratio': The ratio <close-less-open> divided by <high-less-low> is calculated within 
				the interval given by the 'interval' parameter (see below);
			'amplitude': Waiting for price making the given amplitude (see below).
			'return': The return (<close-less-open> value) is calculated for the interval given 
				by the 'interval' parameter (see below). 
		interval (int, default:1) - Interval to look ahead. The parameter takes effect only 
			if method == 'cohl' or 'return'. 
		amplitude (float or function, default:None) - Price amplitude for the 'amplitude' method
		scale (list of float, default:None) - Scale used to group oberved values 
			(calculated by look-ahead functionality) into classes (labels)
	'''
	def addLookAheadOp( self, method="ochl", interval=1, amplitude=None, scale=None, noOvernight=False ):
		self.lookAheadInterval = interval-1
		self.lookAheadMethod = method
		self.lookAheadNoOvernight = noOvernight
		self.lookAheadAmplitude = amplitude
		self.lookAheadLabelScale = scale

		self.lookAheadFunc = None
	# end of def

	'''
		Calculates a one-hot list or numpy array of labels. 
		Example:
			def calc( futureRates ):
				# Doing calculations 
				return [0,1,0,0,0]
	'''
	def addLookAheadFunc( self, func ):
		self.lookAheadFunc = func
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
			"return"
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

		if name == 'high' or name == 'low': # Для индикаторов high и low добавляем значение smoothing по умолчанию
			if not 'smoothing' in lookBackOpDict:
				lookBackOpDict['smoothing'] = 1

		self.lookBackOps.append( lookBackOpDict )
	# end of def


	def run( self, pastRates, futureRates ):
		retErr = None, None, None

		if len(pastRates) == 0: # To avoid exception
			return retErr

		inputs = [] # To be returned as a set of inputs
		labels = None # To be returned by the function
		profit = None # To be returned by the function

		cl0 = pastRates['cl'][0] # Последняя известная нам котировка - цена закрытия последней по времени поступления свечи
		time0 = pastRates['dtm'][0] # Последняя известная нам котировка - ее дата и время

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
				hr = self.tradingTime[i][0]
				if isinstance( hr, int ):
					if time0.hour != hr:
						continue
				elif isinstance( hr, list ):
					if time0.hour < hr[0] or time0.hour > hr[1]:
						continue
				elif hr is not None:
					continue

				mn = self.tradingTime[i][1]
				if isinstance( mn, int ):
					if time0.minute != mn:
						continue
				elif isinstance( mn, list ):
					if time0.minute < mn[0] or time0.minute > mn[1]:
						continue
				elif mn is not None:
					continue
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

			if shift + period > (len(pastRates['cl'])): # If we have to look beyond the possibe bounds of arrays with rates 
				return retErr

			if self.intraDay: # If intra-day trading 
				if time0.day != pastRates['dtm'][shift+period-1].day: # if two different days
					return retErr

			inp = None # Another input to calculate and append to the inputs[] list

			if name == 'high':
				smoothing = self.lookBackOps[i]['smoothing']
				if smoothing == 1:
					inp = np.max( pastRates['hi'][shift:shift+period] ) - cl0
				else:
		 			upper = np.sort( pastRates['hi'][shift:shift+period] )
		 			inp = np.mean( upper[-smoothing:] ) - cl0
		 	elif name == 'low':
		 		smoothing = self.lookBackOps[i]['smoothing']
				if smoothing == 1:
					inp = cl0 - np.min( pastRates['lo'][shift:shift+period] )
				else:
		 			lower = np.sort( pastRates['lo'][shift:shift+period] )
		 			inp = cl0 - np.mean( lower[:smoothing] )
		 	elif name == 'return':
		 		inp = cl0 - pastRates['cl'][shift+period-1]
		 	elif name == 'sma':
		 		sma = taft.sma( period = period, shift = shift, rates=pastRates['cl'] )
		 		if sma is not None:
		 			inp = cl0 - sma
		 	elif name == 'rsi':
		 		res = taft.rsi( period = period, shift = shift, rates=pastRates['cl'] )
		 		if res is not None:
		 			if res['rsi'] is not None:
		 				inp = res['rsi']
		 	elif name == 'stochastic':
		 		res = taft.stochastic( periodK = period, shift = shift, hi=pastRates['hi'], lo=pastRates['lo'], cl=pastRates['cl'] )
		 		if res is not None:
		 			if res['K'] is not None:
		 				inp = res['K']
		 	elif name == 'roc':
		 		inp = taft.roc( period = period, shift = shift, rates=pastRates['cl'] )
		 	elif name == 'vol':
		 		inp = np.sum( pastRates['vol'][shift:shift+period] )

		 	if inp is None:
		 		return retErr

		 	inputs.append(inp)
		# end of for

		if futureRates is None: # The function run 'for real', no future rates are known...
			return inputs

		# ****************************************************************************************************************
		# **** LOOK AHEAD SECTION ****

		if len(futureRates) == 0:
			return retErr

		op0 = futureRates['op'][0] # Ближайшее "доступное" нам "будущее" - это цена открытия 0-го (по futureRates) периода

		if self.lookAheadFunc is not None: # A function calculating label and profit has been given
			labels, profit = self.lookAheadFunc( self, pastRates, futureRates )
		else:
			label = -1
			observed = 0.0
			if self.lookAheadMethod == 'cohl': # close-open / high-low ratio method
				if self.lookAheadInterval is None: # Look ahead period should had been given
					return retErr
				if self.lookAheadInterval >= len(futureRates['cl']):
					return retErr
				clAhead = futureRates['cl'][self.lookAheadInterval]
				hiAhead = np.max( futureRates['hi'][:self.lookAheadInterval+1] )
				loAhead = np.min( futureRates['lo'][:self.lookAheadInterval+1] )
				profit = clAhead - op0
				diapason = hiAhead - loAhead
				if diapason > 0:
					observed = profit / diapason
				else:
					observed = 0.0
				label = int( float( self.numLabels) * ( (observed + 1.0) / (2.0 + 1e-10) ) )
			elif self.lookAheadMethod == 'return': # A return (close-open) at the given interval of time  
				if self.lookAheadInterval >= len(futureRates['cl']): # Can't look beyond the bounds
					return retErr
				profit = futureRates['cl'][self.lookAheadInterval] - op0
				if self.lookAheadLabelScale is not None:
					label = self.getLabelByScale( profit )
				else:
					observed = profit
			elif self.lookAheadMethod == 'amplitude': # Sliding take-profit/stop-loss method
				if self.lookAheadAmplitude is None: 
					return retErr
				amplitude = self.lookAheadAmplitude
				if self.lookAheadInterval is not None:
					lookAhead = self.lookAheadInterval
				else:
					lookAhead = len(futureRates)
				hitUp = 0
				hitDown = 0
				isHit = False
				for ahead in range( lookAhead ):
					if ahead > 0 and self.intraDay: # If intra-day trading
						if futureRates['dtm'][ahead].day != futureRates['dtm'][0].day:
							return retErr
					up = futureRates['hi'][ahead] - op0 
					down = op0 - futureRates['lo'][ahead]
					if up > hitUp:
						hitUp = up
						if hitUp + hitDown >= amplitude:
							profit = amplitude - hitDown
							isHit = True
							break
					if down > hitDown:
						hitDown = down
						if hitUp + hitDown >= amplitude:
							profit = hitUp - amplitude
							isHit = True
							break
				if not isHit:
					return retErr
				label = int( float(self.numLabels) * ( 1.0 + profit/amplitude ) / (2.0 + 1e-10) )
				# end of if 
			if label >= 0:
				labels = np.zeros( shape=[self.numLabels], dtype=np.float32 )
				labels[label] = 1.0
			else:
				labels = observed				
			# end of if

		return inputs, labels, profit
	# end of def run()


	def save( self, fileName ):
	    return utils.save( fileName, self, 'CalcData' )


	def getLabelByScale( self, observed ):
		scale = self.lookAheadLabelScale
		label = -1
		for i in range( len(scale) ):
			if observed < scale[i]:
				label = i
				break
		if label == -1:
			label = len(scale)
		return label

# end of Class CalcData

def load( fileName ):
	return utils.load( fileName, 'CalcData' )

