TNN: Historic Rates Format 
==========================
version 0.0.4
    
> **Important notice**:
> Nothing important yet... :)   

Related documents: [Network Class library](README.md), [CalcData Class library](calcdata.md)   

### Rates Format ###

	Historic rates usually come packed into a dictionary with the following elements:
		rates['op'] (1d numpy array, float) - open rates, the index of the most recent element is 0.
		rates['hi'] (1d numpy array, float) - high rates, the index of the most recent element is 0.
		rates['lo'] (1d numpy array, float) - low rates, the index of the most recent element is 0.
		rates['cl'] (1d numpy array, float) - close rates, the index of the most recent element is 0.
		rates['dtm'] (1d array, datetime) - dates and times. 
		rates['vol']  (1d numpy array, float) - volumes.
		All the arrays are synchronized with each other so that for example:
		rates['op'][0] is the open price of 0th (the most recent) candle, 
		rates['cl'][0] is the close price of the 0th candle,
		rates['dtm'][0] is the date and time for the 0th candle and
		rates['vol'][0] is the volume for the 0th candle.
