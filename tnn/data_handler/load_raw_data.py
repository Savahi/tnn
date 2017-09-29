import pandas as pd
import numpy  as np
import shelve
from data_handler_tim.indicators import indicator, indicators


class InputsShape():
        def __init__(self, indicators, history, num_cand_inds):
                self.indicators = indicators
                self.history    = history
                self.num_cand_inds = num_cand_inds

        def num_inputs(self):
                return len(self.indicators) * self.history



class RawData():
	def __init__(self, filename):
		# check if file exists
		self.df = pd.read_csv(filename)
		self.raw_inputs = None
		self.normalized_inputs = None
		self.inputs_shape = None
		# load it in the dataframe

	def form_inputs(self, inputs_shape):
		self.inputs_shape = inputs_shape
		inputs = []
		print ("Processing raw data file...")
		for i in range(len(self.df)-max(inputs_shape.history, inputs_shape.num_cand_inds)-10): #weird Nones
			inp = []
			for k in range(inputs_shape.history):
				inp += ([ indicator(x) (self.df[i+k:], inputs_shape.num_cand_inds) for x in inputs_shape.indicators ])
			inputs.append(inp)
		self.raw_inputs = inputs
		return inputs

	def normalize_inputs(self, norm=None, std=None):
		if (self.normalized_inputs): return self.normalized_inputs
		if self.raw_inputs is None:
			print ("Please call form_inputs first")
			raise
	        inputs = self.raw_inputs		
		if norm is None or std is None:
			norm = np.mean(self.raw_inputs)
			std  = np.std(self.raw_inputs)
		self.normalized_inputs = [(x-norm)/std for x in inputs]	
		return self.normalized_inputs

	def dump_to_file(self, filename):
		db = shelve.open(filename)
		db["raw_inputs"]   = self.raw_inputs	
		db["normalized_inputs"] = self.normalized_inputs
		db.close()
		print ("Dumped to file")
