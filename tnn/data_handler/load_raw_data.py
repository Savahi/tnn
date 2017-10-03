# -*- coding: utf-8 -*- 
import pandas as pd
import numpy  as np
import random
import shelve
from tnn.data_handler.indicators import indicator, indicators


class InputsShape():
        def __init__(self, indicators, history, num_cand_inds):
                self.indicators = indicators
                self.history    = history
                self.num_cand_inds = num_cand_inds

        def num_inputs(self):
                return len(self.indicators) * self.history


class RawData():
	def __init__(self, filename):
		self.df = pd.read_csv(filename).iloc[::-1] # from present to past
		self.raw_inputs = None
		self.outputs = None
		self.normalized_inputs = None
		self.inputs_shape = None

	def form_inputs(self, inputs_shape):
		self.inputs_shape = inputs_shape
		inputs = []
		outputs= [] # hardcoded for now to predict the <OPEN> price 1 candle forward...
		print ("Processing raw data file...")
		for i in range(1, len(self.df)-max(inputs_shape.history, inputs_shape.num_cand_inds)-10): #weird Nones
			inp = []
			for k in range(inputs_shape.history):
				inp += ([ indicator(x) (self.df[i+k:], inputs_shape.num_cand_inds) for x in inputs_shape.indicators ])
			inputs.append(inp)
			output = self.df["<OPEN>"][i-1] - self.df["<OPEN>"][i]
			output = [1,0] if output<0 else [0,1]
			outputs.append(output)
		self.raw_inputs = inputs
		self.outputs = outputs
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
		db["outputs"] = self.outputs
		db.close()
		print ("Dumped to file")


def example_data():
	import random
	inp = InputsShape(["<OPEN>", "<CLOSE>"], 15, 6)
	rd = RawData("~/RTS_160601_170726.csv+5")
	rd.form_inputs(inp)
	inputs = rd.normalize_inputs()
	observed = rd.outputs
	profits = [random.randint(1,500) for _ in range(len(inputs))]
	return inputs,observed,profits
