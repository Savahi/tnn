# -*- coding: utf-8 -*- 
import numpy as np
from tnn.network import Network 
from tnn.io import loadNetwork

# ...
	nn = loadNetwork( "nn.db" )
	if nn is None:
		print "Can't load network.\nExiting..."
		sys.exit(0)

	input = # Calculating inputs
	
	output = nn.calcOutput(x)
	if output is not None:
		if np.argmax( output ) == numLabels-1:
			print "The network has chosen the biggest label. Going LONG then!" 
			# Doing LONG trade			

