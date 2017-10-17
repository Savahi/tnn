def plot_curves(nn, cb=(lambda plot:plot.show())):
	import matplotlib.pyplot as plt

	# Train
	plt.figure ()

	plt.subplot(231)
	plt.plot (nn.costTrain)
	plt.title("cost train")

	plt.subplot(232)
	plt.plot(nn.accuracyTrain)
	plt.title("accuracy train")

	plt.subplot(233)
	plt.plot(nn.balanceTrain)
	plt.title("plt train")

	# Test

	plt.subplot(234)
	plt.plot (nn.costTest)
	plt.title("cost test")

	plt.subplot(235)
	plt.plot(nn.accuracyTest)
	plt.title("accuracy test")

	plt.subplot(236)
	plt.plot([x-10 for x in nn.balanceTest])
	plt.title("plt test")

	

	"""
	titleText = "fl=%s, lr=%g, bl=%g, opt=%s, ep=%d fl=%d" % (config["raw_file"], config["learningRate"], 0, config["optimizer"], config["numEpochs"], 0)
	plt.figure(  )
	plt.subplot(221)
	plt.scatter( nn.costTrain, nn.costTest, marker = '+', color = 'blue' )
	plt.title( titleText + "\n\ncost-function: train vs test")
	plt.grid()
	plt.subplot(222)
	plt.scatter( nn.accuracyTrain, nn.accuracyTest, marker = '+', color = 'blue' )
	plt.title("accuracy: train vs test")
	plt.grid()
	plt.subplot(223)
	plt.scatter( nn.tradeAccuracyTrain, nn.tradeAccuracyTest, marker = '+', color = 'blue' )
	plt.title("trade accuracy: train vs test")
	plt.grid()
	plt.subplot(224)
	plt.scatter( nn.balanceTrain, nn.balanceTest, marker = '+', color = 'blue' )
	plt.title("balance: train vs test")
	plt.grid()
		
	plt.gcf().set_size_inches( 16, 8 )
	#plt.savefig( nn.learnDir + ".png", bbox_inches='tight' )
	print ("Showing the plot...")
	plt.show()"""

	return cb (plt)


