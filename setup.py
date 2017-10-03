from setuptools import setup

setup( 
	name = 'tnn',
	version = '0.0.2',
	description = 'Tensorflow Neural Network Framework for Algorithmic Traders',
	url = 'http://github.com/Savahi/tnn',
	author = 'Savahi',
	author_email = 'sh@tradingene.ru',
	license = 'MIT',
	classifiers=[
	    'Development Status :: 3 - Alpha',
	    'Intended Audience :: Developers',
	],	
	packages = ['tnn'],
	keywords = 'neural network tensorflow algorithmic trading stock exchange',
	install_requires = ['tensorflow', 'numpy', 'datetime', 'shelve', 'os', 'taft'],
	zip_safe = False )


