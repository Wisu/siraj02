# cd C:\Users\xxsundbw\develop\wisu\siraj\work002
# python btcDemo.py

import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from dateutil.relativedelta import relativedelta


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import numpy as np
import math

sid_btc='btc-usd'
data_root = 'data\\'


def genFileName(sec_id):
	return data_root + sec_id + '.csv'

def loadData(sec_id):
	print("loadData.start")
#	return loadDataWeb(sec_id)
	return loadDataCsv(sec_id)


def loadDataCsv(sec_id):
	print("loadDataCsv.start")

	df = pd.read_csv(genFileName(sec_id)) 
	df.tail()

	return df

def loadDataWeb(sec_id):
	print("loadDataWeb.start:", sec_id)
	now = datetime.datetime.now()
	start = (now - relativedelta(years=6))		# datetime.datetime(2010, 1, 1)

#ts = strftime("%Y-%m-%d %H:%M", gmtime())
#	now = datetime.datetime.now()
#	fn = 'data/ppm-' + now.strftime("%Y%m%d") + '.json'
#	ctrs = [21, 34, 55, 89, 144]
#	for days in ctrs:		
#		print days, now, ":->", now - relativedelta(days=days)
#		lstDates.append((now - relativedelta(days=days)).strftime("%Y%m%d"))

#	end = (now - relativedelta(days=1))
	print('start:', start.strftime("%Y-%m-%d"), ', end:', now.strftime("%Y-%m-%d"))

	df = web.DataReader(sec_id, 'yahoo', start, now)
	df.tail()

	# BUP
	fn = genFileName(sec_id)
	print('save to file:', fn)
	df.to_csv (fn, header=True) 

	return df

def analyseOneSecPredict(sec_id, ds):
	print("analyseOneSecPredict.start:", sec_id)

	print("---1---")
	# 1.
	dfreg = ds.loc[:,['Adj Close','Volume']]
	dfreg['HL_PCT'] = (ds['High'] - ds['Low']) / ds['Close'] * 100.0
	dfreg['PCT_change'] = (ds['Close'] - ds['Open']) / ds['Open'] * 100.0	

	print("---2---", dfreg)
	# 2.
	# Drop missing value
	dfreg.fillna(value=-99999, inplace=True)
	# We want to separate 1 percent of the data to forecast
	forecast_out = int(math.ceil(0.01 * len(dfreg)))
	# Separating the label here, we want to predict the AdjClose
	forecast_col = 'Adj Close'
	dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
	X = np.array(dfreg.drop(['label'], 1))
	# Scale the X so that everyone can have the same distribution for linear regression
	X = preprocessing.scale(X)
	# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
	X_lately = X[-forecast_out:]
	X = X[:-forecast_out]
	# Separate label and identify it as y
	y = np.array(dfreg['label'])
	y = y[:-forecast_out]

#	print("---3---") 	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	# Linear regression
	clfreg = LinearRegression(n_jobs=-1)
	clfreg.fit(X_train, y_train)
	confidencereg = clfreg.score(X_test, y_test)
	print('The linear regression confidence is ', confidencereg)

	# Quadratic Regression 2
	clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
	clfpoly2.fit(X_train, y_train)
	confidencepoly2 = clfpoly2.score(X_test,y_test)
	print('The quadratic regression 2 confidence is ', confidencepoly2)

	# Quadratic Regression 3
	clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
	clfpoly3.fit(X_train, y_train)
	confidencepoly3 = clfpoly3.score(X_test,y_test)
	print('The quadratic regression 3 confidence is ', confidencepoly3)

	# KNN Regression
	clfknn = KNeighborsRegressor(n_neighbors=2)
	clfknn.fit(X_train, y_train)
	confidenceknn = clfknn.score(X_test, y_test)
	print('The knn regression confidence is ', confidenceknn)

	print("Something on the screen...")
	close_px = ds['Adj Close']
	mavg = close_px.rolling(window=100).mean()

	# Adjusting the size of matplotlib
	mpl.rc('figure', figsize=(12, 10))
	mpl.__version__

	# Adjusting the style of matplotlib
	style.use('ggplot')

	close_px.plot(label=sec_id)
	mavg.plot(label='mavg')
	plt.legend()

	print("return on screen...")
	rets = close_px / close_px.shift(1) - 1
	rets.plot(label='return')

	dfreg['Adj Close'].tail(500).plot() 
	dfreg['Forecast'].tail(500).plot()
	plt.legend(loc=4) 
	plt.xlabel('Date') 
	plt.ylabel('Price') 
	plt.show()


def analyseOneSec(sec_id):
	print("analyseOneSec.start:", sec_id)
	ds = loadData(sec_id)

	analyseOneSecPredict(sec_id, ds)



	#fig.savefig("BTC-USD.png")
#	plt.show()



def main():
	print("---START---")
	analyseOneSec(sid_btc)

if __name__ == "__main__":
	main()
