import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
import decomposition


def get_matrix(df, field, excess):
	# Converts the df to a matrix df that can be used for portfolio analysis
	df['Date'] = pd.to_datetime(df['Date'])
	if excess:
		df['rx'] = df[field]-df['NOWA_DayLnrate']
	else:
		df['rx'] = df[field]
	# Remove duplicate entries
	df_unique = df.drop_duplicates(subset=['Date', 'ISIN'])

	# Pivot the dataframe
	pivot_df = df_unique.pivot(index='Date', columns='ISIN', values='rx')
	pivot_df = pivot_df.dropna(axis=1, thresh=int(0.9 * len(pivot_df)))
	pivot_df = pivot_df.fillna(0)
	df_weekly = pivot_df.resample('W').sum()*20 #annualized

	return np.array(df_weekly)

def calc_moments(df, field = 'lnDeltaP', excess = False):
	#Calculate the mean and covariance of the matrix df
	X = get_matrix(df, field, excess)
	cov_matrix = np.cov(X, rowvar=False)
	means = np.mean(X, axis=0).reshape((X.shape[1],1))
	return cov_matrix, means, X

if False:
	df = pd.read_pickle('uit-sok-3011-h24.github.io/finans/output/stocks.df')
	cov_matrix_big, means_big, df_month = calc_moments(df)

	R = decomposition.get_independent_portfolios(cov_matrix_big, 0.000001)
	cov_matrix = R.T @ cov_matrix_big @ R
	means = R.T @ means_big
	a=0