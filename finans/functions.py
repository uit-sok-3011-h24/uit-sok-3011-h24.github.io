import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd


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
	df_monthly = pivot_df.resample('M').sum()
	pivot_df = pivot_df.dropna(axis=1, thresh=int(0.9 * len(pivot_df)))
	pivot_df = pivot_df.fillna(0)
	  
	return df_monthly

def calc_moments(df, field = 'lnDeltaP', excess = False):
	#Calculate the mean and covariance of the matrix df
	X = get_matrix(df, field, excess)
	cov_matrix = np.cov(X, rowvar=False)
	means = np.mean(X, axis=0)
	return cov_matrix, means, X


