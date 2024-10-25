import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
import decomposition


def get_matrix(df, field):
	"""Converts the df to a matrix df that can 
	be used to calculate the covariance matrix"""
	
	df['Date'] = pd.to_datetime(df['Date'])
	df_unique = df.drop_duplicates(subset=['Date', 'ISIN'])
	pivot_df = df_unique.pivot(index='Date', columns='ISIN', values=field)
	pivot_df = pivot_df.dropna()
	df_weekly = pivot_df.resample('W').sum()*52 #annualized

	return np.array(df_weekly)

X = get_matrix(df, 'lnDeltaP')
cov_matrix = np.cov(X, rowvar=False)
means = np.mean(X, axis=0).reshape((X.shape[1],1))
