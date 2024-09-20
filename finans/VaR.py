
from scipy.stats import norm
import numpy as np

def est_var(f, subtract_mean, win_start, df, estimation_win_size):
  pVar = [0.05, 0.01]
  Zscore = norm.ppf(pVar)
  datelist = []
  sigmalist = []
  d95list = []
  d99list = []
  ret = []
  for t in range(win_start, len(df)):
      datelist.append(df['date'].iloc[t])
      #this seems to be the window in the original nb:
      x = df['return'].iloc[t-estimation_win_size-1:t+1]
      #but that would mean we are using future history to predict
      #current returns. In stead it should be lagged one period before:
      #x = df['return'].iloc[t-estimation_win_size-1:t-1]

      if subtract_mean:
        x = x - np.mean(x)
      d95, d99, sigma = f(x, Zscore, pVar, sigmalist)
      sigmalist.append(sigma)
      d95list.append(d95)
      d99list.append(d99)
      ret.append(df['return'].iloc[t])
  return (np.array(d95list),
          np.array(d99list),
          np.array(sigmalist),
          np.array(datelist),
          np.array(ret)
          )





def plot(plt, d95, d99, dates, heading):
  plt.cla()
  yaxis = np.array(d95)
  yaxis1 = np.array(d99)
  plt.plot(dates, yaxis)
  plt.plot(dates, yaxis1)
  plt.ylabel('VaR')
  plt.xlabel('Date')
  plt.legend(['95% Confidence Level', '99% Confidence Level'], loc="lower right")
  plt.title(heading)
  plt.show()