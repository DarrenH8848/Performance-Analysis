import pandas as pd
import numpy as np
import os
from pathlib import Path

current_file = os.path.realpath(__file__)
current_file_path = Path(current_file)
mypath = current_file_path.parent.absolute()
DIR_WORK = (mypath.parent.absolute()).parent.absolute()
DIR_DATA = DIR_WORK/'data'
DIR_OUT = DIR_WORK/'out'

data = pd.read_csv(DIR_DATA/'etf_us.csv')
data.set_index(['Date'],inplace=True)
raw_returns = data.pct_change()
raw_returns.to_csv(DIR_OUT/"raw_return_data.csv")
benchmark_returns = raw_returns.iloc[:,0]
benchmark_returns.to_csv(DIR_OUT/"benchmark_return_data.csv")

data.dropna(how='any', inplace=True)
# daily returns
returns = data.pct_change()
returns.dropna(how='any', inplace=True)
one_return = returns.iloc[:,0]
# store the returns
returns.to_csv(DIR_OUT/"matrix_return_data.csv")
one_return.to_csv(DIR_OUT/'single_return_data.csv')



