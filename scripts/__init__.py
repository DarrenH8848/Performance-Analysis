"""
by default all function works on **log** returns
we try to offer only one, explicit, correct, fast way to users (when possible)
explicitly ask for numpy array only, accepting pandas DF only at the very end user side.
keep results in good shape: (T,N) to (1,N) (not (N,)) in most cases
import tool funcs from pkgs explicitly then no need to search from raw pkg again.
# 
the init scripts gives working directory and test dataframe for this project
"""
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from numpy import expm1, log1p
from pandas import DatetimeIndex, concat, read_csv

__all__ = ['DIR_WORK', 'data_test', 'capm', 'ratio', 'retn', 'risk', 'stat']

#
load_dotenv()
DIR_WORK = Path(getenv('DIR_WORK'))
#
log2simple = expm1
simple2log = log1p
#
tmp1 = read_csv(filepath_or_buffer=DIR_WORK / 'data' /
                'return_simple_last_1d.csv',
                index_col='open_time',
                infer_datetime_format=True)
tmp1.index = DatetimeIndex(tmp1.index)
tmp2 = read_csv(filepath_or_buffer=DIR_WORK / 'data' /
                'srs_return_simple_index_equally_weighted_last_1d.csv',
                index_col='open_time',
                infer_datetime_format=True)
tmp2.index = DatetimeIndex(tmp2.index)
tmp2.columns = ['idx_equally_weighted']
tmp3 = read_csv(filepath_or_buffer=DIR_WORK / 'data' /
                'srs_return_simple_index_volume_weighted_last_1d.csv',
                index_col='open_time',
                infer_datetime_format=True)
tmp3.index = DatetimeIndex(tmp3.index)
tmp3.columns = ['idx_volume_weighted']
#
data_test = simple2log(concat([tmp1, tmp2, tmp3], axis=1))
