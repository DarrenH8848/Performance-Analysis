import pandas as pd
import numpy as np
from .Constant import ANNUALIZATION_FACTORS, DAYS_PER_YEAR
from .Constant import DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY


def AnnualizedFactor(period):
    try:
        factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be {}".format(period)
        )
    return factor


def AnnualizedReturn(returns, period = DAILY, logreturn = False):
    returns.dropna(how = 'any', inplace = True)
    if len(returns) < 1:
        return np.nan

    annual_factor = AnnualizedFactor(period)
    returns_length = len(returns)
    if logreturn:
        result = np.mean(returns) * annual_factor
    else:
        prod_return = (returns+1).prod()
        result = prod_return**(annual_factor/returns_length)-1

    return result
    

# df = pd.read_csv('./out/matrix_return_data.csv')
# df.set_index(['Date'],inplace=True)
# print(AnnualizedReturn(df))


