from cmath import log, sqrt
from re import I
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
    """
    Calculate the nannualized return

    Args:
        returns (pd.Series, pd.Dataframe): Periodical returns time series data.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to DAILY. Also can be "weekly", "monthly", "quarterly" or "yearly".
        logreturn (bool, optional): Define the type of returns.
         False for simple return, True for log return. Defaults to False.

    Returns:
        float, pd.Series: Annualized return for each time series.
    """    
    returns.dropna(how = 'all', inplace = True)
    if len(returns) < 1:
        return np.full([1,returns.ndim], np.nan)

    annual_factor = AnnualizedFactor(period)
    if isinstance(returns, pd.DataFrame):
        returns_length = returns.count(axis = 0)
    else:
        returns_length = len(returns)
    if logreturn:
        result = np.nanmean(returns, axis = 0) * annual_factor
    else:
        prod_return = np.nanprod((returns + 1).values, axis = 0)
        result = prod_return**(annual_factor/returns_length)-1

    return result

    
def AnnualizedSD(returns, period = DAILY):
    """
    Calculate the nannualized standard deviation

    Args:
        returns (pd.Series, pd.Dataframe): Periodical returns time series data.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to DAILY. Also can be "weekly", "monthly", "quarterly" or "yearly".

    Returns:
        float, pd.Series: Annualized standard deviation for each time series.
    """    
    returns.dropna(how = 'all', inplace = True)
    if len(returns) < 2:
        return np.full([1,returns.ndim], np.nan)
    
    annual_factor = AnnualizedFactor(period)
    if isinstance(returns, pd.DataFrame):
        result = np.nanstd(returns, axis = 0, ddof = 1) * np.sqrt(annual_factor)
    else:
        result = np.nanstd(returns, ddof = 1) * np.sqrt(annual_factor)
    return result

def CumulativeReturns(returns, logreturn = False):
    """
    _summary_

    Args:
        returns (_type_): _description_
        logreturn (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    returns.dropna(how = 'all', inplace = True)
    if len(returns) < 1:
        return returns.copy()
    
    if logreturn:
        result = returns.cumsum(axis = 0)
    else:
        result = returns.copy() + 1
        result = result.cumprod(axis = 0) - 1

    return result

def Drawdown(returns, logreturn = False):
    """
    _summary_

    Args:
        returns (_type_): _description_
        logreturn (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    if len(returns) < 1:
        return np.full([1,returns.ndim], np.nan)
    
    cumulativereturns = CumulativeReturns(returns, logreturn) + 1
    maxcumulativereturns = cumulativereturns.cummax(axis = 0)
    result = (cumulativereturns - maxcumulativereturns) / maxcumulativereturns
    
    return result

def AverageDrawDown(returns, logreturn = False):
    """
    _summary_

    Args:
        returns (_type_): _description_
        logreturn (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    if len(returns) < 1:
        return np.full([1,returns.ndim], np.nan)

    drawdown = Drawdown(returns, logreturn)
    if isinstance(drawdown, pd.DataFrame):
        result = pd.DataFrame(index = drawdown.index, columns = drawdown.columns)
        for asset in drawdown.columns:
            for i in range(len(drawdown)):
                result[asset][i] = np.nanmean(drawdown[asset][:i+1])
    else:
        result = pd.Series(index = drawdown.index)
        for j in range(len(drawdown)):
            result[j] = np.nanmean(drawdown[:j+1])
    return result
    
def MaxDrawdown(returns, logreturn = False):
    """
    _summary_

    Args:
        returns (_type_): _description_
        logreturn (bool, optional): _description_. Defaults to False.
    """
    if len(returns) < 1:
        return np.full([1,returns.ndim], np.nan)

    result = Drawdown(returns, logreturn)
    result = result.min(axis = 0)
    return result

def SharpeRatio(returns, risk_free = 0, logreturn = False, period = DAILY):
    """
    _summary_

    Args:
        returns (_type_): _description_
        risk_free (int, optional): _description_. Defaults to 0.
        logreturn (bool, optional): _description_. Defaults to False.
        period (_type_, optional): _description_. Defaults to DAILY.

    Returns:
        _type_: _description_
    """    
    returns.dropna(how = 'all', inplace = True)
    if len(returns) < 2:
        return np.full([1,returns.ndim], np.nan)
    
    result = np.divide(AnnualizedReturn(returns, period, logreturn)-risk_free, AnnualizedSD(returns, period))
    return result
 
def CalmarRatio(returns, period = DAILY, logreturn = False):
    """
    _summary_

    Args:
        returns (_type_): _description_
        period (_type_, optional): _description_. Defaults to DAILY.
        logreturn (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    max_drawdown = MaxDrawdown(returns, logreturn)
    result = AnnualizedReturn(returns, period, logreturn) / abs(max_drawdown)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result

def BurkeRatio(returns, risk_free = 0, period = DAILY, logreturn = False, modified = False):
    """
    _summary_

    Args:
        returns (_type_): _description_
        risk_free (int, optional): _description_. Defaults to 0.
        period (_type_, optional): _description_. Defaults to DAILY.
        logreturn (bool, optional): _description_. Defaults to False.
        modified (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    rp = AnnualizedReturn(returns, period, logreturn)
    drawdown = Drawdown(returns, logreturn)
    result = (rp - risk_free) / np.sqrt(np.nansum(drawdown**2, axis = 0))
    if modified:
        result = result * np.sqrt(drawdown.count(axis = 0))
    result = result.replace([np.inf, -np.inf], np.nan)
    return result



