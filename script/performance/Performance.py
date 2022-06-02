import pandas as pd
import numpy as np
from .Constant import ANNUALIZATION_FACTORS, DAYS_PER_YEAR
from .Constant import DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY


def _annualized_factor(period):
    try:
        factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be {}".format(period)
        )
    return factor

def _adjust_returns(returns, adjustment_factor):
    return returns - adjustment_factor

def annualized_return(returns, period = DAILY, logreturn = False):
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
        If input returns is pd.Dataframe, you will get output in pd.Series;
        If input returns is pd.Series, you will get output in float.   
    """     
    # if len(returns) < 1:
    #     return np.full([1,returns.ndim], np.nan)
    annual_factor = _annualized_factor(period)

    if isinstance(returns, pd.DataFrame):
        returns_length = returns.count(axis = 0)
    else:
        returns_length = returns.count()

    if logreturn:
        result = np.nanmean(returns, axis = 0) * annual_factor
    else:
        prod_return = np.nanprod((returns + 1).values, axis = 0)
        result = prod_return**(annual_factor/returns_length)-1
    
    if isinstance(returns, pd.DataFrame):
        result[returns_length < 1] = np.nan
    elif returns_length < 1:
        result = np.nan

    # if returns.ndim == 1 or len(returns.columns) == 1:
    #      return result.item()

    return result

    
def annualized_sd(returns, period = DAILY):
    """
    Calculate the nannualized standard deviation

    Args:
        returns (pd.Series, pd.Dataframe): Periodical returns time series data.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to DAILY. Also can be "weekly", "monthly", "quarterly" or "yearly".

    Returns:
        float, pd.Series: Annualized standard deviation for each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float. 
    """    
    # if len(returns) < 2:
    #     return np.full([1,returns.ndim], np.nan)
    
    annual_factor = _annualized_factor(period)
    if isinstance(returns, pd.DataFrame):
        returns_length = returns.count(axis = 0)
        result = np.nanstd(returns, axis = 0, ddof = 1) * np.sqrt(annual_factor)
    else:
        returns_length = returns.count()
        result = np.nanstd(returns, ddof = 1) * np.sqrt(annual_factor)
    
    if isinstance(returns, pd.DataFrame):
        result[returns_length < 2] = np.nan
    elif returns_length < 2:
        result = np.nan

    # if returns.ndim == 1 or len(returns.columns) == 1:
    #     return result.item()
    return result

def cumulative_returns(returns, logreturn = False):
    """
    caculate the cumulative returns at each times

    Args:
        returns (pd.Series, pd.DataFrame): Periodical returns time series data.
        logreturn (bool, optional): Define the type of returns.
         False for simple return, True for log return. Defaults to False.

    Returns:
        pd.Series, pd.DataFrame: Cumulative returns at each time.
         The type of output is the same as the input returns.
    """    
    returns.replace(np.nan, 0, inplace = True)
    if len(returns) < 1:
        return returns.copy()
    
    if logreturn:
        result = returns.cumsum(axis = 0)
    else:
        result = returns.copy() + 1
        result = result.cumprod(axis = 0) - 1

    return result

def drawdown(returns, logreturn = False):
    """
    caculate the drawdown of each time series at each time point

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        logreturn (bool, optional): Define the type of returns.
         False for simple return, True for log return. Defaults to False.
    Returns:
        pd.Series, pd.DataFrame: drawdown at each time.
         The type of output is the same as the input returns. 
    """   
    if len(returns) < 1:
        return np.full([1,returns.shape[1] if isinstance(returns, pd.DataFrame) else 1], np.nan)
    
    cumulativereturns = cumulative_returns(returns, logreturn) + 1
    maxcumulativereturns = cumulativereturns.cummax(axis = 0)
    result = (cumulativereturns - maxcumulativereturns) / maxcumulativereturns
    
    return result

def average_drawdown(returns, logreturn = False):
    """
    caculate the average drawdown of each time series at each time point

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        logreturn (bool, optional): Define the type of returns.
         False for simple return, True for log return. Defaults to False.

    Returns:
        pd.Series, pd.DataFrame: average drawdown at each time.
         The type of output is the same as the input returns. 
    """    
    if len(returns) < 1:
        return np.full([1,returns.shape[1] if isinstance(returns, pd.DataFrame) else 1], np.nan)

    drawdown_data = drawdown(returns, logreturn)
    if isinstance(drawdown_data, pd.DataFrame):
        result = pd.DataFrame(index = drawdown_data.index, columns = drawdown_data.columns)
        for asset in drawdown_data.columns:
            for i in range(len(drawdown_data)):
                result[asset][i] = np.nanmean(drawdown_data[asset][:i+1])
    else:
        result = pd.Series(index = drawdown_data.index)
        for j in range(len(drawdown_data)):
            result[j] = np.nanmean(drawdown_data[:j+1])
    return result
    
def max_drawdown(returns, logreturn = False):
    """
    caculate the maximum drawdown of each time series

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        logreturn (bool, optional): Define the type of returns.
         False for simple return, True for log return. Defaults to False.
    
    Returns:
        float, pd.Series: maximum drawdown of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float. 
    """
    if len(returns) < 1:
        return np.full([1,returns.shape[1] if isinstance(returns, pd.DataFrame) else 1], np.nan)

    result = drawdown(returns, logreturn)
    result = result.min(axis = 0)
    return result

def sharpe_ratio(returns, risk_free = 0., logreturn = False, period = DAILY):
    """
    caculate the sharpe ratio of each time series

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        risk_free (float, optional): Risk free interest rate. Defaults to 0.
        logreturn (bool, optional): Define the type of returns.
         False for simple return, True for log return. Defaults to False.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to "daily". Also can be "weekly", "monthly", "quarterly" or "yearly".

    Returns:
        float, pd.Series: sharpe ratio of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float. 
    """    
    # if len(returns) < 2:
    #     return np.full([1,returns.ndim], np.nan)
    # if isinstance(returns, pd.DataFrame):
    #     returns_length = returns.count(axis = 0)
    # else:
    #     returns_length = returns.count()

    annualized_return_data = annualized_return(returns, period, logreturn)
    adj_annualized_return = _adjust_returns(annualized_return_data, risk_free)
    result = np.divide(adj_annualized_return, annualized_sd(returns, period))

    if isinstance(result, pd.Series):
        result = result.replace([np.inf, -np.inf], np.nan)
    else:
        if result in [np.inf, -np.inf]:
            result = np.nan

    return result
 
def calmar_ratio(returns, period = DAILY, logreturn = False):
    """
    caculate the calmar ratio of each time series

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to "daily". Also can be "weekly", "monthly", "quarterly" or "yearly".
        logreturn (bool, optional): Define the type of returns.
         False for simple return, True for log return. Defaults to False.

    Returns:
        float, pd.Series: calmar ratio of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float. 
    """    
    maxdrawdown = max_drawdown(returns, logreturn)
    result = annualized_return(returns, period, logreturn) / abs(maxdrawdown)

    if isinstance(result, pd.Series):
        result = result.replace([np.inf, -np.inf], np.nan)
    else:
        if result in [np.inf, -np.inf]:
            result = np.nan

    return result

def burke_ratio(returns, risk_free = 0, period = DAILY, logreturn = False, modified = False):
    """
    caculate the burke ratio of each time series

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        risk_free (float, optional): Risk free interest rate. Defaults to 0.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to "daily". Also can be "weekly", "monthly", "quarterly" or "yearly".
        logreturn (bool, optional): Define the type of returns.
         False for simple return, True for log return. Defaults to False.
        modified (bool, optional): Decide to calculate Burke ratio or modified Burke ratio. Defaults to False.

    Returns:
        float, pd.Series: burke ratio of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float. 
    """ 
    # if isinstance(returns, pd.DataFrame):
    #     returns_length = returns.count(axis = 0)
    # else:
    #     returns_length = returns.count()
    
    if isinstance(returns, pd.DataFrame):
        returns_length = returns.count(axis = 0)
    else:
        returns_length = returns.count()
   
    rp = annualized_return(returns, period, logreturn)
    drawdown_data = drawdown(returns, logreturn)
    result = _adjust_returns(rp, risk_free) / np.sqrt(np.nansum(drawdown_data ** 2, axis = 0))
    if modified:
        result = result * np.sqrt(returns_length)

    if isinstance(result, pd.Series):
        result = result.replace([np.inf, -np.inf], np.nan)
    elif result in [np.inf, -np.inf]:
        result = np.nan

    return result

def downside_risk(returns, required_return = 0., period = DAILY):
    """
    Calculate the downside deviation below a threshold

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        required_return (float, optional): Minimum acceptable return. Defaults to 0.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to "daily". Also can be "weekly", "monthly", "quarterly" or "yearly".

    Returns:
        float, pd.Series: downside deviation of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float. 
    """    
    annual_factor = _annualized_factor(period)

    downside_diff = np.clip(
        _adjust_returns(
            np.asanyarray(returns),
            np.asanyarray(required_return),
        ),
        -np.inf,
        0,
    )

    downside_diff_square = np.square(downside_diff)
    result = np.sqrt(np.nanmean(downside_diff_square, axis = 0)) * np.sqrt(annual_factor)

    if isinstance(returns, pd.DataFrame):
        result = pd.Series(result, index = returns.columns)
    else:
        result = result.item()

    return result

def sortino_ratio(returns, required_return = 0, period = DAILY):
    """
    Calculate the sotino ratio of each time series

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        required_return (float, optional): Minimum acceptable return. Defaults to 0.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to "daily". Also can be "weekly", "monthly", "quarterly" or "yearly".

    Returns:
        float, pd.Series: sortino ratio of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """    
    adj_returns = np.asanyarray(_adjust_returns(returns, required_return))
    annual_factor = _annualized_factor(period)
    average_annual_return = np.nanmean(adj_returns, axis = 0) * annual_factor
    annualized_downside_risk = downside_risk(returns, required_return, period)
    result = np.divide(average_annual_return, annualized_downside_risk)

    if isinstance(returns, pd.DataFrame):
        result = pd.Series(result, index = returns.columns)
        result = result.replace([np.inf, -np.inf], np.nan)
    elif result.item() in [np.inf, -np.inf]:
        result = np.nan
    else:
        result = result.item()
    
    return result

def tracking_error(returns, bench_returns, period = DAILY):
    """
    Calculate Tracking Error of returns against a benchmark

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        bench_returns (pd.Series, pd.DataFrame): Returns of the benchmark.
         If input pd.DataFrame, let the benchmark data in the first columns.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to "daily". Also can be "weekly", "monthly", "quarterly" or "yearly".

    Returns:
        float, pd.Series: tracking error of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """

    annual_factor = _annualized_factor(period)

    if isinstance(bench_returns, pd.DataFrame):
        active_return = returns.sub(bench_returns.iloc[:,0], axis = 0)
    else:
        active_return = returns.sub(bench_returns, axis = 0)
    result = np.nanstd(active_return, ddof = 1, axis = 0) * np.sqrt(annual_factor)

    if isinstance(returns, pd.DataFrame):
        result = pd.Series(result, index = returns.columns)
    else:
        result = result.item()

    return result

def information_ratio(returns, bench_returns, period = DAILY):
    """
    Calculate the information ratio of each time series

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        bench_returns (pd.Series, pd.DataFrame): Returns of the benchmark.
         If input pd.DataFrame, let the benchmark data in the first columns.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to "daily". Also can be "weekly", "monthly", "quarterly" or "yearly".

    Returns:
        float, pd.Series: information ratio of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """    
    tracking_error_data = tracking_error(returns, bench_returns, period)
    active_premium = _adjust_returns(
            annualized_return(returns, period),
            annualized_return(bench_returns, period).item()
        )
    result = np.divide(active_premium, tracking_error_data)

    if isinstance(returns, pd.DataFrame):
        result = pd.Series(result, index = returns.columns)
    else:
        result = result.item()

    return result


def capm_beta(returns, bench_returns):
    """
    Calculate the CAPM beta

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        bench_returns (pd.Series, pd.DataFrame): Returns of the benchmark.
         If input pd.DataFrame, let the benchmark data in the first columns.

    Returns:
        float, pd.Series: beta of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """
    if isinstance(bench_returns, pd.Series):
        bench_returns = pd.DataFrame(bench_returns)
    else:
        bench_returns = pd.DataFrame(bench_returns.iloc[:,0])

    if isinstance(returns, pd.Series):
        returns = pd.DataFrame(returns)

    benchmark = np.where(np.isnan(returns), np.nan, bench_returns)
    benchmark_residual = benchmark - np.nanmean(benchmark, axis = 0)
    covariance = np.nanmean(benchmark_residual * returns, axis = 0)
    benchmark_var = np.nanmean(np.square(benchmark_residual), axis = 0)
    result = covariance / benchmark_var

    if isinstance(returns, pd.DataFrame):
        result = pd.Series(result, index = returns.columns)
    else:
        result = result.item()

    return result

def capm_alpha(returns, bench_returns, period = DAILY, risk_free = 0, beta = None): 
    """
    Calculate the CAPM alpha

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        bench_returns (pd.Series, pd.DataFrame): Returns of the benchmark.
         If input pd.DataFrame, let the benchmark data in the first columns.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to "daily". Also can be "weekly", "monthly", "quarterly" or "yearly".
        risk_free (float, optional): Risk free interest rate. Defaults to 0.
        beta (float, optional): The beta for the given inputs, if already known.
         Defaults to None and Will be calculated.

    Returns:
        float, pd.Series: alpha of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """     
    if isinstance(bench_returns, pd.Series):
        bench_returns = pd.DataFrame(bench_returns)
        # benchmark = pd.Series(np.where(np.isnan(returns), np.nan, bench_returns), index = bench_returns.index)
    else:
        bench_returns = pd.DataFrame(bench_returns.iloc[:,0])
        # benchmark = pd.DataFrame(np.where(np.isnan(returns), np.nan, bench_returns), index = bench_returns.index, columns = bench_returns.columns)
    
    benchmark = np.where(np.isnan(returns), np.nan, bench_returns)

    if isinstance(returns, pd.Series):
        returns = pd.DataFrame(returns)

    if beta is None:
        beta = capm_beta(returns, bench_returns)

    annual_factor = _annualized_factor(period)
    adj_returns = _adjust_returns(returns, risk_free)
    adj_bench_returns = _adjust_returns(benchmark, risk_free)
    alphas = adj_returns - (beta * adj_bench_returns)
    result = (np.nanmean(alphas, axis = 0) + 1) ** annual_factor - 1

    if isinstance(returns, pd.DataFrame):
        result = pd.Series(result, index = returns.columns)
    else:
        result = result.item()

    return result

def treynor_ratio(returns, bench_returns, period = DAILY, risk_free = 0):
    """
    Calculate the treynor ratio

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        bench_returns (pd.Series, pd.DataFrame): Returns of the benchmark.
         If input pd.DataFrame, let the benchmark data in the first columns.
        period (str, optional): Defines the periodicity of the 'returns' data.
         Defaults to "daily". Also can be "weekly", "monthly", "quarterly" or "yearly".
        risk_free (float, optional): Risk free interest rate. Defaults to 0.

    Returns:
        float, pd.Series: treynor ratio of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """    
    annual_returns = annualized_return(returns, period)
    adj_returns = _adjust_returns(annual_returns, risk_free)
    beta = capm_beta(returns, bench_returns)
    result = adj_returns / beta

    if isinstance(returns, pd.DataFrame):
        result = pd.Series(result, index = returns.columns)
    else:
        result = result.item()

    return result

def skewness(returns, method = "moment"):
    """
    Calculate skewness of each time series

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        method (str, optional): Specify the method of computation. 
         "moment", "fisher" or "sample". Defaults to "moment".
    
    Returns:
        float, pd.Series: skewness of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """    
    if isinstance(returns, pd.DataFrame):
        returns_length = returns.count(axis = 0)
    else:
        returns_length = returns.count() 
    
    if method == 'moment':
        result = (np.nansum((returns - np.nanmean(returns, axis = 0)) ** 3, axis = 0) / (np.nanstd(returns, axis = 0, ddof = 0) ** 3)) / returns_length
    elif method == 'fisher':
        result = ((np.sqrt(returns_length * (returns_length - 1))) / (returns_length - 2)) * ((np.nansum(returns ** 3, axis = 0)/returns_length) / ((np.nansum(returns ** 2, axis = 0)/returns_length)** 1.5))
    elif method == 'sample':
        result = (np.nansum((returns - np.nanmean(returns, axis = 0)) ** 3, axis = 0) / (np.nanstd(returns, axis = 0, ddof = 1) ** 3)) * (returns_length / ((returns_length - 1) * (returns_length - 2)))
    
    return result

def kurtosis(returns, method = "excess"):
    """
    Calculate kurtosis of each time series

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        method (str, optional): Specify the method of computation. 
         "excess", "moment", "fisher", "sample" or "sample_excess". Defaults to "excess".

    Returns:
        float, pd.Series: kurtosis of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """
    if isinstance(returns, pd.DataFrame):
        returns_length = returns.count(axis = 0)
    else:
        returns_length = returns.count() 

    if method == "excess":
        result = (np.nansum((returns - np.nanmean(returns, axis = 0)) ** 4, axis = 0) / (np.nanstd(returns, axis = 0, ddof = 0) ** 4)) / returns_length - 3
    elif method == "moment":
        result = (np.nansum((returns - np.nanmean(returns, axis = 0)) ** 4, axis = 0) / (np.nanstd(returns, axis = 0, ddof = 0) ** 4)) / returns_length
    elif method == "fisher":
        result = (returns_length + 1) * (returns_length - 1) * ((np.nansum(returns ** 4, axis = 0) / returns_length) / ((np.nansum(returns ** 2, axis = 0) / returns_length) ** 2) - 3 * (returns_length - 1) / (returns_length + 1)) / ((returns_length - 2) * (returns_length - 3))
    elif method == "sample":
        result = returns_length * (returns_length + 1) * (np.nansum((returns - np.nanmean(returns, axis = 0)) ** 4, axis = 0) / (np.nanstd(returns, axis = 0, ddof = 1) ** 4)) / ((returns_length -1) * (returns_length - 2) * (returns_length - 3))
    elif method == "sample_excess":
        result = returns_length * (returns_length + 1) * (np.nansum((returns - np.nanmean(returns, axis = 0)) ** 4, axis = 0) / (np.nanstd(returns, axis = 0, ddof = 1) ** 4)) / ((returns_length -1) * (returns_length - 2) * (returns_length - 3)) - 3 * ((returns_length - 1) ** 2) / ((returns_length - 2) * (returns_length - 3))
    
    return result

def value_at_risk(returns, significance_level = 0.05):
    """
    calculate the value at risk

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        significance_level (float, optional): significance level for calculation. Defaults to 0.05.

    Returns:
        float, pd.Series: VaR of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """    
    if isinstance(returns, pd.DataFrame):
        result = pd.Series(index = returns.columns, dtype = float)
        for asset in returns.columns:
            result[asset] = np.percentile(returns[asset].dropna(), 100 * significance_level)
    else:
        result = np.percentile(returns.dropna(), 100 * significance_level)
    
    return result

def conditional_value_at_risk(returns, significance_level = 0.05):
    """
    calculate the expected shortfall

    Args:
        returns (pd.Series, pd.DataFrame): Returns time series data.
        significance_level (float, optional): significance level for calculation. Defaults to 0.05.

    Returns:
        float, pd.Series: CVaR of each time series.
         If input returns is pd.Dataframe, you will get output in pd.Series;
         If input returns is pd.Series, you will get output in float.
    """    
    if isinstance(returns, pd.DataFrame):
        var = value_at_risk(returns, significance_level)
        result = pd.Series(index = returns.columns, dtype = float)
        for asset in returns.columns:
            result[asset] = returns.loc[returns[asset] < var[asset],asset].mean()
    else:
        var = value_at_risk(returns, significance_level)
        result = returns[returns < var].mean()
    
    return result