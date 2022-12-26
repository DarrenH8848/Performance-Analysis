# TODO: hypothesis tests: stationarity, unit root, cointegration
from numpy import apply_along_axis, array, log, nanmean, sqrt, square, arange, empty, append, diff
from statsmodels.tsa.arima.model import ARIMA
from .risk import drawdown_peak
from numpy.lib.stride_tricks import as_strided
from scipy.stats import chi2
from statsmodels.tsa.stattools import adfuller, kpss, bds, coint, breakvar_heteroskedasticity_test
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas import Series


def kurtosis(arr_retn: array) -> array:
    # ~ corrected at 3 (excess)
    dev = square(arr_retn - nanmean(a=arr_retn, axis=0, keepdims=True))
    return (
        nanmean(a=square(dev), axis=0, keepdims=True) / 
        square(nanmean(a=dev, axis=0, keepdims=True)) 
        - 3
        )


def skewness(arr_retn: array) -> array:
    # ~ using unbiased estimator (Fisher)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
    dev = arr_retn - nanmean(a=arr_retn, axis=0, keepdims=True)
    dev2 = square(dev)
    N = len(arr_retn)
    N = sqrt(N * (N - 1)) / (N - 2)
    return (
        nanmean(a=dev * dev2, axis=0, keepdims=True) / 
        nanmean(a=dev2, axis=0, keepdims=True)**1.5 
        * N)


def index_hurst(arr_retn: array) -> array:
    return log(
        (
            arr_retn.max(axis=0, keepdims=True) - 
            arr_retn.min(axis=0, keepdims=True)
        ) /
        arr_retn.std(axis=0, keepdims=True)
        ) / log(len(arr_retn))
    

# TODO needs draw down
def index_pain(arr_retn: array) -> array:
    return abs(drawdown_peak(arr_retn)).mean(axis=0, keepdims=True)

# TODO
def index_smoothing(arr_retn: array, 
                    neg_theta: bool = False,
                    MAorder: int = 2) -> array:
    if arr_retn.ndim == 1:
        model = ARIMA(arr_retn, order=(0, 0, MAorder))
        res = model.fit_constrained({"const":0})
        coefma2 = res.polynomial_ma
        if neg_theta == False:
            for i in arange(1,len(coefma2)):
                coefma2[i] = max(0, coefma2[i])
        thetas = coefma2/coefma2.sum()
        return square(thetas).sum(keepdims=True)
    else:
        return apply_along_axis(func1d=index_smoothing, axis=0, arr=arr_retn)
    
# TODO needs draw down
def index_ulcer(arr_retn: array) -> array:
    return sqrt(square(drawdown_peak(arr_retn)).mean(axis=0, keepdims=True))
    

# TODO rsp
# ! https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
def min_TRL(arr_retn: array) -> array:
    pass

def rolling_window(array, win_len):

    orig_shape = array.shape
    win_num = orig_shape[0]-win_len+1
    
    new_shape = (win_num, win_len) + orig_shape[1:]
    new_strides = (array.strides[0],) + array.strides

    return as_strided(array, new_shape, new_strides)
    

def create_rolling_function(function):
    def rolling(arr, window, **kwargs):
        if len(arr):
            rolling_arr = rolling_window(arr,window)
            result = empty(0, dtype='float64')
            for i in rolling_window(arr,window):
                temp = function(i, **kwargs)
                result = append(result, temp)
            result = result.reshape([len(rolling_arr),1,-1])
        else:
            result = empty(0, dtype='float64')

        return result
    return rolling

def durbin_watson(arr_retn: array, 
                  axis: int) -> array:
    diff_retn = diff(arr_retn, 1, axis=axis)
    dw = sum(diff_retn**2, axis=axis, keepdims=True) / sum(arr_retn**2, axis=axis, keepdims=True)
    return dw

def jarque_bera(arr_retn: array,
                axis: int) -> tuple:
    skew = skewness(arr_retn)
    kurt = kurtosis(arr_retn)
    n = arr_retn.shape[axis]
    jb = (n / 6) * (skew ** 2 + 0.25 * kurt ** 2)
    jb_pv = chi2.sf(jb, 2)

    return jb, jb_pv, skew, kurt

def adf_test(arr_retn: array,
             res_show: bool = False) -> Series:
    
    dftest = adfuller(arr_retn, autolag="AIC")
    adf_result = Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        adf_result["Critical Value (%s)" % key] = value
        
    if res_show:
        print("Results of Dickey-Fuller Test:")
        print(adf_result)

    return adf_result

def kpss_test(arr_retn: array,
              res_show: bool = False) -> Series:

    kpsstest = kpss(arr_retn, regression="c", nlags="auto")
    kpss_result = Series(
        kpsstest[0:3], 
        index=[
            "Test Statistic", 
            "p-value", 
            "Lags Used"
        ]
    )
    for key, value in kpsstest[3].items():
        kpss_result["Critical Value (%s)" % key] = value

    if res_show:
        print("Results of KPSS Test:")
        print(kpss_result)
    
    return kpss_result

def bds_test(arr_retn: array,
             res_show: bool = False) -> Series:

    bdstest = bds(arr_retn)
    bds_result = Series(
        bdstest[0:2], 
        index=[
            "Test Statistic", 
            "p-value"
        ]
    )

    if res_show:
        print("Results of BDS Test:")
        print(bds_result)
    
    return bds_result

def heteroskedasticity_test(arr_retn: array,
                            res_show: bool = False) -> Series:

    hetertest = breakvar_heteroskedasticity_test(arr_retn)
    heter_result = Series(
        hetertest[0:2], 
        index=[
            "Test Statistic", 
            "p-value"
        ]
    )

    if res_show:
        print("Results of heteroskedasticity Test:")
        print(heter_result)
    
    return heter_result

def ljung_box(arr_retn: array,
              res_show: bool = False) -> Series:
    lbtest = acorr_ljungbox(arr_retn)
    lb_result = Series(
        lbtest[0:2], 
        index=[
            "Test Statistic", 
            "p-value"
        ]
    )

    if res_show:
        print("Results of ljung box Test:")
        print(lb_result)
    
    return lb_result

def coint_test(arr_retn: array,
               remaining_arr: array,
               res_show: bool = False) -> Series:
    cointtest = coint(arr_retn, remaining_arr)
    coint_result = Series(
        cointtest[0:2], 
        index=[
            "Test Statistic", 
            "p-value"
        ]
    )
    for key, value in cointtest[2].items():
        coint_result["Critical Value (%s)" % key] = value

    if res_show:
        print("Results of cointegration Test:")
        print(coint_result)
    
    return coint_result

if __name__ == '__main__':
    import logging

    from scripts import *
    logging.basicConfig(level=logging.DEBUG)
    arr = data_test.values
    # 
    tmp = kurtosis(arr).shape
    logging.log(level=logging.INFO, msg='kurtosis')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,16)
    # 
    tmp = skewness(arr).shape
    logging.log(level=logging.INFO, msg='skewness')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,16)
    # 
    tmp = index_hurst(arr).shape
    logging.log(level=logging.INFO, msg='index_hurst')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,16)
    # 
    tmp = index_pain(arr).shape
    logging.log(level=logging.INFO, msg='index_pain')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,16)
    #
    tmp = index_smoothing(arr,
                          neg_theta=False,
                          MAorder=2).shape
    logging.log(level=logging.INFO, msg='index_smoothing')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,16)
    #
    tmp = index_ulcer(arr).shape
    logging.log(level=logging.INFO, msg='index_ulcer')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,16)