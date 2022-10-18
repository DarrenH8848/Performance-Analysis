# TODO: hypothesis tests: stationarity, unit root, cointegration
from numpy import apply_along_axis, array, log, nanmean, sqrt, square


def kurtosis(arr_retn: array) -> array:
    # ~ corrected at 3 (excess)
    dev = square(arr_retn - nanmean(a=arr_retn, axis=0, keepdims=True))
    return (nanmean(a=square(dev), axis=0, keepdims=True) /
            square(nanmean(a=dev, axis=0, keepdims=True)) - 3)


def skewness(arr_retn: array) -> array:
    # ~ using unbiased estimator (Fisher)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
    dev = arr_retn - nanmean(a=arr_retn, axis=0, keepdims=True)
    dev2 = square(dev)
    N = len(arr_retn)
    N = sqrt(N * (N - 1)) / (N - 2)
    return (nanmean(a=dev * dev2, axis=0, keepdims=True) /
            nanmean(a=dev2, axis=0, keepdims=True)**1.5 * N)


def index_hurst(arr_retn: array) -> array:
    return log((arr_retn.max(axis=0, keepdims=True) -
                arr_retn.min(axis=0, keepdims=True)) /
               arr_retn.std(axis=0, keepdims=True)) / log(len(arr_retn))


# TODO needs draw down
def index_pain(arr_retn: array) -> array:
    return log((arr_retn.max(axis=0, keepdims=True) -
                arr_retn.min(axis=0, keepdims=True)) /
               arr_retn.std(axis=0, keepdims=True)) / log(len(arr_retn))


# TODO index_smoothing
def index_smoothing(arr_retn: array) -> array:
    return log((arr_retn.max(axis=0, keepdims=True) -
                arr_retn.min(axis=0, keepdims=True)) /
               arr_retn.std(axis=0, keepdims=True)) / log(len(arr_retn))


# TODO needs draw down
def index_ulcer(arr_retn: array) -> array:
    return log((arr_retn.max(axis=0, keepdims=True) -
                arr_retn.min(axis=0, keepdims=True)) /
               arr_retn.std(axis=0, keepdims=True)) / log(len(arr_retn))


# TODO needs draw down
def index_pain(arr_retn: array) -> array:
    return log((arr_retn.max(axis=0, keepdims=True) -
                arr_retn.min(axis=0, keepdims=True)) /
               arr_retn.std(axis=0, keepdims=True)) / log(len(arr_retn))


# TODO rsp
# ! https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
def min_TRL(arr_retn: array) -> array:
    pass


if __name__ == '__main__':
    import logging

    from scripts import *
    logging.basicConfig(level=logging.DEBUG)
    arr = data_test.values
    #
    tmp = kurtosis(arr).shape
    logging.log(level=logging.INFO, msg='kurtosis')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 16)
    #
    tmp = skewness(arr).shape
    logging.log(level=logging.INFO, msg='skewness')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 16)
    #
    tmp = index_hurst(arr).shape
    logging.log(level=logging.INFO, msg='index_hurst')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 16)
