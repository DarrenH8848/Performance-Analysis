# TODO: hypothesis tests: stationarity, unit root, cointegration

from itertools import cycle

from numpy import (array, ceil, exp, geomspace, int0, isfinite, log, nanmean,
                   nanstd, random, sqrt, square, unique, vectorize)
from scipy.special import ndtr


def cdf_func_kernel(vec: array):
    # * spline inside ppf needs sort; cdf no need
    vec_clean = vec[isfinite(vec)]
    band_width = nanstd(vec_clean) * 0.6973425390765554 * (len(vec_clean))**(
        -0.1111111111111111)

    @vectorize
    def cdf_k(q: array) -> array:
        return ndtr((q - vec_clean) / band_width).mean()

    return cdf_k


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


def _index_tail_upper(vec: array, len_sample_size: int, num_sim: int) -> float:
    vec_clean = vec[vec > 0]
    vec_clean = vec_clean[isfinite(vec_clean)]
    vec_sample_size = len(vec_clean) * 0.6321205588285577
    vec_sample_size = unique(
        int0(
            ceil(geomspace(start=1, stop=vec_sample_size,
                           num=len_sample_size))))

    @vectorize
    def _MPMR(sample_size: int) -> float:

        @vectorize
        def _subsample_blockmaxima_log(seed: int) -> float:
            random.seed(seed=seed)
            return max(
                random.choice(a=vec_clean, size=sample_size, replace=False))

        vec = log(_subsample_blockmaxima_log(range(num_sim)))
        band_width = nanstd(vec) * 0.6973425390765554 * (vec.size)**(
            -0.1111111111111111)
        mode_0 = 1.0
        for _ in range(1000):
            mode_1 = mode_0
            tmp = exp(-square(vec - mode_1) / band_width).clip(min=1e-16)
            mode_0 = (vec @ tmp) / tmp.sum()
            if abs(mode_0 - mode_1) < 1e-8:
                break
        return mode_0

    ydev = _MPMR(vec_sample_size)
    ydev -= ydev.mean()
    xdev = log(vec_sample_size)
    xdev -= xdev.mean()
    return square(xdev).sum() / (xdev @ ydev)


def index_tail(arr_retn: array,
               flag_lower: bool = True,
               len_sample_size: int = 11,
               num_sim: int = 600) -> array:
    if flag_lower:
        arr_retn = -arr_retn
    arr_retn = arr_retn.clip(min=0)
    return array((*map(_index_tail_upper, arr_retn.T, cycle([len_sample_size]),
                       cycle([num_sim])), ),
                 ndmin=2)


# TODO rsp
# ! https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
def min_TRL(arr_retn: array) -> array:
    pass


if __name__ == '__main__':
    import logging

    from numpy import isfinite

    from scripts import *
    logging.basicConfig(level=logging.DEBUG)
    arr = data_test.values
    #
    tmp = cdf_func_kernel(arr[:, [0]])(arr[:, [0]])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='get_ecdf_func')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1542, 1)
    #
    tmp = kurtosis(arr)
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='kurtosis')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 16)
    #
    tmp = skewness(arr)
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='skewness')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 16)
    #
    tmp = index_hurst(arr)
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='index_hurst')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 16)
    #
    for flag_lower in [True, False]:
        tmp = index_tail(arr, flag_lower=flag_lower)
        assert isfinite(tmp).all()
        tmp = tmp.shape
        logging.log(level=logging.INFO, msg=f'index_tail: {flag_lower}')
        logging.log(level=logging.INFO, msg=tmp)
        assert tmp == (1, 16)
