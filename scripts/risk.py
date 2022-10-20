from itertools import cycle

from numpy import (apply_along_axis, array, exp, isfinite, log, maximum,
                   nanmean, nanstd, quantile, sort, sqrt, square, unique,
                   vectorize)
from scipy.interpolate import CubicSpline
from scipy.special import ndtr
from scipy.stats import norm, t

from .capm import beta_capm
from .stat import cdf_func_kernel


def downside_dev(arr_ts: array, lst_idx_retn: list,
                 lst_idx_bcmk: list) -> array:
    return sqrt(
        square((arr_ts[:, lst_idx_retn] -
                arr_ts[:, lst_idx_bcmk]).clip(max=0)).mean(axis=0,
                                                           keepdims=True))


def downside_var(arr_ts: array, lst_idx_retn: list,
                 lst_idx_bcmk: list) -> array:
    return square((arr_ts[:, lst_idx_retn] -
                   arr_ts[:, lst_idx_bcmk]).clip(max=0)).mean(axis=0,
                                                              keepdims=True)


def downside_potential(arr_ts: array, lst_idx_retn: list,
                       lst_idx_bcmk: list) -> array:
    return (arr_ts[:, lst_idx_retn] -
            arr_ts[:, lst_idx_bcmk]).clip(max=0).mean(axis=0, keepdims=True)


def downside_freq(arr_ts: array, lst_idx_retn: list,
                  lst_idx_bcmk: list) -> array:
    arr = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    return ((arr < 0).sum(axis=0, keepdims=True) /
            isfinite(arr).sum(axis=0, keepdims=True))


def upside_dev(arr_ts: array, lst_idx_retn: list, lst_idx_bcmk: list) -> array:
    return sqrt(
        square((arr_ts[:, lst_idx_retn] -
                arr_ts[:, lst_idx_bcmk]).clip(min=0)).mean(axis=0,
                                                           keepdims=True))


def upside_var(arr_ts: array, lst_idx_retn: list, lst_idx_bcmk: list) -> array:
    return square((arr_ts[:, lst_idx_retn] -
                   arr_ts[:, lst_idx_bcmk]).clip(min=0)).mean(axis=0,
                                                              keepdims=True)


def upside_potential(arr_ts: array, lst_idx_retn: list,
                     lst_idx_bcmk: list) -> array:
    return (arr_ts[:, lst_idx_retn] -
            arr_ts[:, lst_idx_bcmk]).clip(min=0).mean(axis=0, keepdims=True)


def upside_freq(arr_ts: array, lst_idx_retn: list,
                lst_idx_bcmk: list) -> array:
    arr = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    return ((arr > 0).sum(axis=0, keepdims=True) /
            isfinite(arr).sum(axis=0, keepdims=True))


# TODO needs draw down
def drawdown_dev(arr_ts: array, lst_idx_retn: list,
                 lst_idx_bcmk: list) -> array:
    arr = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    return ((arr < 0).sum(axis=0, keepdims=True) /
            isfinite(arr).sum(axis=0, keepdims=True))


# TODO needs draw down
def drawdown_peak(arr_ts: array, lst_idx_retn: list,
                  lst_idx_bcmk: list) -> array:
    arr = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    return ((arr < 0).sum(axis=0, keepdims=True) /
            isfinite(arr).sum(axis=0, keepdims=True))


# TODO needs draw down
def drawdown(arr_ts: array, lst_idx_retn: list, lst_idx_bcmk: list) -> array:
    arr = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    return ((arr < 0).sum(axis=0, keepdims=True) /
            isfinite(arr).sum(axis=0, keepdims=True))


# TODO needs draw down
def drawdown_max(arr_ts: array, lst_idx_retn: list,
                 lst_idx_bcmk: list) -> array:
    arr = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    return ((arr < 0).sum(axis=0, keepdims=True) /
            isfinite(arr).sum(axis=0, keepdims=True))


def _fit_ppf_expect(vec_retn: array, method: str, alpha: float,
                    flag_ppf: bool) -> float:
    vec_clean = sort(vec_retn[isfinite(vec_retn)])
    if method == 'hist':
        VaR = quantile(a=vec_clean, q=alpha, method='median_unbiased')
        if flag_ppf:
            return VaR
        else:
            vec_clean = vec_clean[vec_clean <= VaR]
            return (vec_clean.sum() + VaR) / (len(vec_clean) + 1)
    elif method == 'kernel':
        band_width = nanstd(vec_clean) * 0.6973425390765554 * (
            len(vec_clean))**(-0.1111111111111111)
        # empirical cdf,  semi parametric approach, from mixture
        # * https://stats.stackexchange.com/questions/296285/find-cdf-from-an-estimated-pdf-estimated-by-kde
        @vectorize
        def ecdf(quantile: array):
            # ±∞ return to (0,1) cdf
            return ndtr((quantile - vec_clean) / band_width).mean()

        vec_F, idx = unique(ecdf(vec_clean), return_index=True)
        VaR = CubicSpline(x=vec_F, y=vec_clean[idx])(alpha)
        if flag_ppf:
            return VaR
        else:
            vec_clean = vec_clean[vec_clean <= VaR]
            return (vec_clean.sum() + VaR) / (len(vec_clean) + 1)
    elif method == 'norm':
        par = norm.fit(vec_clean)
        VaR = norm.ppf(q=alpha, loc=par[0], scale=par[1])
        if flag_ppf:
            return VaR
        else:
            # * https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
            return norm.expect(func=lambda x: x,
                               loc=par[0],
                               scale=par[1],
                               ub=VaR,
                               conditional=True)
    elif method == 't':
        par = t.fit(vec_clean)
        VaR = t.ppf(q=alpha, df=par[0], loc=par[1], scale=par[2])
        if flag_ppf:
            return VaR
        else:
            # * https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
            return t.expect(func=lambda x: x,
                            args=(par[0], ),
                            loc=par[1],
                            scale=par[2],
                            ub=VaR,
                            conditional=True)
    else:
        raise NotImplementedError


def value_at_risk(arr_ts: array, lst_idx_retn: list, alpha: float,
                  method: str) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    return apply_along_axis(func1d=_fit_ppf_expect,
                            axis=0,
                            arr=arr_retn,
                            alpha=alpha,
                            method=method,
                            flag_ppf=True).reshape(1, -1)


def expected_shortfall(arr_ts: array, lst_idx_retn: list, alpha: float,
                       method: str) -> array:
    # TODO CornishFisher
    arr_retn = arr_ts[:, lst_idx_retn]
    return apply_along_axis(func1d=_fit_ppf_expect,
                            axis=0,
                            arr=arr_retn,
                            alpha=alpha,
                            method=method,
                            flag_ppf=False).reshape(1, -1)


def beta_fama(arr_ts: array, lst_idx_retn: list, lst_idx_bcmk: list) -> array:
    return (nanstd(a=arr_ts[:, lst_idx_retn], axis=0, keepdims=True) /
            nanstd(a=arr_ts[:, lst_idx_bcmk], axis=0, keepdims=True))


def mean_absolute_dev(arr_ts: array, lst_idx_retn: list) -> array:
    return abs(arr_ts[:, lst_idx_retn] -
               arr_ts[:, lst_idx_retn].mean(axis=0, keepdims=True)).mean(
                   axis=0, keepdims=True)


def risk_systematic(arr_ts: array, lst_idx_retn: list, lst_idx_bcmk: list,
                    lst_idx_rf: list) -> array:

    return (nanstd(a=arr_ts[:, lst_idx_bcmk] - arr_ts[:, lst_idx_rf],
                   axis=0,
                   keepdims=True) * beta_capm(arr_ts=arr_ts,
                                              lst_idx_retn=lst_idx_retn,
                                              lst_idx_bcmk=lst_idx_bcmk,
                                              lst_idx_rf=lst_idx_rf,
                                              method='all'))


def tracking_error(arr_ts: array,
                   lst_idx_retn: list,
                   lst_idx_bcmk: list,
                   freq_reb: float = 1.0) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    te = sqrt(
        square(arr_retn - arr_ts[:, lst_idx_bcmk]).mean(axis=0, keepdims=True))
    te *= sqrt(freq_reb)
    return te


def _tail_dependence_lower(vec_retn: array, vec_bcmk: array,
                           alpha: float) -> float:
    idx = vec_bcmk <= quantile(a=vec_bcmk, q=alpha, method='median_unbiased')
    return ((vec_retn[idx] <= quantile(
        a=vec_retn, q=alpha, method='median_unbiased')).sum() / idx.sum())


def _tail_dependence_upper(vec_retn: array, vec_bcmk: array,
                           alpha: float) -> float:
    q = 1 - alpha
    idx = vec_bcmk >= quantile(a=vec_bcmk, q=q, method='median_unbiased')
    return ((vec_retn[idx] >= quantile(
        a=vec_retn, q=q, method='median_unbiased')).sum() / idx.sum())


def tail_dependence(arr_ts: array,
                    lst_idx_retn: list,
                    lst_idx_bcmk: list,
                    alpha: float,
                    flag_lower: bool = True) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    arr_bcmk = arr_ts[:, lst_idx_bcmk]
    tmp_len = len(lst_idx_retn)
    if (tmp_len > 1) and (len(lst_idx_bcmk) < 2):
        arr_bcmk = arr_bcmk.repeat(tmp_len, axis=1)
    if flag_lower:
        return array((*map(_tail_dependence_lower, arr_retn.T, arr_bcmk.T,
                           cycle([alpha])), ),
                     ndmin=2)
    else:
        return array((*map(_tail_dependence_upper, arr_retn.T, arr_bcmk.T,
                           cycle([alpha])), ),
                     ndmin=2)


def _tail_dependence_coefficient_FF(vec_i: array, vec_j: array):
    return 3.0 - 1.0 / (1.0 - nanmean(
        maximum(cdf_func_kernel(vec_i)(vec_i),
                cdf_func_kernel(vec_j)(vec_j)).clip(min=0.0, max=1.0)))


def _tail_dependence_coefficient_CFG(vec_i: array, vec_j: array):
    ecdf_i = cdf_func_kernel(vec_i)(vec_i)
    ecdf_j = cdf_func_kernel(vec_j)(vec_j)
    p_max = maximum(ecdf_i, ecdf_j).clip(min=0.0, max=1.0)
    return 2.0 - 2.0 * exp(
        nanmean(log(sqrt(log(ecdf_i) * log(ecdf_j)) / (-2.0 * log(p_max)))))


def tail_dependence_coefficient(arr_ts: array,
                                lst_idx_retn: list,
                                lst_idx_bcmk: list,
                                flag_lower: bool = True,
                                method: str = "FF") -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    arr_bcmk = arr_ts[:, lst_idx_bcmk]
    tmp_len = len(lst_idx_retn)
    if (tmp_len > 1) and (len(lst_idx_bcmk) < 2):
        arr_bcmk = arr_bcmk.repeat(tmp_len, axis=1)
    if flag_lower:
        arr_retn = -arr_retn
        arr_bcmk = -arr_bcmk
    if method == 'FF':
        return array(
            (*map(_tail_dependence_coefficient_FF, arr_retn.T, arr_bcmk.T), ),
            ndmin=2)
    elif method == 'CFG':
        return array(
            (*map(_tail_dependence_coefficient_CFG, arr_retn.T, arr_bcmk.T), ),
            ndmin=2)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    import logging

    from scripts import *
    logging.basicConfig(level=logging.DEBUG)
    arr = data_test.values
    N = data_test.shape[1] - 2
    #
    tmp = downside_dev(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='downside_dev')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = downside_var(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='downside_var')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = downside_potential(arr_ts=arr,
                             lst_idx_retn=range(N),
                             lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='downside_potential')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = downside_freq(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='downside_freq')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    for method in ['hist', 'kernel', 'norm', 't']:
        tmp = value_at_risk(arr_ts=arr,
                            lst_idx_retn=range(N),
                            alpha=.5,
                            method=method)
        assert isfinite(tmp).all()
        tmp = tmp.shape
        logging.log(level=logging.INFO, msg=f'value_at_risk: {method}')
        logging.log(level=logging.INFO, msg=tmp)
        assert tmp == (1, 14)
        tmp = expected_shortfall(arr_ts=arr,
                                 lst_idx_retn=range(N),
                                 alpha=.5,
                                 method=method)
        assert isfinite(tmp).all()
        tmp = tmp.shape
        logging.log(level=logging.INFO, msg=f'expected_shortfall: {method}')
        logging.log(level=logging.INFO, msg=tmp)
        assert tmp == (1, 14)
    #
    tmp = beta_fama(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='beta_fama')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = mean_absolute_dev(
        arr_ts=arr,
        lst_idx_retn=range(N),
    )
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='mean_absolute_dev')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = risk_systematic(arr_ts=arr,
                          lst_idx_retn=range(N),
                          lst_idx_bcmk=[N],
                          lst_idx_rf=[N + 1])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='risk_systematic')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = tracking_error(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='tracking_error')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    for flag_lower in (True, False):
        tmp = tail_dependence(arr_ts=arr,
                              lst_idx_retn=range(N),
                              lst_idx_bcmk=[N],
                              alpha=.1,
                              flag_lower=flag_lower)
        assert isfinite(tmp).all()
        tmp = tmp.shape
        logging.log(level=logging.INFO, msg=f'tail_dependence: {flag_lower}')
        logging.log(level=logging.INFO, msg=tmp)
        assert tmp == (1, 14)
        #
    for flag_lower in (True, False):
        for method in ['FF', 'CFG']:
            tmp = tail_dependence_coefficient(arr_ts=arr,
                                              lst_idx_retn=range(N),
                                              lst_idx_bcmk=[N],
                                              flag_lower=flag_lower,
                                              method=method)
            assert isfinite(tmp).all()
            tmp = tmp.shape
            logging.log(level=logging.INFO,
                        msg=f'tail_dependence: {flag_lower}: {method}')
            logging.log(level=logging.INFO, msg=tmp)
            assert tmp == (1, 14)
