# TODO: check frequency of rebalancing & scaling of those ratio
# TODO try to modify ratios into trade-offs between return & risk
# TODO try to modify ratios such that the bigger the better

from numpy import apply_along_axis, array, nanmean, sqrt, square

from .capm import beta_capm
from .risk import downside_dev, expected_shortfall
from .stat import kurtosis, skewness


def ratio_sharpe(arr_ts: array,
                 lst_idx_retn: list,
                 lst_idx_bcmk: list,
                 ddof: int = 1,
                 freq_reb: float = 1.0,
                 flag_adjusted: bool = False) -> array:
    # https://en.wikipedia.org/wiki/Sharpe_ratio
    arr_retn = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    r_sharpe = (arr_retn.mean(axis=0, keepdims=True) /
                arr_retn.std(axis=0, keepdims=True, ddof=ddof))
    if flag_adjusted:
        # ~ using the 3-corrected Kurtosis
        r_sharpe *= (1 + skewness(arr_retn=arr_retn) / 6 * r_sharpe -
                     kurtosis(arr_retn=arr_retn) / 24 * square(r_sharpe))
    r_sharpe *= sqrt(freq_reb)
    return r_sharpe


# TODO rsp
# ! https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
def ratio_sharpe_prob(arr_ts: array,
                      lst_idx_retn: list,
                      lst_idx_bcmk: list,
                      ddof: int = 1,
                      freq_reb: float = 1.0,
                      flag_adjusted: bool = False) -> array:
    # here the bcmk is risk free rate
    arr_retn = arr_ts[:, lst_idx_retn]
    r_sharpe = (
        (arr_retn - arr_ts[:, lst_idx_bcmk]).mean(axis=0, keepdims=True) /
        (arr_retn).std(axis=0, keepdims=True, ddof=ddof))
    if flag_adjusted:
        # ~ using the 3-corrected Kurtosis
        r_sharpe *= (1 + skewness(arr_retn=arr_retn) / 6 * r_sharpe -
                     kurtosis(arr_retn=arr_retn) / 24 * square(r_sharpe))
    r_sharpe *= sqrt(freq_reb)
    return r_sharpe


def ratio_bernadoledoit(arr_ts: array, lst_idx_retn: list,
                        lst_idx_bcmk: list) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    arr_bcmk = arr_ts[:, lst_idx_bcmk]
    return -(arr_retn.clip(min=arr_bcmk).sum(axis=0, keepdims=True) /
             arr_retn.clip(max=arr_bcmk).sum(axis=0, keepdims=True))


def ratio_burke(arr_ts: array,
                lst_idx_retn: list,
                lst_idx_bcmk: list,
                freq_reb: float = 1.0) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    arr_bcmk = arr_ts[:, lst_idx_bcmk]
    # * drawdown only
    dd = sqrt(square(arr_retn.clip(max=0)).mean(axis=0, keepdims=True))
    r_burke = (arr_retn.mean(axis=0, keepdims=True) -
               arr_bcmk.mean(axis=0, keepdims=True)) / dd
    r_burke *= sqrt(freq_reb)
    return r_burke


# TODO needs draw down
# ! rolling window based
def ratio_calmar(
    arr_ts: array,
    lst_idx_retn: list,
    lst_idx_bcmk: list,
    # freq_reb: float = 1.0
) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    arr_bcmk = arr_ts[:, lst_idx_bcmk]
    # TODO * maxdrawdown only
    mdd = sqrt(square(arr_retn.clip(max=0)).mean(axis=0, keepdims=True))
    r_calmar = (arr_retn.sum(axis=0, keepdims=True) -
                arr_bcmk.sum(axis=0, keepdims=True)) / mdd
    # r_calmar *= sqrt(freq_reb)
    return r_calmar


# TODO needs draw down
# ! rolling window based
def ratio_sterling(
    arr_ts: array,
    lst_idx_retn: list,
    lst_idx_bcmk: list,
    # freq_reb: float = 1.0
) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    arr_bcmk = arr_ts[:, lst_idx_bcmk]
    # TODO * maxdrawdown only
    mdd = sqrt(square(arr_retn.clip(max=0)).mean(axis=0, keepdims=True))
    r_sterling = (arr_retn.sum(axis=0, keepdims=True) -
                  arr_bcmk.sum(axis=0, keepdims=True)) / mdd
    # r_calmar *= sqrt(freq_reb)
    return r_sterling


def ratio_d(
    arr_ts: array,
    lst_idx_retn: list,
) -> array:
    # ~ modified as larger the better
    arr_retn = arr_ts[:, lst_idx_retn]
    #
    arr = arr_retn.clip(min=0)
    r_d = (arr > 0).sum(axis=0, keepdims=True) * arr.sum(axis=0, keepdims=True)
    arr = arr_retn.clip(max=0)
    r_d /= -(
        (arr < 0).sum(axis=0, keepdims=True) * arr.sum(axis=0, keepdims=True))
    return r_d


def ratio_information(arr_ts: array,
                      lst_idx_retn: list,
                      lst_idx_bcmk: list,
                      freq_reb: float = 1.0) -> array:
    # ~ modified as larger the better
    arr_retn = arr_ts[:, lst_idx_retn]
    arr_bcmk = arr_ts[:, lst_idx_bcmk]
    #
    arr = arr_retn - arr_bcmk
    r_info = arr.mean(axis=0, keepdims=True) / arr.std(axis=0, keepdims=True)
    r_info *= sqrt(freq_reb)
    return r_info


def ratio_kappa(arr_ts: array, lst_idx_retn: list, lst_idx_bcmk: list,
                coeff: float) -> array:
    # Kaplan & Knowles
    arr = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]

    return (nanmean(a=arr, axis=0, keepdims=True) / nanmean(
        a=(-arr).clip(min=0)**coeff, axis=0, keepdims=True)**(1 / coeff))


def ratio_kelly(
    arr_ts: array,
    lst_idx_retn: list,
    lst_idx_bcmk: list,
    # freq_reb: float = 1.0
) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    #
    return ((arr_retn - arr_ts[:, lst_idx_bcmk]).mean(axis=0, keepdims=True) /
            arr_retn.var(axis=0, keepdims=True))


# TODO needs draw down
def ratio_martin(
    arr_ts: array,
    lst_idx_retn: list,
    lst_idx_bcmk: list,
    # freq_reb: float = 1.0
) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    #
    return ((arr_retn - arr_ts[:, lst_idx_bcmk]).mean(axis=0, keepdims=True) /
            arr_retn.var(axis=0, keepdims=True))


def ratio_omega(
    arr_ts: array,
    lst_idx_retn: list,
    lst_idx_bcmk: list,
) -> array:
    arr = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    return -(apply_along_axis(
        func1d=lambda x: x[x > 0].mean(), axis=0, arr=arr) / apply_along_axis(
            func1d=lambda x: x[x < 0].mean(), axis=0, arr=arr)).reshape(1, -1)


# TODO needs draw down
def ratio_pain(
    arr_ts: array,
    lst_idx_retn: list,
    lst_idx_bcmk: list,
    # freq_reb: float = 1.0
) -> array:
    # ~ modified as larger the better
    arr_retn = arr_ts[:, lst_idx_retn]
    #
    return ((arr_retn - arr_ts[:, lst_idx_bcmk]).mean(axis=0, keepdims=True) /
            arr_retn.var(axis=0, keepdims=True))


def ratio_prospect(arr_ts: array,
                   lst_idx_retn: list,
                   lst_idx_bcmk: list,
                   freq_reb: float = 1.0) -> array:
    arr_retn = arr_ts[:, lst_idx_retn]
    r_prospect = ((arr_retn.clip(min=0) + 2.25 * arr_retn.clip(max=0) -
                   arr_ts[:, lst_idx_bcmk]).mean(axis=0, keepdims=True) /
                  downside_dev(arr_ts=arr_ts,
                               lst_idx_retn=lst_idx_retn,
                               lst_idx_bcmk=lst_idx_bcmk))
    # TODO scale?
    r_prospect *= sqrt(freq_reb)
    return r_prospect


def ratio_rachev(arr_ts: array, lst_idx_retn: list, lst_idx_bcmk: list,
                 alpha: float, beta: float, method: str) -> array:
    # ~ modified as larger the better
    arr_retn = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    return (expected_shortfall(arr_ts=-arr_retn,
                               lst_idx_retn=range(arr_retn.shape[1]),
                               alpha=alpha,
                               method=method) /
            expected_shortfall(arr_ts=arr_retn,
                               lst_idx_retn=range(arr_retn.shape[1]),
                               alpha=beta,
                               method=method))


def ratio_sortino(arr_ts: array,
                  lst_idx_retn: list,
                  lst_idx_bcmk: list,
                  freq_reb: float = 1.0) -> array:
    r_sortino = ((arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]).mean(
        axis=0, keepdims=True) / downside_dev(arr_ts=arr_ts,
                                              lst_idx_retn=lst_idx_retn,
                                              lst_idx_bcmk=lst_idx_bcmk))
    # TODO scale?
    r_sortino *= sqrt(freq_reb)
    return r_sortino


def ratio_treynor(arr_ts: array,
                  lst_idx_retn: list,
                  lst_idx_bcmk: list,
                  lst_idx_rf: list,
                  freq_reb: float = 1.0) -> array:
    r_treynor = ((arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_rf]).mean(
        axis=0, keepdims=True) / beta_capm(arr_ts=arr_ts,
                                           lst_idx_retn=lst_idx_retn,
                                           lst_idx_bcmk=lst_idx_bcmk,
                                           lst_idx_rf=lst_idx_rf,
                                           method='all'))
    # TODO scale?
    r_treynor *= freq_reb
    return r_treynor


def _ratio_updown_capture(vec_retn: array, vec_bcmk: array) -> float:
    idx = vec_bcmk > 0
    r_ud = vec_retn[idx].clip(min=0).sum() / vec_bcmk[idx].sum()
    idx = vec_bcmk < 0
    r_ud /= vec_retn[idx].clip(max=0).sum() / vec_bcmk[idx].sum()
    return r_ud


def _ratio_updown_number(vec_retn: array, vec_bcmk: array) -> float:
    idx = vec_bcmk > 0
    r_ud = (vec_retn[idx] > 0).sum() / idx.sum()
    idx = vec_bcmk < 0
    r_ud /= (vec_retn[idx] < 0).sum() / idx.sum()
    return r_ud


def _ratio_updown_percent(vec_retn: array, vec_bcmk: array) -> float:
    idx = vec_bcmk > 0
    r_ud = (vec_retn[idx] > vec_bcmk[idx]).sum() / idx.sum()
    idx = vec_bcmk < 0
    r_ud /= (vec_retn[idx] < vec_bcmk[idx]).sum() / idx.sum()
    return r_ud


def ratio_updown(
    arr_ts: array,
    lst_idx_retn: list,
    lst_idx_bcmk: list,
    method: str,
    # freq_reb: float = 1.0
) -> array:
    # ~ modified to be a trade-off between return ~ risk
    arr_retn = arr_ts[:, lst_idx_retn]
    arr_bcmk = arr_ts[:, lst_idx_bcmk]
    tmp_len = len(lst_idx_retn)
    if (tmp_len > 1) and (len(lst_idx_bcmk) < 2):
        arr_bcmk = arr_bcmk.repeat(tmp_len, axis=1)
    if method == 'capture':
        return array((*map(_ratio_updown_capture, arr_retn.T, arr_bcmk.T), ),
                     ndmin=2)
    elif method == 'number':
        return array((*map(_ratio_updown_number, arr_retn.T, arr_bcmk.T), ),
                     ndmin=2)
    elif method == 'percent':
        return array((*map(_ratio_updown_percent, arr_retn.T, arr_bcmk.T), ),
                     ndmin=2)
    else:
        raise NotImplementedError


def ratio_upsidepotential(arr_ts: array,
                          lst_idx_retn: list,
                          lst_idx_bcmk: list,
                          freq_reb: float = 1.0) -> array:
    arr = arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]
    r_upp = (arr.clip(min=0).mean(axis=0, keepdims=True) /
             sqrt(square(arr.clip(max=0)).mean(axis=0, keepdims=True)))
    r_upp *= sqrt(freq_reb)
    return r_upp


if __name__ == '__main__':
    import logging

    from numpy import isfinite

    from scripts import *
    logging.basicConfig(level=logging.DEBUG)
    arr = data_test.values
    N = data_test.shape[1] - 2
    #
    tmp = ratio_sharpe(arr_ts=arr,
                       lst_idx_retn=range(N),
                       lst_idx_bcmk=[N],
                       freq_reb=365)
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_sharpe')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = ratio_bernadoledoit(arr_ts=arr,
                              lst_idx_retn=range(N),
                              lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_bernadoledoit')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = ratio_burke(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_burke')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = ratio_d(arr_ts=arr, lst_idx_retn=range(N))
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_d')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = ratio_information(arr_ts=arr,
                            lst_idx_retn=range(N),
                            lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_information')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = ratio_kappa(arr_ts=arr,
                      lst_idx_retn=range(N),
                      lst_idx_bcmk=[N],
                      coeff=1.5)
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_kappa')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = ratio_kelly(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_kelly')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = ratio_omega(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_omega')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = ratio_prospect(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_prospect')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    for method in ['hist', 'kernel', 'norm', 't']:
        tmp = ratio_rachev(arr_ts=arr,
                           lst_idx_retn=range(N),
                           lst_idx_bcmk=[N],
                           alpha=.1,
                           beta=.1,
                           method=method)
        assert isfinite(tmp).all()
        tmp = tmp.shape
        logging.log(level=logging.INFO, msg=f'ratio_rachev: {method}')
        logging.log(level=logging.INFO, msg=tmp)
        assert tmp == (1, 14)
    #
    tmp = ratio_sortino(arr_ts=arr, lst_idx_retn=range(N), lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_sortino')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    tmp = ratio_treynor(
        arr_ts=arr,
        lst_idx_retn=range(N),
        lst_idx_bcmk=[N],
        lst_idx_rf=[N + 1],
    )
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_treynor')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    #
    for method in ['capture', 'number', 'percent']:
        tmp = ratio_updown(arr_ts=arr,
                           lst_idx_retn=range(N),
                           lst_idx_bcmk=[N],
                           method=method)
        assert isfinite(tmp).all()
        tmp = tmp.shape
        logging.log(level=logging.INFO, msg=f'ratio_updown: {method}')
        logging.log(level=logging.INFO, msg=tmp)
        assert tmp == (1, 14)
    #
    tmp = ratio_upsidepotential(arr_ts=arr,
                                lst_idx_retn=range(N),
                                lst_idx_bcmk=[N])
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='ratio_upsidepotential')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
