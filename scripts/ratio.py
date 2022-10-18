# TODO: check frequency of rebalancing & scaling of those ratio

from numpy import (apply_along_axis, array, nanmean, nanstd, nansum, sqrt,
                   square)

from .capm import beta_capm
from .risk import downside_dev, expected_shortfall
from .stat import kurtosis, skewness


def ratio_sharpe(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list, 
    ddof: int = 1, 
    freq_reb: float = 1.0, 
    flag_adjusted: bool = False
    ) -> array:
    # https://en.wikipedia.org/wiki/Sharpe_ratio
    arr_retn = arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_bcmk]
    r_sharpe = (
        arr_retn.mean(axis=0, keepdims=True) / 
        arr_retn.std(axis=0, keepdims=True, ddof=ddof)
        )
    if flag_adjusted:
        # ~ using the 3-corrected Kurtosis
        r_sharpe *= (
            1 +
            skewness(arr_retn=arr_retn) / 6 * r_sharpe - 
            kurtosis(arr_retn=arr_retn) / 24 * square(r_sharpe)
            ) 
    r_sharpe *= sqrt(freq_reb)
    return r_sharpe

# TODO rsp
# ! https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
def ratio_sharpe_prob(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list, 
    ddof: int = 1, 
    freq_reb: float = 1.0, 
    flag_adjusted: bool = False
    ) -> array:
    # here the bcmk is risk free rate
    arr_retn = arr_ts[:,lst_idx_retn]
    r_sharpe = (
        (arr_retn - arr_ts[:,lst_idx_bcmk]).mean(axis=0, keepdims=True) / 
        (arr_retn).std(axis=0, keepdims=True, ddof=ddof)
        )
    if flag_adjusted:
        # ~ using the 3-corrected Kurtosis
        r_sharpe *= (
            1 +
            skewness(arr_retn=arr_retn) / 6 * r_sharpe - 
            kurtosis(arr_retn=arr_retn) / 24 * square(r_sharpe)
            ) 
    r_sharpe *= sqrt(freq_reb)
    return r_sharpe


def ratio_bernadoledoit(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list
    ) -> array:
    arr_retn = arr_ts[:,lst_idx_retn]
    arr_bcmk = arr_ts[:,lst_idx_bcmk]
    return -(
        arr_retn.clip(min=arr_bcmk).sum(axis=0, keepdims=True) /
        arr_retn.clip(max=arr_bcmk).sum(axis=0, keepdims=True)
        )
    
    
def ratio_burke(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    freq_reb: float = 1.0
    ) -> array:
    arr_retn = arr_ts[:,lst_idx_retn]
    arr_bcmk = arr_ts[:,lst_idx_bcmk]
    # * drawdown only
    dd = sqrt(square(arr_retn.clip(max=0)).mean(axis=0, keepdims=True))
    r_burke = (
        arr_retn.mean(axis=0, keepdims=True) - 
        arr_bcmk.mean(axis=0, keepdims=True)
        )/dd
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
    arr_retn = arr_ts[:,lst_idx_retn]
    arr_bcmk = arr_ts[:,lst_idx_bcmk]
    # TODO * maxdrawdown only
    mdd = sqrt(square(arr_retn.clip(max=0)).mean(axis=0, keepdims=True))
    r_calmar = (
        arr_retn.sum(axis=0, keepdims=True) - 
        arr_bcmk.sum(axis=0, keepdims=True)
        )/mdd
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
    arr_retn = arr_ts[:,lst_idx_retn]
    arr_bcmk = arr_ts[:,lst_idx_bcmk]
    # TODO * maxdrawdown only
    mdd = sqrt(square(arr_retn.clip(max=0)).mean(axis=0, keepdims=True))
    r_sterling = (
        arr_retn.sum(axis=0, keepdims=True) - 
        arr_bcmk.sum(axis=0, keepdims=True)
        )/mdd
    # r_calmar *= sqrt(freq_reb)
    return r_sterling 


def ratio_d(
    arr_ts: array, 
    lst_idx_retn: list, 
    ) -> array:
    # ~ modified as larger the better
    arr_retn = arr_ts[:,lst_idx_retn]
    # 
    arr = arr_retn.clip(min=0)
    r_d = (arr>0).sum(axis=0, keepdims=True) * arr.sum(axis=0, keepdims=True)
    arr = arr_retn.clip(max=0)
    r_d /= -(
        (arr<0).sum(axis=0, keepdims=True) * arr.sum(axis=0, keepdims=True)
        )
    return r_d


def ratio_information(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    freq_reb: float = 1.0
    ) -> array:
    # ~ modified as larger the better
    arr_retn = arr_ts[:,lst_idx_retn]
    arr_bcmk = arr_ts[:,lst_idx_bcmk]
    # 
    arr = arr_retn - arr_bcmk
    r_info = arr.mean(axis=0, keepdims=True) / arr.std(axis=0, keepdims=True)
    r_info *= sqrt(freq_reb)
    return r_info


def ratio_kappa(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    coeff: float
    ) -> array:
    # Kaplan & Knowles
    arr = arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_bcmk]
    
    return (
        nanmean(
            a=arr, 
            axis=0, 
            keepdims=True) /
        nanmean(
            a=(-arr).clip(min=0) ** coeff, 
            axis=0, 
            keepdims=True) ** (1/coeff)
    )
    

def ratio_kelly(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    # freq_reb: float = 1.0
    ) -> array:
    arr_retn = arr_ts[:,lst_idx_retn]
    # 
    return (
        (
            arr_retn - arr_ts[:,lst_idx_bcmk]
        ).mean(axis=0, keepdims=True) /
        arr_retn.var(axis=0, keepdims=True)
        )
    
    
# TODO needs draw down
def ratio_martin(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    # freq_reb: float = 1.0
    ) -> array:
    arr_retn = arr_ts[:,lst_idx_retn]
    # 
    return (
        (
            arr_retn - arr_ts[:,lst_idx_bcmk]
        ).mean(axis=0, keepdims=True) /
        arr_retn.var(axis=0, keepdims=True)
        )


def ratio_omega(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    ) -> array:
    arr = arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_bcmk]
    return -(
        apply_along_axis(func1d= lambda x: x[x>0].mean(), axis=0, arr=arr) /
        apply_along_axis(func1d= lambda x: x[x<0].mean(), axis=0, arr=arr)
    ).reshape(1,-1)
        
        
# TODO needs draw down
def ratio_pain(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    # freq_reb: float = 1.0
    ) -> array:
    # ~ modified as larger the better
    arr_retn = arr_ts[:,lst_idx_retn]
    # 
    return (
        (
            arr_retn - arr_ts[:,lst_idx_bcmk]
        ).mean(axis=0, keepdims=True) /
        arr_retn.var(axis=0, keepdims=True)
        )
    

def ratio_prospect(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    freq_reb: float = 1.0
    ) -> array:
    arr_retn = arr_ts[:,lst_idx_retn]
    r_prospect = (
        (
            arr_retn.clip(min=0) + 
            2.25 * arr_retn.clip(max=0) - 
            arr_ts[:,lst_idx_bcmk]
        ).mean(axis=0, keepdims=True) /
        downside_dev(
            arr_ts=arr_ts, 
            lst_idx_retn=lst_idx_retn,
            lst_idx_bcmk=lst_idx_bcmk
        )
    )
    # TODO scale?
    r_prospect *= sqrt(freq_reb)
    return r_prospect
    
def ratio_rachev(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    alpha: float,
    beta: float,
    method: str
    ) -> array:
    # ~ modified as larger the better
    arr_retn = arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_bcmk]
    return (
        expected_shortfall(
            arr_ts=-arr_retn, 
            lst_idx_retn=range(arr_retn.shape[1]), 
            alpha=alpha, 
            method=method
            ) / 
        expected_shortfall(
            arr_ts=arr_retn, 
            lst_idx_retn=range(arr_retn.shape[1]), 
            alpha=beta, 
            method=method
            ) 
        )
    
    
def ratio_sortino(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    freq_reb: float = 1.0
    ) -> array:
    r_sortino = (
        (
            arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_bcmk]
        ).mean(axis=0, keepdims=True) /
        downside_dev(
            arr_ts=arr_ts, 
            lst_idx_retn=lst_idx_retn,
            lst_idx_bcmk=lst_idx_bcmk
        )
    )
    # TODO scale?
    r_sortino *= sqrt(freq_reb)
    return r_sortino


def ratio_treynor(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    lst_idx_rf: list,
    freq_reb: float = 1.0
    ) -> array:
    r_treynor = (
        (arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_rf]) /
        beta_capm(
        arr_ts=arr_ts, 
        lst_idx_retn=lst_idx_retn, 
        lst_idx_bcmk=lst_idx_bcmk,
        lst_idx_rf=lst_idx_rf, 
        method='all'
        )
    )
    # TODO scale?
    r_treynor *= freq_reb
    return r_treynor
    
    
# TODO
def ratio_updown(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    lst_idx_rf: list,
    freq_reb: float = 1.0
    ) -> array:
    r_treynor = (
        (arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_rf]) /
        beta_capm(
        arr_ts=arr_ts, 
        lst_idx_retn=lst_idx_retn, 
        lst_idx_bcmk=lst_idx_bcmk,
        lst_idx_rf=lst_idx_rf, 
        method='all'
        )
    )
    # TODO scale?
    r_treynor *= freq_reb
    return r_treynor


# TODO
def ratio_upsidepotential(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    lst_idx_rf: list,
    freq_reb: float = 1.0
    ) -> array:
    r_treynor = (
        (arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_rf]) /
        beta_capm(
        arr_ts=arr_ts, 
        lst_idx_retn=lst_idx_retn, 
        lst_idx_bcmk=lst_idx_bcmk,
        lst_idx_rf=lst_idx_rf, 
        method='all'
        )
    )
    # TODO scale?
    r_treynor *= freq_reb
    return r_treynor


    
if __name__ == '__main__':
    import logging

    from scripts import *
    logging.basicConfig(level=logging.DEBUG)
    arr = data_test.values
    N = data_test.shape[1] - 2
    # 
    logging.log(level=logging.INFO, msg='ratio_sharpe')
    tmp = ratio_sharpe(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N], 
            freq_reb=365
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_bernadoledoit')
    tmp = ratio_bernadoledoit(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N]
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_burke')
    tmp = ratio_burke(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N]
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_d')
    tmp = ratio_d(
            arr_ts=arr, 
            lst_idx_retn=range(N)
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_information')
    tmp = ratio_information(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N]
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_kappa')
    tmp = ratio_kappa(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N],
            coeff=1.5
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_kelly')
    tmp = ratio_kelly(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N]
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_omega')
    tmp = ratio_omega(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N]
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_prospect')
    tmp = ratio_prospect(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N]
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_rachev')
    tmp = ratio_rachev(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N],
            alpha=.1,
            beta=.1,
            method='t'
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_sortino')
    tmp = ratio_sortino(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N]
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)
    # 
    logging.log(level=logging.INFO, msg='ratio_treynor')
    tmp = ratio_treynor(
            arr_ts=arr, 
            lst_idx_retn=range(N),
            lst_idx_bcmk=[N]
            ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1,14)