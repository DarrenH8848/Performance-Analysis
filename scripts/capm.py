from numpy import array, square


def alpha_capm(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    lst_idx_rf: list,
    # freq_reb: float = 1.0
    ) -> array:
    arr_retn = arr_ts[:,lst_idx_retn]
    arr_bcmk = arr_ts[:,lst_idx_bcmk]
    arr_rf = arr_ts[:,lst_idx_rf]
    # risk premium
    yrp = arr_retn - arr_rf
    ym = yrp.mean(axis=0, keepdims=True)
    yrp -= ym
    xrp = arr_bcmk - arr_rf
    xm = xrp.mean(axis=0, keepdims=True)
    xrp -= xm
    # beta = (x * y).sum(axis=0) / (x**2).sum(axis=0)
    return (
        ym - 
        xm * 
        (xrp * yrp).sum(axis=0, keepdims=True) / 
        square(xrp).sum(axis=0, keepdims=True)
        )

def _slope_yxidx(
    vec_y: array, 
    vec_x: array, 
    vec_idx: array) -> float:
    yrp = vec_y[vec_idx]
    yrp -= yrp.mean()
    xrp = vec_x[vec_idx]
    xrp -= xrp.mean()
    return (xrp*yrp).sum()/square(xrp).sum()
        
        
def beta_capm(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    lst_idx_rf: list,
    method: str = 'all'
    ) -> array:
    arr_retn = arr_ts[:,lst_idx_retn]
    arr_bcmk = arr_ts[:,lst_idx_bcmk]
    if arr_bcmk.shape[1] < 2:
        arr_bcmk = arr_bcmk.repeat(arr_retn.shape[1], axis=1)
    arr_rf = arr_ts[:,lst_idx_rf]
    # risk premium
    yrp = arr_retn - arr_rf
    xrp = arr_bcmk - arr_rf
    if method == 'all':
        yrp -= yrp.mean(axis=0, keepdims=True)
        xrp -= xrp.mean(axis=0, keepdims=True)
        return (
            (xrp * yrp).sum(axis=0, keepdims=True) / 
            square(xrp).sum(axis=0, keepdims=True)
            )
    elif method == 'bull':
        arr_idx = arr_bcmk>0
    elif method == 'bear':
        arr_idx = arr_bcmk<0
    return array((*map(
        _slope_yxidx, 
        yrp.T, 
        xrp.T, 
        arr_idx.T),)).reshape((1,-1))


def ratio_timing(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    lst_idx_rf: list
    ) -> array:
    return (
        beta_capm(
            arr_ts=arr_ts, 
            lst_idx_retn=lst_idx_retn, 
            lst_idx_bcmk=lst_idx_bcmk,
            lst_idx_rf=lst_idx_rf,
            method='bull') /
        beta_capm(
            arr_ts=arr_ts, 
            lst_idx_retn=lst_idx_retn, 
            lst_idx_bcmk=lst_idx_bcmk,
            lst_idx_rf=lst_idx_rf,
            method='bear')
        )


def epsilon_capm(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list,
    lst_idx_rf: list
) -> array:
    arr_retn = arr_ts[:,lst_idx_retn]
    arr_bcmk = arr_ts[:,lst_idx_bcmk]
    if arr_bcmk.shape[1] < 2:
        arr_bcmk = arr_bcmk.repeat(arr_retn.shape[1], axis=1)
    arr_rf = arr_ts[:,lst_idx_rf]
    # risk premium
    yrp = arr_retn - arr_rf
    xrp = arr_bcmk - arr_rf
    ym = yrp.mean(axis=0, keepdims=True)
    xm = xrp.mean(axis=0, keepdims=True)
    xdev = xrp - xm
    ydev = yrp - ym
    return (
        ydev - 
        xdev * (
            (xdev * ydev).sum(axis=0, keepdims=True)/
            square(xdev).sum(axis=0, keepdims=True)
            )
        )
    
    


if __name__ == '__main__':
    import logging

    from scripts import *
    logging.basicConfig(level=logging.DEBUG)
    arr = data_test.values
    N = data_test.shape[1]-2
    # 
    logging.log(level=logging.INFO, msg='alpha_capm')
    tmp = alpha_capm(
        arr_ts=arr, 
        lst_idx_retn=range(N), 
        lst_idx_bcmk=[N], 
        lst_idx_rf=[N+1],
        ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    # 
    logging.log(level=logging.INFO, msg='beta_capm')
    tmp = beta_capm(
        arr_ts=arr, 
        lst_idx_retn=range(N), 
        lst_idx_bcmk=[N], 
        lst_idx_rf=[N+1],
        ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    # 
    logging.log(level=logging.INFO, msg='ratio_timing')
    tmp = ratio_timing(
        arr_ts=arr, 
        lst_idx_retn=range(N), 
        lst_idx_bcmk=[N], 
        lst_idx_rf=[N+1],
        ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1, 14)
    # 
    logging.log(level=logging.INFO, msg='epsilon_capm')
    tmp = epsilon_capm(
        arr_ts=arr, 
        lst_idx_retn=range(N), 
        lst_idx_bcmk=[N], 
        lst_idx_rf=[N+1],
        ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1542, 14)