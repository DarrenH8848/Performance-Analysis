from numpy import array, multiply


def return_active(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list, 
    freq_reb: float = 252.0
    ) -> array:
    return (arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_bcmk]) * freq_reb

# TODO
def return_geltner(
    arr_ts: array, 
    ) -> array:
    arr = arr_ts[1:]
    arr_lag = arr_ts[0:-1]

    cov = multiply(
        arr-arr.mean(axis=0,keepdims=True),
        arr_lag-arr_lag.mean(axis=0,keepdims=True)
        ).mean(axis=0,keepdims=True)

    rho = cov/arr.var(axis=0,keepdims=True)
    
    return (arr - arr_lag*rho)/(1-rho)

# TODO
# ~ maybe not necessary (might need simple retn)
def return_portfolio(
    arr_ts: array, 
    lst_idx_retn: list, 
    lst_idx_bcmk: list, 
    freq_reb: float = 252.0
    ) -> array:
    return (arr_ts[:,lst_idx_retn] - arr_ts[:,lst_idx_bcmk]) * freq_reb

if __name__ == '__main__':
    import logging

    from scripts import *
    logging.basicConfig(level=logging.DEBUG)
    arr = data_test.values
    N = data_test.shape[1]-2
    # 
    logging.log(level=logging.INFO, msg='return_active')
    tmp = return_active(
        arr_ts=arr, 
        lst_idx_retn=range(N), 
        lst_idx_bcmk=[N], 
        freq_reb=365.0
        ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1542, 14)
    tmp = return_active(
        arr_ts=arr, 
        lst_idx_retn=range(N), 
        lst_idx_bcmk=range(2,N+2), 
        freq_reb=1.0
        ).shape
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1542, 14)