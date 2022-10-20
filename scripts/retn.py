from numpy import array


def return_active(arr_ts: array,
                  lst_idx_retn: list,
                  lst_idx_bcmk: list,
                  freq_reb: float = 252.0) -> array:
    return (arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]) * freq_reb


# TODO return_geltner
def return_geltner(arr_ts: array,
                   lst_idx_retn: list,
                   lst_idx_bcmk: list,
                   freq_reb: float = 252.0) -> array:
    return (arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]) * freq_reb


# TODO return_portfolio
# ~ maybe not necessary (might need simple retn)
def return_portfolio(arr_ts: array,
                     lst_idx_retn: list,
                     lst_idx_bcmk: list,
                     freq_reb: float = 252.0) -> array:
    return (arr_ts[:, lst_idx_retn] - arr_ts[:, lst_idx_bcmk]) * freq_reb


if __name__ == '__main__':
    import logging

    from numpy import isfinite

    from scripts import *
    logging.basicConfig(level=logging.DEBUG)
    arr = data_test.values
    N = data_test.shape[1] - 2
    #
    tmp = return_active(arr_ts=arr,
                        lst_idx_retn=range(N),
                        lst_idx_bcmk=[N],
                        freq_reb=365.0)
    assert isfinite(tmp).all()
    tmp = tmp.shape
    logging.log(level=logging.INFO, msg='return_active')
    logging.log(level=logging.INFO, msg=tmp)
    assert tmp == (1542, 14)
