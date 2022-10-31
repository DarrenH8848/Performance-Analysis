from numpy.lib.stride_tricks import as_strided
from numpy import empty, append


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
