import pytest 
import pandas as pd
import numpy as np
from performance import Performance
from performance import add


daily_returns = pd.Series(
    np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

asset1 = [0., 1., 10., -4., 2., 3., 2., 1., -10.]
asset2 = [1., 4., 2., -3., 1., 4., 1., 0., -8.]
daily_returns_matrix = pd.DataFrame(
    np.array([asset1, asset2]).T / 100, columns = ['asset1', 'asset2'],
    index = pd.date_range('2000-1-30', periods=9, freq='D'))

weekly_returns = pd.Series(
    np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.])/100,
    index=pd.date_range('2000-1-30', periods=9, freq='W'))

monthly_returns = pd.Series(
    np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.])/100,
    index=pd.date_range('2000-1-30', periods=9, freq='M'))

one_return = pd.Series(
    np.array([1.])/100,
    index=pd.date_range('2000-1-30', periods=1, freq='D'))

empty_returns = pd.Series(
    np.array([])/100,
    index=pd.date_range('2000-1-30', periods=0, freq='D'))


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (1,1,2),
        (2,2,4),
        (10,10,20),
    ]
)
def test_add(x,y,expected):
    assert add.add(x,y) == expected


@pytest.mark.parametrize(
    "x,y,z,expected",
    [
        (daily_returns, 'daily', False, 2.330292797300439),
        (weekly_returns, 'weekly', False, 0.24690830513998208),
        (monthly_returns, 'monthly', False, 0.052242061386048144),
        (one_return, 'daily', False, 15.173072621219674),
        (weekly_returns, 'weekly', True, 0.288888888888889),
        (empty_returns, 'daily', False, np.nan),
        (daily_returns_matrix, 'daily', False, [1.913593,0.490510]),
    ]
)
def test_AnnualizedReturn(x,y,z,expected):
    if np.isnan(expected).all():
        assert np.isnan(Performance.AnnualizedReturn(x,y,z))
    elif isinstance(x, pd.DataFrame):
        assert (Performance.AnnualizedReturn(x,y,z) - expected < 10e-6).all()
    else:
        assert Performance.AnnualizedReturn(x,y,z) - expected < 10e-6 

