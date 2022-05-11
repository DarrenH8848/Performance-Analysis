import pytest 
import pandas as pd
import numpy as np
from performance import Performance
from performance import add
from performance import Constant


# daily_returns = pd.Series(
#     np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
#     index=pd.date_range('2000-1-30', periods=9, freq='D'))

daily_returns = pd.Series(
    np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
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

flat_returns = pd.Series(
        np.linspace(0.01, 0.01, num=100),
        index=pd.date_range('2000-1-30', periods=100, freq='D')
    )

MAXERROR = 10e-8

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
        (daily_returns, Constant.DAILY, False, 2.330292797300439),
        (weekly_returns, Constant.WEEKLY, False, 0.24690830513998208),
        (monthly_returns, Constant.MONTHLY, False, 0.052242061386048144),
        (one_return, Constant.DAILY, False, 15.173072621219674),
        (weekly_returns, Constant.WEEKLY, True, 0.288888888888889),
        (empty_returns, Constant.DAILY, False, np.nan),
        (daily_returns_matrix, Constant.DAILY, False, [1.9135925373194578,0.4905101332607782]),
    ]
)
def test_AnnualizedReturn(x,y,z,expected):
    if np.isnan(expected).all():
        assert np.isnan(Performance.AnnualizedReturn(x,y,z))
    elif isinstance(x, pd.DataFrame):
        assert (Performance.AnnualizedReturn(x,y,z) - expected < MAXERROR).all()
    else:
        assert Performance.AnnualizedReturn(x,y,z) - expected < MAXERROR


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (flat_returns, Constant.DAILY, 0.0),
        (daily_returns, Constant.DAILY, 0.9136465399704637),
        (weekly_returns, Constant.WEEKLY, 0.38851569394870583),
        (monthly_returns, Constant.MONTHLY, 0.18663690238892558),
        (empty_returns, Constant.DAILY, np.nan),
        (daily_returns_matrix, Constant.DAILY, [0.8063635794244568,0.5588877844035917]),
    ]
)
def test_AnnualizedSD(x,y,expected):
    if np.isnan(expected).all():
        assert np.isnan(Performance.AnnualizedSD(x,y))
    elif isinstance(x, pd.DataFrame):
        assert (Performance.AnnualizedSD(x,y) - expected < MAXERROR).all()
    else:
        assert Performance.AnnualizedSD(x,y) - expected < MAXERROR


print(Performance.AnnualizedSD(daily_returns, Constant.DAILY))

