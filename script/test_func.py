from operator import index
from re import S
import pytest 
import pandas as pd
import numpy as np
from pandas import testing as tm
from performance import Performance
from performance import Constant


daily_returns = pd.Series(
    np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# daily_returns = pd.Series(
#     np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
#     index=pd.date_range('2000-1-30', periods=9, freq='D'))

asset1 = [0., 1., 10., -4., 2., 3., 2., 1., -10.]
asset2 = [1., 4., 2., -3., 1., 4., 1., 0., -8.]
daily_returns_matrix = pd.DataFrame(
    np.array([asset1, asset2]).T / 100, columns = ['asset1', 'asset2'],
    index = pd.date_range('2000-1-30', periods=9, freq='D'))

asset1 = [np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]
asset2 = [np.nan, np.nan, 2., -3., 1., 4., 1., 0., -8.]
daily_returns_matrix_withnan = pd.DataFrame(
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

asset1 = []
asset2 = []
empty_returns_matrix = pd.DataFrame(
    np.array([asset1, asset2]).T / 100, columns = ['asset1', 'asset2'],
    index = pd.date_range('2000-1-30', periods=0, freq='D'))

flat_returns = pd.Series(
        np.linspace(0.01, 0.01, num=100),
        index=pd.date_range('2000-1-30', periods=100, freq='D')
    )

MAXERROR = 10e-5

def list_to_series_or_dataframe(expect):
    array = np.array(expect)
    if array.ndim > 1:
        return pd.DataFrame(np.array(expect),columns = ['asset1', 'asset2'])
    else:
        return pd.Series(np.array(expect), index = ['asset1', 'asset2'])

@pytest.mark.parametrize(
    "x,y,z,expected",
    [
        (daily_returns, Constant.DAILY, False, 2.330292797300439),
        (weekly_returns, Constant.WEEKLY, False, 0.24690830513998208),
        (monthly_returns, Constant.MONTHLY, False, 0.052242061386048144),
        (one_return, Constant.DAILY, False, 15.173072621219674),
        (weekly_returns, Constant.WEEKLY, True, 0.288888888888889),
        (empty_returns, Constant.DAILY, False, np.nan),
        (empty_returns_matrix, Constant.DAILY, False, [[np.nan, np.nan]]),
        (daily_returns_matrix, Constant.DAILY, False, [1.9135925373194578,0.4905101332607782]),
    ]
)
def test_AnnualizedReturn(x,y,z,expected):
    if np.isnan(expected).all():
        assert np.isnan(Performance.AnnualizedReturn(x,y,z)).all()
    elif isinstance(x, pd.DataFrame):
        assert (Performance.AnnualizedReturn(x,y,z) - expected < MAXERROR).all()
        # tm.assert_series_equal(Performance.AnnualizedReturn(x,y,z),expected)
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
        (daily_returns_matrix, Constant.DAILY, [0.85527773,0.59279001]),
    ]
)
def test_AnnualizedSD(x,y,expected):
    if np.isnan(expected).all():
        assert np.isnan(Performance.AnnualizedSD(x,y))
    elif isinstance(x, pd.DataFrame):
        assert (Performance.AnnualizedSD(x,y) - expected < MAXERROR).all()
    else:
        assert Performance.AnnualizedSD(x,y) - expected < MAXERROR

@pytest.mark.parametrize(
    "x,y,expected",
    [
        (daily_returns, False, pd.Series([0.01, 0.111, 0.066560, 0.087891, 0.120528, 0.142938, 0.154368, 0.038931])),
        (empty_returns, False, pd.Series([])),
        (daily_returns_matrix, False, pd.DataFrame([[0.0, 0.01, 0.111, 0.066559, 0.08789, 0.12052,
                            0.14293, 0.15436, 0.03893],[0.01, 0.0504, 0.071408, 0.039266, 0.049658, 0.091645, 0.102561, 0.102561, 0.014356]]).T),
        (daily_returns_matrix_withnan, False, pd.DataFrame([[0.01, 0.111, 0.066559, 0.08789, 0.12052,
                            0.14293, 0.15436, 0.03893],[np.nan, 0.02, -0.0106, -0.000706, 0.039266, 0.049658, 0.049658, -0.034314]]).T),
    ]
)
def test_CumulativeReturns(x, y, expected):
    cum_returns = Performance.CumulativeReturns(x, y)
    if isinstance(x, pd.DataFrame):
        expected.index =  cum_returns.index
        expected.columns = cum_returns.columns
        tm.assert_frame_equal(cum_returns, expected, atol = MAXERROR)
    else:
        expected.index = cum_returns.index
        tm.assert_series_equal(cum_returns, expected, atol = MAXERROR)

# todo: more tests following functions
@pytest.mark.parametrize(
    "x,y,expected",
    [
        (daily_returns, False, pd.Series([0., 0., -0.04, -0.0208, 0., 0., 0., -0.1])),
        (daily_returns_matrix, False, pd.DataFrame([[0., 0., 0., -0.04, -0.0208, 0.,
                            0., 0., -0.1],[0., 0., 0., -0.03, -0.0203, 0., 0., 0., -0.08]]).T),
        (daily_returns_matrix_withnan, False, pd.DataFrame([[0., 0., -0.04, -0.0208, 0.,
                            0., 0., -0.1],[np.nan, 0., -0.03, -0.0203, 0., 0., 0., -0.08]]).T),
    ]
)
def test_Drawdown(x, y, expected):
    drawdown = Performance.Drawdown(x, y)
    if isinstance(x, pd.DataFrame):
        expected.index = drawdown.index
        expected.columns = drawdown.columns
        tm.assert_frame_equal(drawdown, expected, check_dtype = False, atol = MAXERROR)
    else:
        expected.index = drawdown.index
        tm.assert_series_equal(drawdown, expected, check_dtype = False, atol = MAXERROR)

@pytest.mark.parametrize(
    "x,y,expected",
    [
        (daily_returns, False, pd.Series([0., 0., -0.013333, -0.015200, -0.012160, -0.010133, -0.008686, -0.020100])),
        (daily_returns_matrix, False, pd.DataFrame([[0., 0., 0., -0.01, -0.01216, -0.010133, -0.008686, -0.0076, -0.017867],
                                            [0., 0., 0., -0.0075, -0.01006, -0.008383, -0.007186, -0.006288, -0.014478]]).T),
        (daily_returns_matrix_withnan, False, pd.DataFrame([[0., 0., -0.013333, -0.0152, -0.01216, -0.010133, -0.008686, -0.0201],
                                                    [np.nan, 0., -0.015, -0.016767, -0.012575, -0.01006, -0.008383, -0.018614]]).T),
    ]
)
def test_AverageDrawDown(x, y, expected):
    avg_drawdown = Performance.AverageDrawDown(x, y)
    if isinstance(x, pd.DataFrame):
        expected.index = avg_drawdown.index
        expected.columns = avg_drawdown.columns
        tm.assert_frame_equal(avg_drawdown, expected, check_dtype = False, atol = MAXERROR)
    else:
        expected.index = avg_drawdown.index
        tm.assert_series_equal(avg_drawdown, expected, check_dtype = False, atol = MAXERROR)


print(Performance.AverageDrawDown(daily_returns_matrix_withnan, False))


