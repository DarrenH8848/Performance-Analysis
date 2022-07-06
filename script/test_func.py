from turtle import pos
import pytest 
import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal
from numpy.testing import assert_almost_equal
from performance import Performance
from performance import Constant

MAXERROR = 10e-5

simple_benchmark = pd.Series(
        np.array([0., 1., 0., 1., 0., 1., 0., 1., 0.]) / 100,
        index=pd.date_range('2000-1-30', periods=9, freq='D'))

daily_returns = pd.Series(
    np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

daily_returns_series = pd.Series(
    np.array([3., 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

# daily_returns = pd.Series(
#     np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
#     index=pd.date_range('2000-1-30', periods=9, freq='D'))

positive_returns = pd.Series(
    np.array([1., 2., 1., 1., 1., 1., 1., 1., 1.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

negative_returns = pd.Series(
    np.array([0., -6., -7., -1., -9., -2., -6., -8., -5.]) / 100,
    index=pd.date_range('2000-1-30', periods=9, freq='D'))

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

flat_line = pd.Series(
        np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
        index=pd.date_range('2000-1-30', periods=9, freq='D'))

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

upward_returns_1 = pd.Series(
        np.linspace(0.01, 0.1, num=100),
        index=pd.date_range('2000-1-30', periods=100, freq='D')
    )

upward_returns_2 = pd.Series(
        np.linspace(-0.1, 0.1, num=100),
        index=pd.date_range('2000-1-30', periods=100, freq='D')
    )

upward_return_matrix = pd.DataFrame({
    'asset1': upward_returns_1,
    'asset2': upward_returns_2})

return_benchmark = pd.Series(
        [-0.00171614, 0.01322056, 0.03063862, -0.01422057, -0.00489779,
        0.01268925, -0.03357711, 0.01797036, -0.02658932, 0.01283923] * 10,
        index=pd.date_range('2000-1-30', periods=100, freq='D')
    )


asset1 = [-0.00171614, 0.01322056, 0.03063862, -0.01422057, -0.00489779,
        0.01268925, -0.03357711, 0.01797036]
asset2 = [0.01846232, 0.00793951, -0.01448395, 0.00422537, -0.00339611,
        0.03756813, 0.0151531, 0.03549769]

df_index_simple = pd.date_range('2000-1-30', periods=8, freq='D')
df_index_week = pd.date_range('2000-1-30', periods=8, freq='W')
df_index_month = pd.date_range('2000-1-30', periods=8, freq='M')

df_simple = pd.DataFrame({
    'asset1': pd.Series(asset1, index=df_index_simple),
    'asset2': pd.Series(asset2, index=df_index_simple)})

df_week = pd.DataFrame({
    'asset1': pd.Series(asset1, index=df_index_week),
    'asset2': pd.Series(asset2, index=df_index_week)})

df_month = pd.DataFrame({
    'asset1': pd.Series(asset1, index=df_index_month),
    'asset2': pd.Series(asset2, index=df_index_month)})

    
def list_to_series(expect):
    # array = np.array(expect)
    # if array.ndim > 1:
    #     return pd.DataFrame(np.array(expect),columns = ['asset1', 'asset2'])
    # else:
    #     return pd.Series(np.array(expect), index = ['asset1', 'asset2'])
    return pd.Series(np.array(expect), index = ['asset1', 'asset2'])


parameters = {
    "ann_rtn_params" : [
            (daily_returns, Constant.DAILY, False, 2.330292797300439),
            (weekly_returns, Constant.WEEKLY, False, 0.24690830513998208),
            (monthly_returns, Constant.MONTHLY, False, 0.052242061386048144),
            (one_return, Constant.DAILY, False, 11.274002099240244),
            (weekly_returns, Constant.WEEKLY, True, 0.288888888888889),
            (empty_returns, Constant.DAILY, False, np.nan),
            (empty_returns_matrix, Constant.DAILY, False, list_to_series([np.nan, np.nan])),
            (daily_returns_matrix, Constant.DAILY, False, list_to_series([1.9135925373194578,0.4905101332607782])),
        ],
    "ann_sd_params" : [
            (flat_returns, Constant.DAILY, 0.0),
            (daily_returns, Constant.DAILY, 0.9136465399704637),
            (weekly_returns, Constant.WEEKLY, 0.38851569394870583),
            (monthly_returns, Constant.MONTHLY, 0.18663690238892558),
            (empty_returns, Constant.DAILY, np.nan),
            (daily_returns_matrix, Constant.DAILY, list_to_series([0.85527773,0.59279001])),
        ],
    "cum_rtn_params" : [
            (daily_returns, False, 
            pd.Series([0.00, 0.01, 0.111, 0.066560, 0.087891, 0.120528, 0.142938, 0.154368, 0.038931])),
            (empty_returns, False, pd.Series([])),
            (daily_returns_matrix, False, 
            pd.DataFrame([[0.0, 0.01, 0.111, 0.066559, 0.08789, 0.12052, 0.14293, 0.15436, 0.03893],
                        [0.01, 0.0504, 0.071408, 0.039266, 0.049658, 0.091645, 0.102561, 0.102561, 0.014356]]).T),
            (daily_returns_matrix_withnan, False, 
            pd.DataFrame([[0.00, 0.01, 0.111, 0.066559, 0.08789, 0.12052, 0.14293, 0.15436, 0.03893],
                        [0.00, 0.00, 0.02, -0.0106, -0.000706, 0.039266, 0.049658, 0.049658, -0.034314]]).T),
        ],
    "drawd_params" : [
            (daily_returns, False, pd.Series([0., 0., 0., -0.04, -0.0208, 0., 0., 0., -0.1])),
            (daily_returns_matrix, False, pd.DataFrame([[0., 0., 0., -0.04, -0.0208, 0.,
                                0., 0., -0.1],[0., 0., 0., -0.03, -0.0203, 0., 0., 0., -0.08]]).T),
            (daily_returns_matrix_withnan, False, pd.DataFrame([[0., 0., 0., -0.04, -0.0208, 0.,
                                0., 0., -0.1],[0., 0., 0., -0.03, -0.0203, 0., 0., 0., -0.08]]).T),
        ],
    "avr_drawd_params" : [
            (daily_returns, False, 
            pd.Series([0., 0., 0., -0.010000, -0.012160, -0.010133, -0.008686, -0.007600, -0.017867])),
            (daily_returns_matrix, False, 
            pd.DataFrame([[0., 0., 0., -0.01, -0.01216, -0.010133, -0.008686, -0.0076, -0.017867],
                        [0., 0., 0., -0.0075, -0.01006, -0.008383, -0.007186, -0.006288, -0.014478]]).T),
            (daily_returns_matrix_withnan, False, 
            pd.DataFrame([[0., 0., 0., -0.01, -0.01216, -0.010133, -0.008686, -0.0076, -0.017867],
                        [0., 0., 0., -0.0075, -0.01006, -0.008383, -0.007186, -0.006288, -0.014478]]).T),
        ],
    "max_drawd_params" : [
            (daily_returns, False, -0.1),
            (daily_returns_matrix, False, list_to_series([-0.1,-0.08])),
            (daily_returns_matrix_withnan, False, list_to_series([-0.1,-0.08])),
        ],
    "sharp_params" : [
            (empty_returns, 0.0, False, Constant.DAILY, np.nan),
            (one_return, 0.0,False, Constant.DAILY, np.nan),
            (daily_returns, 0.0,False, Constant.DAILY, 2.550540822248145),
            (daily_returns, 0.1, False, Constant.DAILY,2.441089305030958),
            (daily_returns_matrix, 0.0, False, Constant.DAILY, 
            list_to_series([2.237393, 0.827460])),
            (daily_returns_matrix_withnan, 0.0, False, Constant.DAILY, 
            list_to_series([2.550541, -1.140458])),
        ],
    "calmar_params" : [
            (empty_returns, Constant.DAILY, False, np.nan),
            (one_return, Constant.DAILY, False, np.nan),
            (daily_returns, Constant.DAILY, False, 19.135925373194233),
            (weekly_returns, Constant.WEEKLY, False, 2.4690830513998208),
            (monthly_returns, Constant.MONTHLY, False, 0.52242061386048144),
            (daily_returns_matrix, Constant.DAILY, False, 
            list_to_series([19.135925, 6.131377])),
            (daily_returns_matrix_withnan, Constant.DAILY, False, 
            list_to_series([19.135925, -7.797671])),
        ],
    "burke_params" : [
            (empty_returns, 0.0, Constant.DAILY, False, False, np.nan),
            (one_return, 0.0, Constant.DAILY, False, False, np.nan),
            (weekly_returns, 0.0, Constant.WEEKLY, False, False, 2.250895000942517),
            (monthly_returns, 0.1, Constant.MONTHLY, False, True, -1.3061298835677062),
            (daily_returns_matrix, 0.0, Constant.DAILY, False, False, 
            list_to_series([17.444921, 5.585498])),
            (daily_returns_matrix_withnan, 0.0, Constant.DAILY, False, True, 
            list_to_series([60.086240, -21.556077])),
        ],
    "dow_risk_params" : [
            (empty_returns, 0.0, Constant.DAILY, np.nan),
            (one_return, 0.0, Constant.DAILY, 0.0),
            (daily_returns, 0.0, Constant.DAILY, 0.60448325038829653),
            (daily_returns, 0.1, Constant.DAILY, 1.7161730681956295),
            (weekly_returns, 0.0, Constant.WEEKLY, 0.25888650451930134),
            (weekly_returns, 0.1, Constant.WEEKLY, 0.7733045971672482),
            (monthly_returns, 0.0, Constant.MONTHLY, 0.1243650540411842),
            (monthly_returns, 0.1, Constant.MONTHLY, 0.37148351242013422),
            (df_simple, 0.0, Constant.DAILY,
            list_to_series([0.20671788246185202, 0.083495680595704475])),
            (df_week, 0.0, Constant.WEEKLY,
            list_to_series([0.093902996054410062, 0.037928477556776516])),
            (df_month, 0.0, Constant.MONTHLY,
            list_to_series([0.045109540184877193, 0.018220251263412916])),
        ],
    "sortino_params" : [
            (empty_returns, 0.0, Constant.DAILY, np.nan),
            (one_return, 0.0, Constant.DAILY, np.nan),
            (daily_returns, 0.0, Constant.DAILY, 2.605531251673693),
            (positive_returns, 0.0, Constant.DAILY, np.nan),
            (negative_returns, 0.0, Constant.DAILY, -13.532743075043401),
            (simple_benchmark, 0.0, Constant.DAILY, np.nan),
            (weekly_returns, 0.0, Constant.WEEKLY, 1.1158901056866439),
            (monthly_returns, 0.0, Constant.MONTHLY, 0.53605626741889756),
            (daily_returns_matrix_withnan, 0.0, Constant.DAILY,
            list_to_series([2.605531251673693,-2.1067406495303502])),
            (df_simple, 0.0, Constant.DAILY,
            list_to_series([3.0639640966566306, 38.090963117002495])),
            (df_week, 0.0, Constant.WEEKLY,
            list_to_series([1.3918264112070571, 17.303077589064618])),
            (df_month, 0.0, Constant.MONTHLY,
            list_to_series([0.6686117809312383, 8.3121296084492844])),
        ],
    "tra_err_params" : [
            (empty_returns, simple_benchmark, Constant.DAILY, np.nan),
            (one_return, one_return, Constant.DAILY, np.nan),
            (daily_returns, flat_line, Constant.DAILY, 0.9136465399704637),
            (daily_returns, daily_returns, Constant.DAILY, 0.0),
            (daily_returns, -daily_returns, Constant.DAILY, 1.8272930799409275),
            (daily_returns_matrix, simple_benchmark, Constant.DAILY,
            list_to_series([0.8638286867197685,0.5766281297335398])),
            (daily_returns_matrix_withnan, simple_benchmark, Constant.DAILY,
            list_to_series([0.9234446382972832,0.61773780845922]))
        ],
    "info_params" : [
            (empty_returns, simple_benchmark, Constant.DAILY, np.nan),
            (one_return, one_return, Constant.DAILY, np.nan),
            (daily_returns, flat_line, Constant.DAILY, -9.789025526467713),
            (daily_returns, daily_returns, Constant.DAILY, np.nan),
            (daily_returns, -daily_returns, Constant.DAILY, 1.7443975579702573),
            (daily_returns_matrix, simple_benchmark, Constant.DAILY,
            list_to_series([-0.15542362691108277,-2.700773186868549])),
            (daily_returns_matrix_withnan, simple_benchmark, Constant.DAILY,
            list_to_series([0.3058557716918939,-4.473333398830916]))
        ],
    "beta_params" : [
            (empty_returns, simple_benchmark, np.nan),
            (one_return, one_return, np.nan),
            (daily_returns, flat_line, np.nan),
            (daily_returns, daily_returns, 1.0),
            (daily_returns, -daily_returns, -1.0),
            (daily_returns_matrix, simple_benchmark,
            list_to_series([-0.55,1.85])),
            (daily_returns_matrix_withnan, simple_benchmark,
            list_to_series([-0.750000,1.333333]))
        ],
    "alpha_params" : [
            (empty_returns, simple_benchmark, Constant.DAILY, 0.0, None, np.nan),
            (one_return, one_return, Constant.DAILY, 0.0, None, np.nan),
            (daily_returns, flat_line, Constant.DAILY, 0.0, None, np.nan),
            (daily_returns, daily_returns, Constant.DAILY, 0.0, None, 0.0),
            (daily_returns, daily_returns, Constant.DAILY, 0.0, 1.0, 0.0),
            (daily_returns, -daily_returns, Constant.DAILY, 0.0, None, 0.0),
            (daily_returns_matrix, simple_benchmark, Constant.DAILY, 0.0, None,
            list_to_series([6.448247364237094,-0.7805331814084672])),
            (daily_returns_matrix_withnan, simple_benchmark, Constant.DAILY, 0.0, None,
            list_to_series([11.274002099240244,-0.9205545483094462])),
        ],
    "treynor_params" : [
            (empty_returns, simple_benchmark, Constant.DAILY, 0.0, np.nan),
            (one_return, one_return, Constant.DAILY, 0.0, np.nan),
            (daily_returns, flat_line, Constant.DAILY, 0.0, np.nan),
            (daily_returns, daily_returns, Constant.DAILY, 0.0, 2.330292797300439),
            (daily_returns, -daily_returns, Constant.DAILY, 0.1, -2.230292797300439),
            (daily_returns, simple_benchmark, Constant.DAILY, 0.0, -3.1070570630672534),
            (daily_returns_matrix, simple_benchmark, Constant.DAILY, 0.0,
            list_to_series([-3.479259158762647, 0.2651406125733936])),
            (daily_returns_matrix_withnan, simple_benchmark, Constant.DAILY, 0.0,
            list_to_series([-3.1070570630672534, -0.5366214340964719])),
        ],
    "skew_params" : [
            (empty_returns, "moment", np.nan),
            (one_return, "moment", np.nan),
            (daily_returns, "moment", -0.3650572690177866),
            (daily_returns, "fisher", -0.01860542367998986),
            (daily_returns, "sample", -0.45530640907197434),
            (daily_returns_matrix, "sample", 
            list_to_series([-0.41883881865707645,-1.4644030498958307])),
            (daily_returns_matrix, "moment", 
            list_to_series([-0.34552439704601157,-1.2080708814668715])),
            (daily_returns_matrix_withnan, "sample", 
            list_to_series([-0.45530640907197434,-1.3001751805052328])),
            (daily_returns_matrix_withnan, "fisher", 
            list_to_series([-0.01860542367998986,-1.7221509113551186])),
        ],
    "kurt_params" : [
            (empty_returns, "excess", np.nan),
            (one_return, "excess", np.nan),
            (daily_returns, "excess", 0.11953473165699258),
            (daily_returns, "moment", 3.1195347316569926),
            (daily_returns, "fisher", 1.2970629244001808),
            (daily_returns, "sample", 6.551022936479684),
            (daily_returns, "sample_excess", 1.6510229364796833),
            (daily_returns_matrix, "sample", 
            list_to_series([6.626664093901564,7.02081826365568])),
            (daily_returns_matrix, "moment", 
            list_to_series([3.4789986492983207,3.685929588419233])),
            (daily_returns_matrix, "sample_excess", 
            list_to_series([2.0552355224729926,2.4493896922271086])),
            (daily_returns_matrix_withnan, "excess", 
            list_to_series([0.11953473165699258,-0.0048055472932775345])),
            (daily_returns_matrix_withnan, "fisher", 
            list_to_series([1.2970629244001808,2.8855180055401637])),
        ],
    "var_params" : [
            (flat_returns, 0.05, 0.01),
            (flat_returns, 0.10, 0.01),
            (upward_returns_1, 0.05, 0.0145),
            (upward_returns_2, 0.05, -0.09),
            (upward_return_matrix, 0.05, 
            list_to_series([0.0145, -0.09])),
        ],
    "cvar_params" : [
            (flat_returns, 0.05, 0.01),
            (flat_returns, 0.10, 0.01),
            (upward_returns_1, 0.05, 0.01181818181818182),
            (upward_returns_2, 0.05, -0.09595959595959595),
            (upward_return_matrix, 0.05, 
            list_to_series([0.01181818181818182, -0.09595959595959595])),
        ],
    "omega_params" : [
            (empty_returns, 0.0, Constant.DAYS_PER_YEAR, np.nan),
            (one_return, 0.0, Constant.WEEKS_PER_YEAR, np.nan),
            (daily_returns, 0.0, Constant.DAYS_PER_YEAR, 1.357142857142857),
            (daily_returns, 0.05, Constant.DAYS_PER_YEAR, 1.3451235931095717),
            (daily_returns_matrix, 0.0, Constant.DAYS_PER_YEAR,
            list_to_series([1.357142857142857, 1.1818181818181819])),
            (daily_returns_matrix_withnan, 0.05, Constant.DAYS_PER_YEAR,
            list_to_series([1.3451235931095717, 0.7164481754947586])),
        ],
    "tdc_params" : [
            (empty_returns, simple_benchmark, 0.05, "lower", np.nan),
            (one_return, one_return, 0.95, "upper", np.nan),
            (daily_returns_matrix, simple_benchmark, 0.2, "lower",
            list_to_series([0.5, 0.5])),
            (daily_returns_matrix, simple_benchmark, 0.9, "upper",
            list_to_series([1.0, 2.0]))
        ],
    "TDC_params" : [
            (empty_returns, simple_benchmark,"FF", np.nan),
            (daily_returns_series, negative_returns, "FF", -0.10344827586207028),
            (daily_returns_series, negative_returns, "CFG", 0.2772697634735837)
        ],
}


class TestClass:
    @pytest.mark.parametrize(
        "x,y,z,expected",
        parameters["ann_rtn_params"]
    )
    def test_annualized_return(self,x,y,z,expected):
        if np.isnan(expected).all():
            assert np.isnan(Performance.annualized_return(x,y,z)).all()
        if isinstance(x, pd.DataFrame):
            # assert (Performance.annualized_return(x,y,z) - expected < MAXERROR).all()
            assert_series_equal(Performance.annualized_return(x,y,z),expected, check_dtype = False, atol = MAXERROR)
        else:
            # assert Performance.annualized_return(x,y,z) - expected < MAXERROR
            assert_almost_equal(Performance.annualized_return(x,y,z), expected, MAXERROR)


    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["ann_sd_params"]
    )
    def test_annualized_sd(self,x,y,expected):
        if np.isnan(expected).all():
            assert np.isnan(Performance.annualized_sd(x,y))
        elif isinstance(x, pd.DataFrame):
            # assert (Performance.annualized_sd(x,y) - expected < MAXERROR).all()
            assert_series_equal(Performance.annualized_sd(x,y), expected, check_dtype = False, atol = MAXERROR)
        else:
            # assert Performance.annualized_sd(x,y) - expected < MAXERROR
            assert_almost_equal(Performance.annualized_sd(x,y), expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["cum_rtn_params"]
    )
    def test_cumulative_returns(self, x, y, expected):
        cum_returns = Performance.cumulative_returns(x, y)
        if isinstance(x, pd.DataFrame):
            expected.index =  cum_returns.index
            expected.columns = cum_returns.columns
            assert_frame_equal(cum_returns, expected, atol = MAXERROR)
        else:
            expected.index = cum_returns.index
            assert_series_equal(cum_returns, expected, atol = MAXERROR)

    # todo: more tests following functions
    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["drawd_params"]
    )
    def test_drawdown(self, x, y, expected):
        drawdown = Performance.drawdown(x, y)
        if isinstance(x, pd.DataFrame):
            expected.index = drawdown.index
            expected.columns = drawdown.columns
            assert_frame_equal(drawdown, expected, check_dtype = False, atol = MAXERROR)
        else:
            expected.index = drawdown.index
            assert_series_equal(drawdown, expected, check_dtype = False, atol = MAXERROR)

    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["avr_drawd_params"]
    )
    def test_average_drawdown(self, x, y, expected):
        avg_drawdown = Performance.average_drawdown(x, y)
        if isinstance(x, pd.DataFrame):
            expected.index = avg_drawdown.index
            expected.columns = avg_drawdown.columns
            assert_frame_equal(avg_drawdown, expected, check_dtype = False, atol = MAXERROR)
        else:
            expected.index = avg_drawdown.index
            assert_series_equal(avg_drawdown, expected, check_dtype = False, atol = MAXERROR)

    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["max_drawd_params"]
    )
    def test_max_drawdown(self, x, y, expected):
        max_drawdown = Performance.max_drawdown(x, y)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(max_drawdown, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(max_drawdown, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,z,m,expected",
        parameters["sharp_params"]
    )
    def test_sharpe_ratio(self, x, y, z, m, expected):
        sharpe_ratio = Performance.sharpe_ratio(x,y,z,m)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(sharpe_ratio, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(sharpe_ratio, expected, MAXERROR)


    @pytest.mark.parametrize(
        "x,y,z,expected",
        parameters["calmar_params"]
    )
    def test_calmar_ratio(self, x, y, z, expected):
        calmar_ratio = Performance.calmar_ratio(x,y,z)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(calmar_ratio, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(calmar_ratio, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,z,m,n,expected",
        parameters["burke_params"]
    )
    def test_burke_ratio(self, x, y, z, m, n, expected):
        burke_ratio = Performance.burke_ratio(x, y, z, m, n)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(burke_ratio, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(burke_ratio, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,z,expected",
        parameters["dow_risk_params"]
    )
    def test_downside_risk(self, x, y, z, expected):
        downside_risk = Performance.downside_risk(x,y,z)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(downside_risk, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(downside_risk, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,z,expected",
        parameters["sortino_params"]
    )
    def test_sortino_ratio(self, x,y,z,expected):
        sortino_ratio = Performance.sortino_ratio(x,y,z)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(sortino_ratio, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(sortino_ratio, expected, MAXERROR)
        

    @pytest.mark.parametrize(
        "x,y,z,expected",
        parameters["tra_err_params"]
    )
    def test_tracking_error(self, x,y,z,expected):
        tracking_error = Performance.tracking_error(x,y,z)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(tracking_error, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(tracking_error, expected, MAXERROR)    

    @pytest.mark.parametrize(
        "x,y,z,expected",
        parameters["info_params"]
    )
    def test_information_ratio(self, x,y,z,expected):
        information_ratio = Performance.information_ratio(x,y,z)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(information_ratio, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(information_ratio, expected, MAXERROR)  

    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["beta_params"]
    )
    def test_capm_beta(self, x,y,expected):
        capm_beta = Performance.capm_beta(x,y)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(capm_beta, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(capm_beta, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,z,m,n,expected",
        parameters["alpha_params"]
    )
    def test_capm_alpha(self, x,y,z,m,n,expected):
        capm_alpha = Performance.capm_alpha(x,y,z,m,n)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(capm_alpha, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(capm_alpha, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,z,m,expected",
        parameters["treynor_params"]
    )
    def test_treynor_ratio(self, x, y, z, m, expected):
        treynor_ratio = Performance.treynor_ratio(x,y,z,m)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(treynor_ratio, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(treynor_ratio, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["skew_params"]
    )
    def test_skewness(self, x, y, expected):
        skewness = Performance.skewness(x, y)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(skewness, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(skewness, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["kurt_params"]
    )
    def test_kurtosis(self, x, y, expected):
        kurtosis = Performance.kurtosis(x, y)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(kurtosis, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(kurtosis, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["var_params"]
    )
    def test_value_at_risk(self, x, y, expected):
        value_at_risk = Performance.value_at_risk(x, y)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(value_at_risk, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(value_at_risk, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,expected",
        parameters["cvar_params"]
    )
    def test_conditional_value_at_risk(self, x, y, expected):
        conditional_value_at_risk = Performance.conditional_value_at_risk(x, y)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(conditional_value_at_risk, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(conditional_value_at_risk, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,z,expected",
        parameters["omega_params"]
    )
    def test_omega_ratio(self, x, y, z, expected):
        omega_ratio = Performance.omega_ratio(x, y, z)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(omega_ratio, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(omega_ratio, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,z,m,expected",
        parameters["tdc_params"]
    )
    def test_tail_dependence(self, x, y, z, m, expected):
        tail_dependence = Performance.tail_dependence(x, y, z, m)
        if isinstance(x, pd.DataFrame):
            assert_series_equal(tail_dependence, expected, atol = MAXERROR) 
        else:
            assert_almost_equal(tail_dependence, expected, MAXERROR)

    @pytest.mark.parametrize(
        "x,y,z,expected",
        parameters["TDC_params"]
    )
    def test_TDC(self, x, y, z, expected):
        tail_dependence = Performance.TDC(x, y, z)
        assert_almost_equal(tail_dependence, expected, MAXERROR)
