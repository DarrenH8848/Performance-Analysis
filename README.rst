performance-analysis
====================


Features
--------

* Performance indicators calculation
* Supports single time series data and multiple time series data with different time spans
* Time series visualization


Installation
------------

You can install "performance-analysis" via 'pip'_ from 'PyPI'_::

    $ pip install performance-analysis


Usage
-----

* Performance part

.. code-block:: python

    from performance_analysis.Performance import Performance
    # Input return data
    raw_return_data = pd.read_csv("./raw_return_data.csv")
    # Just some examples. For more functions, you can explore the package
    ann_rtn = Performance.annualized_return(raw_return_data, period = Constant.DAILY, logreturn = False)
    var = Performance.value_at_risk(raw_return_data, significance_level = 0.05)
    sharpe = Performance.sharpe_ratio(raw_return_data, risk_free = 0., logreturn = False)
    calmar = Performance.calmar_ratio(raw_return_data, period = Constant.DAILY, logreturn = False)

* Computes personal specified indicators

.. code-block:: python

    '''
    indicators = {
            0 : annualized_return,
            1 : annualized_sd,
            2 : max_drawdown,
            3 : sharpe_ratio,
            4 : calmar_ratio,
            5 : burke_ratio,
            6 : downside_risk,
            7 : sortino_ratio,
            8 : tracking_error,
            9 : information_ratio,
            10 : capm_beta,
            11 : capm_alpha,
            12 : treynor_ratio,
            13 : skewness,
            14 : kurtosis,
            15 : value_at_risk,
            16 : conditional_value_at_risk,
            17 : omega_ratio,
            18 : tail_dependence,
            19 : TDC,
        }
    '''

    args = (0,1,2,3,4)
    kwargs = {
        "annualized_return" : {"returns" : single_return_data},
        "annualized_sd" : {"returns" : single_return_data},
        "max_drawdown" : {"returns" : single_return_data},
        "sharpe_ratio" : {"returns" : single_return_data},
        "calmar_ratio" : {"returns" : single_return_data}
    }
    performance = Performance.performance_dashboard(*args, **kwargs)

* Plotting part

.. code-block:: python

    from performance_analysis.Plotting import Plotting
    # read data, set index and convert to datatime
    single_return_data = pd.read_csv("./single_return_data.csv")
    single_return_data.set_index(['Date'],inplace=True)
    single_return_data.index = pd.to_datetime(single_return_data.index, format='%Y%m%d', errors='coerce')

    Plotting.plot_cum_return_and_drawdown(single_return_data)
    Plotting.plot_monthly_return_heatmap(single_return_data, show_text = True)
    Plotting.plot_seasonal_effect(single_return_data)


License
-------

Distributed under the terms of the 'MIT'_ license, "performance-analysis" is free and open source software
