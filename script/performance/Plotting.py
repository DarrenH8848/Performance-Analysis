import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from .Performance import cumulative_returns, drawdown, annualized_return


class Plotting(object):

    @classmethod
    def plot_cum_returns(cls, returns):
        """
        plot cumulative returns for single asset or multiple assets

        Args:
            returns (pd.Series, pd.DataFrame): periodical returns
        """        
        cum_returns = cumulative_returns(returns)
        if isinstance(cum_returns, pd.DataFrame):
            fig = px.line(cum_returns.reset_index(), x='Date', y=cum_returns.columns,
                title='cumulative return')
            fig.show()

        else:
            fig = px.line(cum_returns.reset_index(), x='Date', y=cum_returns.name,
                title='cumulative return')
            fig.show()

    @classmethod
    def plot_drawdown(cls, returns):
        """
        plot drawdown for single asset or multiple assets

        Args:
            returns (pd.Series, pd.DataFrame): periodical returns
        """  
        dd = drawdown(returns)
        if isinstance(dd, pd.DataFrame):
            fig = px.line(dd.reset_index(), x='Date', y=dd.columns,
                title='drawdown')
            fig.show()

        else:
            fig = px.line(dd.reset_index(), x='Date', y=dd.name,
                title='drawdown')
            fig.show()


    @classmethod
    def plot_cum_return_and_drawdown(cls, returns):
        """
        plot cumulative returns and drawdown for single asset

        Args:
            returns (pd.Series): single asset periodical return
        """        
        cum_returns = cumulative_returns(returns)
        dd = drawdown(returns)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=cum_returns.index, y=cum_returns, name = 'cumulative return'),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=dd.index, y=dd, name="drawdown"),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="cumulative return and drawdown"
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Date")

        # Set y-axes titles
        fig.update_yaxes(title_text="cumulative return", secondary_y=False)
        fig.update_yaxes(title_text="drawdown", secondary_y=True)

        fig.show()


    @classmethod
    def plot_yearly_return_heatmap(cls, returns, show_text = False):
        """
        plot yearly return heatmao for multiple assets

        Args:
            returns (pd.DataFrame): multiple asset returns
        """        

        temp_data = returns.groupby([returns.index.year]).apply(annualized_return).round(3)
        fig = px.imshow(temp_data,
                labels=dict(x="asset type", y="year", color="annualized return"),
                x=temp_data.columns.to_list(),
                y=temp_data.index.to_list(),
                text_auto=show_text
               )

        fig.update_layout(
            title_text="yearly return heatmap"
        )

        fig.show()
    

    @classmethod
    def plot_monthly_return_heatmap(cls, returns, show_text = False, modify_return = True):
        '''
        plot monthly return heatmap for single asset

        Args:
            returns (pd.Series): single asset return

        '''
        cum_returns = cumulative_returns(returns)
        max_day = cum_returns.index.max()
        end = pd.Timestamp('{}-{}'.format(max_day.year, max_day.month + 1)) - pd.Timedelta('1d')
        date = pd.DataFrame(pd.date_range(start=cum_returns.index.min(), end=end), columns=['Date'])
        data = pd.merge(date, cum_returns.reset_index(), on='Date', how='outer').sort_values(by='Date', ascending=True).fillna(method='ffill')
        data['Date'] = data['Date'].apply(lambda x: pd.Timestamp(x))
        data = data.set_index('Date')
        df = data.resample('1M').asfreq()
        monthly_return = df.iloc[:, 0].pct_change().round(3)

        new_index_year = []
        new_index_month = []
        for date in monthly_return.index:
            new_index_year.append(date.year)
            new_index_month.append(date.month)
    
        monthly_return.index = pd.MultiIndex.from_arrays(
            [new_index_year, new_index_month],
            names=["year", "month"])
        
        temp_data = monthly_return.unstack()
        if modify_return:
            temp_data.mask(temp_data > 1, 1, inplace = True)

        fig = px.imshow(temp_data,
                labels=dict(x="month", y="year", color="monthly return"),
                x=temp_data.columns.to_list(),
                y=temp_data.index.to_list(),
                text_auto=show_text, aspect="auto",
               )

        fig.update_layout(
            title_text="monthly return heatmap"
        )

        fig.show()


    @classmethod
    def plot_seasonal_effect(cls, returns):
        '''
        plot seasonal effect, a bar chart showing returns in different months

        Args:
            returns (pd.Series): single asset return

        '''
        cum_returns = cumulative_returns(returns)
        date =  pd.DataFrame(pd.date_range(start=cum_returns.index.min(), end=cum_returns.index.max()), columns=['Date'])
        data = pd.merge(date, cum_returns.reset_index(), on='Date', how='outer').sort_values(by='Date', ascending=True).fillna(method='ffill')
        data['Date'] = data['Date'].apply(lambda x: pd.Timestamp(x))
        data = data.set_index('Date')
        df = data.resample('1M').asfreq()
        df['monthly_return'] = df.iloc[:, 0].pct_change()
        df = df.reset_index()
        df.loc[:, 'month'] = df.loc[:, 'Date'].apply(lambda x: x.month)
        df_median = df.loc[:, ['monthly_return', 'month']].groupby('month').median()

        fig = px.bar(df_median, 
                    x=df_median.index, 
                    y="monthly_return", 
                    color=df_median.index, 
                    title="seasonal effect")
        fig.show()