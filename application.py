import dash
import dash_core_components as dcc
from dash_core_components.Graph import Graph
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objs as go
from datetime import datetime
import pandas as pd
import yfinance as yf
import import_funcs as be

app = dash.Dash(__name__)
server=app.server

tickers = pd.read_csv('tickers.csv')
tickers.set_index('Ticker', inplace=True)

my_tickers = []

for tic in tickers.index:
    #{'label': 'user sees', 'value': 'script sees'}
    mydict = {}
    mydict['label'] = tic #Apple Co. AAPL
    mydict['value'] = tic
    my_tickers.append(mydict)

options1 = [{'label': 'LR', 'value': 'LR'},
 {'label': 'SVR', 'value': 'SVR'},
 {'label': 'KNN', 'value': 'KNN'},
 {'label': 'XGB', 'value': 'XGB'},
 {'label': 'ARIMA', 'value': 'ARIMA'}]

options2 = [{'label': 'True', 'value': 'True'}, {'label': 'False', 'value': 'False'}]

hyperparameter_ids = {
    "LR": ["LR1", "LR2"],
    "ARIMA": ["ARIMA1", "ARIMA2", "ARIMA3"],
    "KNN": ["KNN1", "KNN2", "KNN3"],
    "SVR": ["SVR1", "SVR2", "SVR3"],
    "XGB": ["XGB1", "XGB2", "XGB3"]
}



app.layout = html.Div([
            html.H1('Algorithm Comparison'),
            dcc.Markdown(''' --- '''), 
            html.H2('Portfolio DashBoard'),
            html.Div([html.H3('Enter a stock symbol:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                      id='my_ticker_symbol',
                      options = my_tickers,
                      value = 'FB',
                      multi = False
                      # style={'fontSize': 24, 'width': 75}
            )

            ], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
            html.Br(),
            
            html.Div([html.H3('Enter Start / End Date:'),
                dcc.DatePickerRange(id='my_date_picker',
                                    min_date_allowed = datetime(2015,1,1),
                                    max_date_allowed = datetime.today(),
                                    start_date = datetime(2018, 1, 1),
                                    end_date = datetime.today()
                )

            ], style={'display':'inline-block'}), 
            html.Br(),
            html.Br(),

            html.Div([
                html.Button(id='submit-button',
                            n_clicks = 0,
                            children = 'Submit',
                            style = {'fontSize': 16, 'marginLeft': '30px'}
                           )
            ], style={'display': 'inline-block'}),
            html.Br(),

            html.Div([
                dcc.Checklist(
                    id='toggle-rangeslider',
                    options=[{'label': 'Include Rangeslider', 
                              'value': 'slider'}],
                    value=['slider']
                ),
            ]),

            html.Div(id='graph'),
            html.Br(),

    dcc.Markdown(''' --- '''), 
    html.H1('Algorithm DashBoard'),

    html.Div([html.H3('Choose an Evaluation Metric:', style={'paddingRight': '30px'}),
        dcc.Dropdown(
                  id='my_metric',
                  options = [{'label': 'RMSE', 'value':'RMSE'}, {'label': 'MAPE', 'value':'MAPE'}],
                  value = '',
                  multi = False)],
                 style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
    html.Br(), 

    html.Div([html.H3('Choose the First Algorithm:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                id='my_algo_1',
                options = options1,
                value = '', 
                multi = False)
                     ],style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
    html.Br(),

    html.Div(id='hyper_params_1'),
    html.Br(),

    html.Div([html.H3('Choose the Second Algorithm:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                id='my_algo_2',
                options = options1,
                value = 'Pick', 
                multi = False)
                     ],style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
    html.Br(),

    html.Div(id='hyper_params_2'),
    html.Br(),

    html.Div([
    html.Button(id='submit-button-2',
                n_clicks = 0,
                children = 'Submit',
                style = {'fontSize': 16, 'marginLeft': '30px'})], 
                style={'display': 'inline-block'}),
    html.Br(),

    dcc.Graph(id='plot-1'),
    html.Br(),

    dcc.Graph(id='plot-2'),
    html.Br(),

    dcc.Graph(id='plot-3'),
    html.Br()
])

@app.callback(Output('graph', 'children'),
              Input('submit-button', 'n_clicks'),
              Input("toggle-rangeslider", "value"),
              State('my_ticker_symbol', 'value'),
              State('my_date_picker', 'start_date'),
              State('my_date_picker', 'end_date')
            )

def update_graph(n_clicks, value, stock_ticker, start_date, end_date):
    start = datetime.strptime(start_date[:10], '%Y-%m-%d')
    end = datetime.strptime(end_date[:10], '%Y-%m-%d')
    stock_new = yf.download(stock_ticker, start=start, end=end, progress=False)
    stock_sp = yf.download('^GSPC',start=start, end=end, progress=False)
    stock_sp['pct_change'] = (stock_sp['Close'] - stock_sp['Open']) / stock_sp['Open'] * 100.0
    stock_new['pct_change'] = float
    stock_new['pct_change'] = (stock_new['Close'] - stock_new['Open']) / stock_new['Open'] * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_sp.index,
        y=stock_sp['pct_change'],
        name='S&P 500'
    ))
    fig.add_trace(go.Scatter(
        x=stock_new.index,
        y=stock_new['pct_change'],
        name=stock_ticker
    ))
    fig.update_layout(
    xaxis_rangeslider_visible='slider' in value)
    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(
            x=stock_sp.index,
            open=stock_sp['Open'],
            high=stock_sp['High'],
            low=stock_sp['Low'],
            close=stock_sp['Close'],
            name='S&P 500'
            ))
    fig1.add_trace(go.Candlestick(
            x=stock_new.index,
            open=stock_new['Open'],
            high=stock_new['High'],
            low=stock_new['Low'],
            close=stock_new['Close'],
            name=stock_ticker
            ))
    
    fig1.update_layout(
    xaxis_rangeslider_visible='slider' in value)
    
    return html.Div([
        dcc.Graph(id='graph1',figure=fig),
        dcc.Graph(id='graph2',figure=fig1)
    ])


@app.callback(
    Output('hyper_params_1', 'children'),
    Input('my_algo_1', 'value')) 

def get_hyperparameters(algo):
    if(algo =='LR'):
        return html.Div([
            html.H3('fit intercept:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                      id= {'type': 'ALGO_1', 'index': 1},
                      options = [{'label':'True','value':'True'},{'label':'False','value':'False'}],
                      multi = False),
            html.H3('y norm:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                      id={'type': 'ALGO_1', 'index': 2},
                      options = [{'label':'True','value':'True'},{'label':'False','value':'False'}], 
                      multi = False),
            
        ])    
    elif algo=='ARIMA':
        return html.Div([
            html.H3('P Value:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_1', 'index': 1},
                      placeholder='p value'
                      ),
            html.H3('D Value:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_1', 'index': 2},
                      placeholder='d value'
                      ),
            html.H3('q Value:', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_1', 'index': 3},
                  placeholder='q value'
                  )
        ])
    elif algo=='KNN':
        return html.Div([
            html.H3('P Value:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_1', 'index': 1},
                      placeholder='p value'
                      ),
            html.H3('Weights:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                      id={'type': 'ALGO_1', 'index': 2},
                      placeholder='Weights',
                      options = [{'label':'uniform','value':'uniform'},{'label':'distance','value':'distance'}], 
                      multi = False
                      ),
            html.H3('Number of Neighbours:', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_1', 'index': 3},
                  placeholder='Number of Neighbours'
                  )
        ])
    elif algo=='SVR':
        return html.Div([
            html.H3('Kernel:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                      id={'type': 'ALGO_1', 'index': 1},
                      options = [{'label':'linear','value':"linear"},{'label':'poly','value':'poly'},
                                {'label':'rbf','value':"rbf"},{'label':'sigmoid','value':"sigmoid"},
                                {'label':'precomputed','value':"precomputed"}], 
                      multi = False)
                      ,
            html.H3('C value:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_1', 'index': 2},
                      placeholder='C value'
                      ),
            html.H3('Epsilon', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_1', 'index': 3},
                  placeholder='Epsilon'
                  )
        ])
    elif algo=='XGB':
        return html.Div([
            
            html.H3('Maximum Depth:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_1', 'index': 1},
                      placeholder='Maximum Dept'
                      ),
            html.H3('Learning rate:', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_1', 'index': 2},
                  placeholder='Learning rate'
                  ),
            html.H3('Minimun Child Weight', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_1', 'index': 3},
                  placeholder='Minimun child weight'
                  ),
        ])


@app.callback(
    Output('hyper_params_2', 'children'),
    Input('my_algo_2', 'value')) 

def get_hyperparameters(algo):
    if(algo =='LR'):
        return html.Div([
            html.H3('fit intercept:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                      id= {'type': 'ALGO_2', 'index': 1},
                      options = [{'label':'True','value':'True'},{'label':'False','value':'False'}],
                      multi = False),
            html.H3('y norm:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                      id={'type': 'ALGO_2', 'index': 2},
                      options = [{'label':'True','value':'True'},{'label':'False','value':'False'}], 
                      multi = False),
            
        ])    
    elif algo=='ARIMA':
        return html.Div([
            html.H3('P Value:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_2', 'index': 1},
                      placeholder='p value'
                      ),
            html.H3('D Value:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_2', 'index': 2},
                      placeholder='d value'
                      ),
            html.H3('q Value:', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_2', 'index': 3},
                  placeholder='q value'
                  )
        ])
    elif algo=='KNN':
        return html.Div([
            html.H3('P Value:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_2', 'index': 1},
                      placeholder='p value'
                      ),
            html.H3('Weights:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                      id={'type': 'ALGO_2', 'index': 2},
                      placeholder='Weights',
                      options = [{'label':'uniform','value':'uniform'},{'label':'distance','value':'distance'}], 
                      multi = False
                      ),
            html.H3('Number of Neighbours:', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_2', 'index': 3},
                  placeholder='Number of Neighbours'
                  )
        ])
    elif algo=='SVR':
        return html.Div([
            html.H3('Kernel:', style={'paddingRight': '30px'}),
            dcc.Dropdown(
                      id={'type': 'ALGO_2', 'index': 1},
                      options = [{'label':'linear','value':"linear"},{'label':'poly','value':'poly'},
                                {'label':'rbf','value':"rbf"},{'label':'sigmoid','value':"sigmoid"},
                                {'label':'precomputed','value':"precomputed"}],
                      multi = False)
                      ,
            html.H3('C value:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_2', 'index': 2},
                      placeholder='C value'
                      ),
            html.H3('Epsilon', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_2', 'index': 3},
                  placeholder='Epsilon'
                  )
        ])
    elif algo=='XGB':
        return html.Div([
            
            html.H3('Maximum Depth:', style={'paddingRight': '30px'}),
            dcc.Input(
                      id={'type': 'ALGO_2', 'index': 1},
                      placeholder='Maximum Dept'
                      ),
            html.H3('Learning rate:', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_2', 'index': 2},
                  placeholder='Learning rate'
                  ),
            html.H3('Minimun Child Weight', style={'paddingRight': '30px'}),
            dcc.Input(
                  id={'type': 'ALGO_2', 'index': 3},
                  placeholder='Minimun child weight'
                  ),
        ])


@app.callback(
    Output('plot-1', 'figure'),
    Output('plot-2', 'figure'),
    Output('plot-3', 'figure'),
    Input('submit-button-2', 'n_clicks'),
    State('my_metric', 'value'),
    State('my_algo_1', 'value'),
    State('my_algo_2', 'value'),
    State({'type': 'ALGO_1', 'index': ALL}, 'value'),
    State({'type': 'ALGO_2', 'index': ALL}, 'value'),
    State('my_ticker_symbol', 'value'),
    State('my_date_picker', 'start_date'),
    State('my_date_picker', 'end_date'), prevent_initial_call=True)

def update_output_div(n_clicks, metric, algo_1, algo_2, values1, values2, ticker, start, end):
    start = datetime.strptime(start[:10], '%Y-%m-%d')
    end = datetime.strptime(end[:10], '%Y-%m-%d')
    stock_data = yf.download(ticker, start=start, end=end, progress=False)

    return (be.get_output(algo_1, algo_2, metric, stock_data, values1, values2))

if __name__ == '__main__':
    app.run_server(debug=True)