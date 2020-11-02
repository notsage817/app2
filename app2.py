

#use stock price of apple for plot

import numpy as np
import pandas as pd

import dash
import dash_auth
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import dash_bootstrap_components as dbc
import base64

import pymongo
import dns
import json
import plotly
from plotly.offline import plot
import random
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

origin = pd.read_csv('raw data.csv',intel_col = 0)

pre_nosdg = pd.read_csv('prediction_nosdg.csv')

pre_sdg = pd.read_csv('prediction_sdg.csv')

covid = pd.read_csv('covid.csv')

df_sdg = pd.read_csv('df_sdg.csv')

df_adj = pd.read_csv('df_adj.csv')

outlier_sdg = pd.read_csv('outlier_sdg.csv',intel_col = 0)

outlier_adj = pd.read_csv('outlier_adj.csv',intel_col = 0)


outlier_senti = pd.read_csv('outlier_senti.csv',intel_col = 0)


sdg_rank = pd.read_csv('sdg_rank.csv')

sdg_adj_rank = pd.read_csv('sdg_adj_rank.csv')

senti_rank = pd.read_csv('sentiment_rank.csv')

sdg_major = pd.read_csv('sdg_major.csv')

sdg_adj_major = pd.read_csv('sdg_adj_major.csv',intel_col = 0)


senti_major = pd.read_csv('senti_major.csv',intel_col = 0)


aapl = origin[origin['Ticker'] == 'AAPL'][['Timestamp','Adj. Close','Price to Book Value','P/E']]

p1 = go.Scatter(x = aapl['Timestamp'].values,y=aapl['Adj. Close'].values,name = 'Adjusted close price',mode = 'lines')

p2 = go.Scatter(x = aapl['Timestamp'].values,y=aapl['P/E'].values,name = 'P/E',mode = 'lines')

p3 = go.Scatter(x = aapl['Timestamp'].values,y=aapl['Price to Book Value'].values,name = 'P/BV',mode = 'lines')

fig1 = {'data':[p1,p2,p3],
            'layout':go.Layout(xaxis={'title':'date'},yaxis={'title':'stock price'},
                               title = 'adj price, P/B and P/E for AAPL', hovermode='closest')}

n_train = int(pre_nosdg.shape[0]*0.9)
ts = []
ts.append(go.Scatter(x=pre_nosdg['date'][:n_train].values,y=pre_nosdg['actual'][:n_train].values,name='Train',mode='lines'))
ts.append(go.Scatter(x=pre_nosdg['date'][n_train:].values,y=pre_nosdg['actual'][n_train:].values,name='Test',mode='lines'))
ts.append(go.Scatter(x=pre_nosdg['date'][n_train:].values,y=pre_nosdg['model'][n_train:].values,name='Forecasted',mode='lines'))

line_fig = {'data':ts,
            'layout':go.Layout(xaxis={'title':'date'},yaxis={'title':'stock price'},
                               title = 'LSTM model result', hovermode='closest')}

n_trains = int(pre_sdg.shape[0]*0.9)
tss = []
tss.append(go.Scatter(x=pre_sdg['date'][:n_trains].values,y=pre_sdg['actual'][:n_trains].values,name='Train',mode='lines'))
tss.append(go.Scatter(x=pre_sdg['date'][n_trains:].values,y=pre_sdg['actual'][n_trains:].values,name='Test',mode='lines'))
tss.append(go.Scatter(x=pre_sdg['date'][n_trains:].values,y=pre_sdg['model'][n_trains:].values,name='Forecasted',mode='lines'))

line_figs = {'data':tss,
            'layout':go.Layout(xaxis={'title':'date'},yaxis={'title':'stock price'},
                               title = 'LSTM model result (with SDG scores as risk factors)', hovermode='closest')}

major = ['FB','AMZN','WMT','XOM','GS','CVS','GE','AAPL','ECL','AMT','NEE']
industry_ticker = ['Communication Services','Consumer Discretionary','Consumer Staples','Energy','Financials',
                  'Health Care','Industrials','Information Technology','Materials','Real Estate','Utilities']
SDG_choice = ['SDG','SDG_adj','sentiment']

encoded_image1 = base64.b64encode(open('sdg_covid.png', 'rb').read())
sdg_covid = 'data:image/png;base64,{}'.format(encoded_image1.decode())

encoded_image2 = base64.b64encode(open('adj_covid.png', 'rb').read())
adj_covid = 'data:image/png;base64,{}'.format(encoded_image2.decode())

sdgmaj_fig = tools.make_subplots(rows=3, cols=4, subplot_titles=industry_ticker,
                                 specs=[[{'type': 'xy'}]*4]*3 )
for i in range(len(major)):
    sdgmaj_fig.append_trace(
        go.Scatter(
            name = major[i], mode = 'lines',
            x=sdg_major[sdg_major['Ticker']==major[i]]['Timestamp'].values,
            y=sdg_major[sdg_major['Ticker']==major[i]]['SDG_Mean_change'].values
        ),(i//4)+1, (i%4)+1
    )
sdgmaj_fig['layout'].update(title='major companies per sector',title_x=0.5)

# SDG_adj
sdgadjmaj_fig = tools.make_subplots(rows=3, cols=4, subplot_titles=industry_ticker,
                                 specs=[[{'type': 'xy'}]*4]*3 )
for i in range(len(major)):
    sdgadjmaj_fig.append_trace(
        go.Scatter(
            name = major[i], mode = 'lines',
            x=sdg_adj_major[sdg_adj_major['Ticker']==major[i]]['Timestamp'].values,
            y=sdg_adj_major[sdg_adj_major['Ticker']==major[i]]['SDG_Mean_change'].values
        ),(i//4)+1, (i%4)+1
    )
sdgadjmaj_fig['layout'].update(title='major companies per sector',title_x=0.5)

# Sentiment
stmmaj_fig = tools.make_subplots(rows=3, cols=4, subplot_titles=industry_ticker,specs=[[{'type': 'xy'}]*4]*3)
for i in range(len(major)):
    stmmaj_fig.append_trace(
        go.Scatter(
            name = major[i], mode = 'lines',
            x=senti_major[senti_major['Ticker']==major[i]]['Timestamp'].values,
            y=senti_major[senti_major['Ticker']==major[i]]['Sentiment_change'].values
        ),(i//4)+1, (i%4)+1
    )
stmmaj_fig['layout'].update(title='major companies per sector',title_x=0.5)

app = dash.Dash()
server = app.server()
app.layout = html.Div([
    html.Span(id="vir_span_view", style={"display": "none"}, children=0),
    html.H1(
        'Stock Analysis',
        style={
            'textAlign': 'center', 'color': '#333', 'font-size': '30px',
            'height': '20px', 'line-height': '30px', 'padding-top': "20px"
        }
    ),
    html.Div([
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='For Everybody', value='tab-1',
                    style={"height": 60,'font-size': '20px'},
                    selected_style={"height": 60,'font-size': '20px'}),
            dcc.Tab(label='Data Science', value='tab-2',
                    style={"height": 60,'font-size': '20px'},
                    selected_style={"height": 60,'font-size': '20px'}),
        ]),
    ],style={"height": "40px"}),
    html.Div(style=dict(clear="both")),
    html.Div(id='tabs-content')
],style={"padding": "0 40px", "background": "#F2F2F2"})

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab1_layout
    elif tab == 'tab-2':
        return tab2_layout
    
tab1_layout = html.Div([  
    
    html.Div([
        html.Div([html.H2('Part 1: Stock Fundamental Information')],
                 style = {'textAlign': 'center','padding-top': "25px"}),
    
    html.Div([
        html.Div([html.H3('stock price, P/E and P/B')],
                 style = {'width':'48%','display':'inline-block','margin-top':5})
        ]),
    html.Div([
        html.Div([
            
            html.Div([
            dash_table.DataTable(
                id='datatable-paging',
                columns=[{"name": i, "id": i} for i in list(origin.columns)],
                page_current=0, page_size=10, page_action='custom', fixed_columns={'headers': True,'data': 1},
                style_table={'height': '380px','overflowY': 'auto','width': '100%','minWidth': '100%'})   
        ], style = {'margin-top':5,'width':'80%','display':'inline-block',}),
            
            dcc.Graph(id='stock-fund-info',figure = fig1,
                            style={"height":"90%","width":"90%","padding-left":"5%"})],
                 style={"float": "left", "width": "80%",'display':'inline-block'})
        
    ]),

    # part2 time series and machine learning prediction
    html.Div([
    
    html.Div([
        html.Div([
         html.Div([html.H3('machine learning prediction result ')],
                 style = {'width':'48%','display':'inline-block','margin-top':20})
        ]),
        html.Div([dcc.Graph(id='stock-lstm',figure = line_fig,
                            style={"height":"90%","width":"90%","padding-left":"5%"})],
                 style={"float": "left", "width": "80%"}),
         ],style={'padding-top': "20px","padding-bottom":"10px","background": "#F2F2F2"}),
    
        html.Div(style=dict(clear="both")),
    
    # image and figure
        html.Div([html.Div([dcc.Graph(id='stock-lstm-sdg',figure = line_figs,
                            style={"height":"90%","width":"90%","padding-left":"5%"})],
                 style={"float": "left", "width": "80%"}),
              ],
             style={'padding-top': "20px","padding-bottom":"10px","background": "#F2F2F2"}),
    
    html.Div(style=dict(clear="both")),
    html.Div([html.Div([html.H3('')],style = {'width':'52%','display':'inline-block'})]) 
])])])
app.config['suppress_callback_exceptions'] = True
@app.callback(
    Output('datatable-paging', 'data'),
    [Input('datatable-paging', "page_current"),
     Input('datatable-paging', "page_size")])

def update_1(page_current,page_size):
    data = origin.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
    return data

tab2_layout = html.Div([ 
    
    html.Div([
        
        html.Div([
            html.Div([html.H3('Select  score type')],
                     style = {'width':'52%','display':'inline-block','margin-top':10})
        ]),
        
        html.Div([
            html.Div([
                dcc.Dropdown(id='ticker',options = [{'label':i,'value':i} for i in SDG_choice],value = 'SDG')
            ],
                style = {'width':'48%','display':'inline-block'}),
            html.Div([
                html.Button(id = 'submit_button',n_clicks = 1,children = 'Submit',
                            style = {'height':'40px','margin-left':80})
            ],style = {'width':'48%','display':'inline-block','float':'right'})
    ],style={'padding-top': "10px","background": "#F2F2F2"})
        ]),


    # phase1
    html.Div([
        html.Div([html.H2('Phase 1: COVID-19 Timeline and SDG Scores Change at Key Dates')],
                 style = {'textAlign': 'center','padding-top': "25px"})
        ]),
    
    # COVID-19 timeline table and SDG change table
    # title
    html.Div([
        html.Div([html.H3('COVID-19 key dates and events')],
                 style = {'width':'52%','display':'inline-block','margin-top':5}),
        html.Div([html.H3('SDG scores change at COVID-19 key dates')],
                 style = {'width':'48%','display':'inline-block','margin-top':5})
        ]),
    
    html.Div([
            html.Div([dash_table.DataTable(
                id='covid-timeline-table',
                columns=[{"name": i, "id": i} for i in list(covid.columns)],
                page_current=0, page_size=10, page_action='custom', fixed_columns={'headers': True,'data': 1},
                style_table={'height': '360px','overflowY': 'auto','width': '100%','minWidth': '100%'})   
                     ],style={"float": "left", "width": "45%"}),
        
            html.Div([dash_table.DataTable(
                id='SDG-change-table',
                page_current=0, page_size=10, page_action='custom', fixed_columns={'headers': True,'data': 1},
                style_table={'height': '360px','overflowY': 'auto','padding-right':"5%",
                             'width': '100%','minWidth': '100%'})
                     ],style={"float": "right", "width": "50%"}),
        
        ],style={'padding-top': "5px","padding-bottom":"10px","background": "#F2F2F2"}),
    html.Div(style=dict(clear="both")),
    
    
    
    # phase2
    html.Div([
        html.Div([html.H2('Phase 2: Identify Outlier, Rank Stocks and Plot Major Companies')],
                 style = {'textAlign': 'center','padding-top': "25px"})
        ]),
    
    html.Div([
        html.Div([html.H3('Identify dates for SDG score outlier')],
                 style = {'width':'52%','display':'inline-block','margin-top':5}),
        html.Div([html.H3('Top 10 stocks ranked by SDG score change at COVID-19 key dates')],
                 style = {'width':'48%','display':'inline-block','margin-top':5})
        ]),
    
    html.Div([
            html.Div([dash_table.DataTable(
                id='SDG-outlier-table',
                page_current=0, page_size=16, page_action='custom', fixed_rows={'headers': True,'data': 0},
                style_table={'height': '400px','overflowY': 'auto','overflowX': 'scroll','width': '100%','minWidth': '100%'})  
                     ],style={"float": "left", "width": "45%"}),
        
            html.Div([dash_table.DataTable(
                id='top10-stock-table',
                page_current=0, page_size=16, page_action='custom', fixed_rows={'headers': True,'data': 0},
                style_table={'height': '400px','overflowY': 'auto','padding-right':"5%",
                             'width': '100%','minWidth': '100%'})
                     ],style={"float": "right", "width": "50%"}),
        
        ],style={'padding-top': "5px","padding-bottom":"10px","background": "#F2F2F2"}),
    html.Div(style=dict(clear="both")),
    
    
    # show graphs of major companies per sector
    # title
    html.Div([
        html.Div([html.H3('show graphs of major companies per sector')],
                 style = {'width':'52%','display':'inline-block'})
        ]),
    # figure
    html.Div([
        html.Div([dcc.Graph(id='major-company-fig')])
    ],style={'padding-top': "20px","background": "#F2F2F2"}),

    html.Div([html.Span(id="vir_span2", style={"display": "none"})
             ],style={'padding-top': "20px","background": "#F2F2F2"})
    
])

#---------------------------------------- tab2 callback function ----------------------------------------
@app.callback(
    Output('covid-timeline-table', 'data'),
    [Input('covid-timeline-table', "page_current"),
     Input('covid-timeline-table', "page_size")])
def update_2(page_current,page_size):
    return covid.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')

@app.callback(
    [Output('SDG-change-table', 'data'),
    Output('SDG-change-table','columns')],
    [Input('SDG-change-table', "page_current"),
     Input('SDG-change-table', "page_size"),
     Input('submit_button','n_clicks')],
    [State('ticker','value')])
def update_sdg_chg_table(page_current,page_size,n_clicks,ticker):
    # sentiment.csv does not include per sector data, so if ticker==sentiment, it make no sense
    if ticker == 'SDG':
        data = df_sdg.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_sdg.columns)]
    else:
        data = df_adj.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_adj.columns)]
    return data,columns


@app.callback(
    [Output('SDG-outlier-table', 'data'),
     Output('SDG-outlier-table', 'columns')],
    [Input('SDG-outlier-table', "page_current"),
     Input('SDG-outlier-table', "page_size"),
     Input('submit_button','n_clicks')],
    [State('ticker','value')])
def update_outlier_table(page_current,page_size,n_clicks,ticker):
    
    if ticker == 'SDG':
        data = outlier_sdg.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(outlier_sdg.columns)]
    
    elif ticker == 'SDG_adj':
        data = outlier_adj.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(outlier_adj.columns)]
    
    else:
        data = outlier_senti.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(outlier_senti.columns)]
    return data, columns

@app.callback(
    [Output('top10-stock-table', 'data'),
     Output('top10-stock-table','columns')],
    [Input('top10-stock-table', "page_current"),
     Input('top10-stock-table', "page_size"),
     Input('submit_button','n_clicks')],
    [State('ticker','value')])
def update_top10_table(page_current,page_size,n_clicks,ticker):
    if ticker == 'SDG':
        data = sdg_rank.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(sdg_rank.columns)]
    elif ticker == 'SDG_adj':
        data = sdg_adj_rank.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(sdg_adj_rank.columns)]
    else:
        data = senti_rank.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(senti_rank.columns)]
    return data, columns

@app.callback(
    Output('major-company-fig','figure'),
    [Input('submit_button','n_clicks')],
    [State('ticker','value')])
def update_maj_company_fig(n_clicks,ticker):
    if ticker == 'SDG':
        return sdgmaj_fig
    elif ticker == 'SDG_adj':
        return sdgadjmaj_fig
    else:
       return stmmaj_fig


if __name__ == '__main__':
    app.run_server(debug=True)

