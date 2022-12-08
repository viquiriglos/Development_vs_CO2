import time
from dash import Dash
import plotly.express as px
from jupyter_dash import JupyterDash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import dash
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import string

app = dash.Dash(external_stylesheets=[dbc.themes.SOLAR])

# Incorporate data into App
anios=['1990', '1995', '2000', '2005', '2010', '2015', '2019']
Global_CO2_abs =[20344510.0, 21360120.0, 23129880.0, 27045550.0, 30625880.0, 32551680.0, 33882010.0] #la suma de todo el CO2
Global_CO2_abs_mean=[109970.3, 112421.7, 121736.2, 141599.7, 160344.9, 170427.6, 177392.7] #kt, el promedio: (suma CO2)/(paises)
Global_CO2_abs_mean_scaled=[10.99703, 11.24217, 12.17362, 14.15997, 16.03449, 17.04276, 17.73927] #en t, es lo anterior dividido por 10000 para ajustar la escala
Global_CO2_cap_mean=[4.4, 4.1, 4.2, 4.5, 4.4, 4.2, 4.1] #metric tons per capita, es el promedio: (suma CO2_cap)/(paises)
Global_CO2_cap_prom=[3.9, 3.8, 3.8, 4.2, 4.4, 4.4, 4.4] #este es el que vale, es la suma de todo el CO2 dividido la poblacion mundial

Global_GDP=[4273.7, 5405.8, 5494.0, 7300.6, 9584.3, 10156.8, 11320.5] #per capita nuevo, aqui obtuve el GDP abs en cada pais, sume todo y lo dividi por la poblacion mundial
Global_GDP_abs=[22469937580111.3, 30722822543926.3, 33460099083318.8, 47360314077575.9, 66100758771199.1, 74336847498721.0, 86655914798562.8] #nuevo

Global_population=[5257781390.0, 5683323062.0, 6090303076.0, 6487144835.0, 6896780812.0, 7318921042.0, 7654757154.0]

AE_median=[100.0, 100.0, 97.0, 97.8, 98.8, 99.6, 100.0]

df = pd.read_csv('CO2_GDP_AE_CO2_abs.csv')
paises=df['Country Name'].unique()
opciones=[]

for i in range(len(paises)):
    opcion_i={"label": paises[i], "value": paises[i]}
    opciones.append(opcion_i)

# Load data for top 10 lowCO2 and highGDP
topCO2_every_year = pd.read_csv('Top10_CO2_cap_all_years.csv')
topGDP_every_year = pd.read_csv('Top10_GDP_cap_all_years.csv')

#Build figure in introduction (not interactive)
fig0 = make_subplots(specs=[[{"secondary_y": True}]])
fig0.add_trace(go.Scatter(x=anios, y=Global_CO2_abs, name='Global CO2 emissions (kt)', mode="lines"), secondary_y=True)
fig0.add_trace(go.Scatter(x=anios, y=Global_GDP, name="Global GDP per capita (U$S)", mode="lines"), secondary_y=False)
fig0.update_xaxes(title_text="years")
fig0.update_yaxes(title_text="CO2_cap", secondary_y=False)
fig0.update_yaxes(title_text="GDP_cap", secondary_y=True)

#Especial Characters:
def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)

diox='CO{}'.format(get_sub('2'))

def get_super(x):
    normal = "0123456789" #ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

diez_7='10{}'.format(get_super('7'))

# Build the layout to define what will be displayed on the page
app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
       
        dbc.Row([
            dbc.Col(html.H1("Would we survive our Development?"), width=6),

            dbc.Col([
                dcc.Dropdown(id="slct_country",
                    options=opciones,
                    placeholder="Select a Country",
                    multi=False,
                    value=1990,
                )],
            width=2),

            dbc.Col([
                dcc.Dropdown(id="slct_year",
                    options=[
                        {"label": "1990", "value": 1990},
                        {"label": "1995", "value": 1995},
                        {"label": "2000", "value": 2000},
                        {"label": "2005", "value": 2005},
                        {"label": "2010", "value": 2010},
                        {"label": "2015", "value": 2015},
                        {"label": "2019", "value": 2019}],
                    placeholder="Select a year",
                    multi=False,
                    #value=1990,
                )], 
            width=2),

            dbc.Col(
                dbc.Button(
                    "Generate graphs",
                    color="primary",
                    id="button",
                    className="mb-3"),
            width=2),
            ]),

        dbc.Tabs(
            [
                dbc.Tab(label="Introduction", tab_id="intro"),
                dbc.Tab(label="by Country", tab_id="by_country"),
                dbc.Tab(label="Low CO2", tab_id="lowCO2"),
                dbc.Tab(label="High GDP", tab_id="highGDP"),
                dbc.Tab(label="Conclusions", tab_id="last_graph"),
            ],
            id="tabs",
            active_tab="intro",
        ),
        html.Div(id="tab-content", className="p-4"),
    ]
)

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), 
    Input("store", "data")],
)

def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        if active_tab == "intro":
            return dbc.Container(
                [
                
                html.H3("1. Introduction", style={'textAlign': 'left'}),
                html.Hr(),
                dbc.Row([
                    dbc.Row([
                        dbc.Col([                       
                        html.H5("""Is it possible to keep our current way of life without irreversibly damaging the planet? 
                        Are our efforts to reduce green house gases paying off? 

                        To answer these questions I have gathered some time series data from World Bank Data Base 
                        (https://data.worldbank.org/indicator/?tab=all). 

                        We will interactively compare the time evolution of two development indicators 
                        ('Gross Domestic Product per capita' or GDP/cap, and 'Access to Electricity or AE)
                        with the {} emissions in each country. GDP/cap unit is US dolar, AE is presented as
                        a percentage of the total population, {} emissions when speaking about the absolute value
                        will be in metric kt, and {} emissions per capita (that is the emissions produced in a certain
                        country divided by the population of that country) will be in metric tons, t.
                        """.format(diox, diox, diox),
                        style={'textAlign': 'justify'}),

                        html.H6("""Note: The graphics are interactive, this means that we can zoom in, zoom out, select a region, autoscale or
                        even download the image. In the graphics in Section 2, the {} and {} per capita data sets are accompanied by the global mean evolution of
                        these indicators. In order to plot the total {} together with the {}/cap, the {} emissions are divided by a
                        factor of {}.""".format(diox, diox, diox, diox, diox, diez_7), style={'textAlign': 'justify'})

                        ], width=6),

                        dbc.Col(dcc.Graph(figure=fig0, style={"height": "55vh"}), width=6),

                    ])
                    
                ]),

            # dbc.Row([
            #     dbc.Col([
            #         html.H6("""Note: The graphics are interactive, this means that we can zoom in, zoom out, select a region, autoscale or
            #             even download the image. In the graphics in Section 2, the {} and {} per capita data sets are accompanied by the global mean evolution of
            #             these indicators. In order to plot the total {} together with the {}/cap, the {} emissions are divided by a
            #             factor of {}.""".format(diox, diox, diox, diox, diox, diez_7), style={'textAlign': 'justify'})
            #     ], width=12),
            # ]),
            ],#fluid=True,
            )     

        elif active_tab == "by_country":
             return dbc.Container(
                [
                    dbc.Row([
                        dbc.Col([html.H3("""2. Emissions and GDP per country""", style={'textAlign': 'left'})], width=8),
                        dbc.Col([html.H3("""""", style={'textAlign': 'center'})], width=1)
                    ]),
                    #html.Hr(),

                    dbc.Row([

                        dbc.Col([
                            html.H5(
                            """The idea here is to 'zoom in' inside of every country's situation and analyze if it is possible to achieve 
                            good development indicators without harming the environment, despite what the general trend might indicate.

                            Is it possible to have high GDP/cap without having high {} emissions? Or do these indicators always evolve
                            hand by hand?

                            In case of finding a country with a high GDP/cap and low {} emissions, does it really mean that we are
                            dealing with a developed country? Is the Access to Electricity high or low?
                            
                            Let's choose a country, and 'have fun!...' """.format(diox, diox),
                    style={'textAlign': 'justify'})], width=12),
                    ]),

                    dbc.Row([
                        dbc.Col([html.H3("""Environment""", style={'textAlign': 'center'})], width=6),
                        dbc.Col([html.H3("""Development """, style={'textAlign': 'center'})], width=6),
                    ]),

                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=data["fig_1"], style={"height": "55vh"}), width=6),
                        dbc.Col(dcc.Graph(figure=data["fig_2"], style={"height": "55vh"}), width=6),
                    ])

            ])
            
            

        elif active_tab == "lowCO2":
            return dbc.Container(
                [
                    dbc.Row([
                        dbc.Col([html.H3("""3. Countrys with the lowest emissions""", style={'textAlign': 'left'})], width=8)   
                    ]),
                    #html.Hr(),

                    dbc.Row([
                        dbc.Col([
                            html.H5("""In this section, we will analyze for each selected year how were the development indicators in those
                                    countries that present lower {} emissions and observe how were the development indicators in these cases.""".format(diox),
                            style={'textAlign': 'justify'})
                        ], width=12),               
                    ]),

                    dbc.Row([
                        dbc.Col([html.H3("""Environment""", style={'textAlign': 'center'})], width=6),
                        dbc.Col([html.H3("""Development """, style={'textAlign': 'center'})], width=6),
                    ]),

                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=data["fig_4"], style={"height": "60vh"}), width=6),
                        dbc.Col(dcc.Graph(figure=data["fig_5"], style={"height": "60vh"}), width=6)
                    ])
                ]  
            )
            
        elif active_tab == "highGDP":
            return dbc.Container(
                [
                    dbc.Row([
                        dbc.Col([html.H3("""4. Countrys with the highest GDP per capita""", style={'textAlign': 'left'})], width=8)    
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H5("""In the same way as in the previous section, here we will observe the emissions and AE for those countries
                            having a high GDP/cap. It's worth noting that some of these wealthy countries don't report their emissions.""", style={'textAlign': 'justify'})
                        ], width=12),
                    ]),

                    dbc.Row([
                        dbc.Col([html.H3("""Environment""", style={'textAlign': 'center'})], width=6),
                        dbc.Col([html.H3("""Development """, style={'textAlign': 'center'})], width=6),
                    ]),
            
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=data["fig_6"], style={"height": "60vh"}), width=6),
                        dbc.Col(dcc.Graph(figure=data["fig_7"], style={"height": "60vh"}), width=6),
                    ])
                ]
            )
            
            
        elif active_tab == "last_graph":
            return dbc.Container(
                [
                    dbc.Row([
                        dbc.Col([html.H3("""5. Final Remarks""", style={'textAlign': 'left'})], width=5),
                    ]),
                
                    #html.Hr(),

                    dbc.Row([
                        dbc.Row([
                            dbc.Col([
                                html.H5("""So, are we completely lost? Do we have to get back to the Stone Age?""", style={'textAlign': 'justify'}),
                                html.H5("""Do not panic! Eventhough, for most of the countries {} emissions seem to grow with development
                                        it is possible to find an equilibrium without detriment to our welfare.

                                        As shown in the bar chart in this section, it can be seen that every year there are more countries
                                        that achieve reasonable development indicators (AE>90 and GDP/cap>3000 U$S) and, at the same time,
                                        preserve their emissions per capita under the global mean value.
                                        """.format(diox),
                                                    style={'textAlign': 'justify'}),

                                html.H5("""Furthermore, GDP/cap and {}/cap does not always show the same trend, meaning it can be development and an adequate
                                level of commitment to the environment simultaneously. Examples of growing GDP/cap and reducing {}/cap can be found in: 
                                Armenia, Venezuela, USA, Singapore, Uzbekistan, Ukraine, Netherlands, Moldova, Azerbajan, Belgium, Bulgaria, Luxembourg,
                                Liechtenstein, Bielorrusia, Switzerland, Cuba, Rumania, Check Rep., Germany, Tuvalu, Denmark, Estonia, Finland, France,
                                Sweden, UK, Nauru, Georgia, Slovak Rep.,Malta, Greece, Poland, Hungary, Ireland, Iceland, Serbia, Jamaica, Arab Emirates, 
                                Russia and Latvia. Of course not all of these countries show low {} values (total or per capita) but they have managed to reduce their {}/cap
                                while improving their devlopment indicators.
                                """.format(diox, diox, diox, diox),
                                            style={'textAlign': 'justify'})
                                ], width=7),

                            dbc.Col(dcc.Graph(figure=data["fig_8"]), width=5)
                                ]),    
                    ]),
                ],#fluid=True,
            )  
    return "No tab selected"

@app.callback(Output("store", "data"),
             [Input("button", "n_clicks"),
             Input('slct_country', 'value'),
             Input('slct_year', 'value')
             ])
def generate_graphs(n, option_slctd, option_year):
    """
    This callback generates three simple graphs from random data.
    """
    if not n:
        # generate empty graphs when app loads
        return {k: go.Figure(data=[]) for k in ["fig_1", "fig_2", "fig_4", "fig_5", "fig_6", "fig_7", "fig_8"]} #"scatter", "hist_1", "hist_2"

    # simulate expensive graph generation process
    time.sleep(2)

    # generate 100 multivariate normal samples
    #data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)

    dff = df.copy()
    dff_country = dff[dff["Country Name"] == option_slctd]

    dff_year=dff.loc[dff['years']==option_year]
    #sorted_dff_year=dff_year.sort_values(by='GDP/cap', ascending=False)
    sorted_dff_year=dff_year.sort_values(by='CO2', ascending=True)
    filtered_dff_year=sorted_dff_year.loc[(sorted_dff_year['CO2']<4.5) & (sorted_dff_year['AE']>90) & (sorted_dff_year['GDP/cap']>3000)]

    dff_top_CO2=topCO2_every_year.copy()
    dff_top_CO2_year = dff_top_CO2[dff_top_CO2["year"] == option_year]

    dff_top_GDP=topGDP_every_year.copy()
    dff_top_GDP_year = dff_top_GDP[dff_top_GDP["year"] == option_year]

    max_CO2_abs=10000
    if(dff_country.max()['CO2_abs']):
        max_CO2_abs=max_CO2_abs + dff_country.max()['CO2_abs']
    
    max_AE=100
    max_GDP=100 #(dff_country.max()['GDP/cap'])
    if(dff_country.max()['GDP/cap']):
        max_GDP=max_GDP + (dff_country.max()['GDP/cap'])

    min_GDP=0

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=dff_country['years'], y=dff_country['CO2'], name = '{}/cap (t)'.format(diox), marker_color='firebrick'))
    fig1.add_trace(go.Scatter(x=dff_country['years'], y=Global_CO2_cap_prom, name='Global mean {}/cap (t)'.format(diox), line=dict(color='indianred', width=4)))
    fig1.add_trace(go.Bar(x=dff_country['years'], y=dff_country['CO2_abs_new'], name='Total {} (t/{})'.format(diox, diez_7), marker_color='royalblue'))
    fig1.add_trace(go.Scatter(x=dff_country['years'], y=Global_CO2_abs_mean_scaled, name = 'Global mean {} (t/{})'.format(diox, diez_7), line=dict(color='darkblue', width=4)))
    fig1.update_layout(title='Evolution of {} emissions'.format(diox),
                        xaxis_title='Year',
                        yaxis_title='{} (t/{}) and {}/cap (t)'.format(diox, diez_7, diox))

    fig2=px.bar(dff_country, y='GDP/cap', x='years', color='AE', range_color=[0,100])
    #Global_CO2_cap_prom=[3.9, 3.8, 3.8, 4.2, 4.4, 4.4, 4.4]
    anio_global_dict_int={1990: 3.9, 1995: 3.8, 2000: 3.8 , 2005: 4.2, 2010: 4.4, 2015: 4.4 , 2019: 4.4}

    def get_val(llave):
        for key, value in anio_global_dict_int.items():
            if llave == key:
                return value

    CO2_val=get_val(option_year)
    list_CO2_anio=[CO2_val]*10

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=dff_top_CO2_year['Country Name'], y=dff_top_CO2_year['CO2_cap'], name='{}/cap'.format(diox), marker_color='firebrick'))
    fig4.add_trace(go.Scatter(x=dff_top_CO2_year['Country Name'], y=list_CO2_anio, name='Global mean {}/cap'.format(diox), line=dict(color='lightsalmon', width=4)))
    fig4.add_trace(go.Bar(x=dff_top_CO2_year['Country Name'], y=dff_top_CO2_year['CO2_abs_new'], name='{}/{}'.format(diox, diez_7) , marker_color='royalblue'))
    fig4.update_layout(title='Evolution of {} emissions'.format(diox),
                        xaxis_title='Country Name',
                        yaxis_title='{} (t/{}) and {}/cap (t)'.format(diox, diez_7, diox))

    fig7=px.bar(dff_top_GDP_year, y='GDP_cap', x='Country Name', color='AE', range_color=[0,100])

    CO2_val2=get_val(option_year)
    list_CO2_anio2=[CO2_val2]*10

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=dff_top_GDP_year['Country Name'], y=dff_top_GDP_year['CO2_cap'], name='{}/cap'.format(diox), marker_color='firebrick'))
    fig6.add_trace(go.Scatter(x=dff_top_GDP_year['Country Name'], y=list_CO2_anio2, name='Global mean {}/cap'.format(diox), line=dict(color='lightsalmon', width=4)))
    fig6.add_trace(go.Bar(x=dff_top_GDP_year['Country Name'], y=dff_top_GDP_year['CO2_abs_new'], name='CO2/{}'.format(diez_7) , marker_color='royalblue'))
    fig6.update_layout(title='Evolution of {} emissions per capita'.format(diox),
                        xaxis_title='Country Name',
                        yaxis_title='{}/cap (t) and {} (t/{})'.format(diox, diox, diez_7))

    fig5=px.bar(dff_top_CO2_year, y='GDP_cap', x='Country Name', color='AE', range_color=[0,100])

    figBar1=px.bar(filtered_dff_year, x='CO2', y='Country Name', orientation='h')

    # save figures in a dictionary for sending to the dcc.Store
    return {"fig_1":fig1, "fig_2": fig2, "fig_4": fig4, "fig_5": fig5, "fig_6": fig6, "fig_7": fig7, "fig_8": figBar1}

if __name__ == "__main__":
    app.run_server(debug=True) #, port=8888)