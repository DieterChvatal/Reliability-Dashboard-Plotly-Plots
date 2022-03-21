import pandas as pd
import numpy as np
import yaml

from tqdm import tqdm
from utils.supporting_functions import load_pltcfg_from_excel, expand_cylinder, datastr_to_dict, plotly_chart, timestamp_LOC

import os

from utils.my_plant_client import MyPlantClient

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State

#from dash import dash, dcc, html
#from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


'''
----------------------------------------------------
Dash styling dictionaries
----------------------------------------------------
'''

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}


'''
----------------------------------------------------
Class for data download
----------------------------------------------------
'''

class DataPullClient:
    def __init__(self):
        with open('config/user.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # MyPlant Client
        self.Client = MyPlantClient(user=config['MYPLANT_APP_ID'],
                                    password=config['MYPLANT_APP_SECRET'],
                                    application_user=True)

        if not os.path.isfile('data/dataitems.csv'):
            self.create_request_csv()

    def request_assets(self):
        assets = self.Client.get_assets()
        return assets

    def create_request_csv(self):

        data_items = self.Client.get_data_items()
        data_items_localization = self.Client.get_data_items_localization()
        units = self.Client.get_units()

        data_items['unit'] = data_items['unitId'].map(units.set_index('id')['name'])
        data_items['myPlantName'] = data_items['name'].map(
            data_items_localization.set_index('data_item_name')['localization'])
        model = data_items[['id', 'name', 'unit', 'myPlantName']].copy()

        directory = os.path.dirname('data/dataitems.csv')
        if not os.path.exists(directory):
            os.makedirs(directory)

        model.to_csv('data/dataitems.csv', sep=';', index=False)
        print("'dataitems.csv' successfully created")

    def request_data(self, asset_id, datastr, p_from=None, p_to=None):

        assets = self.Client.get_assets()
        sn = assets.loc[assets['id'] == asset_id, 'serialNumber'].item()

        def collect_info():
            # File Metadata
            info = {'p_from': p_from, 'p_to': p_to, 'Timezone': 'UTC',
                    'Export_Date': pd.to_datetime('now', utc=True), 'dataItems': data_item_ids}
            return pd.DataFrame.from_dict(info)

        def check_and_loadfile(p_from, fn):
            ldf = pd.DataFrame([])
            last_p_to = p_from
            try:
                if os.path.exists(fn):
                    dinfo = pd.read_hdf(fn, "info").to_dict()
                    # If the data in the file corresponds to the requested data ...
                    if set(data_item_ids) == set(dinfo['dataItems'].values()):
                        ffrom = list(dinfo['p_from'].values())[0]
                        if ffrom <= p_from:
                            ldf = pd.read_hdf(fn, "data")
                            # Check last lp_to in the file and update the file ....
                            last_p_to = list(dinfo['p_to'].values())[0]
                            print(f'Data file already exists. Downloading from {last_p_to.strftime("%Y-%m-%d %X")} to {p_to.strftime("%Y-%m-%d %X")}"')
                            # new starting point ...
            except:
                pass
            return ldf, last_p_to

        fn = f'./data/{sn}.hdf'

        ans = datastr_to_dict(datastr)
        data_item_ids = list(ans[0].keys())

        df, np_from = check_and_loadfile(p_from, fn)

        if np_from < p_to:
            ndf = pd.DataFrame()
            for data_item_id in tqdm(data_item_ids):
                # Download data item history
                data_item = self.Client.get_data_item_history(
                    asset_id=asset_id,
                    data_item_id=data_item_id,
                    start=np_from,
                    end=p_to,
                    high_res=False)

                data_item.set_index(['datetime'], inplace=True, drop=True)
                df_filtered = data_item

                name = ans[1][ans[0][data_item_id][0]]
                df_filtered.rename(columns={'value': name}, inplace=True)
                ndf = pd.concat([ndf, df_filtered], axis=1)
            df = pd.concat([df, ndf])

        dinfo = collect_info()
        dinfo.to_hdf(fn, "info", complevel=6)
        df.to_hdf(fn, "data", complevel=6)

        return df


'''
----------------------------------------------------
Start of code to build df for plots
----------------------------------------------------
'''

# Initialize client
dataPull = DataPullClient()

# Settings
end_date = pd.to_datetime('today').normalize() - pd.DateOffset(days=1)
start_date = end_date - pd.DateOffset(days=30)  # last 30 days
x_ax = 'Operating hours validation'  # string, can be either 'datetime', 'Operating hours validation' or 'Operating hours engine'
timeCycle = '3600'  # string, in seconds for df resampling to optimize plotting time
LOC_sampling = 100  # integer, in hours to average Myplant LOC

# Loading Engine List from Excel file
df_eng = pd.read_excel('inputs/Input_validation_dashboard.xlsx', sheet_name='Engines',
                       usecols=['Validation Engine', 'serialNumber', 'val start', 'oph@start', 'starts@start'])
df_eng.dropna(subset=['Validation Engine', 'serialNumber', 'val start'], inplace=True)
df_eng.reset_index(drop=True, inplace=True)

# Create list of validation serial # to iterate through later
valSerialNumbers = df_eng['serialNumber'].astype(int).astype(str).to_list()

# Filter assets and create validation fleet dataframe
assets = dataPull.request_assets()
fleet = assets.loc[assets['serialNumber'].isin(valSerialNumbers)]

tab_content = []  # Tabs names for dashboard
dash_content = []  # List containing graphs for every dashboard tab (engine)

for engCount, serNum in enumerate(valSerialNumbers):

    eng = df_eng.loc[engCount, 'Validation Engine']

    try:  # asset_type contains the number of cylinders used to expand cylinders
        asset_type = fleet.loc[fleet['serialNumber'] == serNum, 'Engine Type'].item()
        if pd.isna(asset_type):
            asset_type = 0
    except:
        asset_type = 0

    # Load plot configuration from Excel file
    pltcfg, plt_titles = load_pltcfg_from_excel('inputs/Input_validation_dashboard.xlsx')

    # Creating list of operating parameters to download
    datastr = []

    for cfg in pltcfg:
        for y in cfg:
            y = expand_cylinder(y, e_type=asset_type)
            datastr += y['col']

    datastr += ['Operating hours engine', 'Starts',
                # data from LOC calc
                'Operating hours engine', 'Oil counter active energy', 'Oil counter power average',
                'Oil counter oil consumption', 'Oil counter oil volume', 'Oil counter operational hours delta',
                # manually add interesting dataitems, for specific calculation or x-axis #eventually method with calls if tems requested (for mean, LOC, filter...)
                'Power current', 'Power nominal',  # for filter
                'Exhaust temperature cyl. average',  # for delta values if string has Exhaust temperature delta
                'Speed current',  # For BMEP
                'Starts',  # for Starts validation
                'Exhaust temperature cyl. maximum', 'Exhaust temperature cyl. minimum',
                # Add custom variable: Mention all required values either here or in the definition excel
                'datetime']  # for value of x_axes

    asset_id = fleet.loc[fleet['serialNumber'] == serNum, 'id'].item()

    print(f'\nDownloading data for {eng} ({engCount+1} out of {len(valSerialNumbers)})')
    df = dataPull.request_data(asset_id, datastr, p_from=start_date, p_to=end_date)

    # Resample to optimize for plot
    df = df.resample(timeCycle+'S').mean().fillna(method='ffill')
    # Remove all zero columns
    df = df.loc[:, (df != 0).any(axis=0)]

    # Calculate EGT delta and spread
    for col in df.columns:
        if 'Exhaust temperature' in col and any(map(str.isdigit, col)) and not 'delta' in col:
            df[f'Exhaust temperature delta cyl. {col[-2:]}'] = df[col].sub(df['Exhaust temperature cyl. average'])
    df['Exhaust temperature spread'] = df['Exhaust temperature cyl. maximum'] - df['Exhaust temperature cyl. minimum']

    # Add Column 'Operating hours validation'
    df['Operating hours validation'] = df['Operating hours engine'] - df_eng.fillna(0).loc[engCount, 'oph@start']

    # Add Column 'Starts validation'
    df['Starts validation'] = df['Starts'] - df_eng.fillna(0).loc[engCount, 'starts@start']

    # Add LOC_average, LOC_raw
    dfres = timestamp_LOC(df, windowsize=LOC_sampling)

    df.sort_index(inplace=True)  # additional sorting of index
    df = pd.merge_asof(df, dfres, left_index=True, right_index=True)

    duplicated = df.duplicated(subset=['LOC_average'])
    df.loc[duplicated, ['LOC_average']] = np.NaN
    df['LOC_average'] = df['LOC_average'].interpolate()

    # Generate plots in Loop
    plotly_plots = []
    for i, cfg in enumerate(pltcfg):
        # Plotly plots
        fig = plotly_chart(df, cfg, x_ax=x_ax, chart_theme='simple_white')
        if "'data': []" in str(fig):
            print(plt_titles[i], 'plot has no data, not shown in the dashboard')
            continue
        plotly_plots.append(html.H4(plt_titles[i]))  # append title
        plotly_plots.append(dcc.Graph(figure=fig))  # append plot

    # List for dashboard, every item contains all plots for the engine
    dash_content.append(html.Div(plotly_plots))

    # Children list for dash tabs
    tab_content.append(
        dcc.Tab(label=eng, value='tab-' + str(engCount), style=tab_style, selected_style=tab_selected_style))


'''
----------------------------------------------------
Start of code to build tab-layout and output in Dash
----------------------------------------------------
'''

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

app.layout = html.Div([
    dcc.Tabs(id='tabs-engine', value='tab-0', children=tab_content),
    html.Div(id='tabs-engine-content')
])


@app.callback(
    Output('tabs-engine-content', 'children'),
    Input('tabs-engine', 'value')
)
def render_content(tab):
    for i in range(len(valSerialNumbers)):
        if tab == 'tab-' + str(i):
            return dash_content[i]


if __name__ == '__main__':
    app.run_server(debug=True)
