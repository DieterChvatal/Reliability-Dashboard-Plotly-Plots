# Standard Library imports
import pandas as pd
import numpy as np


def plotly_chart(df, pltcfg, x_ax='datetime', chart_theme='simple_white'):
    """Generates Plotly charts with multiple axes

        Args:
            df (dataframe): Data , datetime as index and MyPlant Names as columns
            pltcfg ([list of dicts]): the source columns to plot, and range of y-axis
            x_ax (str, optional): x-axis column as string. ex 'Operating hours engine' Default to 'datetime'.
            chart_theme (str, optional): Plotly theme. Default to 'simple_white'.

        Returns:
            bokeh.plotting.figure: Bokeh plot ready to plot or embed in a layout

        Example:
        pltcfg=[{'col': ['BMW REGENSBURG 5_@_Starts', 'ALPRO M2 616F412 BE_@_Starts', 'BMW REGENSBURG_@_Starts']}]
        x_ax='datetime'
        """
    import plotly.graph_objects as go

    dataitems = pd.read_csv('data/dataitems.csv', sep=';')

    fig = go.Figure()

    if x_ax == 'datetime':
        x_axis = df.index
    else:
        x_axis = df[x_ax]

    for j, y in enumerate(pltcfg):

        for k in y['col']:

            if not k in df.columns:
                print(k, 'is either not available or 0. Data will not be plotted.')
                continue

            try:
                y_unit = dataitems.loc[dataitems.myPlantName == k].iat[0, 2]
            except:
                y_unit = '-'
            if y_unit is np.nan: y_unit = '-'
            y_unit = y_unit.join((' [', ']'))
            if 'cyl.' in y['col'][0]:
                y_label = y['col'][0] + y_unit
            else:
                y_label = ', '.join(y['col'][:2]) + y_unit

            if j == 0:
                fig.add_trace(go.Scatter(x=x_axis, y=df[k], name=k))
                fig.update_layout(
                    yaxis=dict(title=y_label
                               ))
                if y.get('ylim') is not None: fig.update_layout(yaxis=dict(range=list(y.get('ylim'))))
            else:
                fig.add_trace(go.Scatter(x=x_axis, y=df[k], yaxis='y2', name=k))
                fig.update_layout(
                    yaxis2=dict(
                        title=y_label,
                        anchor="x",
                        overlaying="y",
                        side="right"
                    ))
                if y.get('ylim') is not None: fig.update_layout(yaxis2=dict(range=list(y.get('ylim'))))

    fig.update_layout(
        xaxis=dict(title=x_ax, domain=[0.05, 0.95]),
        margin=dict(l=20, r=20, t=10, b=10),
        autosize=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02)
    )
    fig.layout.template = chart_theme

    return fig


def load_pltcfg_from_excel(filename):
    """Load plotconfig from Excel Sheet "Input" necessary in same folder

    Returns:
        pltcfg (list of dicts): pltcfg with list of dicts
        plt_titles (list of String): titles of plots
    .....
    """

    import math

    def is_number(s):
        """ Returns True is string is a number. """
        try:
            float(s)
            return math.isfinite(s)
        except ValueError:
            return False

    df_cfg = pd.read_excel(filename, sheet_name='Pltcfg',
                           usecols=['Plot_Nr', 'Axis_Nr', 'Name', 'Unit', 'y-lim min', 'y-lim max'])
    df_cfg.sort_values(by=['Plot_Nr', 'Axis_Nr'], inplace=True)
    df_cfg.dropna(subset=['Plot_Nr', 'Axis_Nr', 'Name'], inplace=True)
    df_cfg['p_equal'] = df_cfg.Plot_Nr.eq(df_cfg.Plot_Nr.shift())
    df_cfg['a_equal'] = df_cfg.Axis_Nr.eq(df_cfg.Axis_Nr.shift())

    pltcfg = []
    plt_titles = []
    for i in range(len(df_cfg)):
        if not df_cfg.p_equal.iloc[i]:
            pltcfg.append([])  # new plot
            if df_cfg.Axis_Nr.iloc[i] == 0:  # append title if axis=0
                plt_titles.append(df_cfg.Name.iloc[i])  # append title
            else:
                plt_titles.append('')

        if df_cfg.Axis_Nr.iloc[i] != 0:
            if df_cfg.a_equal.iloc[i] == False or df_cfg.p_equal.iloc[i] == False:
                pltcfg[-1].append(dict())  # new axis

            y = pltcfg[-1][-1]
            if type(df_cfg.Name.iloc[i]) == str:
                if 'col' in y:
                    y['col'].append(df_cfg.Name.iloc[i].replace('\xa0', ' '))
                else:
                    y['col'] = [df_cfg.Name.iloc[i].replace('\xa0', ' ')]
                if 'unit' not in y and type(df_cfg.Unit.iloc[i]) == str:  # take first occurance of unit
                    y['unit'] = df_cfg.Unit.iloc[i].replace('\xa0', ' ')

                lim_min = df_cfg['y-lim min'].iloc[i]
                lim_max = df_cfg['y-lim max'].iloc[i]
                if 'ylim' not in y and is_number(lim_min) and is_number(lim_max):
                    y['ylim'] = (lim_min, lim_max)  # add tuple y lim
    return pltcfg, plt_titles


def datastr_to_dict(datastr):
    """Generate dict from myPlantNames
    In case name is not valid it gets ignored

    Args:
        datastr (list of str): myPlantNames to be transformed

    Returns:
        dat (dict): dictionary of dataitems
        rename (dict): dict of type {name:myPlantName}

    example:
    .....
    datastr_to_dict(['test123','Exhaust temperature cyl. 23'])

        Output: 
        test123 not available! Please check spelling.

        dat={191: ['Exhaust_TempCyl23', 'C (high)']},
        rename={'Exhaust_TempCyl23': 'Exhaust temperature cyl. 23'}"""

    # updated version, can transform myPlantNames from different languages
    data_str = np.unique(datastr).tolist()

    request_ids = pd.read_csv('data/dataitems.csv', sep=';')
    rel_data = pd.DataFrame()

    rename = {}
    for da in data_str:

        data_id = request_ids.loc[request_ids['myPlantName'] == da]
        if not data_id.empty:
            new = request_ids.loc[request_ids.myPlantName == da]['name'].values[0]
            rename[new] = da
            rel_data = pd.concat((rel_data, data_id), axis=0)  # Changed to concat instead of append

        # else: #uncommented for less output messages
        # print(da+' not available! Please check spelling.')
        # warnings.warn(da+' not available! Please check spelling.')

    dat = {rec['id']: [rec['name'], rec['unit']] for rec in rel_data.to_dict('records')}
    return dat, rename


def expand_cylinder(y, rel_cyl=all, e_type=0):
    """Check if parameter cylinder specific and expand if aplicable
    Args:
        y (dict): one line of a single pltcfg
        rel_cyl (list, optional): Defines relevant cylinders, defaults to all
        engi (dmyplant2.engine, optional): Engine instance to get number of cylinders from

    Returns:
        y (dict): line of a single pltcfg with expanded parameters
    """

    if rel_cyl is all:
        if e_type != 0:
            rel_cyl = list(range(1, int(e_type[1:3]) + 1))
        else:
            rel_cyl = list(range(1, 25))

    add_cyl_short_num = ['Inlet valve closure noise', 'Outlet valve closure noise']
    add_cyl_num = ['Exhaust temperature', 'Exhaust temperature delta', 'Ignition voltage', 'ITP', 'Knock integrator',
                   'Knock noise',  # 'Exhaust temperature delta' added for delta to mean value
                   'Pressure 49° before TDC', 'Mechanical noise', 'Cylinder state', 'Close current gradient',
                   'Inlet valve closure timing', 'Outlet valve closure timing']
    add_num = ['Knock signal', 'P-max', 'AI', 'IMEP', 'Duration of opening', 'Conrod bearing temperature', 'CQ max',
               'CQ', 'Slow down time']
    add_mid = []  # talk with Sebastian what is looked at analyzis

    to_remove = []
    for col in y['col']:
        if col in add_cyl_short_num and not col in to_remove:
            for cyl in rel_cyl:
                y['col'].append(f'{col} cyl. {cyl}')
                to_remove.append(col)

        if col in add_cyl_num and not col in to_remove:
            for cyl in rel_cyl:
                y['col'].append(f'{col} cyl. {cyl:02d}')
                to_remove.append(col)

        if col in add_num and not col in to_remove:
            for cyl in rel_cyl:
                y['col'].append(f'{col} {cyl:02d}')
                to_remove.append(col)

        if col in add_mid and not col in to_remove:
            for cyl in rel_cyl:
                y['col'].append(f'{col} cyl. {cyl:02d}')
                to_remove.append(col)

    y['col'] = [i for i in y['col'] if not i in to_remove]  # remove original column
    return y


def shrink_cylinder(y, rel_cyl=list(range(1, 25))):
    """Sort out some cylinder specific parameters, so that only the ones interested in are displayed
        The rest is loaded beforehand for shorter overall loading time

    Args:
        y (dict): one line of a single pltcfg
        rel_cyl (list, optional): Defines relevant cylinders, defaults to list:[1,2...,23,24]

    Returns:
        y (dict): line of a single pltcfg with eventually less parameters

    example:
    .....
    """

    rel_cyl = [str(cyl).zfill(2) for cyl in rel_cyl]
    add_cyl_short_num = ['Inlet valve closure noise', 'Outlet valve closure noise']
    add_cyl_num = ['Exhaust temperature', 'Exhaust temperature delta', 'Ignition voltage', 'ITP', 'Knock integrator',
                   'Knock noise',  # 'Exhaust temperature delta' added for delta to mean value
                   'Pressure 49° before TDC', 'Mechanical noise', 'Cylinder state', 'Close current gradient',
                   'Inlet valve closure timing', 'Outlet valve closure timing']
    add_num = ['Knock signal', 'P-max', 'AI', 'IMEP', 'Duration of opening', 'Conrod bearing temperature', 'CQ max',
               'CQ', 'Slow down time']
    add_mid = []  # talk with Sebastian what is looked at analyzis
    to_check = add_cyl_num + add_num + add_mid

    to_remove = []
    for col in y['col']:
        if (any(ele in col for ele in to_check) and not col[
                                                        -2:] in rel_cyl):  # check if elemt in expanded elements and not in rel_cyl
            # bug with add_cyl_short_num, exception would need to be added
            to_remove.append(col)

    y['col'] = [i for i in y['col'] if not i in to_remove]  # remove original column
    return y


def timestamp_LOC(df, windowsize=50, return_oph=False):  # starttime, endtime,
    """Oilconsumption vs. Validation period
    Args:
        windowsize (optional): Engine instance to get number of cylinders from
        return_oph (optional): Option to directly return the engine OPH in the dataframe at the LOC-data points
    Returns:
        pd.DataFrame:
    """
    # Lube Oil Consumption data
    try:
        dloc = df[(['Operating hours engine', 'Oil counter active energy', 'Oil counter power average',
                    'Oil counter oil consumption',
                    'Oil counter oil volume', 'Oil counter operational hours delta'])]
        dloc = dloc.drop_duplicates(
            ['Oil counter active energy', 'Oil counter power average', 'Oil counter oil consumption',
             'Oil counter oil volume', 'Oil counter operational hours delta'])

        dloc.drop(dloc[((dloc['Oil counter oil volume'] * 10) % 1 != 0)].index, inplace=True)
        dloc.drop(dloc[(dloc['Oil counter power average'] % 1 != 0)].index, inplace=True)
        dloc.drop(dloc[(dloc['Oil counter operational hours delta'] % 1 != 0)].index, inplace=True)

        dloc.drop(dloc[(dloc['Oil counter oil consumption'] > 5)].index,
                  inplace=True)  # Filter very large LOC, e.g. when refilling over the oil counter. Value according to Edward Rogers and Dieter Chvatal
        dloc.drop(dloc[(dloc['Oil counter oil consumption'] < 0.005)].index,
                  inplace=True)  # Filter very small LOC, according to Dieter Chavatal

        hoursum = 0
        volumesum = 0
        energysum = 0

        LOC_ws = []
        LOC_raw = []
        hours_filtered = []
        OPH_engine = []

        for i in range(len(dloc)):
            hoursum = hoursum + dloc.iloc[i, dloc.columns.get_loc('Oil counter operational hours delta')]
            volumesum = volumesum + dloc.iloc[i, dloc.columns.get_loc('Oil counter oil volume')]
            energysum = energysum + dloc.iloc[i, dloc.columns.get_loc('Oil counter active energy')]

            if hoursum >= windowsize:
                LOC_ws.append(volumesum * 0.886 / energysum)  # only make 3 decimal points
                hoursum = 0
                volumesum = 0
                energysum = 0
            else:
                LOC_ws.append(np.nan)

            LOC_raw.append(dloc.iloc[i, dloc.columns.get_loc('Oil counter oil consumption')])
            OPH_engine.append(dloc.iloc[i, dloc.columns.get_loc('Operating hours engine')])
            hours_filtered.append(dloc.index[i])

        if return_oph:
            dfres = pd.DataFrame(data={'datetime': hours_filtered, 'OPH_engine': OPH_engine, 'LOC_average': LOC_ws,
                                       'LOC_raw': LOC_raw})
        else:
            dfres = pd.DataFrame(data={'datetime': hours_filtered, 'LOC_average': LOC_ws, 'LOC_raw': LOC_raw})

        dfres = dfres.set_index('datetime')

    except:
        raise Exception("Loop Error in Validation_period_LOC")
    return dfres


if __name__ == '__main__':
    pass
