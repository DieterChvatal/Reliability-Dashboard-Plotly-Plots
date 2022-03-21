import pathlib
import random
from typing import List
import multiprocessing as mp
import time
from functools import reduce

import yaml
import pandas as pd

from utils.my_plant_client import MyPlantClient
from utils.misc import check_file_exists, mkdir_for_file_path, to_timestamp, forward_fillna


class MyPlantClientLocal:
    """Layer for persistence used in local development"""
    def __init__(
            self,
            path_user_config='config/user.yaml',
            application_user=False,
            max_age=None,
            path_assets='data/assets.csv',
            path_properties='data/properties.csv',
            path_data_items='data/data_items.csv',
            path_data_items_localization='data/data_items_localization.csv',
            path_units='data/units.csv',
            path_alarms='data/alarms.csv',
            path_users='data/users.csv',
            path_pattern_data_item_history='data/data_item_history/{data_item_id}/{asset_id}_{start}_{end}.csv',
            path_pattern_message_events_history='data/message_events_history/{asset_id}_{start}_{end}.csv',
    ):
        self._path_user_config = path_user_config
        self._max_age = max_age
        self._path_assets = path_assets
        self._path_properties = path_properties
        self._path_data_items = path_data_items
        self._path_data_items_localization = path_data_items_localization
        self._path_units = path_units
        self._path_alarms = path_alarms
        self._path_users = path_users
        self._path_pattern_data_item_history = path_pattern_data_item_history
        self._path_pattern_message_events_history = path_pattern_message_events_history

        user = self._read_user_config()
        self._my_plant_client = MyPlantClient(
            user=user['username'], password=user['password'], application_user=application_user)

    def _read_user_config(self):
        with open(self._path_user_config, 'r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def _get_table(self, path, download_function, **kwargs):
        if check_file_exists(path=path, max_age=self._max_age):
            return pd.read_csv(path)
        df = download_function(**kwargs)
        mkdir_for_file_path(path)
        df.to_csv(path, index=False)
        return df

    def get_access_token(self):
        return self._my_plant_client.access_token

    def get_assets(self, **kwargs):
        return self._get_table(self._path_assets, self._my_plant_client.get_assets, **kwargs)

    def get_properties(self, model_id=23):
        return self._get_table(self._path_properties, self._my_plant_client.get_properties, model_id=model_id)

    def get_data_items(self, model_id=23):
        return self._get_table(self._path_data_items, self._my_plant_client.get_data_items, model_id=model_id)

    def get_data_items_localization(self, model_name='J-Engine', language='en'):
        return self._get_table(
            self._path_data_items_localization,
            self._my_plant_client.get_data_items_localization,
            model_name=model_name,
            language=language,
        )

    def get_units(self):
        return self._get_table(self._path_units, self._my_plant_client.get_units)

    def get_alarms(self, model_name='J-Engine'):
        return self._get_table(self._path_alarms, self._my_plant_client.get_alarms, model_name=model_name)

    def read_data_item_history(
            self,
            asset_id, data_item_id,
            start: pd.Timestamp = None, end: pd.Timestamp = None,
            resample=False, forward_fillna_limit: int = 0,
            verbose=True,
    ):
        path = self._path_pattern_data_item_history.format(
            data_item_id=data_item_id, asset_id=asset_id, start=to_timestamp(start), end=to_timestamp(end))

        try:
            ts = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print('Empty file for asset_id={}, data_item_id={}'.format(asset_id, data_item_id))
            return pd.DataFrame()
        if not len(ts):
            return pd.DataFrame()
        ts['datetime'] = pd.to_datetime(ts['datetime'])

        if start:
            ts = ts.loc[ts['datetime'] >= start]
        if end:
            ts = ts.loc[ts['datetime'] <= end]

        if not len(ts):
            return pd.DataFrame()

        if resample is not False:
            ts = ts.resample(resample, on='datetime').mean()
            if forward_fillna_limit:
                ts = forward_fillna(ts, limit=forward_fillna_limit)
            ts = ts.dropna().reset_index()
            ts.dropna(inplace=True)

        if verbose:
            print('Read data for: asset_id={}, data_item_id={}, len={}'.format(asset_id, data_item_id, len(ts)))
        return ts

    def download_data_item_history(
            self,
            asset_ids: List[int], data_item_ids: List[int],
            starts: List[pd.Timestamp], ends: List[pd.Timestamp],
            high_res=False, num_workers=3,
    ):
        assert len(asset_ids) == len(starts) == len(ends)
        for data_item_id in data_item_ids:
            for asset_id, start, end in zip(asset_ids, starts, ends):
                path = self._path_pattern_data_item_history.format(
                    data_item_id=data_item_id, asset_id=asset_id, start=to_timestamp(start), end=to_timestamp(end))
                if not check_file_exists(path=path, max_age=self._max_age, verbose=False):
                    break
            else:
                continue
            break
        else:
            print('All data item history files already downloaded')
            return

        queue = []
        for data_item_id in data_item_ids:
            for asset_id, start, end in zip(asset_ids, starts, ends):
                queue.append(dict(
                    asset_id=asset_id,
                    data_item_id=data_item_id,
                    start=start,
                    end=end,
                    high_res=high_res,
                ))
        random.shuffle(queue)

        with mp.Pool(min(num_workers, len(queue))) as p:
            p.map(self._download_data_item_history_worker, queue)

    def read_data_item_history_and_merge(
            self, asset_id, start, end, data_item_ids, resample_rule='30s', forward_fillna_limit=0,
    ):
        ts_list = []
        for data_item_id in data_item_ids:
            ts_data_item = self.read_data_item_history(
                asset_id, data_item_id, start, end,
                resample=resample_rule, forward_fillna_limit=forward_fillna_limit,
            )
            if len(ts_data_item):
                ts_data_item = ts_data_item.set_index('datetime').rename(columns={'value': data_item_id})
            ts_list.append(ts_data_item)
        ts = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), ts_list).sort_index()
        ts.rename(columns=self.get_data_items().set_index('id').to_dict()['name'], inplace=True)
        return ts

    def _download_data_item_history_worker(self, job):
        self._download_data_item_history_single(**job)

    def _download_data_item_history_single(
            self,
            asset_id,
            data_item_id,
            start,
            end,
            high_res,
    ):
        tic = time.time()
        path = self._path_pattern_data_item_history.format(
            data_item_id=data_item_id, asset_id=asset_id, start=to_timestamp(start), end=to_timestamp(end))
        if check_file_exists(path=path, max_age=self._max_age, verbose=False):
            print('File already exists: asset_id={}, data_item_id={}, start={}, end={}, path={}'.format(
                asset_id, data_item_id, start, end, path))
            return

        ts = self._my_plant_client.get_data_item_history(
            asset_id=asset_id,
            data_item_id=data_item_id,
            start=start,
            end=end,
            high_res=high_res,
        )
        mkdir_for_file_path(path)
        ts.to_csv(path, index=False)
        #print('{} > Downloaded in {:.1f} s: asset_id={}, data_item_id={}, start={}, end={}, len_ts={}, path={}'.format(
            #mp.current_process().name, time.time() - tic, asset_id, data_item_id, start, end, len(ts), path))

    def read_message_event_history(self, asset_id, start: pd.Timestamp = None, end: pd.Timestamp = None):
        path_file = self._path_pattern_message_events_history.format(
            asset_id=asset_id,
            start=to_timestamp(start),
            end=to_timestamp(end),
        )
        try:
            me = pd.read_csv(path_file, parse_dates=['datetime'])
        except pd.errors.EmptyDataError:
            print('Empty file for asset_id={}'.format(asset_id))
            return pd.DataFrame()
        me['name'] = me['name'].astype(str)

        if start:
            me = me.loc[me['datetime'] >= start]
        if end:
            me = me.loc[me['datetime'] <= end]

        print('Read data for: asset_id={}, len={}'.format(asset_id, len(me)))
        return me

    def download_message_events_history(
            self,
            asset_ids: List[int], starts: List[pd.Timestamp], ends: List[pd.Timestamp],
            names: List[str] = None, severity_min: int = None, severity_max: int = None,
            num_workers=3,
    ):
        assert len(asset_ids) == len(starts) == len(ends)
        for asset_id, start, end in zip(asset_ids, starts, ends):
            path = self._path_pattern_message_events_history.format(
                asset_id=asset_id, start=to_timestamp(start), end=to_timestamp(end))
            if not check_file_exists(path=path, max_age=self._max_age, verbose=False):
                break
        else:
            print('All message event history files already downloaded')
            return

        queue = []
        for asset_id, start, end in zip(asset_ids, starts, ends):
            queue.append(dict(
                asset_id=asset_id,
                start=start,
                end=end,
                names=names,
                severity_min=severity_min,
                severity_max=severity_max,
            ))
        random.shuffle(queue)

        with mp.Pool(min(num_workers, len(queue))) as p:
            p.map(self._download_message_events_history_worker, queue)

    def _download_message_events_history_worker(self, job):
        self._download_message_events_history_single(**job)

    def _download_message_events_history_single(self, **kwargs):
        tic = time.time()
        kwargs_formatted = {}
        for key, value in kwargs.items():
            if type(value) == pd.Timestamp:
                kwargs_formatted[key] = to_timestamp(value)
            else:
                kwargs_formatted[key] = value
        path = self._path_pattern_message_events_history.format(**kwargs_formatted)
        if check_file_exists(path=path, max_age=self._max_age, verbose=False):
            print('File already exists: path={}'.format(path))
            return

        ts = self._my_plant_client.get_message_events_history(**kwargs)

        mkdir_for_file_path(path)
        ts.to_csv(path, index=False)
       #print('{} > Downloaded in {:.1f} s: len_ts={}, path={}'.format(
            #mp.current_process().name, time.time() - tic, len(ts), path))
