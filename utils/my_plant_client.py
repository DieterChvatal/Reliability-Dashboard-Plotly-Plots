import math
from multiprocessing import Pool
import os
from datetime import datetime
import time
import json
from typing import List
import multiprocessing as mp

import requests
import urllib3
import pandas as pd

from utils.misc import df_move_cols_to_begin, to_timestamp


class MyPlantClientException(Exception):
    pass


class MyPlantClient:
    """Client without persistence"""
    def __init__(
            self,
            user,
            password,
            application_user=True,
            max_limit=100000,
    ):
        self.user = user
        self.password = password
        self.application_user = application_user
        self.max_limit = max_limit
        self.proxies = None
        self._access_token = None
        self.verify = True

        self._base_url = 'https://api.myplant.io/'

    @property
    def access_token(self):
        if not self._access_token:
            self.get_access_token()
        return self._access_token

    def get_access_token(self):
        """"""
        if self.application_user:
            url = self._base_url + 'oauth/token'
            auth = (self.user, self.password)
            data = {'grant_type': 'client_credentials'}
            r = requests.post(url, auth=auth, data=data, verify=self.verify, proxies=self.proxies)
            self._access_token = r.json()['access_token']
        else:
            url = self._base_url + 'auth'
            data = {'username': self.user, 'password': self.password}
            r = requests.post(url, json=data, verify=self.verify, proxies=self.proxies)
            try:
                self._access_token = r.json()['token']
            except Exception as e:
                print(r.text)
                raise
        print('Requested new access token.')

    def _request(self, method, endpoint, params=None, json_data=None):
        headers = {'x-seshat-token': self.access_token}

        request_method = {
            'get': requests.get,
            'post': requests.post,
        }
        r = request_method[method](
            self._base_url + endpoint,
            headers=headers,
            params=params,
            json=json_data,
            verify=self.verify,
            proxies=self.proxies
        )
        if r.status_code != 200:
            print('{}: {}'.format(r.status_code, r.text))
            raise MyPlantClientException(r.text)
        return r

    def _request_safe(self, max_tries=10, **kwargs):
        for i in range(max_tries):
            try:
                r = self._request(**kwargs)
            except (MyPlantClientException, requests.exceptions.ConnectionError) as e:
                print('Download failed in {} try ({}): {}'.format(i + 1, e, kwargs))
                time.sleep(i * 5)
            else:
                return r
        print('All download tries failed for: {}'.format(kwargs))
        raise

    def get_assets(
            self,
            properties=(
                "Engine Series",
                "Engine Type",
                "Engine Version",
                "Customer Engine Number",
                "Design Number",
                "Gas Type",
                "Generator Model",
                "Commissioning Date",
                "Contract.Service Contract Type",
                "Control System Type",
                "Application Type",
            ),
            data_items=(
                "OperationalCondition",
                "Count_OpHour",
                "Count_Start",
                "Power_PowerNominal",
            ),
            filters=('{name:"modelId", op:EQUALS, value:"23", comb:AND}',)
    ):
        """Downloads all assets"""
        r = self._request_assets(properties, data_items, filters=filters)

        assets = self._flatten_assets(r)
        assert len(assets) < self.max_limit
        return assets

    def _request_assets(self, properties: List[str], data_items: List[str], limit=None, filters=None):
        if not limit:
            limit = self.max_limit
        if not filters:
            filters = []

        graphQL = """
        {
        assets(
            filter: {
                limit: %s
                filters: [
                    %s
                ]
            }
        ) {
          items {
            id
            serialNumber
            modelId
            model
            timezone
            site {
              id
              name
              country
            }
            customer {
              name
            }
            status {
              lastContactDate
            }
            geoLocation {
              lat
              lon
            }
            properties(names: [
                %s
            ]) {
              name
              value
            }
            dataItems(query: [
                %s
            ]) {
              name
              value
              unit
              timestamp
            }
          }
        }
        }
        """ % (
            limit,
            ','.join(filters),
            ','.join(['"{}"'.format(i) for i in properties]),
            ','.join(['"{}"'.format(i) for i in data_items]),
        )

        r = self._request('post', endpoint='graphql', json_data={'query': graphQL})
        return r

    @staticmethod
    def _flatten_assets(r):
        # Convert response to flat table
        assets_list = []
        for asset in r.json()['data']['assets']['items']:
            asset_dict = {}
            for item, item_value in asset.items():
                if type(item_value) is dict:
                    for key, value in item_value.items():
                        asset_dict[item + '_' + key] = item_value[key]
                elif type(item_value) is list:
                    for sub_item in item_value:
                        asset_dict[sub_item['name']] = sub_item['value']
                elif item_value is None:
                    pass
                else:
                    asset_dict[item] = asset[item]
            assets_list.append(asset_dict)
        assets = pd.DataFrame(assets_list)
        COLS_TO_BEGIN = [
            'id', 'serialNumber', 'customer_name', 'site_name', 'site_country',
            'Engine Type', 'OperationalCondition', 'model', 'modelId',
        ]
        assets = df_move_cols_to_begin(assets, COLS_TO_BEGIN)
        COLS_TO_DATETIME = ['status_lastContactDate']
        for col in COLS_TO_DATETIME:
            assets[col] = pd.to_datetime(assets[col], unit='ms')
        return assets

    def _get_full_table(self, endpoint, params=None, model_id=None):
        """Downloads full table. If a path is defined saves it to CSV else returns it."""
        r = self._request('get', endpoint=endpoint, params=params)

        df = pd.DataFrame(r.json())
        assert len(df) < self.max_limit
        if 'id' in df.columns:
            df.sort_values('id', inplace=True)
        if model_id and 'modelId' in df.columns:
            df = df.loc[df['modelId'] == model_id]
        return df

    def get_properties(self, model_id=23):
        df = self._get_full_table('modelproperty', model_id=model_id)
        return df

    def get_data_items(self, model_id=23):
        return self._get_full_table('dataitem', model_id=model_id)

    def get_data_items_localization(self, model_name='J-Engine', language='en'):
        endpoint = 'system/localization'
        params = {
            'groupResult': True,
            'groups': 'data-items,data-item-values',
            'languages': language,
        }
        r = self._request(
            method='get',
            endpoint=endpoint,
            params=params
        )
        localization_dict = r.json()[language]['groups'][0]['values']
        df = pd.DataFrame({'key': localization_dict.keys(), 'localization': localization_dict.values()})
        key_split = df['key'].str.split('_', n=1, expand=True)
        df['model_name'] = key_split[0]
        df['data_item_name'] = key_split[1]
        df.drop(columns=['key'], inplace=True)
        df = df_move_cols_to_begin(df, ['model_name', 'data_item_name', 'localization'])
        if model_name:
            df = df.loc[df['model_name'] == model_name]
        return df

    def get_units(self):
        return self._get_full_table('unit')

    def get_alarms(self, model_name='J-Engine'):
        endpoint = 'model/{model_name}/alarms'.format(model_name=model_name)
        params = {
            'lang': 'en',
        }
        return self._get_full_table(endpoint, params=params)

    def get_users(self):
        endpoint = 'user'
        params = {
            'all': True,
            'stale': True,
        }
        return self._get_full_table(endpoint, params=params)

    def get_asset_groups(self):
        endpoint = 'asset/groups'
        asset_groups = self._get_full_table(endpoint)
        new_rows = []
        for _, row in asset_groups[['id', 'assetIds']].dropna().iterrows():
            for asset_id in row['assetIds']:
                new_rows.append({
                    'asset_group_id': row['id'],
                    'asset_id': asset_id,
                })
        asset_group_to_asset = pd.DataFrame(new_rows)
        asset_groups.drop(columns=['assetIds'], inplace=True)
        return asset_groups, asset_group_to_asset

    def _get_data_item_history_interval(
            self,
            asset_id: int, data_item_id: int, start: pd.Timestamp, end: pd.Timestamp, high_res=False,
    ):
        ENDPOINT_MULTI_BATCH_ASSET_DATA_ITEMS = 'ws/historic-data/dataitem/{resolution}/batch'
        RESOLUTION_DICT = {
            True: 'HIGHRES',
            False: 'LOWRES'
        }
        json_data = {
            'assetIds': [asset_id],
            'dataItemIds': [data_item_id],
            'range': {
                'from': to_timestamp(start),
                'to': to_timestamp(end),
            }
        }
        endpoint = ENDPOINT_MULTI_BATCH_ASSET_DATA_ITEMS.format(resolution=RESOLUTION_DICT[high_res])

        tic = time.time()
        r = self._request_safe(method='post', endpoint=endpoint, json_data=json_data)
        ts = pd.DataFrame(r.json()['values'][0]['values'], columns=['timestamp', 'value'])
        #print('{} > Downloaded in {:.1f} s: asset_id={}, data_item_id={}, len_ts={}'.format(
            #mp.current_process().name, time.time() - tic, asset_id, data_item_id, len(ts)))
        return ts

    def get_data_item_history(
            self,
            asset_id: int, data_item_id: int, start: pd.Timestamp, end: pd.Timestamp, high_res=False,
            interval_len=pd.to_timedelta('30d'),
    ):
        ts_list = []
        for i in range(math.ceil((end - start) / interval_len)):
            i_start = start + i * interval_len
            i_end = min(i_start + interval_len, end)
            i_ts = self._get_data_item_history_interval(asset_id, data_item_id, i_start, i_end, high_res)
            ts_list.append(i_ts)
        ts = pd.concat(ts_list)

        if not len(ts):
            return ts
        ts.columns = ['timestamp', 'value']
        ts = ts.loc[ts['timestamp'] > 0]
        ts['datetime'] = pd.to_datetime(ts['timestamp'], unit='ms')
        ts.drop(columns=['timestamp'], inplace=True)
        ts = ts[['datetime', 'value']]
        ts = ts.loc[(ts['datetime'] >= start) & (ts['datetime'] <= end)]
        return ts

    def _get_message_events_history_interval(
            self, asset_id: int, start: pd.Timestamp, end: pd.Timestamp, names: List[str] = None,
            severity_min: int = None, severity_max: int = None,
    ):
        ENDPOINT_BATCH_ALARMS = 'ws/historic-alarms/alarms/batch'
        json_data = {
            'assetIds': [asset_id],
            'timeRange': {
                'from': to_timestamp(start),
                'to': to_timestamp(end),
            }
        }
        if names:
            json_data['names'] = names
        if severity_min and severity_max:
            json_data['severityRange'] = {'min': severity_min, 'max': severity_max}
        tic = time.time()
        r = self._request_safe(method='post', endpoint=ENDPOINT_BATCH_ALARMS, json_data=json_data)
        meh = pd.DataFrame(r.json()[0]['data'])
        #print('{} > Downloaded in {:.1f} s: asset_id={}, len_meh={}'.format(
            #mp.current_process().name, time.time() - tic, asset_id, len(meh)))
        return meh

    def get_message_events_history(
            self, asset_id: int, start: pd.Timestamp, end: pd.Timestamp, names: List[str] = None,
            severity_min: int = None, severity_max: int = None,
            interval_len=pd.to_timedelta('90d'),
    ):
        meh_list = []
        for i in range(math.ceil((end - start) / interval_len)):
            i_start = start + i * interval_len
            i_end = min(i_start + interval_len, end)
            i_meh = self._get_message_events_history_interval(asset_id, i_start, i_end, names, severity_min, severity_max)
            meh_list.append(i_meh)
        meh = pd.concat(meh_list)

        if not len(meh):
            return meh
        meh['name'] = meh['name'].astype(str)
        meh['datetime'] = pd.to_datetime(meh['timestamp'], unit='ms')
        meh.drop(columns='timestamp', inplace=True)
        return meh

    def download_reliability(self, request_df, path_pattern, use_timestamps=True, num_workers=5):
        request_df_input = request_df.copy()
        if not use_timestamps:
            request_df_input['timestampFrom'] = 0
            request_df_input['timestampTo'] = 9000000000000

        queue = []
        for index, row in request_df_input.iterrows():
            path = path_pattern.format(
                assetId=int(row['assetId']),
                FROM=int(row['timestampFrom']),
                TO=int(row['timestampTo']),
            )

            queue.append({
                'path': path,
                'assetId': int(row['assetId']),
                'from': row['timestampFrom'],
                'to': row['timestampTo'],
            })

        # Check if dataItem has already been downloaded for all assetIds
        for job in queue:
            if not os.path.isfile(job['path']):
                break
        else:
            print('All files already downloaded.')
            return [True]

        # Download
        print('\n', 'Jobs in queue:', len(queue), '\n')
        with Pool(min(num_workers, len(queue))) as p:
            status = p.map(self._download_reliability_worker, queue)

        return status

    def _download_reliability_worker(self, job, max_attemps=10):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if not os.path.isfile(job['path']):
            # File not existing yet -> download
            for attempt in range(10):
                # time.sleep(random.randint(0, 3))
                tic = time.time()
                try:
                    r = self._download_reliability_single(
                        job['assetId'],
                        timestampFrom=int(job['from']),
                        timestampTo=int(job['to']),
                    )
                except Exception as e:
                    print('{}. try for {}'.format(attempt + 2, job['assetId']))
                    print(e)
                else:
                    if r.status_code != 200:
                        print('{}. try for {}: Status code {} ({})'.format(
                            attempt + 2, job['assetId'], r.status_code, r.url))
                        time.sleep(30 * (attempt + 1))
                    else:
                        # Download successful -> break loop
                        break
            else:
                # For loop was not broken -> download failed
                print('Failed for', job['assetId'])
                return False

            # Download successful -> Write JSON response to file
            with open(job['path'], 'w') as outfile:
                try:
                    json.dump(r.json(), outfile)
                except:
                    print(r.text)
                    raise
            # Verify that JSON file was written correctly and can be read
            with open(job['path'], 'r') as outfile:
                try:
                    json_verify = json.load(outfile)
                except:
                    print('Failed opening', outfile)
                    raise
            assert json_verify == r.json(), 'Corrupt file for {}'.format(job['deviceId'])

            # Json to hdf

            # Download complete, print ouput
            print('Downloaded ({:.1f} s): {}, text-len: {}'.format(time.time() - tic, job['path'], len(r.text)))
            return True
        else:
            # File already exists
            created = datetime.fromtimestamp(os.path.getctime(job['path']))
            print('Already downloaded: {}, Created on: {:%d.%m.%Y %H:%M:%S}'.format(job['path'], created))
            return True

    def _download_reliability_single(self, assetIds, timestampFrom, timestampTo):
        url = self._base_url + 'performance/reliability'
        headers = {'x-seshat-token': self.access_token}
        params = {
            'assetIds': assetIds,
            'from': timestampFrom,
            'to': timestampTo,
        }
        r = requests.get(url, headers=headers, params=params, verify=self.verify, proxies=self.proxies)
        return r

    def get_data_item_current_values(
            self, batch_size=2000,
            filters=('{name:"modelId", op:EQUALS, value:"23", comb:AND}',), data_items=(),
    ):
        if not filters:
            filters = []

        # graphQL query
        graphQL_query = """
        {
        assets(
            filter: {
                limit: %s,
                offset: %s,
                filters: [
        	       %s
                ]
            }
        ) {
          items {
            id
            dataItems(query: [
              %s
            ]) {
              id
              value
            }
          }
        }
        }
        """

        offset = 0
        rows = []
        while True:
            tic = time.time()
            graphQL = graphQL_query % (
                batch_size,
                offset,
                ','.join(filters),
                ', '.join('"{}"'.format(d) for d in data_items)
            )
            url = self._base_url + 'graphql'
            for attempt in range(10):
                try:
                    headers = {'x-seshat-token': self.access_token}
                    r = requests.post(url, headers=headers, json={'query': graphQL}, verify=self.verify, proxies=self.proxies)
                except Exception as e:
                    print('{}. try for offset={}: {}'.format(attempt + 2, offset, repr(e)))
                    self.get_access_token()
                else:
                    if r.status_code != 200:
                        print('{}. try for offset={}: Status code {}'.format(attempt + 2, offset, r.status_code))
                        self.get_access_token()
                    else:
                        # Download successful -> break loop
                        break
            else:
                raise ValueError('Failed downloading current values in all attempts.')

            items = r.json()['data']['assets']['items']
            for item in items:
                row = {'asset_id': item['id']}
                for data_item in item['dataItems']:
                    row[data_item['id']] = data_item['value']
                rows.append(row)

            print('Downloaded CVs for: offset={}, len_items={}, time={:.1f}s'.format(offset, len(items), time.time() - tic))

            if len(items) < batch_size:
                break
            offset += batch_size

        cv = pd.DataFrame(rows)
        cv.set_index('asset_id', inplace=True)
        return cv

    def get_operation_profile(self, public_id):
        if self.application_user:
            raise ValueError('User cannot be application_user for operation-profile endpoint.')
        url = 'https://myplant.io/beta/ws/operation-profile/result/{}'.format(public_id)
        headers = {'x-seshat-token': self.access_token}
        r = requests.get(url, headers=headers, verify=self.verify)
        if r.status_code != 200:
            print(r.text)
            raise ValueError
        return r

    def get_spc_item_matching(self, item_numbers):
        url = 'https://alpha.myplant.io/ws/spc/api/item-matching'
        headers = {'x-seshat-token': self.access_token}
        data = json.dumps(["{}".format(i) for i in item_numbers])
        r = requests.post(url, headers=headers, data=data, verify=self.verify)
        if r.status_code != 200:
            print(r.text)
            raise ValueError
        return r.json()
