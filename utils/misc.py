import pathlib
from uuid import UUID
import os
from datetime import datetime
from math import log10, floor

import numpy as np
import pandas as pd


def to_timestamp(dt: pd.Timestamp):
    """Converts pandas datetime object to timestamp in ms"""
    if dt is not None:
        return round(dt.value / 1e6)


def df_move_cols_to_begin(df: pd.DataFrame, cols_at_begin: list):
    """Helper function to reorder columns in a dataframe"""
    return df[[c for c in cols_at_begin if c in df] + [c for c in df if c not in cols_at_begin]]


def df_move_cols_to_end(df: pd.DataFrame, cols_at_end: list):
    """Helper function to reorder columns in a dataframe"""
    return df[[c for c in df if c not in cols_at_end] + [c for c in cols_at_end if c in df]]


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def right_strip_zeros(n):
    n = str(n)
    return n.rstrip('0').rstrip('.') if '.' in n else n


def round_to_significant_digits(n: float, significant_digits: int):
    assert type(significant_digits) == int
    assert significant_digits >= 1

    log = log10(abs(n))
    if log > significant_digits - 1:
        precision = 1
    else:
        precision = significant_digits - int(floor(log)) - 1
    print('log10={:.2f}, precision={}, significant_digits={}'.format(log, precision, significant_digits))
    return round(n, precision)


def is_valid_uuid(value):
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def forward_fillna(df, limit=0):
    """Only forward fills contiguous nan groups which length is <= than limit
    https://stackoverflow.com/questions/40442278/using-fillna-selectively-in-pandas
    """
    if not limit:
        return df
    nulls = df.isnull()
    filled = df.ffill(limit=limit)
    unfilled = nulls & (~filled.notnull())
    nf = nulls.replace({False: 2.0, True: np.nan})
    do_not_fill = nf.combine_first(unfilled.replace(False, np.nan)).bfill() == 1
    return df.where(do_not_fill, df.ffill())


def check_file_exists(path, max_age=None, verbose=True):
    if not os.path.isfile(path):
        return False
    if not max_age:
        if verbose:
            print('File already downloaded'.format(path))
        return True
    modified = datetime.fromtimestamp(os.path.getmtime(path))
    if modified < max_age:
        return False
    if verbose:
        print('File already downloaded on {:%d.%m.%Y %H:%M:%S}: {}'.format(modified, path))
    return True


def mkdir_for_file_path(path_file):
    pathlib.Path(path_file.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
