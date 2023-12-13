# Editor : Sunkyoung Kim, Eunbi Choi

import logging
import os
import io
import json
from typing import Optional, Sequence, Union
import datasets


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def load_data(filepath):
    data = {}
    # if os.path.isdir(filepath):  # filepath is a directory
    #     files = [
    #         f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))
    #     ]
    #     for f in files:
    #         data.update(load_data(f))
    extension = os.path.splitext(filepath)[-1]
    if extension == ".json":
        return jload(filepath)
    elif extension == ".jsonl":
        return jlload(filepath)
    else:
        return datasets.load_dataset(filepath, download_mode="force_redownload")['train'] # FIXME
#        return datasets.load_dataset(filepath)['train'] # FIXME
    return data


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jlload(f, mode="r"):
    """Load a .json file into a dictionary."""
    data = []
    f = _make_r_io_base(f, mode)
    for line in f:
        jdict = json.loads(line)
        data.append(jdict)
    f.close()
    return data
