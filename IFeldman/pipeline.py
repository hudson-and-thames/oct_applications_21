import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

ROOTDIR = Path().cwd()
# PROJECTDIR = ROOTDIR / os.getenv("PROJECTDIR")
DATADIR = ROOTDIR / os.getenv("DATADIR")
filepath = DATADIR / os.getenv("FILENAME")


def EURUSD():
    """
    ccy1: EUR
    ccy2: USD
    """
    df = pd.read_excel(
        filepath.as_uri(),
        sheet_name="EURUSD",
        index_col=[0],
        header=0,
        usecols=[0, 1],
        skiprows=8,
        parse_dates=[0],
        squeeze=True,
    )
    df = df.sort_index(ascending=True)
    return df


def USDGBP():
    """
    ccy1: USD
    ccy2: GBP
    """
    df = pd.read_excel(
        filepath.as_uri(),
        sheet_name="USDGBP",
        index_col=[0],
        header=0,
        usecols=[0, 1],
        skiprows=8,
        parse_dates=[0],
        squeeze=True,
    )
    df = df.sort_index(ascending=True)
    return df


def USDJPY():
    """
    ccy1: USD
    ccy2: 100 JPY
    """
    df = pd.read_excel(
        filepath.as_uri(),
        sheet_name="USDJPY",
        index_col=[0],
        header=0,
        usecols=[0, 1],
        skiprows=8,
        parse_dates=[0],
        squeeze=True,
    )
    df = df.sort_index(ascending=True)
    return df


def EURJPY():
    """
    ccy1: EUR
    ccy2: 100 JPY
    """
    df = pd.read_excel(
        filepath.as_uri(),
        sheet_name="EURJPY",
        index_col=[0],
        header=0,
        usecols=[0, 1],
        skiprows=8,
        parse_dates=[0],
        squeeze=True,
    )
    df = df.sort_index(ascending=True)
    return df


def EURGBP():
    """
    ccy1: EUR
    ccy2: GBP
    """
    df = pd.read_excel(
        filepath.as_uri(),
        sheet_name="EURGBP",
        index_col=[0],
        header=0,
        usecols=[0, 1],
        skiprows=8,
        parse_dates=[0],
        squeeze=True,
    )
    df = df.sort_index(ascending=True)
    return df


def JPYGBP():
    """
    ccy1: 100 JPY
    ccy2: GBP
    """
    df = pd.read_excel(
        filepath.as_uri(),
        sheet_name="JPYGBP",
        index_col=[0],
        header=0,
        usecols=[0, 1],
        skiprows=8,
        parse_dates=[0],
        squeeze=True,
    )
    df = df.sort_index(ascending=True)
    return df
