import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from sklearn import preprocessing

def plot_gold_future_price(price_df):
    fig, ax = plt.subplots()
    ax.set_title('Gold Futures contract Price (GC=F)')
    ax.set_ylabel('Price (Dollars)')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%0.0f'))
    ax.set_xlabel('Date')
    ax.plot(price_df["GC=F"], label="GC=F", color="blue")
    ax.grid()

def plot_gold_future_to_kca_position(price_df):
    fig, ax = plt.subplots()
    ax.set_title('Gold Futures to KCA Position Fit Comparison')
    ax.set_ylabel('Price (Dollars)')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%0.0f'))
    ax.set_xlabel('Date')
    ax.plot(price_df['GC=F'], label='Gold Futures Price (GC=F)', color='black')
    ax.plot(price_df['KCA_position'], label='KCA Position', color='red', marker='o', markersize=2, linestyle='None')
    ax.legend(loc="lower left")
    ax.grid()

    plt.xticks(rotation=45)

def plot_gold_kca_position_velocity(price_df):
    """"""
    # Normalize kca data to visually compare patterns
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(price_df)
    futures_normalized = pd.DataFrame(x_scaled)
    futures_normalized.index = price_df.index
    normalized_position = futures_normalized[1]
    normalized_velocity = futures_normalized[2]

    fig, ax = plt.subplots()
    ax.set_title('KCA Normalized Position & Velocity')
    ax.set_ylabel('Normalized (Unitless)')
    ax.set_xlabel('Date')
    ax.plot(normalized_position, color='red', label="Position", marker='x', markersize=2, linestyle = 'None')
    ax.plot(normalized_velocity, color='blue', marker='o', markersize=2, linestyle = 'None', label="Velocity")
    ax.legend(loc="lower left")
    ax.grid()
    # plt.xticks(rotation=45)

def plot_gold_kca_strategy(kca_observations, predictions):
    kca = kca_observations
    fig, ax = plt.subplots()

    ax.set_title('KCAStrategy Predictions & Observations (GC=F)')
    ax.set_ylabel('Price (Dollars)')
    ax.set_xlabel('Date')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%0.0f'))
    ax.plot(kca.observations["GC=F"], label="GC=F", color="blue")
    ax.plot(kca.observations["prediction"], label="prediction", linestyle='None', marker="x", color="red")
    ax.plot(predictions["GC=F"], label="observed", linestyle=None, marker="o", color="green")
    ax.legend(loc="lower left")
    ax.grid()
    plt.xlim([kca.observations.index[-20], kca.observations.index[-1]])
    plt.ylim([1400, 2000])
    plt.xticks(rotation=45)