#!/usr/bin/env python3.6

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

def moving_average(data_set, periods=20):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def get_rewards(path):
    df = pd.read_csv(os.path.join(path))
    values = df.iloc[:, 2].values
    values = moving_average(values)[:1000]
    times = df.iloc[:, 1].values
    times = np.linspace(times[0], times[-1], len(values))[:1000]
    return times, values

def load_data(data_dir):
    dirs = [d for d in os.listdir(data_dir) if d.startswith("run")]
    xseries = None
    yseries = []
    for d in dirs:
        path = os.path.join(data_dir, d)
        x, y = get_rewards(path)
        xseries = xseries if xseries is not None else x
        yseries.append(y)

    yseries = np.array(yseries)

    return xseries, yseries

def plot_data(data):
    xseries, yseries = data
    plt.figure()
    plt.title("Forward Model Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    mean_reward = np.mean(yseries, axis=0)
    std_err = scipy.stats.sem(yseries, axis=0)
    h = std_err * scipy.stats.t.ppf((1.0 + 0.95) / 2.0, yseries.shape[0]-1)
    plt.plot(xseries, mean_reward, label="Inverse Model")
    plt.fill_between(xseries, mean_reward + h, mean_reward - h, alpha=0.2)

    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    data = load_data('runs/')
    plot_data(data)
