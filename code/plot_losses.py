from __future__ import print_function
import numpy as np
import argparse
import json
import pathlib
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


eps = np.finfo(np.float32).eps

factors = ['network']


class DF_writer(object):
    def __init__(self, columns):
        self.df = pd.DataFrame(columns=columns)
        self.columns = columns

    def append(self, **row_data):
        if set(self.columns) == set(row_data):
            s = pd.Series(row_data)
            self.df = self.df.append(s, ignore_index=True)

    def df(self):
        return self.df


def load_models(results_dir):
    losses = []
    labels = []
    for json_file in pathlib.Path(results_dir).glob('*.json'):
        with open(json_file, 'r') as stream:
            results = json.load(stream)

        losses.append(results['train_losses'])
        print(len(results['train_losses']))
        labels.append(str(results['args']['bottleneck_size']))

    return losses, labels


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Find lowest loss')

    parser.add_argument(
        'results_dir',
        type=str,
        help='path to results'
    )

    args = parser.parse_args()

    losses, labels = load_models(args.results_dir)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

    losses = np.array(losses)
    # ax = plt.subplot(111)
    handles = plt.plot(losses.T)
    plt.legend(handles, labels, loc=1, title="d")
    plt.xlabel("Number of Iterations (k)")
    plt.ylabel("Sliced-Wasserstein Loss")
    ax = plt.gca()
    ax.set_yscale("log")
    plt.savefig(
        "iterations.pdf",
        bbox_inches='tight',
        dpi=300
    )
