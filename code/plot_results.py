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
    columns = [
        'loss',
        'dataset',
        'subset',
        'bottleneck_size'
    ]

    data = DF_writer(columns)

    for json_file in pathlib.Path(results_dir).glob('*.json'):
        with open(json_file, 'r') as stream:
            results = json.load(stream)

        if results['args']['bottleneck_size'] <= 16 or results['args']['input_dim'] > -1:
            continue

        data.append(
            loss=float(results['train_loss']),
            subset="train",
            dataset=results['args']['dataset'],
            bottleneck_size=int(results['args']['bottleneck_size'])
        )
        data.append(
            loss=float(results['test_loss']),
            subset="test",
            dataset=results['args']['dataset'],
            bottleneck_size=int(results['args']['bottleneck_size'])
        )

    return data.df


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Find lowest loss')

    parser.add_argument(
        'results_dir',
        type=str,
        help='path to results'
    )

    args = parser.parse_args()

    df = load_models(args.results_dir)
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.5)

    print(df)
    g = (sns.catplot(
        x="bottleneck_size", y="loss", sharey=False,
        col="dataset", row="subset", data=df
    ).add_legend())
    plt.show()
    # g.fig.savefig(
    #     "5_train.svg",
    #     bbox_inches='tight',
    #     dpi=300
    # )
