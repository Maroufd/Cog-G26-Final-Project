import argparse
import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")
sns.set(font_scale=1.2)


if __name__ == "__main__":
    for simtype in ['resnet50','mobilenetv2','ESF','GOOD']:
        results = pd.read_csv("results/k_fold_results_"+simtype+"_restaurant.csv")
        results=results[results.columns[-2:]]
        bplot = sns.boxplot(x='model',
                            data=results,
                           y='score',
                     palette="colorblind")
        bplot.set(ylabel="Accuracy Score", xlabel="")
        bplot.set_title(simtype+" descriptor")
        bplot.figure.savefig("plot/"+simtype+".png",
                        format='png',
                        dpi=100)
