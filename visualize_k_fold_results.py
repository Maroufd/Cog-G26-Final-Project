import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")
sns.set(font_scale=1.2)


if __name__ == "__main__":
    for descriptor in ["ESF", "GOOD"]:
        results = pd.read_csv(f"results/k_fold_results_{descriptor}_restaurant.csv")
        results = results[results.columns[-2:]]
        bplot = sns.boxplot(x="model", data=results, y="score", palette="colorblind")
        bplot.set(ylabel="Accuracy Score", xlabel="")
        bplot.set_title(f"{descriptor} descriptor")
        bplot.figure.savefig(f"plot/{descriptor}.png", format="png", dpi=100)
