import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")
sns.set(font_scale=1.2)


if __name__ == "__main__":

    descriptor = "ESF"
    knn = pd.read_csv(f"results/results_open_ended_knn_{descriptor}_washington.csv")
    mlp = pd.read_csv(f"results/results_open_ended_mlp_{descriptor}_washington.csv")
    knn["n_classes"] += 1
    mlp["n_classes"] += 1
    knn.rename(
        columns={
            "n_classes": "Number of classes",
            "accuracy": "Accuracy",
            "clf_time": "Classification time",
            "train_time": "Retrain time",
        },
        inplace=True
    )
    mlp.rename(
        columns={
            "n_classes": "Number of classes",
            "accuracy": "Accuracy",
            "clf_time": "Classification time",
            "train_time": "Retrain time",
        },
        inplace=True
    )

    ax = sns.lineplot(x="Number of classes", y="Accuracy", data=knn, label="KNN")
    sns.lineplot(x="Number of classes", y="Accuracy", data=mlp, ax=ax, label="MLP")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.show()

    ax = sns.lineplot(x="Number of classes", y="Classification time", data=knn, label="KNN")
    sns.lineplot(x="Number of classes", y="Classification time", data=mlp, ax=ax, label="MLP")
    plt.legend()
    plt.show()

    ax = sns.lineplot(x="Number of classes", y="Retrain time", data=knn, label="KNN")
    sns.lineplot(x="Number of classes", y="Retrain time", data=mlp, ax=ax, label="MLP")
    plt.legend()
    plt.show()
    import pdb; pdb.set_trace()
