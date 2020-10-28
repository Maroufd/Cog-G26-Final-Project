from data_utils import extract_X_y
from mlp import make_mlp


if __name__ == "__main__":
    X, y, preprocessor = extract_X_y("GOOD_histograms")
    model = make_mlp(X, y)
