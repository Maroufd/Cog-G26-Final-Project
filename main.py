from data_utils import extract_X_y
from mlp import make_mlp
from lvq import train_lvq

if __name__ == "__main__":
    X, y, preprocessor = extract_X_y("GOOD_washington")
    model = make_mlp(X, y)
    # lvqnet= train_lvq(X,y)
