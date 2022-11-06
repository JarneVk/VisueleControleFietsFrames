from torch.utils.data import DataLoader
import joblib
import matplotlib.pyplot as plt

train_x = joblib.load("python/CV/train_x.joblib")
test_y = joblib.load("python/CV/test_y.joblib")

