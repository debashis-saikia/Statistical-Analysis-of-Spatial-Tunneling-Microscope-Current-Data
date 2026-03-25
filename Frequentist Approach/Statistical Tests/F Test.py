from scipy.stats import f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Gaussian_Analysis2d import PolynomialRegression2D


def f_test(RSS_d, RSS_d1, N, p_d, p_d1):
    k = p_d1 - p_d

    numerator = (RSS_d - RSS_d1) / k
    denominator = RSS_d1 / (N - p_d1)

    F_stat = numerator / denominator

    p_value = 1 - f.cdf(F_stat, k, N - p_d1)

    return F_stat, p_value

from scipy.stats import f

def f_confidence_interval(F_stat, df1, df2, alpha=0.05): #95% Confidence interval
    F_lower = f.ppf(alpha/2, df1, df2)
    F_upper = f.ppf(1 - alpha/2, df1, df2)

    lower = F_stat / F_upper
    upper = F_stat / F_lower

    return lower, upper

models = []

data = pd.read_csv(r"C:\Users\IISER13\OneDrive\Desktop\PHY 5032\processed_stm_dataset_1.csv")

x = data["x_nm"].values
y = data["y_nm"].values
z = data["log_current"].values

x = x[:7000]
y = y[:7000]
z = z[:7000]

max_degree = 5
for d in range(max_degree +1):
    model = PolynomialRegression2D(d)
    model.fit(x, y, z)
    models.append(model)

for d in range(len(models) - 1):
    m1 = models[d]
    m2 = models[d+1]

    p_d = len(m1.beta_hat)
    p_d1 = len(m2.beta_hat)

    F_stat, p_val = f_test(
        m1.RSS,
        m2.RSS,
        m1.N,
        p_d,
        p_d1
    )

    print(f"d={d} vs d={d+1}: p = {p_val}")

for d in range(len(models) - 1):
    m1 = models[d]
    m2 = models[d+1]

    print(f"\n--- d={d} vs d={d+1} ---")
    print("RSS_d      =", m1.RSS)
    print("RSS_d+1    =", m2.RSS)

    print("ΔRSS       =", m1.RSS - m2.RSS)

    print("p_small    =", len(m1.beta_hat))
    print("p_large    =", len(m2.beta_hat))
    print("N          =", m1.N)

for d in range(len(models) - 1):
    m1 = models[d]
    m2 = models[d+1]

    p_d = len(m1.beta_hat)
    p_d1 = len(m2.beta_hat)

    k = p_d1 - p_d
    df2 = m1.N - p_d1

    F_stat, p_val = f_test(
        m1.RSS,
        m2.RSS,
        m1.N,
        p_d,
        p_d1
    )

    lower, upper = f_confidence_interval(F_stat, k, df2)

    print(f"\n--- d={d} vs d={d+1} ---")
    print(f"F = {F_stat:.4f}, p = {p_val:.4e}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
