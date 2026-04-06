import numpy as np
import matplotlib.pyplot as plt
from Gaussian_Analysis2d import PolynomialRegression2D
import pandas as pd
import scipy.stats as stats

data = pd.read_csv(r"C:\Users\IISER13\OneDrive\Desktop\PHY 5032\processed_stm_dataset_1.csv")

x = data["x_nm"].values
y = data["y_nm"].values
z = data["log_current"].values

x = x[:100000]
y = y[:100000]
z = z[:100000]

model = PolynomialRegression2D(degree=k)
model.fit(x, y, z)
model.model_info()
z_pred = model.predict(x, y)
residuals = z - z_pred

# Histogram
plt.hist(residuals, bins=50)
plt.title("Residual Histogram")
plt.show()

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()
