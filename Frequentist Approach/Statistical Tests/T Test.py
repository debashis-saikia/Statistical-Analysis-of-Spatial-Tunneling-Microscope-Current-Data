import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Gaussian_Analysis2d import PolynomialRegression2D

def design_matrix_2d(x, y, degree):
    terms = []

    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x**i) * (y**j))

    X = np.vstack(terms).T
    return X

def beta_covariance_2d(x, y, sigma2, degree):
    X = design_matrix_2d(x, y, degree)

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)

    cov_beta = sigma2 * XtX_inv

    return cov_beta

from scipy.stats import t

def t_test_2d(beta_hat, cov_beta, N):
    """
    Returns t-stats and p-values for each parameter.
    """

    p = len(beta_hat)
    se = np.sqrt(np.diag(cov_beta))
    se = np.where(se < 1e-12, 1e-12, se)

    t_stats = beta_hat / se
    df = N - p
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df))

    return t_stats, p_values

def beta_confidence_intervals(beta_hat, cov_beta, N, alpha=0.05):
    """
    Confidence intervals for regression coefficients.
    """

    p = len(beta_hat)
    df = N - p

    t_crit = t.ppf(1 - alpha/2, df)

    se = np.sqrt(np.diag(cov_beta))

    lower = beta_hat - t_crit * se
    upper = beta_hat + t_crit * se

    return lower, upper

def basis_vector_2d(x, y, degree):
    phi = []

    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            phi.append((x**i) * (y**j))

    return np.array(phi)

def prediction_confidence_interval_2d(x0, y0, beta_hat, cov_beta, sigma2, N, degree, alpha=0.05):
    """
    Confidence interval for prediction at a single (x0, y0).
    """

    p = len(beta_hat)
    df = N - p

    # mean prediction
    phi = basis_vector_2d(x0, y0, degree)
    z_hat = phi @ beta_hat

    # variance of prediction
    var_pred = sigma2 * (phi @ cov_beta @ phi)

    t_crit = t.ppf(1 - alpha/2, df)

    margin = t_crit * np.sqrt(var_pred)

    lower = z_hat - margin
    upper = z_hat + margin

    return z_hat, lower, upper

def prediction_ci_grid(x, y, beta_hat, cov_beta, sigma2, N, degree, alpha=0.05):
    """
    Confidence intervals over grid points.
    """

    p = len(beta_hat)
    df = N - p
    t_crit = t.ppf(1 - alpha/2, df)

    z_hat = []
    lower = []
    upper = []

    for xi, yi in zip(x, y):
        phi = basis_vector_2d(xi, yi, degree)

        mean = phi @ beta_hat
        var = sigma2 * (phi @ cov_beta @ phi)

        margin = t_crit * np.sqrt(var)

        z_hat.append(mean)
        lower.append(mean - margin)
        upper.append(mean + margin)

    return np.array(z_hat), np.array(lower), np.array(upper)



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


for d, model in enumerate(models):
    if d == 0:
        continue

    print(f"\n========== Degree {d} ==========")

    # covariance
    cov_beta = beta_covariance_2d(
        x, y,
        model.sigma2_hat,
        model.degree
    )

    # t-test
    t_stats, p_vals = t_test_2d(
        model.beta_hat,
        cov_beta,
        model.N
    )

    print("\nParameter Significance (t-test)")
    print("--------------------------------")

    for i, (t_val, p_val) in enumerate(zip(t_stats, p_vals)):
        print(f"beta_{i}: t = {t_val:.4f}, p = {p_val:.4e}")

    # confidence intervals for predictions
    z_hat, lower, upper = prediction_ci_grid(
        x, y,
        model.beta_hat,
        cov_beta,
        model.sigma2_hat,
        model.N,
        model.degree
    )

    avg_width = np.mean(upper - lower)
    print(f"\nAverage CI width: {avg_width:.4f}")

ci_widths = []

for d, model in enumerate(models):
    if d == 0:
        continue

    cov_beta = beta_covariance_2d(x, y, model.sigma2_hat, model.degree)

    _, lower, upper = prediction_ci_grid(
        x, y,
        model.beta_hat,
        cov_beta,
        model.sigma2_hat,
        model.N,
        model.degree
    )

    ci_widths.append(np.mean(upper - lower))


plt.figure()
plt.plot(range(1, len(ci_widths)+1), ci_widths, marker='o')
plt.xlabel("Degree")
plt.ylabel("Average CI width")
plt.title("Model Uncertainty vs Degree")
plt.show()
