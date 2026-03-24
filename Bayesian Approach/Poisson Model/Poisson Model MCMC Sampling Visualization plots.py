for degree in range(1, 3):

    print(f"\n--- Degree {degree} ---")

    model = PolynomialRegression2D(degree=degree)
    model.fit(x, y, z)

    curr = np.exp(z)

    samples = run_mcmc_poisson(
        x, y, curr,
        model.polynomial_model,
        degree,
        initial_beta=model.beta_hat
    )

    k = samples.shape[1]   

    # Scatter
    plt.figure()
    plt.plot(samples[:, 0], samples[:, 1], 'k.', alpha=0.1)
    plt.show()

    # Trace
    fig, axes = plt.subplots(k, 1, figsize=(10, 2*k), sharex=True)

    for i in range(k):
        axes[i].plot(samples[:, i], alpha=0.5)
        axes[i].set_ylabel(f"b{i}")

    plt.show()

    # Corner
    labels = [f"b{i}" for i in range(k)]

    corner.corner(
        samples,
        labels=labels,
        bins=30,
        smooth=1.0,
        plot_datapoints=True,
        fill_contours=True
    )
    plt.show()
