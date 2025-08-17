def CLS(query_spec, components_spec, component_names, plot=True, verbose=True):
    """
    Perform classical least squares (CLS) spectral unmixing.

    Decomposes a query spectrum into a linear combination of given component spectra
    using linear regression. Returns the fitted coefficients, residual spectrum, and
    individual component contributions. Optionally displays a plot and prints the results.

    Parameters:
        query_spec (rp.Spectrum): The target spectrum to be decomposed.
        components_spec (list of rp.Spectrum): Reference component spectra.
        component_names (list of str): Names corresponding to each component.
        plot (bool): If True, plots the query, residual, and component spectra.
        verbose (bool): If True, prints component names and their corresponding coefficients.

    Returns:
        tuple:
            - cs (np.ndarray): Fitted CLS coefficients.
            - res_spec (rp.Spectrum): Residual spectrum after fitting.
            - fitted_components_spec (list of rp.Spectrum): Scaled component spectra.
    """
    
    # Check that all components have the same number of datapoints as the query spectrum
    assert all(len(query_spec.spectral_axis) == len(component.spectral_axis) for component in components_spec), "All components must have the same spectral axis length as the mixture spectrum"

    # Get the CLS coefficients
    components = np.array([component.spectral_data for component in components_spec])
    query = query_spec.spectral_data
    cs = linear_model.LinearRegression().fit(components.T, query).coef_

    # Calculate the residual spectrum and fitted components spectra
    res = query_spec.spectral_data.copy()
    fitted_components_spec = []
    for c, component in zip(cs, components):
        res -= c * component
        fitted_components_spec.append(rp.Spectrum(c * component, query_spec.spectral_axis))
    
    res_spec = rp.Spectrum(res, query_spec.spectral_axis)
    
    print("components:\n" + "\n".join(f"{name}, {c}" for name, c in zip(component_names, cs)))

    # First figure: query, residual, and fitted components
    rp.plot.spectra(
        [query_spec, res_spec] + fitted_components_spec,
        label=["query", "residual"] + component_names,
        plot_type="single",
        alpha=0.8,
    )

    return cs, res_spec, fitted_components_spec