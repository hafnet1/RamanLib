"""
Analysis routines for Raman spectral data. This focuses on single-spectrum analysis
methods or methods that don't require heavy grouping or metadata management.

This module currently provides a Classical Least Squares (CLS) unmixing routine
that decomposes a query spectrum into a linear combination of reference
(component) spectra using linear regression. Outputs include the fitted
coefficients, the residual spectrum, and the scaled component spectra that best
fit the query under a least-squares criterion.

See Also
--------
ramanspy.Spectrum
    RamanSPy spectrum class used throughout.
sklearn.linear_model.LinearRegression
    Estimator used to obtain CLS coefficients.

"""

from __future__ import annotations

import numpy as np
import ramanspy as rp
from sklearn import linear_model


def CLS(query_spec, components_spec, component_names, plot=True, verbose=True):
    """
    Classical Least Squares (CLS) spectral unmixing.

    Parameters
    ----------
    query_spec : rp.Spectrum
    components_spec : list[rp.Spectrum]
    component_names : list[str]
    plot : bool, optional
        If True, plot query, residual, and fitted component spectra. Default True.
    verbose : bool, optional
        If True, print component names and coefficients. Default True.

    Returns
    -------
    cs : numpy.ndarray
    res_spec : rp.Spectrum
    fitted_components_spec : list[rp.Spectrum]
    """
    # Basic checks
    assert all(len(query_spec.spectral_axis) == len(c.spectral_axis) for c in components_spec), (
        "All components must have the same spectral axis length as the mixture spectrum"
    )
    if len(component_names) != len(components_spec):
        raise ValueError("component_names must match components_spec in length and order.")

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
    
    # Verbose print
    if verbose:
        print("components:\n" + "\n".join(f"{name}, {c}" for name, c in zip(component_names, cs)))

    # Optional plot
    if plot:
        # query, residual, and fitted components
        rp.plot.spectra(
            [query_spec, res_spec] + fitted_components_spec,
            label=["query", "residual"] + component_names,
            plot_type="single",
            alpha=0.8,
        )

    return cs, res_spec, fitted_components_spec