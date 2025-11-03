"""
Computations over grouped Raman spectra.

This module provides analysis helpers that operate on a
:class:`ramanlib.core.GroupedSpectralContainer` (GSC) and return results designed
to feed directly into plotting functions in :mod:`ramanlib.plot`:

- :func:`outliers_per_group`  →  :func:`ramanlib.plot.outliers_per_group`
- :func:`mean_difference`     →  :func:`ramanlib.plot.mean_difference`
- :func:`mean_correlation_per_group` → :func:`ramanlib.plot.mean_correlation_per_group`

Notes
-----
All grouping semantics mirror :meth:`pandas.DataFrame.groupby`. Unless stated
otherwise, group labels are rendered from the grouping keys (tuples become
comma-separated strings).

Examples
--------
Select outliers and plot them::

    results = outliers_per_group(gsc, metric=rp.metrics.euclidean, by="sample", n_spectra=3)
    ramanlib.plot.outliers_per_group(gsc, results)
"""

from __future__ import annotations

import ramanspy as rp
import numpy as np
import pandas as pd


def outliers_per_group(gsc, metric, by=None, n_spectra=3, highest=True):
    """
    Select per-group outlier indices according to a pairwise metric vs. the group mean.

    For each group (or the entire container if ``by is None``), compute the mean
    spectrum, score each row's spectrum against that mean using ``metric``, and
    return the indices of the top/bottom ``n_spectra`` according to the scores.
    Also returns the group's mean :class:`ramanspy.Spectrum`.

    Parameters
    ----------
    gsc : GroupedSpectralContainer
        Input container with a ``'spectrum'`` column of :class:`ramanspy.Spectrum`.
    metric : callable
        Pairwise metric with signature
        ``metric(spec_a: rp.Spectrum, spec_b: rp.Spectrum) -> float``.
        Typical choices are in :mod:`ramanspy.metrics` (e.g., ``MAE``,
        ``MSE``).
    by : str or list[str] or callable or None, optional
        Grouping key(s) passed to :meth:`pandas.DataFrame.groupby`. If ``None``,
        all rows are treated as one group labeled ``"all"``.
    n_spectra : int, optional
        Number of spectra to select per group (clipped to the group size). Default ``3``.
    highest : bool, optional
        If ``True`` (default), select the largest metric values; if ``False``,
        select the smallest.

    Returns
    -------
    dict[str, tuple[list[int], rp.Spectrum]]
        Mapping
        ``{ group_label: ([row_indices_into_gsc_df], mean_spectrum) }``.
        Indices are global row indices into ``gsc.df``.

    See Also
    --------
    ramanlib.plot.outliers_per_group
        Plot the selected spectra per group and overlay the mean.

    Notes
    -----
    The mean spectrum is computed via :meth:`ramanspy.SpectralContainer.mean`
    after stacking the group's spectra.

    """
    grouped = [("all", gsc.df)] if by is None else list(gsc.df.groupby(by))

    results = {}
    for key, group_df in grouped:
        if group_df.empty:
            continue

        spectra = group_df["spectrum"].tolist()
        cont = rp.SpectralContainer.from_stack(spectra)
        mean_spec = cont.mean

        scores = np.array([metric(spec, mean_spec) for spec in spectra])
        order = np.argsort(scores)
        if highest:
            order = order[::-1]

        k = min(n_spectra, len(group_df))
        pick_local = order[:k]
        pick_global = group_df.index.values[pick_local].tolist()

        label = ", ".join(map(str, key)) if isinstance(key, tuple) else str(key)
        results[label] = (pick_global, mean_spec)

    return results


def mean_difference(group1_stats, group2_stats, ci_z=1.96):
    """
    Compute difference of group mean spectra and a normal-approximation CI band.

    Parameters
    ----------
    group1_stats : GroupedSpectralContainer
        Container with exactly **one row** representing group 1's statistics, as
        produced by :meth:`ramanlib.core.GroupedSpectralContainer.mean` with
        ``include_stats=True``. Must contain columns: ``"spectrum"``, ``"n"``,
        ``"var_vector"``, and ``"std_vector"``.
    group2_stats : GroupedSpectralContainer
        Same format/requirements as ``group1_stats`` for group 2.
    ci_z : float, optional
        Z-score for a two-sided normal CI (e.g., ``1.96`` ≈ 95%). Default ``1.96``.

    Returns
    -------
    rp.Spectrum
        The difference spectrum ``(group1_mean - group2_mean)`` with the same
        spectral axis as the inputs.
    numpy.ndarray
        One-dimensional nonnegative array giving the half-width of the symmetric
        CI band at each wavenumber, computed as
        ``ci_z * sqrt(var1/n1 + var2/n2)``.

    Raises
    ------
    ValueError
        If a stats container is missing required columns or contains
        more/less than one row.

    See Also
    --------
    ramanlib.plot.mean_difference
        Plot the difference spectrum with its CI band.
    ramanlib.core.GroupedSpectralContainer.mean
        Produces the required stats columns when ``include_stats=True``.

    Notes
    -----
    This uses the usual normal approximation for a difference of means with
    independent groups:
    ``Var(diff) = Var(mean1) + Var(mean2) = var1/n1 + var2/n2``.

    """

    if any(stat not in group1_stats.df.columns for stat in ["n", "var_vector", "std_vector"]) or (len(group1_stats) != 1):
        raise ValueError("group1_stats missing statistics columns or includes multiple rows. Use include_stats=True in \
GSC.mean() to ensure stats are included.")
    
    s1 = group1_stats["spectrum"].iloc[0]
    s2 = group2_stats["spectrum"].iloc[0]

    diff = s1.spectral_data - s2.spectral_data
    axis = s1.spectral_axis
    diff_spectrum = rp.Spectrum(diff, axis)

    var1 = group1_stats["var_vector"].iloc[0]
    var2 = group2_stats["var_vector"].iloc[0]
    n1 = group1_stats["n"].iloc[0]
    n2 = group2_stats["n"].iloc[0]

    ci_band = ci_z * np.sqrt((var1 / n1) + (var2 / n2))

    return diff_spectrum, ci_band


def mean_correlation_per_group(gsc, by):
    """
    Pearson correlation matrix between per-group mean spectra.

    Parameters
    ----------
    gsc : GroupedSpectralContainer
        Input container.
    by : str
        Column name to group by when computing the means.

    Returns
    -------
    pandas.DataFrame
        Square correlation matrix (index and columns are group labels) computed
        from the stacked intensity vectors of each group's mean spectrum.

    See Also
    --------
    ramanlib.plot.mean_correlation_per_group
        Heatmap visualization of the returned matrix.
    ramanlib.core.GroupedSpectralContainer.mean
        Computes per-group mean spectra.

    Notes
    -----
    Groups are ordered according to the key order in ``groupby(by)``. The matrix
    is computed by forming a DataFrame whose columns are the intensity vectors of
    each group's mean spectrum and calling :meth:`pandas.DataFrame.corr`
    with ``method="pearson"``.

    """
    group_means_gsc = gsc.mean(by=by)
    spectral_data = [row["spectrum"].spectral_data for _, row in group_means_gsc.df.iterrows()]
    group_keys = [", ".join(map(str, k)) if isinstance(k, tuple) else str(k) for k in group_means_gsc.df.groupby(by).groups.keys()]
    df_group_means = pd.DataFrame({k: v for k, v in zip(group_keys, spectral_data)})
    correlation_matrix = df_group_means.corr(method='pearson')
    return correlation_matrix