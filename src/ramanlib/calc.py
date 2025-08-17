import ramanspy as rp
import numpy as np
import pandas as pd


def outliers_per_group(gsc, metric, by=None, n_spectra=3, highest=True):
    """
    Compute, per group, the indices of the n highest/lowest spectra by a metric
    against the group's mean spectrum. Also returns the group mean (Spectrum).

    Parameters
    ----------
    gsc : GroupedSpectralContainer
    metric : callable
        Like rp.metrics.* with signature metric(spec_a: rp.Spectrum, spec_b: rp.Spectrum) -> float.
    by : str | list[str] | callable | None
        Grouping key for pandas .groupby. If None, the whole dataset is one group.
    n_spectra : int
        Number of spectra to select per group (clipped to group size).
    highest : bool
        If True, select largest metric values; else smallest.

    Returns
    -------
    results : dict[str, tuple[list[int], rp.Spectrum]]
        { group_label: ([row_indices_into_gsc_df], mean_spectrum) }
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
    Compute the difference in mean spectra and 95% confidence interval band.

    group1_stats and group2_stats are gsc rows containing the mean spectrum in column "spectrum" and containing
    stats columns "n", "var_vector" and "std_vector" for group statistics.

    Returns:
        - Spectrum object for the difference
        - np.ndarray for the CI band at each wavenumber
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
    Compute the Pearson correlation matrix between mean spectra of each group in a GroupedSpectralContainer.

    Parameters:
        gsc (GroupedSpectralContainer): The grouped spectral data.
        by (str): Column name to group by.

    Returns:
        correlation_matrix (pd.DataFrame): Correlation matrix between mean spectra of each group.
    """
    group_means_gsc = gsc.mean(by=by)
    spectral_data = [row["spectrum"].spectral_data for _, row in group_means_gsc.df.iterrows()]
    group_keys = [", ".join(map(str, k)) if isinstance(k, tuple) else str(k) for k in group_means_gsc.df.groupby(by).groups.keys()]
    df_group_means = pd.DataFrame({k: v for k, v in zip(group_keys, spectral_data)})
    correlation_matrix = df_group_means.corr(method='pearson')
    return correlation_matrix