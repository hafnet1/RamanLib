import numpy as np
import ramanspy as rp
from .utils import _normalize_axes_obj
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def mean_per_group(gsc, by=None, interval=None, plot_type="separate", ci_z=1.96, **kwargs):
    """
    Plot mean spectrum per group using precomputed statistics from GSC.mean(include_stats=True).
    interval: None | 'ci' | 'sd'
        'ci' plots ± z * (std / sqrt(n)); 'sd' plots ± std.
    """
    # 1) Compute group means + stats once
    means_gsc = gsc.mean(by=by, include_stats=True, ddof=1)
    df = means_gsc.df

    # 2) Prepare means and labels
    group_means = df["spectrum"].tolist()
    
    # Use user labels if provided; otherwise build them. Avoid double-passing via kwargs.
    group_labels = kwargs.pop("label", None)
    if group_labels is None:
        if by is None:
            group_labels = ["all"]
        else:
            grouped = df.groupby(by, dropna=False)
            group_labels = [
                ", ".join(map(str, key)) if isinstance(key, tuple) else str(key)
                for key, _ in grouped
            ]
    spectral_axis = group_means[0].spectral_axis if group_means else None

    # 3) Precompute bands if requested
    bands = [None] * len(df)
    if interval in ("ci", "sd"):
        if "std_vector" not in df or (interval == "ci" and "n" not in df):
            warnings.warn("Required statistics not present; skipping interval bands.")
        else:
            for i, row in df.iterrows():
                std = row["std_vector"]
                if std is None or (isinstance(std, np.ndarray) and std.size == 0):
                    bands[i] = None
                    continue
                if interval == "sd":
                    band = std
                else:  # 'ci'
                    n = int(row["n"])
                    band = (ci_z * std / np.sqrt(n)) if n > 0 else None
                bands[i] = band

    # 4) Plot means
    axes_obj = rp.plot.spectra(group_means, label=group_labels, plot_type=plot_type, **kwargs)
    axes = _normalize_axes_obj(axes_obj)

    # 5) Handle stacked-with-offset limitation
    if (plot_type or "").lower() == "single stacked" and interval is not None:
        warnings.warn("Interval bands disabled for 'single stacked' due to vertical offsets.")
        return axes_obj

    # 6) Overlay bands on the correct axes
    if spectral_axis is not None and any(b is not None for b in bands):
        for ax, mean_spec, band in zip(axes, group_means, bands):
            if band is None:
                continue
            ax.fill_between(
                spectral_axis,
                mean_spec.spectral_data - band,
                mean_spec.spectral_data + band,
                alpha=0.2
            )

    return axes_obj


def random_per_group(gsc, by=None, n_samples=3, plot_type="single", seed=None, **kwargs):
    """
    Plot n random spectra from each group in the container.

    Parameters:
        gsc (GroupedSpectralContainer)
        by (str or list or callable): Metadata column(s) or grouping method.
        n_samples (int): Number of spectra to sample per group.
        plot_type (str): Passed to rp.plot.spectra.
        seed (int or None): Random seed for reproducible sampling. Default None.
        **kwargs: Forwarded to rp.plot.spectra.

    Returns:
        Axes | list[Axes] | Figure: Whatever rp.plot.spectra returns.
    """
    rng = random.Random(seed)  # local RNG

    def _sample_k(spectra, k):
        if len(spectra) == 0:
            return []
        if k <= len(spectra):
            return rng.sample(spectra, k)
        return spectra[:] + rng.choices(spectra, k=k - len(spectra))

    if by is None:
        spectra = gsc.df["spectrum"].tolist()
        spectra_groups = [_sample_k(spectra, n_samples)]
        group_labels = ["all"]
    else:
        grouped = gsc.df.groupby(by)
        spectra_groups, group_labels = [], []
        for key, group_df in grouped:
            sample = _sample_k(group_df["spectrum"].tolist(), n_samples)
            spectra_groups.append(sample)
            group_labels.append(", ".join(map(str, key)) if isinstance(key, tuple) else str(key))

    return rp.plot.spectra(spectra_groups, label=group_labels, plot_type=plot_type, **kwargs)


def outliers_per_group(gsc, results, **kwargs):
    """
    Plot outlier spectra per group and overlay the group mean.
    Layout is fixed to 'separate' for robustness. Additional keyword args
    are forwarded to rp.plot.spectra (e.g., color, linewidth, title, ax, etc.).

    Parameters
    ----------
    gsc : GroupedSpectralContainer
    results : dict[str, tuple[list[int], rp.Spectrum]]
        Output of calc_outlier_indices_by_group: {group_label: (row_indices, mean_spectrum)}
    **kwargs :
        Forwarded to rp.plot.spectra, EXCEPT 'plot_type' which is ignored.

    Returns
    -------
    axes_obj :
        Whatever rp.plot.spectra returns (Axes | list[Axes] | Figure).
    """
    if not results:
        return None

    # Don’t allow plot_type injection here to avoid edge cases.
    if "plot_type" in kwargs:
        warnings.warn("plot_type is fixed to 'separate' for this plot; ignoring provided plot_type.")
        kwargs = {k: v for k, v in kwargs.items() if k != "plot_type"}

    group_labels = list(results.keys())

    spectra_groups = []
    means_for_overlay = []
    for label in group_labels:
        idxs, mean_spec = results[label]
        spectra_groups.append(gsc.df.loc[idxs, "spectrum"].tolist())
        means_for_overlay.append(mean_spec)

    axes_obj = rp.plot.spectra(
        spectra_groups,
        label=group_labels,
        plot_type="separate",
        **kwargs
    )

    # Normalize to a list of Axes to overlay the mean
    if isinstance(axes_obj, list):
        axes_list = axes_obj
    elif hasattr(axes_obj, "fill_between"):   # single Axes
        axes_list = [axes_obj]
    elif hasattr(axes_obj, "get_axes"):       # Figure
        axes_list = axes_obj.get_axes()
    else:
        axes_list = []

    # Overlay the mean line (no labels/legend/tight_layout/show here)
    for ax, mean_spec in zip(axes_list, means_for_overlay):
        ax.plot(mean_spec.spectral_axis, mean_spec.spectral_data, color="red", linewidth=1.5)

    return axes_obj


def baseline(spectrum, baseline_process, **kwargs):
    '''Plot the baseline resulting from a single baseline process alongside its raw and baseline-subtracted spectra'''
    corrected_spectrum = baseline_process.apply(spectrum)
    baseline = rp.Spectrum(spectrum.spectral_data - corrected_spectrum.spectral_data, spectrum.spectral_axis)
    spectra = [spectrum, baseline, corrected_spectrum]
    labels = ["Original spectrum", "removed baseline", "corrected spectrum"]
    return rp.plot.spectra(spectra, label=labels, plot_type="single", alpha=0.9, **kwargs)


def n_baselines(raw_gsc, baseline_process, process_name, n_samples=3, figsize=(8,7), seed=None):
    '''Plot n randomly selected spectra along with their baselines and corrected spectra in a single figure
    with mulitple subplots'''
    spec_samples = raw_gsc.df.sample(n=n_samples)["spectrum"]
    fig, axs = plt.subplots(n_samples, 1, figsize=figsize)
    for i, spec in enumerate(spec_samples):
        baseline(spec, baseline_process, ax=axs[i], title="", xlabel="")
    fig.suptitle(f"{process_name}")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.tight_layout()
    return axs


def compare_baselines(spectrum, baseline_processes, process_names, figsize=(8,7)):
    '''Plot a single spectrum with multiple different baseline processes applied to compare how
    the algorithm handles that spectrum'''
    fig, axs = plt.subplots(len(baseline_processes), 1, figsize=figsize)
    for i, process in enumerate(baseline_processes):
        baseline(spectrum, process, ax=axs[i], title=f"{process_names[i]}", xlabel="")
    plt.xlabel("Wavenumber(cm⁻¹)")
    plt.tight_layout()
    return axs


def mean_difference(diff_spectrum, ci_band, label="Difference in Means", **kwargs):
    """
    Plot the difference between two mean spectra with a 95% CI band centered at 0.

    Parameters:
        diff_spectrum (rp.Spectrum): Difference between two mean spectra.
        ci_band (np.ndarray): 1D array of CI boundsz.
        title (str): Plot title.
        color (str): Line color.
        **kwargs: Additional matplotlib parameters forwarded to rp.plot.spectra().
    """

    if "plot_type" in kwargs.keys():
        warnings.warn('Only plot_type="single" is supported for mean_difference')
        kwargs.pop("plot_type")

    ax_obj = rp.plot.spectra(diff_spectrum, label=label, plot_type="single", **kwargs)
    axs = _normalize_axes_obj(ax_obj)
    axs[0].fill_between(diff_spectrum.spectral_axis, -ci_band, ci_band, color='gray', alpha=0.3, label='95% Confidence Band')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    return ax_obj


def mean_correlation_per_group(
    correlation_matrix,
    title="Correlation Matrix of Raman Spectra",
    vmin=0,
    vmax=1,
    annot=True,
    cmap="coolwarm",
    figsize=(8, 6),
    **kwargs
):
    """
    Plot a heatmap of the correlation matrix between group mean spectra.

    Parameters:
        correlation_matrix (pd.DataFrame): Correlation matrix to plot.
        title (str): Title of the plot.
        vmin (float): Minimum value for heatmap color scale.
        vmax (float): Maximum value for heatmap color scale.
        annot (bool): Whether to annotate cells with values.
        cmap (str): Color map to use for the heatmap.
        figsize (tuple): Figure size.
        **kwargs: Additional keyword arguments passed to seaborn.heatmap().
    """
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    plt.title(title)