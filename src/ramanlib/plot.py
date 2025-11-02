"""
Plotting utilities for RamanLib.

This module provides convenience functions to visualize spectra stored in a
:class:`ramanlib.core.GroupedSpectralContainer` (GSC) and derived statistics
computed by :mod:`ramanlib.calc`. Functions are designed so that common analysis
outputs feed directly into plotting helpers—for example:

- :func:`ramanlib.calc.outliers_per_group` → :func:`ramanlib.plot.outliers_per_group`
- :func:`ramanlib.calc.mean_difference` → :func:`ramanlib.plot.mean_difference`
- :func:`ramanlib.calc.mean_correlation_per_group` → :func:`ramanlib.plot.mean_correlation_per_group`

Most functions delegate the actual drawing to :mod:`ramanspy`’s plotting backend
and Matplotlib/Seaborn, adding light logic for grouping, sampling, and overlays.

Notes
-----
Return types mirror whatever the underlying plotting call returns (usually a
Matplotlib :class:`matplotlib.axes.Axes`, a list of Axes, or a
:class:`matplotlib.figure.Figure`), so these functions compose naturally with
your own Matplotlib pipelines (titles, labels, styling).
"""

from __future__ import annotations

import numpy as np
import ramanspy as rp
from .utils import _normalize_axes_obj
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def mean_per_group(gsc, by=None, interval=None, plot_type="separate", ci_z=1.96, **kwargs):
    """
    Plot mean spectrum per group with optional uncertainty bands.

    The group means and (optionally) per-wavenumber standard deviation and
    confidence intervals are computed via :meth:`gsc.mean` with
    ``include_stats=True``. Bands can represent standard deviation (``±SD``) or
    normal-approximation confidence intervals (``± z * SD / sqrt(n)``).

    Parameters
    ----------
    gsc : GroupedSpectralContainer
        Input container.
    by : str or list of str or None, optional
        Column name(s) used to form groups. If ``None``, all rows form a single
        group named ``"all"``. Passed to :meth:`pandas.DataFrame.groupby`.
    interval : {None, "ci", "sd"}, optional
        Type of band to draw around the mean. ``"sd"`` plots ``± SD``.
        ``"ci"`` plots ``± z * SD / sqrt(n)`` with ``z=ci_z``.
        ``None`` (default) disables bands.
    plot_type : {"separate", "single", "single stacked"}, optional
        Plot style passed to :func:`ramanspy.plot.spectra`. Default is
        ``"separate"`` (one subplot per group). For ``"single stacked"``,
        interval bands are disabled due to vertical offsets.
    ci_z : float, optional
        Z-score for CI bands (e.g., ``1.96`` ≈ 95% CI). Ignored if
        ``interval != "ci"``. Default is ``1.96``.
    **kwargs
        Forwarded to :func:`ramanspy.plot.spectra` and Matplotlib.

    Returns
    -------
    matplotlib.axes.Axes or list[matplotlib.axes.Axes] or matplotlib.figure.Figure
        Whatever :func:`ramanspy.plot.spectra` returns.

    See Also
    --------
    ramanlib.core.GroupedSpectralContainer.mean
        Computes group means and optional per-wavenumber statistics.
    ramanlib.plot.random_per_group
        Sample and plot random spectra per group.

    Notes
    -----
    Bands require the presence of ``'std_vector'`` and, for ``interval="ci"``,
    also ``'n'`` in the dataframe returned by ``gsc.mean(include_stats=True)``.

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


def random_per_group(gsc, by=None, n_samples=3, plot_type="single", seed=None, **kwargs): # Note: fix 'label' attribute if user gives this as input
    """
    Plot a random sample of spectra from each group.

    Parameters
    ----------
    gsc : GroupedSpectralContainer
        Input container.
    by : str or list of str or None, optional
        Column name(s) to group by. If ``None``, samples from all rows as one group.
    n_samples : int, optional
        Number of spectra to sample per group. If a group has fewer than
        ``n_samples`` rows, sampling with replacement is used to reach ``n_samples``.
        Default is ``3``.
    plot_type : {"single", "separate", "single stacked"}, optional
        Plot style passed to :func:`ramanspy.plot.spectra`. Default ``"single"``.
    seed : int or None, optional
        Random seed for reproducibility. Default ``None``.
    **kwargs
        Forwarded to :func:`ramanspy.plot.spectra` and Matplotlib.

    Returns
    -------
    matplotlib.axes.Axes or list[matplotlib.axes.Axes] or matplotlib.figure.Figure
        Whatever :func:`ramanspy.plot.spectra` returns.

    See Also
    --------
    ramanlib.plot.mean_per_group
        Plot group means with optional uncertainty bands.

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
    Plot detected outlier spectra for each group and overlay the group mean.

    This is the plotting counterpart of :func:`ramanlib.calc.outliers_per_group`.
    It expects the exact mapping produced by that function and draws, for each
    group, the selected "outlier" spectra in a separate subplot, with the group's
    mean spectrum overlaid.

    Parameters
    ----------
    gsc : GroupedSpectralContainer
        The container from which spectra are retrieved by global row index.
    results : dict[str, tuple[list[int], rp.Spectrum]]
        Output mapping from :func:`ramanlib.calc.outliers_per_group`, i.e.,
        ``{ group_label: ([row_indices_into_gsc_df], mean_spectrum) }``.
    **kwargs
        Forwarded to :func:`ramanspy.plot.spectra` (e.g., ``color``, ``linewidth``,
        ``title``, ``ax``). Any provided ``plot_type`` is ignored (layout is fixed
        to ``"separate"`` for robustness and clarity).

    Returns
    -------
    matplotlib.axes.Axes or list[matplotlib.axes.Axes] or matplotlib.figure.Figure or None
        Whatever :func:`ramanspy.plot.spectra` returns. Returns ``None`` if
        ``results`` is empty.

    See Also
    --------
    ramanlib.calc.outliers_per_group
        Compute per-group outlier indices and each group's mean spectrum.

    Notes
    -----
    Overlays the supplied per-group mean spectrum in red. No legend/tight layout
    adjustments are performed here, so you can customize them upstream or
    downstream as needed.

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
    axes_list = _normalize_axes_obj(axes_obj)

    # Overlay the mean line (no labels/legend/tight_layout/show here)
    for ax, mean_spec in zip(axes_list, means_for_overlay):
        ax.plot(mean_spec.spectral_axis, mean_spec.spectral_data, color="red", linewidth=1.5)

    return axes_obj


def baseline(spectrum, baseline_process, **kwargs):
    """
    Plot a spectrum, its estimated baseline, and the baseline-corrected spectrum.

    Parameters
    ----------
    spectrum : rp.Spectrum
        Input spectrum.
    baseline_process : object
        Any object exposing ``.apply(Spectrum) -> Spectrum`` that performs baseline
        correction (e.g., a RamanSPy pipeline step).
    **kwargs
        Forwarded to :func:`ramanspy.plot.spectra` (e.g., ``ax``, ``alpha``,
        ``linewidth``, color styling, etc.).

    Returns
    -------
    matplotlib.axes.Axes or list[matplotlib.axes.Axes] or matplotlib.figure.Figure
        Whatever :func:`ramanspy.plot.spectra` returns.

    Notes
    -----
    The baseline is computed as ``baseline = original - corrected`` and plotted
    alongside the original and corrected spectra with fixed labels:
    ``["Original spectrum", "removed baseline", "corrected spectrum"]``.

    """
    corrected_spectrum = baseline_process.apply(spectrum)
    baseline = rp.Spectrum(spectrum.spectral_data - corrected_spectrum.spectral_data, spectrum.spectral_axis)
    spectra = [spectrum, baseline, corrected_spectrum]
    labels = ["Original spectrum", "removed baseline", "corrected spectrum"]
    return rp.plot.spectra(spectra, label=labels, plot_type="single", alpha=0.9, **kwargs)


def n_baselines(raw_gsc, baseline_process, process_name, n_samples=3, figsize=(8,7), seed=None):
    """
    Plot several randomly selected spectra with their baselines in a vertical figure.

    Parameters
    ----------
    raw_gsc : GroupedSpectralContainer
        Container from which spectra are sampled.
    baseline_process : object
        Any object exposing ``.apply(Spectrum) -> Spectrum`` that performs baseline
        correction (e.g., a RamanSPy pipeline step).
    process_name : str
        Title displayed above the figure.
    n_samples : int, optional
        Number of spectra (rows) to sample. Default ``3``.
    figsize : tuple[float, float], optional
        Matplotlib figure size passed to :func:`matplotlib.pyplot.subplots`.
        Default ``(8, 7)``.
    seed : int or None, optional
        Random seed used when sampling. Default ``None``.

    Returns
    -------
    list[matplotlib.axes.Axes]
        The list of axes for each subplot row.

    """
    spec_samples = raw_gsc.df.sample(n=n_samples)["spectrum"]
    fig, axs = plt.subplots(n_samples, 1, figsize=figsize)
    for i, spec in enumerate(spec_samples):
        baseline(spec, baseline_process, ax=axs[i], title="", xlabel="")
    fig.suptitle(f"{process_name}")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.tight_layout()
    return axs


def compare_baselines(spectrum, baseline_processes, process_names, figsize=(8,7)):
    """
    Compare multiple baseline algorithms on the same spectrum.

    Parameters
    ----------
    spectrum : rp.Spectrum
        Input spectrum to be corrected by each baseline process.
    baseline_processes : list
        Sequence of objects exposing ``.apply(Spectrum) -> Spectrum``.
    process_names : list[str]
        Display names for each process. Must have the same length and order as
        ``baseline_processes``.
    figsize : tuple[float, float], optional
        Matplotlib figure size. Default ``(8, 7)``.

    Returns
    -------
    list[matplotlib.axes.Axes]
        The list of axes for each subplot row.

    """
    fig, axs = plt.subplots(len(baseline_processes), 1, figsize=figsize)
    for i, process in enumerate(baseline_processes):
        baseline(spectrum, process, ax=axs[i], title=f"{process_names[i]}", xlabel="")
    plt.xlabel("Wavenumber(cm⁻¹)")
    plt.tight_layout()
    return axs


def mean_difference(diff_spectrum, ci_band, label="Difference in Means", **kwargs):
    """
    Plot a difference-of-means spectrum with a two-sided CI band centered at zero.

    This is the plotting counterpart of :func:`ramanlib.calc.mean_difference`.
    Pass in the tuple returned by that function’s computation step.

    Parameters
    ----------
    diff_spectrum : rp.Spectrum
        The difference spectrum (e.g., group A mean minus group B mean).
        Typically the first element returned by
        :func:`ramanlib.calc.mean_difference`.
    ci_band : numpy.ndarray
        One-dimensional nonnegative array giving the half-width of the symmetric
        confidence band at each wavenumber (i.e., plot ``± ci_band``). Typically
        the second element returned by :func:`ramanlib.calc.mean_difference`.
    label : str, optional
        Legend label for the difference trace. Default ``"Difference in Means"``.
    **kwargs
        Forwarded to :func:`ramanspy.plot.spectra`. If ``plot_type`` is supplied,
        it is ignored and a warning is issued (this plot always uses ``"single"``).

    Returns
    -------
    matplotlib.axes.Axes or list[matplotlib.axes.Axes] or matplotlib.figure.Figure
        Whatever :func:`ramanspy.plot.spectra` returns.

    Notes
    -----
    The shaded band is drawn as ``[-ci_band, +ci_band]`` about zero on the same
    x-axis as ``diff_spectrum``. A horizontal reference line at ``y=0`` is added.

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
    **kwargs,
):
    """
    Plot a heatmap of correlations between per-group mean spectra.

    This is the plotting counterpart of
    :func:`ramanlib.calc.mean_correlation_per_group`. Pass in the
    square correlation matrix that function returns.

    Parameters
    ----------
    correlation_matrix : pandas.DataFrame
        Square matrix of correlation coefficients; index/columns are group labels
        in the same order used to compute the means.
    title : str, optional
        Figure title. Default ``"Correlation Matrix of Raman Spectra"``.
    vmin : float, optional
        Lower bound for the color scale. Default ``0``.
    vmax : float, optional
        Upper bound for the color scale. Default ``1``.
    annot : bool, optional
        Whether to annotate each cell with its numeric value. Default ``True``.
    cmap : str, optional
        Colormap passed to :func:`seaborn.heatmap`. Default ``"coolwarm"``.
    figsize : tuple[float, float], optional
        Matplotlib figure size. Default ``(8, 6)``.
    **kwargs
        Additional keyword arguments forwarded to :func:`seaborn.heatmap`.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the heatmap.

    """
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    plt.title(title)