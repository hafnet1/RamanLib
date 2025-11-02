"""
Core container primitives for RamanLib.

This module defines :class:`GroupedSpectralContainer`, a thin wrapper around a
:pandas:`pandas.DataFrame` whose first-class column is ``"spectrum"`` containing
:mod:`ramanspy` ``Spectrum`` objects (one per row). All other columns are free-form
metadata (strings, numbers, categories, etc.) that stay aligned to each spectrum.

The class provides a small, opinionated surface for:
- Safe construction from lists of spectra plus metadata rows.
- Conversion to a :mod:`ramanspy` ``SpectralContainer`` when axes match.
- Simple grouped reductions (e.g., group-wise mean spectra and optional stats).
- Convenience plotting hooks that defer to :mod:`ramanlib.plot`.

Notes
-----
The design goal is to make typical dataset manipulations ergonomic while keeping
the full power of :class:`pandas.DataFrame` available via the ``.df`` attribute.
For transformations beyond the light helpers here, operate directly on
``GroupedSpectralContainer.df`` and rebuild a container with
:meth:`GroupedSpectralContainer.from_dataframe`.

"""

from __future__ import annotations

import ramanspy as rp
import pandas as pd
import numpy as np

from . import plot


class GroupedSpectralContainer:
    """
    A table of Raman spectra with aligned metadata.

    Each row contains a :mod:`ramanspy` ``Spectrum`` in the ``"spectrum"`` column,
    plus arbitrary metadata columns (e.g., ``"sample"``, ``"region"``, ``"label"``).
    The container exposes a minimal API; for advanced operations, use the
    underlying :class:`pandas.DataFrame` via :attr:`df`.

    Parameters
    ----------
    spectral_list : list of ramanspy.Spectrum
        One spectrum per row.
    metadata : list of dict
        One metadata mapping per spectrum. Each dict's keys become columns in the
        backing DataFrame. The length must match ``spectral_list``.

    Attributes
    ----------
    df : pandas.DataFrame
        Backing table with column ``"spectrum"`` and zero or more metadata columns.

    Raises
    ------
    TypeError
        If any element of ``spectral_list`` is not a :mod:`ramanspy` ``Spectrum``.
    ValueError
        If ``spectral_list`` and ``metadata`` lengths differ.

    See Also
    --------
    GroupedSpectralContainer.from_dataframe : Build from an existing DataFrame.
    GroupedSpectralContainer.to_spectral_container : Convert to ``rp.SpectralContainer``.
    GroupedSpectralContainer.mean : Group-wise mean spectra.
    GroupedSpectralContainer.plot_mean : Plot group means with CIs.
    GroupedSpectralContainer.plot_random : Plot random spectra per group.

    """

    def __init__(self, spectral_list, metadata):   # metadata is a list of dictionaries
        """
        Construct a container from spectra and per-row metadata.

        Notes
        -----
        This initializer validates types/lengths and constructs :attr:`df`. For
        robustness with preexisting DataFrames, prefer
        :meth:`GroupedSpectralContainer.from_dataframe`.

        """
        # Raise errors if data isn't spectrum objects or not the same length as the metadata
        if not all(isinstance(s, rp.Spectrum) for s in spectral_list):
            raise TypeError("All items in spectral_list must be RamanSPy Spectrum objects.")
        if len(spectral_list) != len(metadata):
            raise ValueError("spectral_list and metadata must be the same length.")
        
        rows = [{"spectrum": s, **meta} for s, meta in zip(spectral_list, metadata)]
        self.df = pd.DataFrame(rows)
    
    @classmethod
    def from_dataframe(cls, df) -> "GroupedSpectralContainer":
        """
        Build a container from an existing DataFrame.

        The constructor validates that a ``'spectrum'`` column exists and that each
        entry is a :mod:`ramanspy` ``Spectrum``. All other columns are treated as
        metadata and preserved.

        Parameters
        ----------
        df : pandas.DataFrame
            Input table with a ``'spectrum'`` column of :mod:`ramanspy` ``Spectrum``
            objects and any number of metadata columns.

        Returns
        -------
        GroupedSpectralContainer
            A new container referencing a copy of ``df``'s contents.

        Raises
        ------
        ValueError
            If the DataFrame lacks a ``'spectrum'`` column.
        TypeError
            If any value in ``df['spectrum']`` is not a :mod:`ramanspy` ``Spectrum``.

        """
        # Check for a spectrum column and the type of the spectrum columns
        if "spectrum" not in df.columns:
            raise ValueError("DataFrame must contain a 'spectrum' column.")
        if not all(isinstance(s, rp.Spectrum) for s in df["spectrum"]):
            raise TypeError("All entries in 'spectrum' column must be Spectrum objects.")
        
        spectra = df["spectrum"].tolist()
        metadata = df.drop(columns=["spectrum"]).to_dict(orient="records")
        return cls(spectra, metadata)

    def copy(self) -> "GroupedSpectralContainer":
        """
        Return a deep copy of the container.

        Returns
        -------
        GroupedSpectralContainer
            A new container whose :attr:`df` is a copy of the original.

        """
        return GroupedSpectralContainer.from_dataframe(self.df.copy())

    def to_spectral_container(self) -> rp.SpectralContainer:
        """
        Convert to a :mod:`ramanspy` :class:`~ramanspy.SpectralContainer`.

        All spectra must share an identical spectral axis. The spectra are stacked
        in their current row order.

        Returns
        -------
        ramanspy.SpectralContainer
            A spectral container built by stacking the row spectra.

        Raises
        ------
        ValueError
            If any spectrum has a spectral axis different from the first row's.

        """
        axes = [s.spectral_axis for s in self.df['spectrum']]
        # Check that all spectra within have the same spectral axis.
        if not all((axes[0] == ax).all() for ax in axes[1:]):
            raise ValueError("All spectra must have the same spectral axis to convert to a SpectralContainer.")
        return rp.SpectralContainer.from_stack(self.df["spectrum"].tolist())

    def mean(
        self,
        by: str | list[str] | None = None,
        include_stats: bool = False,
        ddof: int = 1,
    ) -> "GroupedSpectralContainer":
        """
        Compute mean spectra per group.

        Groups the rows by the key(s) in ``by`` (or treats the whole table as a single
        group when ``by=None``), computes the mean spectrum per group, and returns a
        new container with one row per group. Optionally adds group-level statistics.

        Parameters
        ----------
        by : str or list of str or None, optional
            Column name(s) to group by, passed to :meth:`pandas.DataFrame.groupby`.
            If ``None`` (default), all rows belong to a single group named ``"all"``.
        include_stats : bool, optional
            If ``True``, append the following columns to the result:
            ``'n'`` (group size), ``'var_vector'`` and ``'std_vector'``
            (per-wavenumber variance and standard deviation). Default is ``False``.
        ddof : int, optional
            Delta degrees of freedom used for variance/std (as in :func:`numpy.var`);
            ``ddof=1`` gives sample variance. Default is ``1``.

        Returns
        -------
        GroupedSpectralContainer
            A new container where each row holds the group's mean :mod:`ramanspy`
            ``Spectrum`` in ``'spectrum'`` and the group key(s) as metadata columns.

        Notes
        -----
        The mean is computed via :meth:`ramanspy.SpectralContainer.mean`. The
        variance/standard deviation vectors, when requested, are aligned to the
        spectrum's spectral axis and computed over the stacked intensities.

        Examples
        --------
        >>> means = gsc.mean(by=["sample", "region"], include_stats=True)
        >>> list(means.df.columns)
        ['spectrum', 'sample', 'region', 'n', 'var_vector', 'std_vector']

        """
        # Build iterable of (group_key, group_df)
        grouped = [("all", self.df)] if by is None else list(self.df.groupby(by, dropna=False))

        rows = []
        for key, gdf in grouped:
            if gdf.empty:
                continue

            spectra = gdf["spectrum"].tolist()
            container = rp.SpectralContainer.from_stack(spectra)
            mean_spec = container.mean

            # Rehydrate group key(s) into metadata columns
            meta = {}
            if by is None:
                meta["group"] = "all"
            else:
                by_cols = by if isinstance(by, (list, tuple)) else [by]
                # pandas returns tuple keys for multi-column groups
                key_vals = key if isinstance(key, tuple) else (key,)
                meta.update(dict(zip(by_cols, key_vals)))

            # Always include the mean Spectrum
            meta["spectrum"] = mean_spec

            if include_stats:
                var = np.var(container.spectral_data, axis=0, ddof=ddof)
                meta["n"] = container.shape[0]
                meta["var_vector"] = var
                meta["std_vector"] = np.sqrt(var)

            rows.append(meta)

        mean_df = pd.DataFrame(rows)
        return GroupedSpectralContainer.from_dataframe(mean_df)

    def plot_mean(
        self,
        by: str | list[str] | None = None,
        interval: tuple[float, float] | None = None,
        plot_type: str = "separate",
        ci_z: float = 1.96,
        **kwargs,
    ):
        """
        Plot mean spectra per group.

        This is a thin wrapper around :func:`ramanlib.plot.mean_per_group`.

        Parameters
        ----------
        by : str or list of str or None, optional
            Grouping key(s). See :meth:`GroupedSpectralContainer.mean`.
        interval : tuple of (float, float) or None, optional
            Optional spectral axis range (min, max) to display.
        plot_type : {"single", "separate", "stacked", "single stacked"}, optional
            Plot style. ``"separate"`` draws one subplot per group;
            ``"single"`` overlays all groups; ``"stacked"``
            create separate plots stacked vertically; ``"single stacked"`` overlays
            spectra in a single plot with vertical offsets. Default is ``"separate"``. 
        ci_z : float, optional
            Z-score for confidence intervals (e.g., 1.96 ≈ 95% CI). Default is 1.96.
        **kwargs
            Forwarded to the underlying plotting function/matplotlib.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray
            Axes object(s) produced by the plotting backend (RamanSPy).

        See Also
        --------
        ramanlib.plot.mean_per_group : Implementation of the plotting logic.

        """
        return plot.mean_per_group(self, by=by, interval=interval, plot_type=plot_type, ci_z=ci_z, **kwargs)

    def plot_random(
        self,
        by: str | list[str] | None = None,
        n_samples: int = 3,
        plot_type: str = "single",
        seed: int | None = None,
        **kwargs,
    ):
        """
        Plot a random sample of spectra per group.

        This is a thin wrapper around :func:`ramanlib.plot.random_per_group`.

        Parameters
        ----------
        by : str or list of str or None, optional
            Grouping key(s). If ``None``, sample from all rows.
        n_samples : int, optional
            Number of spectra to sample per group. Default is ``3``.
        plot_type : {"single", "separate", "stacked", "single stacked"}, optional
            Plot style. ``"separate"`` draws one subplot per group;
            ``"single"`` overlays all groups; ``"stacked"``
            create separate plots stacked vertically; ``"single stacked"`` overlays
            spectra in a single plot with vertical offsets. Default is ``"separate"``.
        seed : int or None, optional
            Random seed for reproducibility. Default is ``None``.
        **kwargs
            Forwarded to the underlying plotting function/matplotlib.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray
            Axes object(s) produced by the plotting backend.

        See Also
        --------
        ramanlib.plot.random_per_group : Implementation of the plotting logic.

        """
        return plot.random_per_group(self, by=by, n_samples=n_samples, plot_type=plot_type, seed=seed, **kwargs)

    def apply_pipeline(self, pipeline) -> "GroupedSpectralContainer":
        """
        Apply a RamanSPy processing pipeline to each spectrum.

        Parameters
        ----------
        pipeline : object
            Any object exposing an ``.apply(Spectrum) -> Spectrum`` method
            (e.g., a :mod:`ramanspy` pipeline).

        Returns
        -------
        GroupedSpectralContainer
            A new container with transformed spectra and the same metadata.

        Notes
        -----
        The operation is row-wise and does not mutate the original container.

        """
        df = self.df.assign(spectrum=self.df["spectrum"].apply(pipeline.apply))
        return GroupedSpectralContainer.from_dataframe(df)
    
    def __len__(self) -> int:
        """
        Number of rows (spectra) in the container.

        Returns
        -------
        int
            Row count of :attr:`df`.

        """
        return len(self.df)

    def __getitem__(self, key):
        """
        Column/row selection proxy to the underlying DataFrame.

        Parameters
        ----------
        key : Any
            Key accepted by :class:`pandas.DataFrame`'s ``__getitem__`` (e.g., a
            column name, a boolean mask, or a list of columns).

        Returns
        -------
        GroupedSpectralContainer or pandas.Series or pandas.DataFrame
            If the selection yields a DataFrame, it is wrapped back into a
            :class:`GroupedSpectralContainer`. Otherwise the raw pandas object is
            returned (e.g., a Series for a single column).

        Notes
        -----
        This is a convenience to keep simple subsetting ergonomic. For complex
        indexing, slice :attr:`df` directly.

        """
        result = self.df[key]
        if isinstance(result, pd.DataFrame):
            return GroupedSpectralContainer.from_dataframe(result)
        return result

    def __repr__(self) -> str:
        """
        Debug-friendly representation with a small preview of the DataFrame.

        Returns
        -------
        str
            Human-readable summary including a head of :attr:`df`.

        """
        return f"GroupedSpectralContainer({len(self.df)} spectra)\n\n{self.df.head()}"
