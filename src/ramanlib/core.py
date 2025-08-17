import ramanspy as rp
import pandas as pd
import numpy as np

from . import plot


class GroupedSpectralContainer:
    # GroupedSpectralContainer is a wrapper for a pandas DataFrame object, which should always have a "spectrum" column of
    # rp.Spectrum objects, and other columns representing corresponding metadata.

    # Basic data manipulation methods. For more advanced functionality, the user is expected to
    # use GroupedSpectralContainer.df to apply changes to the data.
    def __init__(self, spectral_list, metadata):   # metadata is a list of dictionaries
        # Raise errors if data isn't spectrum objects or not the same length as the metadata
        if not all(isinstance(s, rp.Spectrum) for s in spectral_list):
            raise TypeError("All items in spectral_list must be RamanSPy Spectrum objects.")
        if len(spectral_list) != len(metadata):
            raise ValueError("spectral_list and metadata must be the same length.")
        
        rows = [{"spectrum": s, **meta} for s, meta in zip(spectral_list, metadata)]
        self.df = pd.DataFrame(rows)
    
    @classmethod
    def from_dataframe(cls, df):
        # Create a GroupedSpectralContainer using a dataframe, going through the __init__ constructor
        # for increased robustness

        # Check for a spectrum column and the type of the spectrum columns
        if "spectrum" not in df.columns:
            raise ValueError("DataFrame must contain a 'spectrum' column.")
        if not all(isinstance(s, rp.Spectrum) for s in df["spectrum"]):
            raise TypeError("All entries in 'spectrum' column must be Spectrum objects.")
        
        spectra = df["spectrum"].tolist()
        metadata = df.drop(columns=["spectrum"]).to_dict(orient="records")
        return cls(spectra, metadata)

    def copy(self):
        return GroupedSpectralContainer.from_dataframe(self.df.copy())

    def to_spectral_container(self):
        # Return a SpectralContainer with the data of the GroupedSpectralContainer.
        axes = [s.spectral_axis for s in self.df['spectrum']]
        # Check that all spectra within have the same spectral axis.
        if not all((axes[0] == ax).all() for ax in axes[1:]):
            raise ValueError("All spectra must have the same spectral axis to convert to a SpectralContainer.")
        return rp.SpectralContainer.from_stack(self.df["spectrum"].tolist())

    def mean(self, by=None, include_stats=False, ddof=1):
        """
        Compute mean spectra per group and return a new GroupedSpectralContainer
        with one row per group.

        Parameters
        ----------
        by : str | list[str] | callable | None
            Grouping key(s) for pandas .groupby. If None, the whole container is one group.
        include_stats : bool
            If True, add columns: 'n' (group size), 'std_vector', 'var_vector'.
            (Vectors are numpy arrays aligned with the spectrum's spectral_axis.)
        ddof : int
            Delta degrees of freedom for variance/std (default=1 ⇒ sample stats).

        Returns
        -------
        GroupedSpectralContainer
            A new GSC with mean Spectrum per group. Group keys are preserved
            back into metadata columns.
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

    def plot_mean(self, by=None, interval=None, plot_type="separate", ci_z=1.96, **kwargs):
        return plot.mean_per_group(self, by=by, interval=interval, plot_type=plot_type, ci_z=ci_z, **kwargs)

    def plot_random(self, by=None, n_samples=3, plot_type="single", seed=None, **kwargs):
        return plot.random_per_group(self, by=by, n_samples=n_samples, plot_type=plot_type, seed=seed, **kwargs)

    def apply_pipeline(self, pipeline):
        df = self.df.assign(spectrum=self.df["spectrum"].apply(pipeline.apply))
        return GroupedSpectralContainer.from_dataframe(df)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        result = self.df[key]
        if isinstance(result, pd.DataFrame):
            return GroupedSpectralContainer.from_dataframe(result)
        return result

    def __repr__(self):
        return f"GroupedSpectralContainer({len(self.df)} spectra)\n\n{self.df.head()}"
