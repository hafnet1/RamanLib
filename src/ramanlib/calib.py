from scipy.optimize import curve_fit
import ramanspy as rp
import numpy as np
from .utils import _normalize_axes_obj
from .core import GroupedSpectralContainer

def fit_peak(calib: rp.Spectrum, plot: bool = False):
    """
    Fit a Gaussian to a single peak in a calibration spectrum (e.g., internal Si).

    Parameters
    ----------
    calib : rp.Spectrum
        Has `.spectral_axis` and `.spectral_data` (1D arrays).
    plot : bool
        If True, plot data + Gaussian fit using RamanSPy. (No plt.show() here.)

    Returns
    -------
    peak_center : float
        Estimated center of the peak (cm⁻¹).
    peak_height : float
        Estimated peak intensity (amplitude).
    sigma : float
        Estimated Gaussian σ (related to FWHM via FWHM = 2*sqrt(2*ln2)*σ).
    """

    # Gaussian model
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    x = np.asarray(calib.spectral_axis)
    y = np.asarray(calib.spectral_data)

    # Initial guesses
    a0 = float(np.max(y))
    x0 = float(x[np.argmax(y)])
    sigma0 = 5.0
    p0 = [a0, x0, sigma0]

    # Fit
    popt, _ = curve_fit(gaussian, x, y, p0=p0)
    a_fit, x0_fit, sigma_fit = popt

    if plot:
        # Build a Spectrum for the fitted curve so we can use rp.plot.spectra
        y_fit = gaussian(x, *popt)
        fit_spec = rp.Spectrum(y_fit, x)

        # Plot data + fit in a single axes using RamanSPy
        axes_obj = rp.plot.spectra(
            [calib, fit_spec],
            label=["Data", "Gaussian Fit"],
            plot_type="single"
        )

        # Ensure we have an Axes to draw the vertical line on
        ax = _normalize_axes_obj(axes_obj)[0]
        if ax is not None:
            ax.axvline(x0_fit, linestyle=":", color="r")  # Peak center marker

        # (No plt.show(); let caller decide)

    return x0_fit, a_fit, sigma_fit


def get_wn_shift(calib: rp.Spectrum, expected_x0_value, plot=False):
    '''Fit a gaussian to a single calibration peak of spectrum calib. Return the distance to expected_x0_value.
    
    expected_x0_value is a single number representing the value to correct to.'''
    x0, _, _ = fit_peak(calib, plot=plot)
    return x0 - expected_x0_value


def get_gsc_wn_shifts(
    calib_gsc: GroupedSpectralContainer,
    expected_x0_range,
    expected_x0_value: float,
    in_place: bool = False,
    plot: bool = False,
):
    """
    Fit calibration peaks for all spectra in a GSC and annotate the GSC with:
      - 'x0_fit' : fitted peak center (cm^-1)
      - 'shift'  : x0_fit - expected_x0_value

    Parameters
    ----------
    calib_gsc : GroupedSpectralContainer
        Must have a 'spectrum' column of rp.Spectrum objects.
    expected_x0_range : iterable of two floats
        [low, high] expected cm^-1 window for valid peak centers.
    expected_x0_value : float
        Target cm^-1 value to calibrate to.
    in_place : bool
        If True, modify calib_gsc.df in place. If False, return a new GSC.
    plot : bool
        If True, plot each fit (slow; for debugging small sets).

    Returns
    -------
    fail_gsc : pd.DataFrame                     (if in_place=True)
        Rows from the (now-annotated) calib_gsc.df that fail the range check.
    fail_gsc, new_gsc : (GroupedSpectralContainer, GroupedSpectralContainer)   (if in_place=False)
        A new annotated GSC plus the failing rows DataFrame.
    """
    if expected_x0_range is None or len(expected_x0_range) != 2:
        raise ValueError("expected_x0_range must be a two-element iterable [low, high].")
    low, high = sorted(expected_x0_range)

    # Work on a copy unless modifying in-place
    df = calib_gsc.df if in_place else calib_gsc.df.copy()

    # Compute fits
    x0_vals = []
    shift_vals = []
    for _, spec in df["spectrum"].items():
        try:
            x0_fit, _, _ = fit_peak(spec, plot=plot)
            shift_val = x0_fit - expected_x0_value
        except Exception:
            x0_fit = np.nan
            shift_val = np.nan
        x0_vals.append(x0_fit)
        shift_vals.append(shift_val)

    # Attach columns
    df["x0_fit"] = x0_vals
    df["shift"] = shift_vals

    # Failing mask: NaN or outside range
    mask_fail = ~np.isfinite(df["x0_fit"]) | (df["x0_fit"] < low) | (df["x0_fit"] > high)
    fail_df = df.loc[mask_fail].copy()

    if in_place:
        # Persist columns back to the original df
        calib_gsc.df["x0_fit"] = df["x0_fit"]
        calib_gsc.df["shift"]  = df["shift"]
        return GroupedSpectralContainer.from_dataframe(fail_df)

    # Build new GSCs with the augmented dataframes
    new_gsc = GroupedSpectralContainer.from_dataframe(df)
    fail_gsc = GroupedSpectralContainer.from_dataframe(fail_df)
    
    return fail_gsc, new_gsc