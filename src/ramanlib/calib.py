"""
Calibration utilities for RamanLib.

This module provides helpers to fit a single calibration peak (e.g., Si ~520 cm⁻¹),
derive a wavenumber shift from the fitted center, and annotate an entire
:class:`ramanlib.core.GroupedSpectralContainer` with per-row peak centers and
shifts. The outputs are designed to feed directly into your preprocessing or QA
pipeline and to be visualized with :mod:`ramanspy` plotting.

Functions
---------
fit_peak
    Fit a single Gaussian peak in a calibration spectrum and (optionally) plot the fit.
get_wn_shift
    Compute the wavenumber shift of a calibration peak relative to an expected center.
get_gsc_wn_shifts
    Fit/annotate peak centers and shifts for all spectra in a container; return
    failing rows and (optionally) an augmented container.

Notes
-----
- The Gaussian model is ``a * exp(- (x - x0)^2 / (2 * sigma^2))`` with parameters
  amplitude ``a``, center ``x0`` (cm⁻¹), and width ``sigma`` (cm⁻¹).  The FWHM is
  related by ``FWHM = 2 * sqrt(2 * ln 2) * sigma``.
- Plots use :func:`ramanspy.plot.spectra` and draw a vertical marker at the fitted center.
"""

from __future__ import annotations

from scipy.optimize import curve_fit
import ramanspy as rp
import numpy as np
from .utils import _normalize_axes_obj
from .core import GroupedSpectralContainer

def fit_peak(calib: rp.Spectrum, plot: bool = False):
    """
    Fit a Gaussian to a single peak in a calibration spectrum.

    Parameters
    ----------
    calib : rp.Spectrum
        Input calibration spectrum with ``.spectral_axis`` and ``.spectral_data`` (1D arrays).
    plot : bool, optional
        If ``True``, plot the data and Gaussian fit using :mod:`ramanspy` on a single axis
        and draw a vertical line at the fitted center. Default ``False``.
        (No ``plt.show()`` is called.)

    Returns
    -------
    peak_center : float
        Estimated peak center ``x0`` in cm⁻¹.
    peak_height : float
        Estimated amplitude ``a`` of the Gaussian.
    sigma : float
        Estimated Gaussian width ``σ`` in cm⁻¹. Related to FWHM via
        ``FWHM = 2 * sqrt(2 * ln 2) * σ``.

    Notes
    -----
    Initial parameter guesses are derived from the maximum of the signal.
    The fit is performed with :func:`scipy.optimize.curve_fit`.

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
    """
    Wavenumber shift relative to an expected calibration peak center.

    Parameters
    ----------
    calib : rp.Spectrum
        Input calibration spectrum.
    expected_x0_value : float
        Expected/target peak center in cm⁻¹ (e.g., 520.7 for Si).
    plot : bool, optional
        If ``True``, plot the fitted peak as in :func:`fit_peak`. Default ``False``.

    Returns
    -------
    shift : float
        The signed shift ``(fitted_center - expected_x0_value)`` in cm⁻¹.

    See Also
    --------
    fit_peak
        Underlying Gaussian peak fitting routine.

    """
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
    Fit calibration peaks for all spectra in a container and annotate each row.

    Two new columns are added to the container's dataframe:

    - ``'x0_fit'`` : fitted peak center (cm⁻¹)
    - ``'shift'``  : ``x0_fit - expected_x0_value`` (cm⁻¹)

    You can either modify the provided container in place (``in_place=True``) or
    return a new, augmented container (``in_place=False``). Rows that fail the
    center range check (NaN or outside ``expected_x0_range``) are returned as a
    separate container in both modes.

    Parameters
    ----------
    calib_gsc : GroupedSpectralContainer
        Container with a ``'spectrum'`` column of :class:`ramanspy.Spectrum`.
    expected_x0_range : iterable of float
        Two-element iterable ``[low, high]`` giving the valid cm⁻¹ interval for
        fitted centers. The values are sorted internally.
    expected_x0_value : float
        Target cm⁻¹ value to calibrate to (used to compute ``shift``).
    in_place : bool, optional
        If ``True``, annotate ``calib_gsc.df`` in place and return only the
        failing rows as a container. If ``False`` (default), return both the
        failing rows and a new, annotated container.
    plot : bool, optional
        If ``True``, plot each individual fit (may be slow). Default ``False``.

    Returns
    -------
    fail_gsc : GroupedSpectralContainer
        Container of rows that failed the range check (NaN or outside the bounds).
        Returned in both modes.
    new_gsc : GroupedSpectralContainer
        Only when ``in_place=False``: a new container with ``'x0_fit'`` and
        ``'shift'`` columns attached.

    Raises
    ------
    ValueError
        If ``expected_x0_range`` is not a 2-element iterable.

    Notes
    -----
    On failures/exceptions during per-row fitting, ``x0_fit`` and ``shift`` are set
    to ``NaN`` for that row.

    Examples
    --------
    >>> fail, calibrated = get_gsc_wn_shifts(gsc, expected_x0_range=[515, 530], expected_x0_value=520.7)
    >>> list(calibrated.df.columns)
    ['spectrum', ..., 'x0_fit', 'shift']

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