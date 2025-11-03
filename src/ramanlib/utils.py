"""
Utility helpers for RamanLib.

Contains `_normalize_axes_obj`, which normalizes the variable output of
:func:`ramanspy.plot.spectra` into a consistent list of Matplotlib `Axes`.
"""

from __future__ import annotations

from typing import List
import matplotlib.axes as maxes

def _normalize_axes_obj(axes_obj) -> List[maxes.Axes]:
    """
    Ensure RamanSPy plot output is returned as a list of Axes.

    Parameters
    ----------
    axes_obj : object
        Output from :func:`ramanspy.plot.spectra`. Can be a single Axes, list of
        Axes, or a Figure.

    Returns
    -------
    list[matplotlib.axes.Axes]
        Normalized list of Axes.

    Raises
    ------
    TypeError
        If the object type is not recognized.

    Notes
    -----
    Some RamanSPy plot types return a Figure instead of Axes; this helper
    corrects for that inconsistency.
    
    """
    if isinstance(axes_obj, list):
        return axes_obj
    # Single Axes
    if hasattr(axes_obj, "plot") and hasattr(axes_obj, "fill_between"):
        return [axes_obj]
    # Figure
    if hasattr(axes_obj, "get_axes"):
        return axes_obj.get_axes()
    # Unknown
    raise TypeError(f"Unexpected return type from rp.plot.spectra: {type(axes_obj)}")