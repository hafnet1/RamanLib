from typing import List
import matplotlib.axes as maxes

def _normalize_axes_obj(axes_obj) -> List[maxes.Axes]:
    """
    RamanSPy claims to return Axes or list[Axes], but some plot_types (e.g. 'stacked')
    may return a Figure. Normalize to a list[Axes].
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