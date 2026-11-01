"""Heatmap visualization of a Granger Causality Matrix."""

from __future__ import annotations

import numpy as np


def plot_gcm(
    result,
    alpha: float = 0.05,
    annot: bool = True,
    cmap: str = "RdYlGn",
    figsize: tuple[float, float] | None = None,
    ax=None,
    mask_insignificant: bool = False,
    title: str | None = None,
):
    """Draw the Granger Causality Matrix as a heatmap of p-values.

    Reads like a correlation table: rows are effects, columns are causes,
    and a cell shows the p-value for ``cause -> effect``. Low p-values
    (significant causality) are highlighted.

    Parameters
    ----------
    result : GCMatrixResult
        Output of :func:`grangercm.granger_matrix`.
    alpha : float
        Significance threshold; used for the colour midpoint and masking.
    annot : bool
        Write the numeric p-value in each cell.
    cmap : str
        Any matplotlib/seaborn colormap. Default reversed so green = low p.
    figsize : (w, h), optional
        Figure size; auto-scaled to the number of variables if omitted.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axis.
    mask_insignificant : bool
        If True, blank out cells with p >= alpha.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import TwoSlopeNorm

    pv = result.pvalues
    n = len(pv)
    if figsize is None:
        figsize = (max(5, 0.8 * n + 2), max(4, 0.8 * n + 1))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    mask = np.eye(n, dtype=bool)
    if mask_insignificant:
        mask = mask | (pv.values >= alpha)

    norm = TwoSlopeNorm(vmin=0.0, vcenter=alpha, vmax=1.0)
    sns.heatmap(
        pv, mask=mask, annot=annot, fmt=".3f", cmap=cmap, norm=norm,
        linewidths=0.5, linecolor="white", cbar_kws={"label": "p-value"},
        ax=ax, square=True,
    )
    ax.set_xlabel("Cause (predictor)")
    ax.set_ylabel("Effect (predicted)")
    corr = result.correction or "none"
    ax.set_title(title or f"Granger Causality Matrix "
                          f"(test={result.test}, correction={corr})")
    return ax
