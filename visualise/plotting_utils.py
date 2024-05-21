import matplotlib.pyplot as plt
import numpy as np
from sbi import analysis as analysis


# check the docs for more options
# https://github.com/sbi-dev/sbi/blob/3c1e7250192041d995d8fda368d421919400a499/sbi/analysis/plot.py#L635

# plot posterior/prior samples
fig, ax = analysis.pairplot(
    samples,
    limits=plotting_limits,
    figsize=(4, 4),
    points=[points_1, points_2],
    labels=your_labels,
    points_colors=["red", "green"],  # colors for the points_1 and points_2
    upper="contour",  # 'scatter' hist, scatter, contour, cond, None
    **{  #'title': f'Paramter prediction given model {model_sample.numpy()}', # title
        "kde_offdiag": {"bins": 50},  # bins for kde on the off-diagonal
        "points_offdiag": {"markersize": 3, "marker": "x"},
        "contour_offdiag": {"levels": [0.023, 0.5, 0.977]},
        "points_diag": {"ls": "-", "lw": 1, "alpha": 1},
    }
)


# plot predictives with mean +- std, often +-2std is used, so change factor to 2
def plot_std(x, y, alpha_fill=0.3, factor=1, **kwargs):
    """plots the mean +-std of y

    Args:
        x (array): (l))
        y (array): (n,l)
        factor (float): factor to multiply std with
    """
    mean = np.mean(y, 0)
    std = np.std(y, 0) * factor

    (base_line,) = plt.plot(x, mean, **kwargs)
    kwargs["label"] = None
    kwargs["alpha"] = alpha_fill
    kwargs["facecolor"] = base_line.get_color()
    kwargs["edgecolor"] = None  # "green"
    plt.fill_between(x, mean - std, mean + std, **kwargs)
