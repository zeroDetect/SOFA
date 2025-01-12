import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FEATURE_START_INDEX_IN_FIELDS_FILE = 8


def get_class_list(data_dir, class_choice):
    """
    :param data_dir: str
    :param class_choice: str all or list of chosen classes
    :return: list of chosen classes (sorted)
    """
    if class_choice == 'all':
        all_data = sorted(os.listdir(data_dir))
        class_choice = [os.path.splitext(a)[0] for a in all_data if not a.startswith('.') and a.endswith('.npy')]
        class_choice = [c for c in class_choice if 'scale' not in c]
    class_choice = sorted(class_choice)
    return class_choice


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_fields_list(fields_file):
    with open(fields_file, 'r') as f:
        fields = f.readlines()
    fields = [line.strip('\n') for line in fields]
    fields = fields[FEATURE_START_INDEX_IN_FIELDS_FILE:]
    fields = ['direction'] + fields
    return fields


def parse_str_dim(dim):
    """
    parse str dim to int dim or list of int dims
    :param dim: str
    :return: parse result int or list of int
    """
    if ',' in dim:
        dim = dim.split(',')
        dim = [int(d) for d in dim]
    else:
        dim = int(dim)
    return dim


def fact(x):
    ret = 1
    for i in range(1, x+1):
        ret *= i
    return ret


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=15)
    cbar.ax.tick_params(labelsize=15)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=15)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=15)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def horizontal_distribution_bar_chart(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from a sample to a list of number per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.invert_yaxis()
    # ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    ax.set_xlabel('Feature Importance', fontsize=10)
    ax.tick_params(labelsize=10)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, labels=[str(i)], label_type='center', color=text_color)
    ax.legend(ncol=4, bbox_to_anchor=(0, 1),
              loc='lower left', fontsize=9)

    return fig, ax
