"""Common plot style, output directories and figure saving for the paper figures."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figures')   # figures (PDF for the paper, PNG preview)
DATADIR = os.path.join(HERE, 'figs')     # intermediate numerical results (.npy)


def apply(legend_fontsize=7.5, linewidth=1.4):
    """Serif (Computer Modern) style used for all figures of the paper."""
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                         'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False, 'font.size': 9,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                         'legend.frameon': False, 'legend.fontsize': legend_fontsize,
                         'lines.linewidth': linewidth})


def save(fig, name):
    """Save figures/<name>.pdf and a PNG preview figures/<name>.png."""
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, name + '.png'), dpi=220, bbox_inches='tight')
    print('written:', os.path.join('figures', name + '.pdf'))


def data_path(name):
    """Path of an intermediate data file in figs/ (the directory is created if needed)."""
    os.makedirs(DATADIR, exist_ok=True)
    return os.path.join(DATADIR, name)
