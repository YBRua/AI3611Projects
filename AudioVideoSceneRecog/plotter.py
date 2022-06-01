# %%
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List


matplotlib.style.use('seaborn-white')
sns.set_theme('paper', 'white', font_scale=2.5)


# %%
def plot_grouped_bars(
    ax: plt.Axes,
    data_lists: List[List],
    labels: List[str],
    xlabels: List[str],
    facecolors: List[str],
    max_width: float = 0.8,
    **bar_kwargs
):
    n_bars = len(data_lists)  # how many bars
    width = max_width / n_bars
    step = (2 / n_bars) * max_width / 2
    start = - max_width / 2
    for lid, data_list in enumerate(data_lists):
        for iid, item in enumerate(data_list):
            ax.bar(
                x=iid + start + lid * step + width / 2,
                height=item,
                width=width,
                label=labels[lid] if iid == 0 else None,
                facecolor=facecolors[lid] if facecolors is not None else None,
                linewidth=0,
                **bar_kwargs)
    ax.set_xticks(list(range(len(xlabels))))
    ax.set_xticklabels(xlabels)
    return ax


baseline_f1 = [0.77, 0.89, 0.86, 0.89, 0.91, 0.69, 0.74, 0.71, 0.88, 0.76]
asc_f1 = [0.75, 0.72, 0.67, 0.66, 0.84, 0.63, 0.62, 0.62, 0.81, 0.65]
vsc_f1 = [0.43, 0.73, 0.77, 0.85, 0.82, 0.49, 0.60, 0.58, 0.70, 0.52]


fig, ax = plt.subplots(figsize=(15, 5))
ax = plot_grouped_bars(
    ax,
    [baseline_f1, asc_f1, vsc_f1],
    ['AVSC', 'ASC', 'VSC'],
    [
        'airport', 'bus', 'metro', 'mtr_station', 'park',
        'square', 'mall', 'str_pede', 'str_traf', 'tram'
    ],
    ['C0', 'C1', 'C2'])
ax.legend(bbox_to_anchor=(0.5, -0.5), ncol=3, loc='upper center')
ax.set_ylim([0, 1])
ax.set_xlabel('Class')
ax.set_ylabel('F1 score')
ax.set_title('F1 Score for Different Modals')
fig.tight_layout()
fig.savefig('avsc-f1.pdf', bbox_inches='tight')

# %%
