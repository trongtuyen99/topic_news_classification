import numpy as np
import matplotlib.pyplot as plt


def visualize(groups, name_groups, labels, top_k=10, width=.35, gap=.3):
    def autolabel(rects):

        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    x = np.array([i * (width * len(labels) + gap) for i in range(len(name_groups))])  # the label locations
    fig, ax = plt.subplots(figsize=(12, 7))

    rects = []
    for i in range(len(groups)):
        r = ax.bar(x + (2 * i + 1) * width / 2, groups[i], width, label=labels[i])
        rects.append(r)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and algorithm')
    ax.set_xticks(x + len(groups) * width / 2)
    ax.set_xticklabels(name_groups)
    ax.legend()

    for r in rects:
        autolabel(r)

    fig.tight_layout()
