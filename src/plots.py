import matplotlib.pyplot as plt
from os.path import join as pjoin
from Plotter import summary_table, plot, plot_results_hist, plot_sigma, plot_population_variety
from Experiments import expr
from tqdm import tqdm

path = '../stats2'


def compare_se9():
    se9 = [x for x in expr.keys() if 'se_9' in x]
    se9.remove('se_9mcb')
    se9_plus = [x for x in se9 if 'p' in x]
    se9_comma = [x for x in se9 if 'c' in x]
    y_plot, (plus, comma) = plt.subplots(2)
    plot(plus, se9_plus)
    plot(comma, se9_comma)
    info_plot, (sigma, variety) = plt.subplots(2)
    plot_population_variety(variety, se9)
    plot_sigma(sigma, se9)
    hist_plot, ax2 = plt.subplots(1)
    plot_results_hist(ax2, se9)
    se9.append('se_9mcb')

    y_plot.set_size_inches(7, 7)
    y_plot.suptitle("Zbieżność od µ,λ")
    y_plot.tight_layout()

    handles, labels = variety.get_legend_handles_labels()
    variety.set_title("Wykres odchylenia średniej normy osobników")
    sigma.set_title("Wykres średniej σ(t)")
    info_plot.set_size_inches(7, 9)
    info_plot.legend(handles, labels, loc="upper right")
    info_plot.tight_layout()
    hist_plot.suptitle("Rozłożenie wyników optymalizacji przy wielu uruchomieniach")

    y_plot.savefig(pjoin(path, "se9_plot.png"))
    info_plot.savefig(pjoin(path, "se9_info_plot.png"))
    hist_plot.savefig(pjoin(path, "se9_hist_plot.png"))
    with open(pjoin(path, "se9_summary.md"), 'w') as fh:
        fh.writelines(summary_table(se9).to_markdown())
    summary_table(se9).to_csv(pjoin(path, "se9_summary.csv"))


def compare_se1():
    se1 = [x for x in expr.keys() if 'se_1' in x]
    summary_table(se1)
    se1_plus = [x for x in se1 if 'p' in x]
    se1_comma = [x for x in se1 if 'c' in x]
    y_plot, (plus, comma) = plt.subplots(2)
    plot(plus, se1_plus)
    plot(comma, se1_comma)
    info_plot, (sigma, variety) = plt.subplots(2)
    se1.remove('se_1mcb')
    plot_population_variety(variety, se1)
    plot_sigma(sigma, se1)
    hist_plot, ax2 = plt.subplots(1)
    plot_results_hist(ax2, se1)

    se1.append('se_1mcb')
    y_plot.set_size_inches(7, 7)
    y_plot.suptitle("Zbieżność od µ,λ")
    y_plot.tight_layout()

    handles, labels = variety.get_legend_handles_labels()
    variety.set_title("Wykres odchylenia średniej normy osobników")
    sigma.set_title("Wykres średniej σ(t)")
    info_plot.set_size_inches(7, 9)
    info_plot.legend(handles, labels, loc="upper right")
    info_plot.tight_layout()
    hist_plot.suptitle("Rozłożenie wyników optymalizacji przy wielu uruchomieniach")

    y_plot.savefig(pjoin(path, "se1_plot.png"))
    info_plot.savefig(pjoin(path, "se1_info_plot.png"))
    hist_plot.savefig(pjoin(path, "se1_hist_plot.png"))
    with open(pjoin(path, "se1_summary.md"), 'w') as fh:
        fh.writelines(summary_table(se1).to_markdown())
    summary_table(se1).to_csv(pjoin(path, "se1_summary.csv"))

def compare_cls1():
    cls1 = [x for x in expr.keys() if 'cls_1' in x]
    y_plot, ax = plt.subplots(1)
    plot(ax, cls1)
    info_plot, variety = plt.subplots(1)
    plot_population_variety(variety, cls1)
    hist_plot, ax2 = plt.subplots(1)
    plot_results_hist(ax2, cls1)

    y_plot.set_size_inches(7, 4.5)
    y_plot.suptitle("Zbieżność od wyboru implementacji elementów algorytmu")
    y_plot.tight_layout()

    variety.set_title("Wykres odchylenia średniej normy osobników")
    variety.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)
    info_plot.tight_layout()

    hist_plot.suptitle("Rozłożenie wyników optymalizacji przy wielu uruchomieniach")

    y_plot.savefig(pjoin(path, "cls1_plot.png"))
    info_plot.savefig(pjoin(path, "cls1_info_plot.png"))
    hist_plot.savefig(pjoin(path, "cls1_hist_plot.png"))
    with open(pjoin(path, "cls1_summary.md"), 'w') as fh:
        fh.writelines(summary_table(cls1).to_markdown())
    summary_table(cls1).to_csv(pjoin(path, "cls1_summary.csv"))

def compare_cls9():
    cls9 = [x for x in expr.keys() if 'cls_9' in x]
    y_plot, ax = plt.subplots(1)
    plot(ax, cls9)
    info_plot, variety = plt.subplots(1)
    plot_population_variety(variety, cls9)
    hist_plot, ax2 = plt.subplots(1)
    plot_results_hist(ax2, cls9)

    y_plot.set_size_inches(7, 4.5)
    y_plot.suptitle("Zbieżność od wyboru implementacji elementów algorytmu")
    y_plot.tight_layout()

    variety.set_title("Wykres odchylenia średniej normy osobników")
    variety.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)
    info_plot.tight_layout()

    hist_plot.suptitle("Rozłożenie wyników optymalizacji przy wielu uruchomieniach")

    y_plot.savefig(pjoin(path, "cls9_plot.png"))
    info_plot.savefig(pjoin(path, "cls9_info_plot.png"))
    hist_plot.savefig(pjoin(path, "cls9_hist_plot.png"))
    with open(pjoin(path, "cls9_summary.md"), 'w') as fh:
        fh.writelines(summary_table(cls9).to_markdown())
    summary_table(cls9).to_csv(pjoin(path, "cls9_summary.csv"))


def compare_best():
    best9 = ['se_9ucb', 'se_9upb', 'cls_9tueb', 'gd_9best']
    best1 = ['cls_1rmeb', 'se_1upb', 'gd_1best']
    y_plot, (ax1, ax9) = plt.subplots(2)
    plot(ax9, best9)
    plot(ax1, best1)
    ax9.set_xlim(left=0, right=50)
    ax1.set_xlim(left=0, right=250)
    ax1.set_title("f1")
    ax9.set_title("f9")
    y_plot.set_size_inches(7, 7)
    y_plot.suptitle("Porównanie gradientu, strategii i klasycznego a. ewolucyjnego")
    y_plot.tight_layout()
    # y_plot.show()

    hist_plot, (h1, h9) = plt.subplots(2)
    plot_results_hist(h1, best1)
    plot_results_hist(h9, best9)

    hist_plot.suptitle("Rozłożenie wyników optymalizacji przy wielu uruchomieniach")
    h1.set_title('f1')
    h9.set_title('f9')
    hist_plot.tight_layout()
    # hist_plot.show()
    y_plot.savefig(pjoin(path, "best_plot.png"))
    hist_plot.savefig(pjoin(path, "best_hist_plot.png"))
    with open(pjoin(path, "best1_summary.md"), 'w') as fh:
        fh.writelines(summary_table(best1).to_markdown())
    summary_table(best1).to_csv(pjoin(path, "best1_summary.csv"))
    with open(pjoin(path, "best9_summary.md"), 'w') as fh:
        fh.writelines(summary_table(best9).to_markdown())
    summary_table(best9).to_csv(pjoin(path, "best9_summary.csv"))


if __name__ == '__main__':
    with plt.style.context('seaborn'):
        for func in tqdm([compare_cls1, compare_cls9, compare_se1, compare_se9, compare_best]):
            func()