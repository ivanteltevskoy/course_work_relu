from experiments.experiment_1_scaling import run_scaling_experiment
from visualization.plot_scaling import plot_scaling_results

from experiments.experiment_2_depth import run_depth_experiment
from visualization.plot_depth_comparison import plot_depth_comparison

from experiments.experiment_3_ablation import run_ablation_experiment
from visualization.plot_ablation import plot_ablation_results

from experiments.experiment_1_scaling_v2 import run_scaling_experiment_v2
from visualization.plot_scaling_v2 import plot_scaling_results_v2


def run_all():

    print("\n[1] Эксперимент 1 (Скейлинг)")
    run_scaling_experiment()
    plot_scaling_results(show_fit=True, tail_k=3)

    print("\n[1] Эксперимент 1.2 (Скейлинг на доп. функциях)")
    run_scaling_experiment_v2()
    plot_scaling_results_v2(show_fit=True, tail_k=3)

    print("\n[2] Эксперимент 2 (Глубина)")
    run_depth_experiment()
    plot_depth_comparison()

    print("\n[3] Эксперимент 3 (Абляции)")
    run_ablation_experiment()
    plot_ablation_results()


if __name__ == "__main__":
    run_all()
