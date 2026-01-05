def main():
    
    while True:
        print("\nВЫБЕРИТЕ ЭКСПЕРИМЕНТ:")
        print("1 - Эксперимент 1 (Скейлинг)")
        print("2 - Эксперимент 2 (Глубина)")
        print("3 - Эксперимент 3 (Абляции)")
        print("4 - Эксперимент 1.2 (Скейлинг на доп. функциях)")
        print("5 - Выход")

        choice = input("\nВаш выбор (1-5): ").strip()
        
        if choice == '5':
            print("Выход.")
            break
            
        if choice == '1':
            print("\n[ЭКСПЕРИМЕНТ 1: Скейлинг]")
            from experiments.experiment_1_scaling import run_scaling_experiment
            run_scaling_experiment()
            from visualization.plot_scaling import plot_scaling_results
            plot_scaling_results()
                
        elif choice == '2':
            print("\n[ЭКСПЕРИМЕНТ 2: Глубина]")
            from experiments.experiment_2_depth import run_depth_experiment
            run_depth_experiment()
            from visualization.plot_depth_comparison import plot_depth_comparison
            plot_depth_comparison()
                
        elif choice == '3':
            print("\n[ЭКСПЕРИМЕНТ 3: Абляции]")
            from experiments.experiment_3_ablation import run_ablation_experiment
            run_ablation_experiment()
            from visualization.plot_ablation import plot_ablation_results
            plot_ablation_results()
                
        elif choice == '4':
            print("\n[ЭКСПЕРИМЕНТ 1.2: Скейлинг на доп. функциях]")
            from experiments.experiment_1_scaling_v2 import run_scaling_experiment_v2
            run_scaling_experiment_v2()
            from visualization.plot_scaling_v2 import plot_scaling_results_v2
            plot_scaling_results_v2()
        else:
            print("Неверный выбор. Только 1,2,3,4,5.")

if __name__ == "__main__":
    main()