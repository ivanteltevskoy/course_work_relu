# course_work_relu
## Окружение
- Python 3.13
- Зависимости: см. `requirements.txt`
- Эксперименты используют PyTorch.

## Структура проекта
- `experiments/` — скрипты экспериментов:
  - `experiment_1_scaling.py` — Эксп. 1: ошибка vs число параметров (фикс. глубина / фикс. ширина)
  - `experiment_1_scaling_v2.py` — Эксп. 1.2: то же для дополнительных функций
  - `experiment_2_depth.py` — Эксп. 2: роль глубины при фиксированном бюджете параметров
  - `experiment_3_ablation.py` — Эксп. 3: абляции (noise/weight decay/sampler/grid/seed)
- `functions/` — тестовые функции и генерация датасетов
- `models/` — MLP ReLU и утилиты подсчёта параметров
- `training/` — обучение и метрики L2/L∞
- `visualization/` — построение графиков
- `results/` — результаты (`results/data/*.csv`) и рисунки (`results/figures/*.png`)

## Как запустить
1) Установить зависимости:
```bash
pip install -r requirements.txt

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
