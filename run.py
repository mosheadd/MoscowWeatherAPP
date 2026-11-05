import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

NOTEBOOK = BASE_DIR / "weather_predictor.ipynb"
MODEL = BASE_DIR / "model" / "laml_model.joblib"
DATA = BASE_DIR / "data" / "topredict.csv"
APP = BASE_DIR / "app.py"


def train_model():
    print("Запускаем обучение модели...")

    (BASE_DIR / "data").mkdir(exist_ok=True)
    (BASE_DIR / "model").mkdir(exist_ok=True)

    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", str(NOTEBOOK),
            "--output", "weather_predictor_executed.ipynb",
            "--ExecutePreprocessor.timeout=-1",
        ],
        cwd=BASE_DIR,
        check=True
    )

    if not MODEL.exists() or not DATA.exists():
        raise FileNotFoundError(
            "После обучения модель или topredict.csv не найдены."
        )

    print("Обучение завершено!")


def run_app():
    print("Запускаем Streamlit...")

    subprocess.run(
        [
            sys.executable, "-m", "streamlit",
            "run", str(APP)
        ],
        cwd=BASE_DIR,
        check=True
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        action="store_true",
        help="Принудительно переобучить модель"
    )

    args = parser.parse_args()

    if args.train or not MODEL.exists() or not DATA.exists():
        train_model()
    else:
        print("Модель и данные найдены. Обучение пропущено.")

    run_app()


if __name__ == "__main__":
    main()
