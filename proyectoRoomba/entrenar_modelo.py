
"""Entrenamiento de papel y plastico utilizando la GPU NVIDIA."""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# Dataset combinado: Roboflow + SWiM3
DATA = ROOT / "dataset_combinado" / "data.yaml"

# Modelo anterior para continuar mejorando la detección
MODELO_ANTERIOR = (
    ROOT / "runs" / "papel_plastico-6" / "weights" / "best.pt"
)

if not DATA.is_file():
    raise FileNotFoundError(f"No encuentro el dataset: {DATA}")

if not MODELO_ANTERIOR.is_file():
    raise FileNotFoundError(f"No encuentro el modelo: {MODELO_ANTERIOR}")

if __name__ == "__main__":
    model = YOLO(str(MODELO_ANTERIOR))

    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=640,
        batch=4,
        patience=15,
        workers=0,
        project=str(ROOT / "runs"),
        name="papel_plastico_mejorado_gpu",
        pretrained=True,
        cache=False,
        device=0
    )

    print("¡Entrenamiento terminado!")
