"""Exporta el modelo entrenado a NCNN. Ejecutar en PC compatible o Raspberry Pi."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "runs" / "papel_plastico" / "weights" / "best.pt"
if not BEST.is_file():
    raise FileNotFoundError(f"No existe {BEST}. Ejecuta primero entrenar_modelo.py")
model = YOLO(str(BEST))
destino = model.export(format="ncnn", imgsz=416)
print("Exportación terminada:", destino)
print("Copia la carpeta NCNN completa a la Raspberry Pi y úsala con --modelo.")
