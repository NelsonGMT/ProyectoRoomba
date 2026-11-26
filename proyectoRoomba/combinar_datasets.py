
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

DATASETS = [
    ROOT / "dataset_2_clases",
    ROOT / "dataset_swim3_reducido"
]

DESTINO = ROOT / "dataset_combinado"

DIVISIONES = ["train", "valid", "test"]

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    if DESTINO.exists():
        raise FileExistsError(
            f"Ya existe {DESTINO}. No se sobrescribirá."
        )

    for dataset in DATASETS:
        if not dataset.exists():
            raise FileNotFoundError(dataset)

    for division in DIVISIONES:
        destino_images = DESTINO / division / "images"
        destino_labels = DESTINO / division / "labels"

        destino_images.mkdir(parents=True, exist_ok=True)
        destino_labels.mkdir(parents=True, exist_ok=True)

        total = 0

        for numero, dataset in enumerate(DATASETS, start=1):
            origen_images = dataset / division / "images"
            origen_labels = dataset / division / "labels"

            if not origen_images.exists() or not origen_labels.exists():
                raise FileNotFoundError(
                    f"Faltan images o labels en {dataset / division}"
                )

            for imagen in origen_images.iterdir():
                if not imagen.is_file() or imagen.suffix.lower() not in EXTENSIONES:
                    continue

                etiqueta = origen_labels / (imagen.stem + ".txt")

                if not etiqueta.exists():
                    raise FileNotFoundError(
                        f"Falta la etiqueta de {imagen.name}"
                    )

                # Prefijo para evitar nombres repetidos
                nuevo_nombre = f"ds{numero}_{imagen.stem}"

                destino_imagen = destino_images / (
                    nuevo_nombre + imagen.suffix.lower()
                )
                destino_etiqueta = destino_labels / (
                    nuevo_nombre + ".txt"
                )

                shutil.copy2(imagen, destino_imagen)
                shutil.copy2(etiqueta, destino_etiqueta)

                total += 1

        print(f"{division.upper()}: {total} imágenes copiadas")

    yaml = """train: train/images
val: valid/images
test: test/images

nc: 2
names: ['papel', 'plastico']
"""

    with open(DESTINO / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml)

    print("\n¡COMBINACIÓN TERMINADA!")
    print(f"Dataset creado en: {DESTINO}")

if __name__ == "__main__":
    main()
