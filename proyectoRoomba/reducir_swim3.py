
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent

ORIGEN = ROOT / "dataset_swim3_2_clases"
DESTINO = ROOT / "dataset_swim3_reducido"

# Cantidad de imágenes por división
CANTIDADES = {
    "train": 2000,
    "valid": 400,
    "test": 200
}

random.seed(42)

def main():
    if not ORIGEN.exists():
        raise FileNotFoundError(f"No encuentro: {ORIGEN}")

    if DESTINO.exists():
        raise FileExistsError(
            f"Ya existe {DESTINO}. No se sobrescribirá."
        )

    for division, cantidad in CANTIDADES.items():

        carpeta_images = ORIGEN / division / "images"
        carpeta_labels = ORIGEN / division / "labels"

        if not carpeta_images.exists():
            raise FileNotFoundError(carpeta_images)

        if not carpeta_labels.exists():
            raise FileNotFoundError(carpeta_labels)

        imagenes = [
            img for img in carpeta_images.iterdir()
            if img.is_file()
            and img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            and (carpeta_labels / (img.stem + ".txt")).exists()
        ]

        if len(imagenes) < cantidad:
            raise ValueError(
                f"No hay suficientes imágenes en {division}: "
                f"{len(imagenes)} disponibles."
            )

        seleccionadas = random.sample(imagenes, cantidad)

        destino_images = DESTINO / division / "images"
        destino_labels = DESTINO / division / "labels"

        destino_images.mkdir(parents=True, exist_ok=True)
        destino_labels.mkdir(parents=True, exist_ok=True)

        for imagen in seleccionadas:
            etiqueta = carpeta_labels / (imagen.stem + ".txt")

            shutil.copy2(imagen, destino_images / imagen.name)
            shutil.copy2(etiqueta, destino_labels / etiqueta.name)

        print(f"\n{division.upper()}")
        print(f"Imágenes copiadas: {len(seleccionadas)}")

    yaml = """train: train/images
val: valid/images
test: test/images

nc: 2
names: ['papel', 'plastico']
"""

    with open(DESTINO / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml)

    print("\n¡REDUCCIÓN TERMINADA!")
    print(f"Nuevo dataset: {DESTINO}")


if __name__ == "__main__":
    main()
