
from pathlib import Path
import shutil

# ==========================================
# RUTAS
# ==========================================

ROOT = Path(__file__).resolve().parent

ORIGEN = (
    Path.home()
    / "Downloads"
    / "SWiM3-Dataset-main"
    / "SWiM3-Dataset-main"
    / "Splitted_Dataset"
)

DESTINO = ROOT / "dataset_swim3_2_clases"

# Índices originales de SWiM3:
# 0 = Plastic
# 1 = Paper
# 2 = Metal

# Nuevos índices:
# 0 = papel
# 1 = plastico

MAPEO = {
    1: 0,  # Paper -> papel
    0: 1,  # Plastic -> plastico
}

DIVISIONES = {
    "train": "train",
    "valid": "valid",
    "test": "test"
}


def convertir_etiqueta(archivo, destino):
    objetos = 0

    with open(archivo, "r", encoding="utf-8") as entrada:
        lineas = entrada.readlines()

    nuevas_lineas = []

    for linea in lineas:
        partes = linea.strip().split()

        if len(partes) != 5:
            continue

        try:
            clase = int(partes[0])
        except ValueError:
            continue

        # Ignorar metal y cualquier clase no seleccionada
        if clase not in MAPEO:
            continue

        nueva_clase = MAPEO[clase]

        # Mantener las coordenadas originales
        nueva_linea = (
            f"{nueva_clase} "
            + " ".join(partes[1:])
        )

        nuevas_lineas.append(nueva_linea)
        objetos += 1

    with open(destino, "w", encoding="utf-8") as salida:
        salida.write("\n".join(nuevas_lineas))

    return objetos


def main():
    if not ORIGEN.exists():
        raise FileNotFoundError(
            f"No encuentro el dataset original:\n{ORIGEN}"
        )

    print("Iniciando conversión de SWiM3...")
    print(f"Origen: {ORIGEN}")
    print(f"Destino: {DESTINO}")

    resumen = {}

    for division, nombre in DIVISIONES.items():

        carpeta_origen = ORIGEN / division

        # Aceptar también la carpeta tests, si existe
        if not carpeta_origen.exists() and division == "test":
            carpeta_origen = ORIGEN / "tests"

        imagenes_origen = carpeta_origen / "images"
        etiquetas_origen = carpeta_origen / "labels"

        if not imagenes_origen.exists():
            raise FileNotFoundError(
                f"No encuentro: {imagenes_origen}"
            )

        imagenes_destino = DESTINO / nombre / "images"
        etiquetas_destino = DESTINO / nombre / "labels"

        imagenes_destino.mkdir(parents=True, exist_ok=True)
        etiquetas_destino.mkdir(parents=True, exist_ok=True)

        total_imagenes = 0
        total_papel = 0
        total_plastico = 0

        for imagen in imagenes_origen.iterdir():

            if not imagen.is_file():
                continue

            if imagen.suffix.lower() not in (
                ".jpg", ".jpeg", ".png", ".bmp"
            ):
                continue

            etiqueta = etiquetas_origen / (imagen.stem + ".txt")

            if not etiqueta.exists():
                continue

            destino_imagen = imagenes_destino / imagen.name
            destino_etiqueta = etiquetas_destino / etiqueta.name

            objetos = convertir_etiqueta(
                etiqueta,
                destino_etiqueta
            )

            shutil.copy2(imagen, destino_imagen)

            total_imagenes += 1

            with open(destino_etiqueta, "r", encoding="utf-8") as f:
                for linea in f:
                    if not linea.strip():
                        continue

                    clase = int(linea.split()[0])

                    if clase == 0:
                        total_papel += 1
                    elif clase == 1:
                        total_plastico += 1

        resumen[nombre] = (
            total_imagenes,
            total_papel,
            total_plastico
        )

        print(f"\n{nombre.upper()}")
        print(f"Imágenes: {total_imagenes}")
        print(f"Objetos de papel: {total_papel}")
        print(f"Objetos de plástico: {total_plastico}")

    # Crear data.yaml
    contenido_yaml = """train: train/images
val: valid/images
test: test/images

nc: 2
names: ['papel', 'plastico']
"""

    with open(DESTINO / "data.yaml", "w", encoding="utf-8") as f:
        f.write(contenido_yaml)

    print("\n¡CONVERSIÓN TERMINADA!")
    print(f"Dataset creado en: {DESTINO}")
    print("Clases: papel y plastico")


if __name__ == "__main__":
    main()
