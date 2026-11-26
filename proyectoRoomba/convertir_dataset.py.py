
from pathlib import Path
import yaml
import shutil

BASE = Path(__file__).resolve().parent

ORIGINAL = BASE / "dataset_original"
DESTINO = BASE / "dataset_2_clases"

CLASES_PAPEL = [
    "paper",
    "normal paper",
    "magazine paper",
    "paper bag",
    "paper cup",
    "paper straw",
    "pizza box",
    "corrugated carton",
    "other carton",
    "egg carton",
    "drink carton",
    "meal carton",
    "tissues",
    "toilet tube",
    "wrapping paper",
]

CLASES_PLASTICO = [
    "plastic",
    "clear plastic bottle",
    "other plastic",
    "other plastic bottle",
    "other plastic container",
    "other plastic cup",
    "other plastic wrapper",
    "disposable plastic cup",
    "plastic bottle cap",
    "plastic film",
    "plastic lid",
    "plastic straw",
    "plastic utensils",
    "polypropylene bag",
    "single-use carrier bag",
    "six pack rings",
    "squeezable tube",
    "styrofoam piece",
    "foam cup",
    "foam food container",
    "garbage bag",
]

def normalizar(nombre):
    return str(nombre).strip().lower()

def convertir_a_caja(partes):
    """
    Convierte una etiqueta de segmentación YOLO
    en una caja delimitadora YOLO.
    """

    valores = [float(x) for x in partes]

    # Formato de detección: clase + 4 coordenadas
    if len(valores) == 5:
        return valores[1:]

    # Formato de segmentación: clase + puntos X,Y
    coordenadas = valores[1:]

    if len(coordenadas) < 6 or len(coordenadas) % 2 != 0:
        return None

    xs = coordenadas[0::2]
    ys = coordenadas[1::2]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    centro_x = (x_min + x_max) / 2
    centro_y = (y_min + y_max) / 2
    ancho = x_max - x_min
    alto = y_max - y_min

    return [centro_x, centro_y, ancho, alto]


def main():

    archivo_yaml = ORIGINAL / "data.yaml"

    if not archivo_yaml.exists():
        print("ERROR: No encuentro dataset_original/data.yaml")
        return

    with open(archivo_yaml, "r", encoding="utf-8") as archivo:
        configuracion = yaml.safe_load(archivo)

    nombres = configuracion["names"]

    if isinstance(nombres, dict):
        nombres = [
            nombres[i] for i in sorted(nombres, key=lambda x: int(x))
        ]

    mapa = {}

    for indice, nombre in enumerate(nombres):
        nombre = normalizar(nombre)

        if nombre in CLASES_PAPEL:
            mapa[indice] = 0

        elif nombre in CLASES_PLASTICO:
            mapa[indice] = 1

    print("Clases originales:", len(nombres))
    print("Clases que se conservarán:", len(mapa))

    if DESTINO.exists():
        shutil.rmtree(DESTINO)

    for conjunto in ["train", "valid", "test"]:

        carpeta_imagenes = ORIGINAL / conjunto / "images"
        carpeta_etiquetas = ORIGINAL / conjunto / "labels"

        if not carpeta_imagenes.exists():
            print(f"No existe: {carpeta_imagenes}")
            continue

        destino_imagenes = DESTINO / conjunto / "images"
        destino_etiquetas = DESTINO / conjunto / "labels"

        destino_imagenes.mkdir(parents=True, exist_ok=True)
        destino_etiquetas.mkdir(parents=True, exist_ok=True)

        total_imagenes = 0
        total_objetos = 0
        total_papel = 0
        total_plastico = 0

        for imagen in carpeta_imagenes.iterdir():

            if not imagen.is_file():
                continue

            shutil.copy2(
                imagen,
                destino_imagenes / imagen.name
            )

            total_imagenes += 1

            etiqueta = carpeta_etiquetas / (imagen.stem + ".txt")

            nuevas_lineas = []

            if etiqueta.exists():

                with open(etiqueta, "r", encoding="utf-8") as archivo:

                    for linea in archivo:

                        partes = linea.strip().split()

                        if len(partes) < 5:
                            continue

                        try:
                            clase_original = int(partes[0])

                            if clase_original not in mapa:
                                continue

                            caja = convertir_a_caja(partes)

                            if caja is None:
                                continue

                            # Validar coordenadas normalizadas
                            if not all(0 <= x <= 1 for x in caja):
                                continue

                            clase_nueva = mapa[clase_original]

                            texto = (
                                str(clase_nueva) + " " +
                                " ".join(f"{x:.8f}" for x in caja)
                            )

                            nuevas_lineas.append(texto)

                            if clase_nueva == 0:
                                total_papel += 1
                            else:
                                total_plastico += 1

                        except (ValueError, IndexError):
                            continue

            with open(
                destino_etiquetas / (imagen.stem + ".txt"),
                "w",
                encoding="utf-8"
            ) as archivo:

                archivo.write("\n".join(nuevas_lineas))

            total_objetos += len(nuevas_lineas)

        print(f"\n{conjunto}:")
        print("Imágenes:", total_imagenes)
        print("Objetos:", total_objetos)
        print("Papel:", total_papel)
        print("Plástico:", total_plastico)

    nuevo_yaml = {
        "path": str(DESTINO.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["papel", "plastico"]
    }

    with open(
        DESTINO / "data.yaml",
        "w",
        encoding="utf-8"
    ) as archivo:

        yaml.safe_dump(
            nuevo_yaml,
            archivo,
            allow_unicode=True,
            sort_keys=False
        )

    print("\n¡Conversión terminada!")
    print("Nuevo dataset:", DESTINO)


if __name__ == "__main__":
    main()
