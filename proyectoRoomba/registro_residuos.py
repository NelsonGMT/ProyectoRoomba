
import csv
from datetime import datetime
from pathlib import Path


class RegistroResiduos:


    def reiniciar_deteccion(self):

        # Permitir detectar nuevamente ambas clases
        self.activos = {
            "papel": False,
            "plastico": False
        }

        self.frames_ausentes = {
            "papel": 0,
            "plastico": 0
        }

        # Limpiar eventos que todavía no se han recogido
        self.eventos_nuevos.clear()

        print("[YOLO] Estado de deteccion reiniciado.")


    def __init__(self, frames_rearme=900):

        self.frames_rearme = frames_rearme

        self.conteo = {
            "papel": 0,
            "plastico": 0
        }

        self.activos = {
            "papel": False,
            "plastico": False
        }

        self.frames_ausentes = {
            "papel": 0,
            "plastico": 0
        }

        self.historial = []
        self.eventos_nuevos = []

    def normalizar_clase(self, nombre):

        nombre = str(nombre).lower().strip()

        if nombre in ("papel", "paper"):
            return "papel"

        if nombre in ("plastico", "plástico", "plastic"):
            return "plastico"

        return None

    def actualizar(self, detecciones):

        clases_presentes = set()

        for deteccion in detecciones:

            clase = self.normalizar_clase(
                deteccion.get("clase", "")
            )

            if clase is not None:
                clases_presentes.add(clase)

        for clase in ("papel", "plastico"):

            if clase in clases_presentes:

                self.frames_ausentes[clase] = 0

                if not self.activos[clase]:

                    self.activos[clase] = True
                    self.conteo[clase] += 1

                    confianza = max(
                        (
                            float(d.get("confianza", 0))
                            for d in detecciones
                            if self.normalizar_clase(
                                d.get("clase", "")
                            ) == clase
                        ),
                        default=0
                    )

                    evento = {
                        "fecha_hora": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "residuo": clase,
                        "confianza": round(
                            confianza * 100,
                            2
                        )
                    }

                    self.historial.append(evento)
                    self.eventos_nuevos.append(evento)

                    print(
                        f"Nuevo residuo: {clase} | "
                        f"Confianza: {confianza * 100:.2f}%"
                    )

            else:

                if self.activos[clase]:

                    self.frames_ausentes[clase] += 1

                    if (
                        self.frames_ausentes[clase]
                        >= self.frames_rearme
                    ):

                        self.activos[clase] = False
                        self.frames_ausentes[clase] = 0

    def obtener_eventos(self):

        eventos = self.eventos_nuevos.copy()
        self.eventos_nuevos.clear()

        return eventos

    def obtener_total(self):

        return (
            self.conteo["papel"]
            + self.conteo["plastico"]
        )

    def guardar_csv(self, ruta=None):

        if ruta is None:

            nombre = (
                "registro_residuos_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".csv"
            )

            ruta = Path(__file__).resolve().parent / nombre

        ruta = Path(ruta)

        columnas = [
            "fecha_hora",
            "residuo",
            "confianza"
        ]

        with open(
            ruta,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as archivo:

            escritor = csv.DictWriter(
                archivo,
                fieldnames=columnas
            )

            escritor.writeheader()
            escritor.writerows(self.historial)

        print(f"Registro guardado en: {ruta}")

        return ruta
