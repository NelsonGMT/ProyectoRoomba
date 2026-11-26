
import cv2
import torch
import time
from pathlib import Path
from ultralytics import YOLO


class CamaraYOLO:

    def __init__(
        self,
        ruta_modelo,
        indice=0,
        confianza=0.35,
        tamano=640,
        device="auto"
    ):

        self.confianza = confianza
        self.tamano = tamano
        self.indice = indice

        if device == "auto":

            self.dispositivo = (
                0 if torch.cuda.is_available() else "cpu"
            )

        else:

            self.dispositivo = device

        ruta = Path(ruta_modelo)

        if not ruta.exists():

            raise FileNotFoundError(
                f"No se encuentra el modelo: {ruta}"
            )

        print("Cargando modelo YOLO11n...")

        self.modelo = YOLO(
            str(ruta),
            task="detect"
        )

        print("Modelo cargado:", self.modelo.names)
        print("Dispositivo:", self.dispositivo)

        self.camara = cv2.VideoCapture(
            indice,
            cv2.CAP_ANY
        )

        self.camara.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.camara.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        self.camara.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if not self.camara.isOpened():

            raise RuntimeError(
                "No se pudo abrir la webcam."
            )

        self.fps = 0
        self.cantidad = 0
        self.detecciones = []

        print("Webcam iniciada correctamente.")

    def actualizar(self):

        correcto, frame = self.camara.read()

        if not correcto or frame is None:
            return None

        inicio = time.perf_counter()

        resultado = self.modelo.predict(
            source=frame,
            imgsz=self.tamano,
            conf=self.confianza,
            device=self.dispositivo,
            verbose=False,
            max_det=30
        )[0]

        imagen = resultado.plot()

        duracion = time.perf_counter() - inicio

        self.fps = 1.0 / max(duracion, 1e-6)

        self.cantidad = (
            len(resultado.boxes)
            if resultado.boxes is not None
            else 0
        )

        self.detecciones = []

        if resultado.boxes is not None:

            for caja in resultado.boxes:

                clase_id = int(caja.cls[0])
                confianza = float(caja.conf[0])

                nombre = self.modelo.names[clase_id]

                self.detecciones.append({
                    "clase": nombre,
                    "confianza": confianza
                })

        return cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2RGB
        )

    def cerrar(self):

        if self.camara is not None:

            self.camara.release()
            self.camara = None

        print("Webcam cerrada.")
