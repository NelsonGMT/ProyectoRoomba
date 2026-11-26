
#!/usr/bin/env python3
# Detección de papel y plástico con YOLO.
# Compatible con cámara USB y Raspberry Pi CSI.

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def abrir_camara(indice=0, csi=False, ancho=640, alto=480):
    if csi:
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                "Falta Picamera2. Instala con: "
                "sudo apt install -y python3-picamera2"
            ) from e

        cam = Picamera2(camera_num=indice)
        config = cam.create_video_configuration(
            main={"size": (ancho, alto), "format": "RGB888"},
            buffer_count=2
        )
        cam.configure(config)
        cam.start()
        time.sleep(1)
        return ("csi", cam)

    cap = cv2.VideoCapture(
        indice,
        cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"No se pudo abrir la cámara USB índice {indice}. "
            "Prueba --camara 1."
        )

    return ("usb", cap)


def leer_camara(camara):
    tipo, obj = camara

    if tipo == "csi":
        return obj.capture_array()

    ok, frame = obj.read()

    if not ok or frame is None:
        raise RuntimeError("La cámara USB dejó de entregar imágenes.")

    return frame


def cerrar_camara(camara):
    if not camara:
        return

    tipo, obj = camara

    if tipo == "csi":
        obj.stop()
        obj.close()
    else:
        obj.release()


def main():
    p = argparse.ArgumentParser(
        description="Detecta residuos de papel y plástico."
    )

    p.add_argument(
        "--modelo",
        default="runs/papel_plastico_mejorado_gpu-3/weights/best.pt",
        help="Ruta del modelo best.pt o carpeta NCNN"
    )

    p.add_argument("--csi", action="store_true")
    p.add_argument("--camara", type=int, default=0)
    p.add_argument("--tamano", type=int, default=640)
    p.add_argument("--confianza", type=float, default=0.35)

    p.add_argument(
        "--device",
        default="auto",
        help="auto, cpu o 0 para utilizar la GPU NVIDIA"
    )

    p.add_argument("--sin-ventana", action="store_true")

    args = p.parse_args()

    if not 0 < args.confianza <= 1:
        p.error("--confianza debe estar entre 0 y 1")

    modelo_path = Path(args.modelo).expanduser()

    if not modelo_path.exists():
        raise FileNotFoundError(
            f"No encuentro el modelo: {modelo_path}"
        )

    # Seleccionar dispositivo
    if args.device == "auto":
        dispositivo = 0 if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu":
        dispositivo = "cpu"
    else:
        dispositivo = args.device

    print("Dispositivo:", dispositivo)

    if dispositivo == 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    modelo = YOLO(str(modelo_path), task="detect")

    print("Clases del modelo:", modelo.names)
    print("Pulsa Q o Esc para salir.")
    print("La detección no activa motores.")

    camara = None

    try:
        camara = abrir_camara(args.camara, args.csi)

        while True:
            inicio = time.perf_counter()

            frame = leer_camara(camara)

            resultado = modelo.predict(
                source=frame,
                imgsz=args.tamano,
                conf=args.confianza,
                device=dispositivo,
                verbose=False,
                max_det=30
            )[0]

            anotado = resultado.plot()

            fps = 1.0 / max(time.perf_counter() - inicio, 1e-6)

            cantidad = (
                len(resultado.boxes)
                if resultado.boxes is not None
                else 0
            )

            cv2.putText(
                anotado,
                f"Residuos: {cantidad} | FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            if not args.sin_ventana:
                cv2.imshow(
                    "Deteccion de basura - papel/plastico",
                    anotado
                )

                tecla = cv2.waitKey(1) & 0xFF

                if tecla in (ord("q"), ord("Q"), 27):
                    break

            else:
                print(
                    f"Detecciones={cantidad} FPS={fps:.1f}",
                    flush=True
                )
                time.sleep(0.05)

    finally:
        cerrar_camara(camara)

        if not args.sin_ventana:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma terminado.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
