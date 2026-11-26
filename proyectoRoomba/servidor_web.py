
import cv2
import time
import base64
from pathlib import Path

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

from simulacion.camara_yolo import CamaraYOLO
from simulacion.registro_residuos import RegistroResiduos


# --------------------------------------------------
# CONFIGURACION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RUTA_MODELO = (
    BASE_DIR
    / "runs"
    / "papel_plastico_mejorado_gpu-3"
    / "weights"
    / "best.pt"
)

INDICE_CAMARA = 0
CONFIANZA = 0.35
TAMANO = 640

WEB_DIR = BASE_DIR / "web_esp32"

print("[WEB] Carpeta:", WEB_DIR)
print("[WEB] Existe:", WEB_DIR.exists())
print("[WEB] HTML existe:", (WEB_DIR / "index.html").exists())

print("[YOLO] Modelo:", RUTA_MODELO)
print("[YOLO] Existe:", RUTA_MODELO.exists())

# --------------------------------------------------
# SERVIDOR WEB
# --------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "robot-aspiradora"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)

# --------------------------------------------------
# YOLO Y REGISTRO
# --------------------------------------------------

camara = None
registro = RegistroResiduos(frames_rearme=900)

ejecutando = False


# --------------------------------------------------
# PAGINA PRINCIPAL
# --------------------------------------------------
@app.route("/")
def inicio():
    return send_from_directory(WEB_DIR, "index.html")




@app.route("/<path:archivo>")
def archivos_web(archivo):
    return send_from_directory(WEB_DIR, archivo)

# --------------------------------------------------
# TRANSMISION DE CAMARA
# --------------------------------------------------

def transmitir_camara():

    global ejecutando

    while ejecutando:

        imagen = camara.actualizar()

        if imagen is None:
            socketio.sleep(0.05)
            continue

        # Convertir RGB a JPEG
        imagen_bgr = cv2.cvtColor(
            imagen,
            cv2.COLOR_RGB2BGR
        )

        correcto, buffer = cv2.imencode(
            ".jpg",
            imagen_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        )

        if not correcto:
            socketio.sleep(0.05)
            continue

        imagen_base64 = base64.b64encode(
            buffer
        ).decode("utf-8")

        # Actualizar registro
        registro.actualizar(camara.detecciones)

        # Datos enviados a la pagina
        datos = {
            "imagen": imagen_base64,
            "detecciones": camara.detecciones,
            "fps": round(camara.fps, 2),
            "cantidad": camara.cantidad,
            "conteo": registro.conteo,
            "eventos": registro.obtener_eventos()
        }

        socketio.emit(
            "actualizar_camara",
            datos
        )

        socketio.sleep(0.03)


# --------------------------------------------------
# CONEXION DE CLIENTES
# --------------------------------------------------

@socketio.on("connect")
def conectar():

    print("[WEB] Cliente conectado.")

    emit(
        "estado_servidor",
        {
            "estado": "conectado",
            "mensaje": "YOLO preparado"
        }
    )


# --------------------------------------------------
# INICIAR CAMARA
# --------------------------------------------------

def iniciar_sistema():

    global camara
    global ejecutando

    print("\nIniciando sistema YOLO...")

    camara = CamaraYOLO(
        ruta_modelo=RUTA_MODELO,
        indice=INDICE_CAMARA,
        confianza=CONFIANZA,
        tamano=TAMANO,
        device="auto"
    )

    ejecutando = True

    socketio.start_background_task(
        transmitir_camara
    )

    print("[SISTEMA] Camara y YOLO funcionando.")


# --------------------------------------------------
# CERRAR SISTEMA
# --------------------------------------------------

def cerrar_sistema():

    global ejecutando

    ejecutando = False

    if camara is not None:
        camara.cerrar()

    registro.guardar_csv()


# --------------------------------------------------
# EJECUCION
# --------------------------------------------------

if __name__ == "__main__":

    try:

        iniciar_sistema()

        print("\nServidor disponible en:")
        print("http://127.0.0.1:5000")

        socketio.run(
            app,
            host="0.0.0.0",
            port=5000,
            debug=False,
            use_reloader=False
        )

    except KeyboardInterrupt:

        print("\nCerrando sistema...")

    finally:

        cerrar_sistema()
