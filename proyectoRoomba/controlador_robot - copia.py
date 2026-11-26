
import serial
import time
from datetime import datetime


class ControladorRobot:

    def __init__(
        self,
        modo="simulacion",
        puerto="COM5",
        baudrate=115200,
        timeout=2
    ):

        self.modo = modo
        self.puerto = puerto
        self.baudrate = baudrate
        self.timeout = timeout

        self.conectado = False
        self.serial = None
        self.historial = []

        self.comandos_validos = [
            "AVANZAR",
            "RETROCEDER",
            "GIRAR_IZQUIERDA",
            "GIRAR_DERECHA",
            "DETENER",
            "ASPIRAR_ON",
            "ASPIRAR_OFF"
        ]

        if self.modo == "real":
            self.conectar()

        else:
            print("[SIMULACION] Controlador preparado.")

    def conectar(self):

        try:

            self.serial = serial.Serial(
                self.puerto,
                self.baudrate,
                timeout=0.2,
                write_timeout=2
            )

            time.sleep(2)
            self.serial.reset_input_buffer()

            self.conectado = True

            print("[ESP32] Conexion establecida.")
            print("[ESP32] Puerto:", self.puerto)

        except (serial.SerialException, OSError) as error:

            self.conectado = False
            self.serial = None

            print("[ERROR] No se pudo conectar con la ESP32.")
            print(error)

    def enviar_comando(self, comando):

        comando = comando.upper().strip()

        if comando not in self.comandos_validos:

            print("[ERROR] Comando no valido:", comando)
            return False

        if self.modo == "real" and not self.conectado:

            print("[ERROR] ESP32 no conectada.")
            return False

        evento = {
            "fecha": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "comando": comando,
            "modo": self.modo
        }

        self.historial.append(evento)

        if self.modo == "simulacion":

            print("[SIMULACION] Comando:", comando)
            return True

        try:

            self.serial.write(
                (comando + "\n").encode("utf-8")
            )

            self.serial.flush()

            print("[ESP32] Enviado:", comando)

            inicio = time.time()
            respuesta_ok = False

            while time.time() - inicio < self.timeout:

                respuesta = self.serial.readline().decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                if respuesta:

                    print("[ESP32]", respuesta)

                    if respuesta == "OK":
                        respuesta_ok = True
                        break

            if respuesta_ok:
                return True

            print("[ERROR] La ESP32 no confirmo el comando.")
            return False

        except (serial.SerialException, OSError) as error:

            print("[ERROR SERIAL]", error)
            self.conectado = False

            return False

    def avanzar(self):
        return self.enviar_comando("AVANZAR")

    def retroceder(self):
        return self.enviar_comando("RETROCEDER")

    def girar_izquierda(self):
        return self.enviar_comando("GIRAR_IZQUIERDA")

    def girar_derecha(self):
        return self.enviar_comando("GIRAR_DERECHA")

    def detener(self):
        return self.enviar_comando("DETENER")

    def activar_aspiracion(self):
        return self.enviar_comando("ASPIRAR_ON")

    def desactivar_aspiracion(self):
        return self.enviar_comando("ASPIRAR_OFF")

    def obtener_ultimo_comando(self):

        if not self.historial:
            return None

        return self.historial[-1]

    def obtener_historial(self):
        return self.historial

    def obtener_total_comandos(self):
        return len(self.historial)

    def mostrar_estado(self):

        print("========== ESTADO DEL CONTROLADOR ==========")
        print("Modo:", self.modo)
        print("Puerto:", self.puerto)
        print("Conectado:", self.conectado)
        print("Total de comandos:", self.obtener_total_comandos())
        print("============================================")

    def cerrar(self):

        if self.serial is not None and self.serial.is_open:

            self.serial.close()
            self.conectado = False

            print("[ESP32] Conexion cerrada.")
