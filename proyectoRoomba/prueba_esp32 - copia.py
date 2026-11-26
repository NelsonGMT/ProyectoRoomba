
import serial
import time

# Cambia COM3 por el puerto de tu ESP32
PUERTO = "COM5"
BAUDRATE = 115200

try:
    esp32 = serial.Serial(
        port=PUERTO,
        baudrate=BAUDRATE,
        timeout=2
    )

    time.sleep(2)

    print("================================")
    print(" CONEXION CON ESP32 ESTABLECIDA")
    print("================================")

    while True:

        print("\nComandos disponibles:")
        print("1. AVANZAR")
        print("2. RETROCEDER")
        print("3. GIRAR_IZQUIERDA")
        print("4. GIRAR_DERECHA")
        print("5. DETENER")
        print("6. ASPIRAR_ON")
        print("7. ASPIRAR_OFF")
        print("0. SALIR")

        opcion = input("\nSelecciona un comando: ").strip()

        comandos = {
            "1": "AVANZAR",
            "2": "RETROCEDER",
            "3": "GIRAR_IZQUIERDA",
            "4": "GIRAR_DERECHA",
            "5": "DETENER",
            "6": "ASPIRAR_ON",
            "7": "ASPIRAR_OFF"
        }

        if opcion == "0":
            break

        if opcion not in comandos:
            print("Opcion no valida.")
            continue

        comando = comandos[opcion]

        esp32.write((comando + "\n").encode("utf-8"))

        print("\nEnviado:", comando)

        # Leer respuesta de la ESP32
        while True:
            respuesta = esp32.readline().decode(
                "utf-8", errors="ignore"
            ).strip()

            if respuesta:
                print("ESP32:", respuesta)

            if respuesta == "OK":
                break

    esp32.close()
    print("\nConexion cerrada.")

except serial.SerialException as error:
    print("Error de comunicacion:", error)

except KeyboardInterrupt:
    print("\nPrograma interrumpido.")
