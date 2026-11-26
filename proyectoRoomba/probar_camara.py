
import cv2

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la webcam")
    exit()

print("Camara iniciada correctamente")
print("Presiona ESC para cerrar")

while True:

    correcto, imagen = camara.read()

    if not correcto:
        print("No se pudo obtener imagen")
        break

    cv2.imshow("Prueba de webcam", imagen)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
        break

camara.release()
cv2.destroyAllWindows()
