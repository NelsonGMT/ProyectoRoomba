
import pygame
from camara_yolo import CamaraYOLO

pygame.init()

ventana = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Prueba de camara YOLO integrada")

reloj = pygame.time.Clock()

camara = CamaraYOLO(
    ruta_modelo="../runs/papel_plastico_mejorado_gpu-3/weights/best.pt",
    confianza=0.35,
    device="auto"
)

ejecutando = True

try:

    while ejecutando:

        for evento in pygame.event.get():

            if evento.type == pygame.QUIT:
                ejecutando = False

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    ejecutando = False

        imagen = camara.actualizar()

        if imagen is not None:

            imagen = pygame.surfarray.make_surface(
                imagen.swapaxes(0, 1)
            )

            imagen = pygame.transform.scale(
                imagen,
                (640, 480)
            )

            ventana.fill((20, 24, 32))
            ventana.blit(imagen, (80, 60))

        pygame.display.flip()
        reloj.tick(30)

finally:
    camara.cerrar()
    pygame.quit()
