
import pygame
import sys
import time
import heapq
import math
import random
import csv

from pathlib import Path
from collections import deque
from datetime import datetime

from camara_yolo import CamaraYOLO
from registro_residuos import RegistroResiduos
from controlador_robot import ControladorRobot


# =====================================================
# CONFIGURACION
# =====================================================

pygame.init()

ANCHO = 1600
ALTO = 900

VENTANA = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Smart Vacuum - Simulador")
RELOJ = pygame.time.Clock()

FILAS = 10
COLUMNAS = 10
TAM_CELDA = 50

MAPA_X = 35
MAPA_Y = 155

METROS_POR_CELDA = 0.30

FUENTE = pygame.font.SysFont("Arial", 18)
FUENTE_PEQUENA = pygame.font.SysFont("Arial", 14)
FUENTE_TITULO = pygame.font.SysFont("Arial", 26, bold=True)

FONDO = (20, 24, 32)
PANEL = (29, 35, 46)
PANEL_CLARO = (42, 50, 64)
BLANCO = (240, 243, 248)
GRIS = (155, 165, 180)
AZUL = (55, 145, 255)
AMARILLO = (255, 205, 70)
ROJO = (245, 85, 85)
VERDE = (65, 205, 135)
NEGRO = (12, 15, 20)

POSICION_INICIAL = (0, 0)
CANTIDAD_RESIDUOS = 3

# =====================================================
# ESTADO DEL MAPA
# =====================================================

aspiradora = POSICION_INICIAL

obstaculos = []
residuos = []

tipos_residuos = {}
origen_residuos = {}

obstaculos_originales = []
residuos_originales = []
tipos_residuos_originales = {}
origen_residuos_originales = {}

ruta = []

algoritmo = "BFS"
heuristica = "MANHATTAN"

limpiando = False
mostrar_comparacion = False

estado = "Coloca obstaculos y residuos en el mapa"

pasos = 0
distancia = 0.0

nodos_explorados = 0
tiempo_busqueda = 0.0

resultados_comparacion = []

# =====================================================
# CONTROLADOR
# =====================================================

# Modo simulacion: no intenta conectarse a COM5.
controlador = ControladorRobot(
    modo="simulacion",
    puerto="COM5",
    baudrate=115200
)


def enviar_comando_movimiento(origen, destino):

    df = destino[0] - origen[0]
    dc = destino[1] - origen[1]

    # Estos comandos solo se registran en modo simulacion.
    # El control fisico requiere orientacion y distancia
    # calculadas mediante encoders e IMU.

    if df == -1:
        controlador.avanzar()

    elif df == 1:
        controlador.retroceder()

    elif dc == -1:
        controlador.girar_izquierda()

    elif dc == 1:
        controlador.girar_derecha()


# =====================================================
# YOLO Y REGISTRO
# =====================================================

camara_yolo = None
imagen_camara = None
error_camara = None

registro = RegistroResiduos(
    frames_rearme=1800
)

eventos_pendientes = []


def iniciar_camara():

    global camara_yolo
    global error_camara

    try:

        ruta_modelo = (
            Path(__file__).resolve().parent.parent
            / "runs"
            / "papel_plastico_mejorado_gpu-3"
            / "weights"
            / "best.pt"
        )

        camara_yolo = CamaraYOLO(
            ruta_modelo=str(ruta_modelo),
            indice=0,
            confianza=0.35,
            tamano=640,
            device="auto"
        )

        error_camara = None

    except Exception as error:

        camara_yolo = None
        error_camara = str(error)

        print("[YOLO] Error:", error)


def actualizar_camara():

    global imagen_camara
    global error_camara

    if camara_yolo is None:
        return

    try:

        imagen_camara = camara_yolo.actualizar()

        if imagen_camara is not None:

            registro.actualizar(
                camara_yolo.detecciones
            )

    except Exception as error:

        error_camara = str(error)
        imagen_camara = None

        print("[YOLO] Error al actualizar:", error)


def cerrar_camara():

    global camara_yolo

    if camara_yolo is not None:

        try:
            camara_yolo.cerrar()

        except Exception as error:
            print("[YOLO] Error al cerrar:", error)

        camara_yolo = None



def procesar_eventos_yolo():

    global eventos_pendientes

    # Obtener eventos nuevos del registro
    nuevos_eventos = registro.obtener_eventos()

    if nuevos_eventos:
        eventos_pendientes.extend(nuevos_eventos)

    if not eventos_pendientes:
        return

    # Procesar los eventos pendientes
    eventos_procesados = []

    for evento in eventos_pendientes:

        clase = evento.get("residuo", "papel").lower()

        # Normalizar nombre de clase
        if clase in ["paper"]:
            clase = "papel"
        elif clase in ["plastic"]:
            clase = "plastico"

        posiciones_disponibles = [
            (f, c)
            for f in range(FILAS)
            for c in range(COLUMNAS)
            if (f, c) != aspiradora
            and (f, c) not in obstaculos
            and (f, c) not in residuos
        ]

        if not posiciones_disponibles:
            print("[YOLO] No hay casillas disponibles.")
            continue

        posicion = random.choice(posiciones_disponibles)

        residuos.append(posicion)
        tipos_residuos[posicion] = clase
        origen_residuos[posicion] = "YOLO"

        print(
            f"[YOLO] {clase} agregado al mapa. "
            f"Posicion provisional: {posicion}"
        )

        eventos_procesados.append(evento)

    # Eliminar solamente los eventos que sí se procesaron
    for evento in eventos_procesados:
        eventos_pendientes.remove(evento)



# =====================================================
# FUNCIONES DEL MAPA
# =====================================================

def guardar_estado_original():

    global obstaculos_originales
    global residuos_originales
    global tipos_residuos_originales
    global origen_residuos_originales

    obstaculos_originales = obstaculos.copy()
    residuos_originales = residuos.copy()

    tipos_residuos_originales = tipos_residuos.copy()
    origen_residuos_originales = origen_residuos.copy()


def restaurar_estado_original():

    global obstaculos
    global residuos
    global tipos_residuos
    global origen_residuos
    global aspiradora
    global ruta
    global limpiando
    global pasos
    global distancia
    global estado

    obstaculos = obstaculos_originales.copy()
    residuos = residuos_originales.copy()

    tipos_residuos = tipos_residuos_originales.copy()
    origen_residuos = origen_residuos_originales.copy()

    aspiradora = POSICION_INICIAL

    ruta = []
    limpiando = False

    pasos = 0
    distancia = 0.0

    estado = "Mapa restaurado"


def crear_mapa_vacio():

    global obstaculos
    global residuos
    global tipos_residuos
    global origen_residuos
    global aspiradora
    global ruta
    global limpiando
    global estado
    global POSICION_INICIAL

    obstaculos = []
    residuos = []

    tipos_residuos = {}
    origen_residuos = {}

    aspiradora = (0, 0)
    POSICION_INICIAL = aspiradora

    ruta = []
    limpiando = False

    estado = "Mapa vacio"

    guardar_estado_original()


def crear_mapa_experimental():

    global obstaculos
    global residuos
    global tipos_residuos
    global origen_residuos
    global aspiradora
    global ruta
    global limpiando

    obstaculos = []
    residuos = []

    tipos_residuos = {}
    origen_residuos = {}

    ruta = []
    limpiando = False

    aspiradora = POSICION_INICIAL

    disponibles = [
        (f, c)
        for f in range(FILAS)
        for c in range(COLUMNAS)
        if (f, c) != aspiradora
    ]

    random.shuffle(disponibles)

    cantidad_obstaculos = random.randint(12, 18)

    obstaculos = disponibles[:cantidad_obstaculos]

    alcanzables = obtener_posiciones_alcanzables(
        aspiradora,
        obstaculos
    )

    disponibles_residuos = [
        p for p in alcanzables
        if p != aspiradora
    ]

    random.shuffle(disponibles_residuos)

    residuos = disponibles_residuos[:CANTIDAD_RESIDUOS]

    for i, posicion in enumerate(residuos):

        tipos_residuos[posicion] = (
            "papel" if i % 2 == 0 else "plastico"
        )

        origen_residuos[posicion] = "manual"

    guardar_estado_original()


def posicion_desde_mouse(posicion_mouse):

    x, y = posicion_mouse

    columna = (x - MAPA_X) // TAM_CELDA
    fila = (y - MAPA_Y) // TAM_CELDA

    if (
        0 <= fila < FILAS
        and 0 <= columna < COLUMNAS
    ):

        return fila, columna

    return None


def editar_obstaculo(posicion):

    global ruta

    if posicion is None or posicion == aspiradora:
        return

    if limpiando:
        return

    if posicion in obstaculos:

        obstaculos.remove(posicion)

    else:

        if posicion in residuos:

            residuos.remove(posicion)
            tipos_residuos.pop(posicion, None)
            origen_residuos.pop(posicion, None)

        obstaculos.append(posicion)

    ruta = []


def editar_residuo(posicion):

    global ruta

    if posicion is None or posicion == aspiradora:
        return

    if limpiando:
        return

    if posicion in obstaculos:
        return

    if posicion in residuos:

        residuos.remove(posicion)

        tipos_residuos.pop(posicion, None)
        origen_residuos.pop(posicion, None)

    else:

        residuos.append(posicion)

        tipos_residuos[posicion] = "papel"
        origen_residuos[posicion] = "manual"

    ruta = []


def mover_aspiradora_manual(posicion):

    global aspiradora
    global POSICION_INICIAL
    global ruta
    global limpiando

    if posicion is None:
        return

    if limpiando:
        return

    if posicion in obstaculos or posicion in residuos:
        return

    aspiradora = posicion
    POSICION_INICIAL = posicion

    ruta = []

    guardar_estado_original()


# =====================================================
# BUSQUEDA Y HEURISTICAS
# =====================================================

def vecinos_validos(posicion, mapa_obstaculos):

    fila, columna = posicion

    movimientos = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    resultado = []

    for df, dc in movimientos:

        nf = fila + df
        nc = columna + dc

        vecino = (nf, nc)

        if not (0 <= nf < FILAS and 0 <= nc < COLUMNAS):
            continue

        if vecino in mapa_obstaculos:
            continue

        resultado.append(vecino)

    return resultado


def obtener_posiciones_alcanzables(inicio, mapa_obstaculos):

    visitados = {inicio}
    cola = deque([inicio])

    while cola:

        actual = cola.popleft()

        for vecino in vecinos_validos(
            actual,
            mapa_obstaculos
        ):

            if vecino not in visitados:

                visitados.add(vecino)
                cola.append(vecino)

    return list(visitados)


def heuristica_distancia(a, b, tipo):

    df = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])

    if tipo == "MANHATTAN":
        return df + dc

    return math.sqrt(df ** 2 + dc ** 2)


def reconstruir_ruta(padres, inicio, objetivo):

    camino = []
    actual = objetivo

    while actual != inicio:

        camino.append(actual)

        if actual not in padres:
            return []

        actual = padres[actual]

    camino.reverse()

    return camino


# =====================================================
# BFS
# =====================================================

def bfs(inicio, objetivo, mapa_obstaculos):

    cola = deque([inicio])

    padres = {}
    visitados = {inicio}

    explorados = 0

    while cola:

        actual = cola.popleft()
        explorados += 1

        if actual == objetivo:

            return (
                reconstruir_ruta(
                    padres,
                    inicio,
                    objetivo
                ),
                explorados
            )

        for vecino in vecinos_validos(
            actual,
            mapa_obstaculos
        ):

            if vecino not in visitados:

                visitados.add(vecino)
                padres[vecino] = actual

                cola.append(vecino)

    return [], explorados


# =====================================================
# UCS
# =====================================================

def ucs(inicio, objetivo, mapa_obstaculos):

    frontera = [(0, inicio)]

    padres = {}
    costos = {inicio: 0}

    explorados = 0

    while frontera:

        costo_actual, actual = heapq.heappop(frontera)

        if costo_actual > costos.get(
            actual,
            float("inf")
        ):
            continue

        explorados += 1

        if actual == objetivo:

            return (
                reconstruir_ruta(
                    padres,
                    inicio,
                    objetivo
                ),
                explorados
            )

        for vecino in vecinos_validos(
            actual,
            mapa_obstaculos
        ):

            nuevo_costo = costo_actual + 1

            if nuevo_costo < costos.get(
                vecino,
                float("inf")
            ):

                costos[vecino] = nuevo_costo
                padres[vecino] = actual

                heapq.heappush(
                    frontera,
                    (nuevo_costo, vecino)
                )

    return [], explorados


# =====================================================
# GREEDY
# =====================================================

def greedy(
    inicio,
    objetivo,
    mapa_obstaculos,
    tipo_heuristica
):

    frontera = [
        (
            heuristica_distancia(
                inicio,
                objetivo,
                tipo_heuristica
            ),
            inicio
        )
    ]

    padres = {}
    visitados = set()

    explorados = 0

    while frontera:

        _, actual = heapq.heappop(frontera)

        if actual in visitados:
            continue

        visitados.add(actual)
        explorados += 1

        if actual == objetivo:

            return (
                reconstruir_ruta(
                    padres,
                    inicio,
                    objetivo
                ),
                explorados
            )

        for vecino in vecinos_validos(
            actual,
            mapa_obstaculos
        ):

            if vecino not in visitados:

                padres[vecino] = actual

                prioridad = heuristica_distancia(
                    vecino,
                    objetivo,
                    tipo_heuristica
                )

                heapq.heappush(
                    frontera,
                    (prioridad, vecino)
                )

    return [], explorados


# =====================================================
# A*
# =====================================================

def a_estrella(
    inicio,
    objetivo,
    mapa_obstaculos,
    tipo_heuristica
):

    frontera = [
        (
            heuristica_distancia(
                inicio,
                objetivo,
                tipo_heuristica
            ),
            0,
            inicio
        )
    ]

    padres = {}
    costos = {inicio: 0}

    explorados = 0

    while frontera:

        _, costo_actual, actual = heapq.heappop(frontera)

        if costo_actual > costos.get(
            actual,
            float("inf")
        ):
            continue

        explorados += 1

        if actual == objetivo:

            return (
                reconstruir_ruta(
                    padres,
                    inicio,
                    objetivo
                ),
                explorados
            )

        for vecino in vecinos_validos(
            actual,
            mapa_obstaculos
        ):

            nuevo_costo = costo_actual + 1

            if nuevo_costo < costos.get(
                vecino,
                float("inf")
            ):

                costos[vecino] = nuevo_costo
                padres[vecino] = actual

                prioridad = (
                    nuevo_costo
                    + heuristica_distancia(
                        vecino,
                        objetivo,
                        tipo_heuristica
                    )
                )

                heapq.heappush(
                    frontera,
                    (
                        prioridad,
                        nuevo_costo,
                        vecino
                    )
                )

    return [], explorados


# =====================================================
# SELECCION DE ALGORITMO
# =====================================================

def buscar_ruta(inicio, objetivo):

    global nodos_explorados
    global tiempo_busqueda

    tiempo_inicio = time.perf_counter()

    if algoritmo == "BFS":

        camino, explorados = bfs(
            inicio,
            objetivo,
            obstaculos
        )

    elif algoritmo == "UCS":

        camino, explorados = ucs(
            inicio,
            objetivo,
            obstaculos
        )

    elif algoritmo == "GREEDY":

        camino, explorados = greedy(
            inicio,
            objetivo,
            obstaculos,
            heuristica
        )

    else:

        camino, explorados = a_estrella(
            inicio,
            objetivo,
            obstaculos,
            heuristica
        )

    tiempo_busqueda = (
        time.perf_counter() - tiempo_inicio
    )

    nodos_explorados = explorados

    return camino


# =====================================================
# PLANIFICACION DE OBJETIVOS
# =====================================================

def seleccionar_objetivo_cercano():

    if not residuos:
        return None, []

    mejor_objetivo = None
    mejor_ruta = []
    mejor_costo = float("inf")

    for objetivo in residuos:

        camino = buscar_ruta(
            aspiradora,
            objetivo
        )

        if camino and len(camino) < mejor_costo:

            mejor_costo = len(camino)
            mejor_objetivo = objetivo
            mejor_ruta = camino

    return mejor_objetivo, mejor_ruta


def iniciar_limpieza():

    global ruta
    global limpiando
    global estado
    global pasos
    global distancia

    if not residuos:

        estado = "No hay residuos para limpiar"
        return

    objetivo, camino = seleccionar_objetivo_cercano()

    if objetivo is None:

        estado = "No hay residuos alcanzables"
        return

    ruta = camino
    limpiando = True

    pasos = 0
    distancia = 0.0

    estado = f"Dirigiendose al residuo {objetivo}"


# =====================================================
# MOVIMIENTO Y RECOLECCION
# =====================================================

def finalizar_objetivo():

    global ruta
    global limpiando
    global estado

    if aspiradora in residuos:

        tipo = tipos_residuos.get(
            aspiradora,
            "desconocido"
        )

        residuos.remove(aspiradora)

        tipos_residuos.pop(aspiradora, None)
        origen_residuos.pop(aspiradora, None)

        print(f"[LIMPIEZA] Residuo recogido: {tipo}")

        controlador.activar_aspiracion()
        controlador.desactivar_aspiracion()

    ruta = []

    if residuos:

        # Replanificar despues de terminar el objetivo actual.
        iniciar_limpieza()

    else:

        limpiando = False

        controlador.detener()

        estado = "Limpieza terminada. Todos los residuos recogidos."


def actualizar_movimiento():

    global aspiradora
    global ruta
    global pasos
    global distancia

    if not limpiando:
        return

    if not ruta:

        finalizar_objetivo()
        return

    siguiente = ruta.pop(0)

    enviar_comando_movimiento(
        aspiradora,
        siguiente
    )

    aspiradora = siguiente

    pasos += 1
    distancia = pasos * METROS_POR_CELDA

    if not ruta:

        finalizar_objetivo()


# =====================================================
# COMPARACION
# =====================================================

def comparar_algoritmos():

    global resultados_comparacion
    global mostrar_comparacion

    resultados_comparacion = []

    if not residuos:
        return

    objetivo = residuos[0]

    for nombre in ["BFS", "UCS", "GREEDY", "A*"]:

        inicio_tiempo = time.perf_counter()

        if nombre == "BFS":

            camino, explorados = bfs(
                aspiradora,
                objetivo,
                obstaculos
            )

        elif nombre == "UCS":

            camino, explorados = ucs(
                aspiradora,
                objetivo,
                obstaculos
            )

        elif nombre == "GREEDY":

            camino, explorados = greedy(
                aspiradora,
                objetivo,
                obstaculos,
                heuristica
            )

        else:

            camino, explorados = a_estrella(
                aspiradora,
                objetivo,
                obstaculos,
                heuristica
            )

        duracion = (
            time.perf_counter() - inicio_tiempo
        )

        resultados_comparacion.append({
            "algoritmo": nombre,
            "nodos": explorados,
            "tiempo": duracion,
            "pasos": len(camino)
        })

    mostrar_comparacion = True


# =====================================================
# EXPERIMENTOS CON 10 MAPAS
# =====================================================

def ejecutar_experimentos():

    global aspiradora
    global obstaculos
    global residuos
    global tipos_residuos
    global origen_residuos
    global POSICION_INICIAL

    resultados = []

    obstaculos_guardados = obstaculos.copy()
    residuos_guardados = residuos.copy()

    tipos_guardados = tipos_residuos.copy()
    origen_guardado = origen_residuos.copy()

    aspiradora_guardada = aspiradora
    posicion_inicial_guardada = POSICION_INICIAL

    for numero in range(1, 11):

        crear_mapa_experimental()

        if not residuos:
            continue

        objetivo = residuos[0]

        for nombre in ["BFS", "UCS", "GREEDY", "A*"]:

            inicio_tiempo = time.perf_counter()

            if nombre == "BFS":

                camino, explorados = bfs(
                    aspiradora,
                    objetivo,
                    obstaculos
                )

            elif nombre == "UCS":

                camino, explorados = ucs(
                    aspiradora,
                    objetivo,
                    obstaculos
                )

            elif nombre == "GREEDY":

                camino, explorados = greedy(
                    aspiradora,
                    objetivo,
                    obstaculos,
                    heuristica
                )

            else:

                camino, explorados = a_estrella(
                    aspiradora,
                    objetivo,
                    obstaculos,
                    heuristica
                )

            duracion = (
                time.perf_counter() - inicio_tiempo
            )

            resultados.append({
                "mapa": numero,
                "algoritmo": nombre,
                "nodos": explorados,
                "tiempo": duracion,
                "pasos": len(camino)
            })

    obstaculos = obstaculos_guardados
    residuos = residuos_guardados

    tipos_residuos = tipos_guardados
    origen_residuos = origen_guardado

    aspiradora = aspiradora_guardada
    POSICION_INICIAL = posicion_inicial_guardada

    guardar_estado_original()

    nombre_archivo = (
        "experimentos_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )

    with open(
        nombre_archivo,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        campos = [
            "mapa",
            "algoritmo",
            "nodos",
            "tiempo",
            "pasos"
        ]

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos
        )

        escritor.writeheader()
        escritor.writerows(resultados)

    print(
        f"[EXPERIMENTOS] Resultados guardados en: "
        f"{nombre_archivo}"
    )


# =====================================================
# DIBUJAR MAPA
# =====================================================

def dibujar_mapa():

    pygame.draw.rect(
        VENTANA,
        PANEL,
        (
            MAPA_X - 10,
            MAPA_Y - 10,
            COLUMNAS * TAM_CELDA + 20,
            FILAS * TAM_CELDA + 20
        ),
        border_radius=12
    )

    for fila in range(FILAS):

        for columna in range(COLUMNAS):

            posicion = (fila, columna)

            x = MAPA_X + columna * TAM_CELDA
            y = MAPA_Y + fila * TAM_CELDA

            rect = pygame.Rect(
                x,
                y,
                TAM_CELDA,
                TAM_CELDA
            )

            color = (225, 230, 238)

            if posicion in obstaculos:

                color = NEGRO

            elif posicion in residuos:

                if origen_residuos.get(posicion) == "YOLO":
                    color = VERDE

                else:
                    color = ROJO

            elif posicion in ruta:

                color = AMARILLO

            pygame.draw.rect(
                VENTANA,
                color,
                rect
            )

            pygame.draw.rect(
                VENTANA,
                (120, 130, 145),
                rect,
                1
            )

    x = MAPA_X + aspiradora[1] * TAM_CELDA
    y = MAPA_Y + aspiradora[0] * TAM_CELDA

    centro = (
        x + TAM_CELDA // 2,
        y + TAM_CELDA // 2
    )

    pygame.draw.circle(
        VENTANA,
        AZUL,
        centro,
        TAM_CELDA // 2 - 6
    )

    pygame.draw.circle(
        VENTANA,
        BLANCO,
        centro,
        TAM_CELDA // 2 - 6,
        2
    )


# =====================================================
# PANEL DE INFORMACION
# =====================================================

def dibujar_texto(
    texto,
    x,
    y,
    color=BLANCO,
    fuente=FUENTE
):

    imagen = fuente.render(
        str(texto),
        True,
        color
    )

    VENTANA.blit(imagen, (x, y))


def dibujar_panel():

    pygame.draw.rect(
        VENTANA,
        PANEL,
        (570, 20, 1000, 850),
        border_radius=12
    )

    dibujar_texto(
        "SMART VACUUM",
        600,
        40,
        BLANCO,
        FUENTE_TITULO
    )

    dibujar_texto(
        "Mapa de 3 x 3 metros | 10 x 10 casillas",
        600,
        80,
        GRIS
    )

    dibujar_texto(
        "Cada casilla representa 30 x 30 cm",
        600,
        105,
        GRIS
    )

    dibujar_texto(
        "CONTROLES DEL MAPA",
        600,
        155,
        AMARILLO,
        FUENTE_TITULO
    )

    dibujar_texto(
        "Clic izquierdo: colocar/quitar obstaculo",
        600,
        200
    )

    dibujar_texto(
        "Clic derecho: colocar/quitar basura manual",
        600,
        230
    )

    dibujar_texto(
        "Clic central: mover aspiradora",
        600,
        260
    )

    dibujar_texto(
        "ALGORITMOS",
        600,
        320,
        AMARILLO,
        FUENTE_TITULO
    )

    dibujar_texto(
        f"Algoritmo: {algoritmo}",
        600,
        365
    )

    dibujar_texto(
        f"Heuristica: {heuristica}",
        600,
        395
    )

    dibujar_texto(
        "Tecla 1: BFS     Tecla 2: UCS",
        600,
        435
    )

    dibujar_texto(
        "Tecla 3: Greedy  Tecla 4: A*",
        600,
        465
    )

    dibujar_texto(
        "Tecla M: Manhattan   Tecla E: Euclidiana",
        600,
        495
    )

    dibujar_texto(
        "ESTADISTICAS",
        600,
        555,
        AMARILLO,
        FUENTE_TITULO
    )

    dibujar_texto(
        f"Nodos explorados: {nodos_explorados}",
        600,
        600
    )

    dibujar_texto(
        f"Tiempo de busqueda: {tiempo_busqueda:.6f} s",
        600,
        630
    )

    dibujar_texto(
        f"Pasos realizados: {pasos}",
        600,
        660
    )

    dibujar_texto(
        f"Distancia recorrida: {distancia:.2f} m",
        600,
        690
    )

    dibujar_texto(
        f"Residuos pendientes: {len(residuos)}",
        600,
        720
    )

    dibujar_texto(
        f"Obstaculos: {len(obstaculos)}",
        600,
        750
    )

    dibujar_texto(
        f"Estado: {estado}",
        600,
        795,
        VERDE
    )


# =====================================================
# PANEL DE CAMARA
# =====================================================

def dibujar_panel_camara():

    pygame.draw.rect(
        VENTANA,
        PANEL,
        (1120, 120, 420, 300),
        border_radius=10
    )

    dibujar_texto(
        "CAMARA YOLO",
        1140,
        135,
        AMARILLO,
        FUENTE_TITULO
    )

    if imagen_camara is not None:

        try:

            imagen = pygame.surfarray.make_surface(
                imagen_camara.swapaxes(0, 1)
            )

            imagen = pygame.transform.scale(
                imagen,
                (380, 220)
            )

            VENTANA.blit(
                imagen,
                (1140, 180)
            )

            if camara_yolo is not None:

                dibujar_texto(
                    f"FPS: {camara_yolo.fps:.1f} | "
                    f"Detecciones: {camara_yolo.cantidad}",
                    1140,
                    405,
                    VERDE,
                    FUENTE_PEQUENA
                )

        except Exception as error:

            dibujar_texto(
                f"Error de imagen: {str(error)[:35]}",
                1140,
                200,
                ROJO,
                FUENTE_PEQUENA
            )

    elif error_camara:

        dibujar_texto(
            "No se pudo iniciar la camara:",
            1140,
            195,
            ROJO
        )

        dibujar_texto(
            error_camara[:45],
            1140,
            225,
            ROJO,
            FUENTE_PEQUENA
        )

    else:

        dibujar_texto(
            "Esperando imagen...",
            1140,
            210,
            GRIS
        )

    dibujar_texto(
        "Verde: residuo YOLO",
        1140,
        380,
        VERDE,
        FUENTE_PEQUENA
    )


# =====================================================
# PANEL DE COMPARACION
# =====================================================

def dibujar_comparacion():

    if not mostrar_comparacion:
        return

    pygame.draw.rect(
        VENTANA,
        PANEL_CLARO,
        (1120, 450, 420, 350),
        border_radius=10
    )

    dibujar_texto(
        "COMPARACION",
        1140,
        465,
        AMARILLO,
        FUENTE_TITULO
    )

    y = 520

    for resultado in resultados_comparacion:

        texto = (
            f"{resultado['algoritmo']}: "
            f"{resultado['nodos']} nodos | "
            f"{resultado['pasos']} pasos"
        )

        dibujar_texto(
            texto,
            1140,
            y,
            BLANCO,
            FUENTE_PEQUENA
        )

        y += 30

        dibujar_texto(
            f"Tiempo: {resultado['tiempo']:.6f} s",
            1160,
            y,
            GRIS,
            FUENTE_PEQUENA
        )

        y += 35


# =====================================================
# BOTONES
# =====================================================

BOTON_INICIAR = pygame.Rect(35, 690, 150, 42)
BOTON_REINICIAR = pygame.Rect(195, 690, 150, 42)
BOTON_NUEVO_MAPA = pygame.Rect(355, 690, 150, 42)

BOTON_COMPARAR = pygame.Rect(35, 745, 220, 42)
BOTON_EXPERIMENTOS = pygame.Rect(265, 745, 240, 42)


def dibujar_boton(texto, rect, seleccionado=False):

    color = VERDE if seleccionado else PANEL_CLARO

    pygame.draw.rect(
        VENTANA,
        color,
        rect,
        border_radius=9
    )

    pygame.draw.rect(
        VENTANA,
        (90, 100, 115),
        rect,
        1,
        border_radius=9
    )

    imagen = FUENTE_PEQUENA.render(
        texto,
        True,
        BLANCO
    )

    VENTANA.blit(
        imagen,
        imagen.get_rect(center=rect.center)
    )


def dibujar_botones():

    dibujar_boton(
        "Iniciar limpieza",
        BOTON_INICIAR,
        limpiando
    )

    dibujar_boton(
        "Reiniciar",
        BOTON_REINICIAR
    )

    dibujar_boton(
        "Mapa aleatorio",
        BOTON_NUEVO_MAPA
    )

    dibujar_boton(
        "Comparar algoritmos",
        BOTON_COMPARAR
    )

    dibujar_boton(
        "Experimento 10 mapas",
        BOTON_EXPERIMENTOS
    )


# =====================================================
# LEYENDA
# =====================================================

def dibujar_leyenda():

    y = 825

    elementos = [
        (AZUL, "Aspiradora"),
        (NEGRO, "Obstaculo"),
        (ROJO, "Basura manual"),
        (VERDE, "Basura YOLO"),
        (AMARILLO, "Ruta")
    ]

    for i, (color, nombre) in enumerate(elementos):

        x = MAPA_X + i * 105

        pygame.draw.rect(
            VENTANA,
            color,
            (x, y, 18, 18)
        )

        dibujar_texto(
            nombre,
            x + 23,
            y,
            BLANCO,
            FUENTE_PEQUENA
        )


# =====================================================
# EVENTOS
# =====================================================

def manejar_eventos():

    global algoritmo
    global heuristica
    global estado
    global pasos
    global distancia
    global mostrar_comparacion

    for evento in pygame.event.get():

        if evento.type == pygame.QUIT:

            cerrar_camara()
            controlador.cerrar()

            pygame.quit()
            sys.exit()

        if evento.type == pygame.KEYDOWN:

            if evento.key == pygame.K_ESCAPE:

                cerrar_camara()
                controlador.cerrar()

                pygame.quit()
                sys.exit()

            elif evento.key == pygame.K_1:
                algoritmo = "BFS"

            elif evento.key == pygame.K_2:
                algoritmo = "UCS"

            elif evento.key == pygame.K_3:
                algoritmo = "GREEDY"

            elif evento.key == pygame.K_4:
                algoritmo = "A*"

            elif evento.key == pygame.K_m:
                heuristica = "MANHATTAN"

            elif evento.key == pygame.K_e:
                heuristica = "EUCLIDIANA"

            elif evento.key == pygame.K_SPACE:
                iniciar_limpieza()

            elif evento.key == pygame.K_r:

                restaurar_estado_original()

            elif evento.key == pygame.K_n:

                crear_mapa_experimental()
                estado = "Nuevo mapa aleatorio"

            elif evento.key == pygame.K_c:

                comparar_algoritmos()

            elif evento.key == pygame.K_s:

                guardar_estado_original()
                estado = "Mapa guardado"

            elif evento.key == pygame.K_g:

                registro.guardar_csv()

        if evento.type == pygame.MOUSEBUTTONDOWN:

            posicion = posicion_desde_mouse(evento.pos)

            if posicion is not None:

                if evento.button == 1:
                    editar_obstaculo(posicion)

                elif evento.button == 3:
                    editar_residuo(posicion)

                elif evento.button == 2:
                    mover_aspiradora_manual(posicion)

            if evento.button == 1:

                if BOTON_INICIAR.collidepoint(evento.pos):

                    iniciar_limpieza()

                elif BOTON_REINICIAR.collidepoint(evento.pos):

                    restaurar_estado_original()

                elif BOTON_NUEVO_MAPA.collidepoint(evento.pos):

                    crear_mapa_experimental()
                    estado = "Nuevo mapa aleatorio"

                elif BOTON_COMPARAR.collidepoint(evento.pos):

                    comparar_algoritmos()

                elif BOTON_EXPERIMENTOS.collidepoint(evento.pos):

                    ejecutar_experimentos()
                    estado = "Experimentos terminados y CSV guardado"


# =====================================================
# BUCLE PRINCIPAL
# =====================================================

def main():

    global estado

    iniciar_camara()

    tiempo_ultimo_movimiento = 0
    intervalo_movimiento = 0.35

    try:

        while True:

            RELOJ.tick(30)

            manejar_eventos()

            actualizar_camara()
            procesar_eventos_yolo()

            tiempo_actual = time.time()

            if (
                limpiando
                and tiempo_actual - tiempo_ultimo_movimiento
                >= intervalo_movimiento
            ):

                actualizar_movimiento()

                tiempo_ultimo_movimiento = tiempo_actual

            VENTANA.fill(FONDO)

            dibujar_mapa()
            dibujar_panel()
            dibujar_panel_camara()
            dibujar_comparacion()
            dibujar_botones()
            dibujar_leyenda()

            pygame.display.flip()

    finally:

        cerrar_camara()
        controlador.cerrar()

        pygame.quit()


if __name__ == "__main__":

    main()
