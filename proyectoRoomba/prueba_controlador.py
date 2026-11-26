
from controlador_robot import ControladorRobot

# Crear controlador en modo simulacion
robot = ControladorRobot(modo="simulacion")

# Probar movimientos
robot.avanzar()
robot.girar_izquierda()
robot.girar_derecha()
robot.retroceder()

# Probar aspiracion
robot.activar_aspiracion()
robot.desactivar_aspiracion()

# Detener robot
robot.detener()

# Mostrar informacion
robot.mostrar_estado()

# Mostrar historial
print("HISTORIAL DE COMANDOS:")

for evento in robot.obtener_historial():
    print(evento)

# Cerrar controlador
robot.cerrar()
