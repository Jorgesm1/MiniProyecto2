# ✋🖥️ Sistema de Conteo de Dedos con Python, OpenCV, MediaPipe y Arduino

Este proyecto implementa un **sistema de visión por computadora** para **detectar el número de dedos levantados** (de 0 a 5) usando Python y enviarlo a un **Arduino** que muestra el número en una **matriz LED 8x8 MAX7219** y enciende LEDs físicos para representar la cantidad detectada.

---

## 🎯 Funcionalidad del Proyecto

- Python detecta los dedos usando **MediaPipe** y **OpenCV** en tiempo real.
- Cada vez que cambia el número de dedos, se envía vía **puerto serial** al Arduino.
- El Arduino:
  - **Enciende LEDs** según la cantidad detectada.
  - **Muestra el número** en una **matriz LED 8x8** usando el chip **MAX7219**.
- El número y los LEDs **se mantienen activos** hasta que se detecte un nuevo cambio.

---

## 🛠️ Requerimientos

### 1. Software

- **Python 3.x**
- **Arduino IDE**

#### Librerías de Python

Instalar las siguientes librerías ejecutando:

```bash
pip install opencv-python mediapipe pyserial

opencv-python: para el procesamiento de imagen.

mediapipe: para la detección de la mano y dedos.

pyserial: para comunicación serial con Arduino.

Librerías de Arduino
Instalar la librería:

LedControl (de Eberhard Fahle)


🔌 Componentes de Hardware
1x Arduino UNO (o compatible)
1x Módulo matriz LED 8x8 con chip MAX7219
5x LEDs
5x Resistencias de 220Ω a 330Ω
Protoboard
Jumpers
Cable USB para Arduino

⚡ Conexiones Arduino
LEDS

LED	Pin Arduino
LED 1	2
LED 2	3
LED 3	4
LED 4	5
LED 5	6

Cada LED debe tener una resistencia en serie.

Matriz LED 8x8 MAX7219

Pin del módulo MAX7219	Pin del Arduino UNO
VCC	5V
GND	GND
DIN	11
CS	10
CLK	13


******** 📜 Funcionamiento del Código
Python
Captura video de la cámara web.

Detecta la cantidad de dedos levantados.

Envía el número actual vía Serial solo si cambia respecto al anterior.

El formato de envío es un número seguido de salto de línea (\n).

Arduino
Escucha constantemente por serial.

Actualiza LEDs y matriz sólo cuando llega un número nuevo.

Mantiene encendidos los LEDs y el número mostrado hasta nuevo cambio.

🚀 Ejecución del Proyecto
Sube el programa Arduino (.ino) al microcontrolador.

Ejecuta el script de Python (.py).

Coloca tu mano frente a la cámara.

Observa los LEDs y el número en la matriz actualizarse dinámicamente.#   M i n i - P r o y e c t o - 2  
 #   M i n i - P r o y e c t o - 2  
 #   M i n i - P r o y e c t o - 2  
 #   M i n i - P r o y e c t o - 2  
 #   M i n i - P r o y e c t o - 2  
 #   M i n i P r o y e c t o 2  
 #   M i n i P r o y e c t o 2  
 #   M i n i P r o y e c t o 2  
 #   M i n i P r o y e c t o 2  
 