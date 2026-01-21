# Sequencer

## _process_single_streamline
La decisión de normalizar la distancia radial es para que sea invariante al tamaño del cerebro. Por otro lado, normalizar los ángulos con seno y coseno es para que sean invariables a la orientación del cerebro. Se podría normalizar haciendo clip() pero es provocaría que 0 sea codificado como 0, y 2pi sea codificado como 1, sin embargo son el mismo ángulo.


# _utils
## radius_normalization
En radius_normalization.py se guardan los valores de r_min y r_max para cada sujeto así como su COM. De esta manera no hace falta calcularlos cada vez que se quiera generar la secuencia.