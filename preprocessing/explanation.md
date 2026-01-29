# Sequencer

## Porqué usar coordenadas polares normalizadas con el centro de masas?
* Invariante a la traslación
* Separar "qué tan lejos" de "en qué dirección". r es la distancia al centro y los ángulos es la dirección espacial. Las redes neuronales aprenden mejor factores desacoplados.
* r normalizado es invariante al tamaño del cerebro, evita gradientes enormes o diminutos, por ende, hace el modelo más estable.
* Usar seno y coseno para que el espacio angular se vuelva continuo y no haya saltos en ±π: dos direcciones parecidas=vectores parecidos. 
* [r_norm, cosθ, sinθ, cosφ, sinφ] preserva geometría, elimina ruido irrelevante y reduce complejidad de aprendizaje:
    * Importa cómo se curva la fibra
    * Importa hacia dónde apunta
    * Importa que tan lejos está
* NO ES COMPLETAMENTE INVARIANTE A LO ROTACIÓN. Si se quisiera esto: armónicos esféricos, frames locales o PCA por sujeto.

## _process_single_streamline
La decisión de normalizar la distancia radial es para que sea invariante al tamaño del cerebro. Por otro lado, normalizar los ángulos con seno y coseno es para que sean invariables a la orientación del cerebro. Se podría normalizar haciendo clip() pero es provocaría que 0 sea codificado como 0, y 2pi sea codificado como 1, sin embargo son el mismo ángulo.

## process_and_save_subject
Los guardo en un archivo .hdf5 para que sea más rápido cargarlos y además permite cargar streamlines individuales o subconjuntos de streamlines sin cargar el archivo entero en memoria. El formato de arcicho .npz no permite esto.


# _utils
## radius_normalization
En radius_normalization.py se guardan los valores de r_min y r_max para cada sujeto así como su COM. De esta manera no hace falta calcularlos cada vez que se quiera generar la secuencia.

# Conclusiones sobre coordanadas polares normalizadas (mirar sequencer_analysis.ipynb)
## Traslación
Coordenadas euclidianas cambian drásticamente con la traslación, con un cambio medio de 12.57, mientras uqe el cambio medio de las coordenadas polares normalizadas es de 4.3e-16 (ruido numérico del punto flotante): **INVARIANZA PERFECTA A LA TRASLACIÓN**
## Rotación alrededor del eje z
r y theta permanece exactamente igual, porque al rotar alrededor del eje z no se altera la distancia radial ni la dirección angular en el plano horizontal. Por otro lado, phi cambia proporcionalmente al ángulo de rotación . Esto demuestra que el encoding desacopla correctamente la magnitud de la orientación (solo se tiene en cuenta la geometría relevante).
## Perturbaciones locales / curvatura
La representación en coordenadas polares suviza las perturbaciones locales: los cambios en las features son más pequeños que en (x, y, z). Los cambios en theta y phi son proporcionalmente menores que en (x, y, z): sin/cos evita saltos angulares. **Robustez frente a jitters (ruido)**

