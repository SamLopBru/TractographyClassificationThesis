import nibabel as nib
from dipy.io.streamline import load_trk
from dipy.viz import window, actor

tract_path = "/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives/trainset/sub-1056/tractography/sub-1056__PYT_R.trk"
# Cargar tractografía (.trk)
tractogram = load_trk(tract_path, "same")
streamlines = tractogram.streamlines

# Crear escena
scene = window.Scene()

# Crear actor de líneas
stream_actor = actor.line(streamlines)

scene.add(stream_actor)

# Mostrar ventana interactiva
window.show(scene)
