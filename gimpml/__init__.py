from .tools.kmeans import get_kmeans as kmeans
from .tools.deblur import get_deblur as deblur
from .tools.coloring import get_deepcolor as deepcolor
from .tools.dehaze import get_dehaze as dehaze
from .tools.denoise import get_denoise as denoise
from .tools.matting import get_matting as matting
from .tools.enlighten import get_enlighten as enlighten
from .tools.canny import get_edge as edge
from .tools.faceparse import get_face as parseface
from .tools.interpolation import get_inter as interpolateframe
from .tools.monodepth import get_mono_depth as depth
from .tools.complete_install import setup_python_weights
from .tools.semseg import get_seg as semseg
from .tools.superresolution import get_super as super
from .tools.inpainting import get_inpaint as inpaint
from .tools.detectobjects import get_detect_objects as detect_objects
from .filters import *
__version__ = "0.0.8"
