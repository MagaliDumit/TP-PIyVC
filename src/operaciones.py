# src/operaciones.py
import numpy as np
from PIL import Image
from imagen import Imagen

class Operaciones:
    @staticmethod
    def subtract(img1: Imagen, img2: Imagen) -> Imagen:
        """
        Resta img1 - img2 sin truncamiento: normaliza el resultado a 0..255.
        Devuelve Imagen (PIL.Image) con el resultado.
        """
        a1 = img1.to_numpy().astype(int)
        a2 = img2.to_numpy().astype(int)

        if a1.shape != a2.shape:
            raise ValueError("Las imágenes deben tener el mismo tamaño (o igual shape) para restar")

        diff = a1 - a2  # signed
        dmin = diff.min()
        dmax = diff.max()
        if dmax == dmin:
            norm = np.zeros_like(diff, dtype=np.uint8)
        else:
            norm = ((diff - dmin) * 255.0 / (dmax - dmin)).astype(np.uint8)
        return Imagen(Image.fromarray(norm))

    @staticmethod
    def modificar_pixel_copy(img: Imagen, x:int, y:int, valor) -> Imagen:
        """Devuelve una copia de la imagen con el pixel (x,y) modificado."""
        if isinstance(img.data, list):
            nueva = [row.copy() for row in img.data]
            nueva[y][x] = valor
            return Imagen(nueva)
        else:
            cp = img.data.copy()
            cp.putpixel((x,y), valor)
            return Imagen(cp)

    @staticmethod
    def negative(img1: Imagen) -> Imagen:
        """
        Convierte img1 a negativo sin truncamiento: normaliza el resultado a 0..255.
        Devuelve Imagen (PIL.Image) con el resultado.
        """
        a1 = img1.to_numpy().astype(int)
      
        norm = (255.0 -a1 ).astype(np.uint8)
        
        return Imagen(Image.fromarray(norm))

