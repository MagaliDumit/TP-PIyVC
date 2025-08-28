# src/imagen.py
from PIL import Image
import numpy as np
from typing import Tuple

class Imagen:
    def __init__(self, data):
        """
        data: puede ser
          - lista 2D (grises) [[...],[...]]  => internal grayscale
          - PIL.Image instance
        """
        self.data = data

    def get_size(self) -> Tuple[int,int]:
        if isinstance(self.data, list):
            alto = len(self.data)
            ancho = len(self.data[0]) if alto>0 else 0
            return (ancho, alto)
        return self.data.size  # (w,h)

    def get_pixel(self, x:int, y:int):
        if isinstance(self.data, list):
            return self.data[y][x]
        return self.data.getpixel((x,y))

    def set_pixel(self, x:int, y:int, valor):
        if isinstance(self.data, list):
            self.data[y][x] = valor
        else:
            self.data.putpixel((x,y), valor)

    def copy_region(self, box):
        """box = (x1,y1,x2,y2) en coordenadas de la imagen original"""
        x1,y1,x2,y2 = box
        if isinstance(self.data, list):
            region = [row[x1:x2] for row in self.data[y1:y2]]
            return Imagen(region)
        else:
            region = self.data.crop((x1,y1,x2,y2))
            return Imagen(region)

    def to_pil(self):
        """Convierte lista 2D a PIL.Image 'L' o devuelve la PIL si ya lo es."""
        if isinstance(self.data, list):
            alto = len(self.data)
            ancho = len(self.data[0]) if alto>0 else 0
            img = Image.new("L", (ancho, alto))
            for y in range(alto):
                for x in range(ancho):
                    img.putpixel((x,y), int(self.data[y][x]))
            return img
        return self.data

    def to_numpy(self):
        """Devuelve numpy array (H,W) o (H,W,3) dtype=uint8"""
        pil = self.to_pil()
        arr = np.array(pil)
        return arr
