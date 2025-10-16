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
        """Convierte data a PIL.Image o devuelve la instancia si ya lo es."""
        if isinstance(self.data, list):
            alto = len(self.data)
            ancho = len(self.data[0]) if alto>0 else 0
            img = Image.new("L", (ancho, alto))
            for y in range(alto):
                for x in range(ancho):
                    img.putpixel((x, y), int(self.data[y][x]))
            return img
        # Si ya es una instancia de PIL.Image, devuélvela
        if isinstance(self.data, Image.Image):
             return self.data
        # Si es un numpy array, conviértelo
        if isinstance(self.data, np.ndarray):
            if self.data.ndim == 3: # Color
                return Image.fromarray(self.data.astype(np.uint8), 'RGB')
            else: # Grayscale
                return Image.fromarray(self.data.astype(np.uint8), 'L')
        return None # O manejar error

    def to_numpy(self):
        """Convierte data a numpy array. Respeta el modo si es PIL.Image."""
        if isinstance(self.data, list):
            # Para datos RAW/PGM, que son grayscale
            return np.array(self.data, dtype=np.uint8)
        else:
            # Para PIL.Image, convierte a array sin cambiar el modo (puede ser 'L' o 'RGB')
            return np.array(self.data)