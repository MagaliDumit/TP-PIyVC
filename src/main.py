import tkinter as tk
from interfaz import ImageApp
import os

if __name__ == "__main__":
    root = tk.Tk()
    # ajustar la ruta a tu carpeta "Imágenes" desde src/
    carpeta_imagenes = "../Imágenes"
    carpeta_imagenes = os.path.abspath(carpeta_imagenes)
    app = ImageApp(root, carpeta_imagenes)
    root.mainloop()
