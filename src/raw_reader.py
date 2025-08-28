# src/raw_reader.py
import os

class RAWReader:
    @staticmethod
    def leer_raw(path, ancho, alto):
        """Lee RAW 8-bit (grayscale) y devuelve lista 2D."""
        with open(path, "rb") as f:
            datos = f.read()
        if len(datos) != ancho * alto:
            raise ValueError(f"Tamaño RAW no coincide: esperado {ancho*alto}, obtenido {len(datos)}")
        return [[datos[y*ancho + x] for x in range(ancho)] for y in range(alto)]

    @staticmethod
    def leer_pgm(path):
        """Lee PGM binario (P5) 8-bit y devuelve lista 2D."""
        with open(path, "rb") as f:
            header = f.readline().decode(errors='ignore').strip()
            if header != "P5":
                raise ValueError("Solo se soporta PGM binario (P5)")
            # saltar comentarios
            line = f.readline().decode(errors='ignore')
            while line.startswith("#"):
                line = f.readline().decode(errors='ignore')
            ancho, alto = map(int, line.split())
            maxval = int(f.readline().decode(errors='ignore').strip())
            if maxval > 255:
                raise ValueError("Solo PGM 8 bits soportado")
            data = f.read()
            if len(data) != ancho * alto:
                raise ValueError("Tamaño PGM no coincide con ancho x alto")
            return [[data[y*ancho + x] for x in range(ancho)] for y in range(alto)]

    @staticmethod
    def leer_readme(path_readme):
        """Lee README (latin-1) y devuelve dict {NOMBRE.RAW: (w,h)}"""
        info = {}
        if not os.path.exists(path_readme):
            return info
        with open(path_readme, "r", encoding="latin-1", errors="ignore") as f:
            for linea in f:
                partes = linea.split()
                # busca líneas tipo: NOMBRE   ancho   alto
                if len(partes) >= 3 and partes[0].upper().endswith(".RAW"):
                    try:
                        nombre = partes[0]
                        ancho = int(partes[1])
                        alto = int(partes[2])
                        info[nombre] = (ancho, alto)
                    except:
                        continue
        return info
