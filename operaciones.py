# src/operaciones.py
import numpy as np
from PIL import Image
from imagen import Imagen
from filtroDeslizante import filtro_deslizante
import math
from typing import Tuple, Union

class Operaciones:

    # --- KERNELS CONSTANTES (Reemplazo de la clase Kernels) ---
    PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    PREWITT_Y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    LAPLACIANO_PLUS = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) # Realce de bordes TP1
    LAPLACIANO_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # Laplaciano clásico
    LAPLACIANO_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) # Laplaciano 8-conectado

    # -------------------------------------------------------------------------
    # --- Utilidades y Funciones Comunes ---
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        """ 
        Normaliza un array flotante o entero al rango [0, 255] de forma lineal 
        (Min-Max scaling).
        """
        arr = arr.astype(float)
        amin = arr.min()
        amax = arr.max()
        if amax == amin:
            return np.zeros_like(arr, dtype=np.uint8)
        
        norm = 255.0 * (arr - amin) / (amax - amin)
        return np.rint(np.clip(norm, 0, 255)).astype(np.uint8)

    @staticmethod
    def _gaussiano_kernel(sigma: float = 1.0, k_size: int = 5) -> np.ndarray:
        """ Genera un kernel Gaussiano k_size x k_size dinámicamente. """
        ax = np.arange(-k_size // 2 + 1., k_size // 2 + 1.)
        yy, xx = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / np.sum(kernel)

    # -------------------------------------------------------------------------
    # --- TP1: BÁSICAS Y FILTROS ---
    # -------------------------------------------------------------------------

    @staticmethod
    def subtract(img1: Imagen, img2: Imagen) -> Imagen:
        a1 = img1.to_numpy().astype(int)
        a2 = img2.to_numpy().astype(int)
        if a1.shape != a2.shape:
            raise ValueError("Las imágenes deben tener el mismo tamaño")
        diff = a1 - a2
        norm = Operaciones._normalize_to_uint8(diff)
        return Imagen.from_numpy(norm)

    @staticmethod
    def negativo(img: Imagen) -> Imagen:
        arr = img.to_numpy().astype(np.uint8)
        return Imagen.from_numpy(255 - arr)
        
    @staticmethod
    def gamma(img: Imagen, y: float) -> Imagen:
        arr = img.to_numpy().astype(float) / 255.0
        out = 255.0 * (arr ** y)
        return Imagen.from_numpy(out)

    @staticmethod
    def binarizar(img: Imagen, T: int = 127) -> Imagen:
        arr = img.to_numpy().astype(np.uint8)
        return Imagen.from_numpy(np.where(arr > T, 255, 0).astype(np.uint8))
        
    @staticmethod
    def realce_bordes(img: Imagen) -> Imagen:
        """ Realce de bordes usando el Laplaciano con valor central positivo. """
        return filtro_deslizante.aplicar(img, Operaciones.LAPLACIANO_PLUS, normalize=True)

    @staticmethod
    def filtro_media(img: Imagen, k_size: int = 3) -> Imagen:
        """ Versión completa de filtro de la media (3x3 o KxK). """
        if k_size == 3:
            return filtro_deslizante.aplicar(img, modo="media")
        
        kernel = np.ones((k_size, k_size)) / (k_size * k_size)
        return filtro_deslizante.aplicar(img, kernel, normalize=True)
    
    # Alias para compatibilidad de menú
    media = filtro_media 

    @staticmethod
    def filtro_mediana(img: Imagen, k_size: int = 3) -> Imagen:
        return filtro_deslizante.aplicar_mediana(img, k_size=k_size) 
    
    @staticmethod
    def filtro_gaussiano(img: Imagen, k_size: int = 5, sigma: float = 1.0) -> Imagen:
        kernel = Operaciones._gaussiano_kernel(sigma=sigma, k_size=k_size)
        return filtro_deslizante.aplicar(img, kernel, normalize=True)
    
    # -------------------------------------------------------------------------
    # --- TP2: GRADIENTES Y UMBRALIZACIÓN AVANZADA ---
    # -------------------------------------------------------------------------

    @staticmethod
    def prewitt_x(img: Imagen) -> Imagen:
        return filtro_deslizante.aplicar(img, Operaciones.PREWITT_X, normalize=True)

    @staticmethod
    def prewitt_y(img: Imagen) -> Imagen:
        return filtro_deslizante.aplicar(img, Operaciones.PREWITT_Y, normalize=True)

    @staticmethod
    def prewitt_magnitud(img: Imagen) -> Imagen:
        # Aplicamos el filtro deslizante SIN normalización para obtener Gx y Gy en float
        gx_img = filtro_deslizante.aplicar(img, Operaciones.PREWITT_X, normalize=False)
        gy_img = filtro_deslizante.aplicar(img, Operaciones.PREWITT_Y, normalize=False)
        
        gx = gx_img.to_numpy(dtype=float)
        gy = gy_img.to_numpy(dtype=float)
        
        mag = np.sqrt(gx**2 + gy**2)
        
        # Normalizamos la magnitud resultante a 0-255
        return Imagen.from_numpy(Operaciones._normalize_to_uint8(mag))

    @staticmethod
    def laplaciano(img: Imagen) -> Imagen:
        """ Laplaciano con kernel clásico (suma 0) """
        return filtro_deslizante.aplicar(img, Operaciones.LAPLACIANO_4, normalize=True)

    @staticmethod
    def laplaciano_cruce_cero(img: Imagen, umbral_pendiente: float = 15.0) -> Imagen:
        """ Detección de bordes por cruce por cero del Laplaciano 8-conectado. """
        
        # Aplicamos el Laplaciano SIN normalización (obtenemos valores signed float)
        I_lap = filtro_deslizante.aplicar(img, Operaciones.LAPLACIANO_8, normalize=False).to_numpy(dtype=float)
        
        h, w = I_lap.shape
        out = np.zeros_like(I_lap, dtype=np.uint8)
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                ventana = I_lap[y-1:y+2, x-1:x+2]
                
                # Cruce por Cero: la ventana debe tener valores positivos y negativos
                cruce_cero = (ventana.max() > 0 and ventana.min() < 0)
                # Pendiente: la diferencia entre el máximo y el mínimo debe superar el umbral
                pendiente = ventana.max() - ventana.min() 
                
                if cruce_cero and pendiente > umbral_pendiente:
                    out[y, x] = 255 # Borde detectado
                    
        return Imagen.from_numpy(out)

    @staticmethod
    def laplaciano_gaussiano(img: Imagen, sigma: float = 1.0, k_size: int = 5) -> Imagen:
        """ Suavizado Gaussiano seguido de Laplaciano de Cruce por Cero (LoG). """
        img_smooth = Operaciones.filtro_gaussiano(img, k_size=k_size, sigma=sigma)
        # Reutilizamos la lógica del cruce por cero en la imagen suavizada
        return Operaciones.laplaciano_cruce_cero(img_smooth, umbral_pendiente=15.0) # El umbral es fijo o configurable en la interfaz

    # --- UMBRALIZACIÓN AVANZADA ---

    @staticmethod
    def umbral_iterativo(img: Imagen, T0: Union[float, None] = None, tol: float = 1.0) -> Tuple[Imagen, float]:
        """ Calcula el umbral de forma iterativa y binariza. """
        I = img.to_numpy().astype(float)
        T = T0 if T0 is not None else I.mean()

        while True:
            # Separar píxeles según el umbral actual T
            G1 = I[I >= T]; G2 = I[I < T]
            
            # Calcular las medias (m1 y m2)
            m1 = G1.mean() if G1.size > 0 else T
            m2 = G2.mean() if G2.size > 0 else T
            
            # Nuevo umbral
            T_new = 0.5 * (m1 + m2)
            
            # Criterio de parada
            if abs(T - T_new) < tol: break
            T = T_new
        
        out = (I >= T) * 255
        return Imagen.from_numpy(out.astype(np.uint8)), T

    @staticmethod
    def umbral_otsu(img: Imagen) -> Tuple[Imagen, int]:
        """ Calcula el umbral óptimo de Otsu y binariza. """
        I = img.to_numpy().astype(np.uint8)
        hist, _ = np.histogram(I.flatten(), bins=256, range=(0, 256))
        total = I.size
        sum_total = np.dot(np.arange(256), hist)
        sumB = wB = varMax = 0
        threshold = 0
        
        for i in range(256):
            wB += hist[i]
            if wB == 0: continue
            wF = total - wB
            if wF == 0: break
            sumB += i * hist[i]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            # Varianza interclase
            var = wB * wF * (mB - mF) ** 2
            
            if var > varMax:
                varMax = var
                threshold = i
                
        out = (I >= threshold) * 255
        return Imagen.from_numpy(out.astype(np.uint8)), threshold

    @staticmethod
    def segmentacion_rgb(img: Imagen, umbral_defecto: int = 100) -> Imagen:
        """ Placeholder simple para binarización de imagen a color por umbral de media. """
        print(f"Segmentación RGB (Placeholder): umbral={umbral_defecto}")
        # Convertir a grises simple y binarizar (como placeholder)
        arr = img.to_numpy(dtype=float)
        if arr.ndim == 3: # Si es color, tomamos la media
             arr = np.mean(arr, axis=2).astype(np.uint8)
        
        out = np.where(arr > umbral_defecto, 255, 0).astype(np.uint8)
        return Imagen.from_numpy(out)


    # --- DIFUSIÓN (Placeholders con firmas corregidas) ---
    
    @staticmethod
    def difusion_isotropica(img: Imagen, iterations: int = 10, lambda_param: float = 0.2) -> Imagen:
        print(f"Difusión Isotropica (Placeholder): {iterations} iteraciones, lambda={lambda_param}")
        return img

    @staticmethod
    def difusion_anisotropica(img: Imagen, iterations: int = 10, lambda_param: float = 0.25, kappa: float = 30.0, option: int = 1) -> Imagen:
        print(f"Difusión Anisotropica (Placeholder): {iterations} iteraciones, lambda={lambda_param}, kappa={kappa}, option={option}")
        return img
    
    @staticmethod
    def filtro_bilateral(img: Imagen, sigma_s: float = 5.0, sigma_r: float = 50.0, k_size: int = 5) -> Imagen:
        print(f"Filtro Bilateral (Placeholder): sigma_s={sigma_s}, sigma_r={sigma_r}, k_size={k_size}")
        return img