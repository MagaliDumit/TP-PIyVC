import numpy as np
from PIL import Image, ImageDraw
from imagen import Imagen
from collections import deque
class Operaciones:
    @staticmethod
    def subtract(img1: Imagen, img2: Imagen) -> Imagen:
        a1 = img1.to_numpy().astype(int)
        a2 = img2.to_numpy().astype(int)
        if a1.shape != a2.shape:
            raise ValueError("Las imágenes deben tener el mismo tamaño para restar")
        diff = a1 - a2
        dmin, dmax = diff.min(), diff.max()
        if dmax == dmin:
            norm = np.zeros_like(diff, dtype=np.uint8)
        else:
            norm = ((diff - dmin) * 255.0 / (dmax - dmin)).astype(np.uint8)
        return Imagen(Image.fromarray(norm))

    #----------------------------- TP1 ----------------------------
    
    @staticmethod
    def gamma(img1: Imagen, y: float) -> Imagen:
        a1 = img1.to_numpy().astype(float) / 255.0
        b = np.power(a1, y)
        norm = (255 * b).astype(np.uint8)
        return Imagen(Image.fromarray(norm))

    @staticmethod
    def negative(img1: Imagen) -> Imagen:
        a1 = img1.to_numpy()
        norm = 255 - a1
        return Imagen(Image.fromarray(norm.astype(np.uint8)))

    @staticmethod
    def histograma(img1: Imagen) -> np.ndarray:
        arr = np.array(img1.to_pil().convert('L'))
        counts = np.bincount(arr.ravel(), minlength=256)
        return counts / arr.size
    
    @staticmethod
    def umbral(img1: Imagen, u: int) -> Imagen:
        arr = np.array(img1.to_pil().convert('L'))
        norm = np.where(arr >= u, 255, 0).astype(np.uint8)
        return Imagen(Image.fromarray(norm))
    
    @staticmethod
    def ecualizar_histograma(img1: Imagen) -> Imagen:
        arr = np.array(img1.to_pil().convert('L'))
        hist = np.bincount(arr.ravel(), minlength=256)
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        if cdf[-1] - cdf_min == 0: return img1
        lookup = np.round(255 * (cdf - cdf_min) / (cdf[-1] - cdf_min)).astype(np.uint8)
        result = lookup[arr]
        return Imagen(Image.fromarray(result))

    @staticmethod
    def aplicar_ruido_gaussiano(img: Imagen, mu: float, sigma: float, porcentaje: float) -> Imagen:
        arr = np.array(img.to_pil().convert('L')).astype(float)
        h, w = arr.shape
        num_pixels = int(h * w * porcentaje)
        ruido = np.random.normal(mu, sigma, num_pixels)
        coords_x = np.random.randint(0, w, num_pixels)
        coords_y = np.random.randint(0, h, num_pixels)
        arr[coords_y, coords_x] += ruido
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Imagen(Image.fromarray(arr))
    
    def _manual_convolve(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Aplica una convolución 2D manual.
        Asume padding 'edge' (replicar bordes).
        """
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        # Aplicar padding replicando el borde
        arr_padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        out = np.zeros_like(arr)
        
        # Volteamos el kernel para una convolución correcta
        kernel_flipped = np.flip(kernel, (0, 1))
        
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                # La sub-matriz a multiplicar
                sub_matrix = arr_padded[y : y + k_h, x : x + k_w]
                # Multiplicación elemento a elemento y suma
                out[y, x] = np.sum(sub_matrix * kernel_flipped)
                
        return out

    # ---------------------------------------------------------------
    # FUNCIÓN HELPER: Kernel Gaussiano
    # ---------------------------------------------------------------

    @staticmethod
    def _gauss_kernel(sigma):
        """Función helper para crear un kernel Gaussiano (sin scipy)"""
        k_size = 2 * int(3.7 * sigma - 0.5) + 1
        
        ax = np.linspace(-(k_size - 1) / 2., (k_size - 1) / 2., k_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel / np.sum(kernel)

    @staticmethod
    def aplicar_ruido_multiplicativo(img: Imagen, generador_ruido, porcentaje: float) -> Imagen:
        arr = np.array(img.to_pil().convert('L')).astype(float)
        h, w = arr.shape
        num_pixels = int(h * w * porcentaje)
        ruido = generador_ruido(num_pixels)
        coords_x = np.random.randint(0, w, num_pixels)
        coords_y = np.random.randint(0, h, num_pixels)
        arr[coords_y, coords_x] *= ruido
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Imagen(Image.fromarray(arr))

    @staticmethod
    def aplicar_ruido_rayleigh(img: Imagen, xi: float, porcentaje: float) -> Imagen:
        return Operaciones.aplicar_ruido_multiplicativo(img, lambda n: np.random.rayleigh(xi, n), porcentaje)

    @staticmethod
    def aplicar_ruido_exponencial(img: Imagen, lambd: float, porcentaje: float) -> Imagen:
        return Operaciones.aplicar_ruido_multiplicativo(img, lambda n: np.random.exponential(1/lambd, n), porcentaje)

    @staticmethod
    def aplicar_ruido_sal_y_pimienta(img: Imagen, density: float) -> Imagen:
        arr = np.array(img.to_pil().convert('L'))
        h, w = arr.shape
        num_total = int(h * w * density)
        num_salt = num_total // 2
        
        coords_x_s = np.random.randint(0, w, num_salt)
        coords_y_s = np.random.randint(0, h, num_salt)
        arr[coords_y_s, coords_x_s] = 255
        
        coords_x_p = np.random.randint(0, w, num_total - num_salt)
        coords_y_p = np.random.randint(0, h, num_total - num_salt)
        arr[coords_y_p, coords_x_p] = 0
        return Imagen(Image.fromarray(arr))

    @staticmethod
    def aplicar_filtro_deslizante(img: Imagen, kernel: np.ndarray, normalizar=True):
        arr = np.array(img.to_pil().convert('L')).astype(float)
        h, w = arr.shape
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        
        padded_arr = np.pad(arr, ((ph, ph), (pw, pw)), mode='edge')
        output = np.zeros_like(arr, dtype=float)
        
        for y in range(h):
            for x in range(w):
                output[y, x] = np.sum(padded_arr[y:y+kh, x:x+kw] * kernel)
        
        if normalizar:
            out_min, out_max = output.min(), output.max()
            if out_max - out_min > 0:
                output = 255 * (output - out_min) / (out_max - out_min)
            return Imagen(Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)))
        else:
            return output

    @staticmethod
    def filtro_media(img: Imagen, k_size: int = 3) -> Imagen:
        kernel = np.ones((k_size, k_size)) / (k_size * k_size)
        return Operaciones.aplicar_filtro_deslizante(img, kernel)

    @staticmethod
    def filtro_mediana(img: Imagen, k_size: int = 3) -> Imagen:
        arr = np.array(img.to_pil().convert('L'))
        h, w = arr.shape
        ph = k_size // 2
        padded_arr = np.pad(arr, ((ph, ph), (ph, ph)), mode='edge')
        output = np.zeros_like(arr)
        for y in range(h):
            for x in range(w):
                output[y, x] = np.median(padded_arr[y:y+k_size, x:x+k_size])
        return Imagen(Image.fromarray(output.astype(np.uint8)))

    @staticmethod
    def filtro_gaussiano(img: Imagen, k_size: int = 3, sigma: float = 1.0) -> Imagen:
        k_size = 2 * int(3.7 * sigma - 0.5) + 1

        ax = np.linspace(-(k_size - 1) / 2., (k_size - 1) / 2., k_size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        kernel /= np.sum(kernel)
        return Operaciones.aplicar_filtro_deslizante(img, kernel)

#------------ TP 2 ---------------
    
    @staticmethod
    def detector_prewitt(img: Imagen) -> Imagen:
        arr = np.array(img.to_pil().convert('L')).astype(float)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        gx = Operaciones.aplicar_filtro_deslizante(Imagen(arr), kernel_x, normalizar=False)
        gy = Operaciones.aplicar_filtro_deslizante(Imagen(arr), kernel_y, normalizar=False)
        magnitud = np.sqrt(gx**2 + gy**2)
        magnitud = (255.0 * magnitud / magnitud.max()).astype(np.uint8)
        #umbral = np.mean(magnitud) * 1.2
        #bordes = np.where(magnitud > umbral, 255, 0).astype(np.uint8)
        return Imagen(Image.fromarray(magnitud))

    @staticmethod
    def detector_sobel(img: Imagen) -> Imagen:
        arr = np.array(img.to_pil().convert('L')).astype(float)
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = Operaciones.aplicar_filtro_deslizante(Imagen(arr), kernel_x, normalizar=False)
        gy = Operaciones.aplicar_filtro_deslizante(Imagen(arr), kernel_y, normalizar=False)
        magnitud = np.sqrt(gx**2 + gy**2)
        magnitud = (255.0 * magnitud / magnitud.max()).astype(np.uint8)
        #umbral = np.mean(magnitud) * 1.2
        #bordes = np.where(magnitud > umbral, 255, 0).astype(np.uint8)
        return Imagen(Image.fromarray(magnitud))

    @staticmethod
    def _zero_crossing_detector(image_array, threshold):
        """
        Función helper para detectar cruces por cero de forma robusta.
        Busca cambios de signo entre pixeles vecinos.
        """
        bordes = np.zeros(image_array.shape, dtype=np.uint8)
        # Umbral para la magnitud del cambio (evita marcar ruido)
        #threshold = np.max(np.abs(image_array)) * 0.10

        # Iterar sobre la imagen (excepto los bordes)
        for y in range(1, image_array.shape[0] - 1):
            for x in range(1, image_array.shape[1] - 1):
                # Chequeo Horizontal
                if (image_array[y, x-1] * image_array[y, x+1] < 0) and \
                   (abs(image_array[y, x-1]) + abs(image_array[y, x+1]) > threshold):
                    bordes[y, x] = 255
                    continue
                # Chequeo Vertical
                if (image_array[y-1, x] * image_array[y+1, x] < 0) and \
                   (abs(image_array[y-1, x]) + abs(image_array[y+1, x]) > threshold):
                    bordes[y, x] = 255
                    continue
                # Chequeo Diagonal /
                if (image_array[y-1, x+1] * image_array[y+1, x-1] < 0) and \
                   (abs(image_array[y-1, x+1]) + abs(image_array[y+1, x-1]) > threshold):
                    bordes[y, x] = 255
                    continue
                # Chequeo Diagonal \
                if (image_array[y-1, x-1] * image_array[y+1, x+1] < 0) and \
                   (abs(image_array[y-1, x-1]) + abs(image_array[y+1, x+1]) > threshold):
                    bordes[y, x] = 255
        return bordes
    

    @staticmethod
    def detector_laplaciano(img: Imagen, threshold: float) -> Imagen:
        """
        Detecta bordes usando el operador Laplaciano y cruces por cero.
        Permite definir manualmente el umbral de detección.
        """
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        # Aplicar el filtro Laplaciano sin normalizar
        arr_laplace = Operaciones.aplicar_filtro_deslizante(img, kernel, normalizar=False)
        # Detectar los cruces por cero con el umbral definido por el usuario
        bordes = Operaciones._zero_crossing_detector(arr_laplace, threshold)
        return Imagen(Image.fromarray(bordes))

    @staticmethod
    def detector_laplaciano_pendiente(img: 'Imagen', threshold: float) -> 'Imagen':
        """
        Detector de bordes Laplaciano con evaluación de la pendiente.
        Basado en: ΔI(x,y) = 4I(x,y) - I(x-1,y) - I(x+1,y) - I(x,y-1) - I(x,y+1)
        y marca un borde si hay cambio de signo y |a + b| > threshold.
        """
        arr = img.to_numpy().astype(float)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)
        
        h, w = arr.shape
        lap = np.zeros_like(arr, dtype=float)

        # Aplicar la fórmula discreta del Laplaciano
        lap[1:-1, 1:-1] = (
            4 * arr[1:-1, 1:-1]
            - arr[1:-1, 0:-2]
            - arr[1:-1, 2:]
            - arr[0:-2, 1:-1]
            - arr[2:, 1:-1]
        )

        # Detectar cruces por cero con evaluación de pendiente
        bordes = np.zeros_like(lap, dtype=np.uint8)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # Revisar pares de vecinos
                vecinos = [
                    (lap[y, x - 1], lap[y, x + 1]),   # horizontal
                    (lap[y - 1, x], lap[y + 1, x]),   # vertical
                    (lap[y - 1, x - 1], lap[y + 1, x + 1]), # diagonal \
                    (lap[y - 1, x + 1], lap[y + 1, x - 1])  # diagonal /
                ]

                for a, b in vecinos:
                    if a * b < 0:  # cambio de signo
                        pendiente = abs(a + b)
                        if pendiente > threshold:
                            bordes[y, x] = 255
                            break

        return Imagen(Image.fromarray(bordes))


    @staticmethod
    def detector_log(img: 'Imagen', sigma: float, threshold: float) -> 'Imagen':
        """
        Implementación del detector de bordes Laplaciano del Gaussiano (LoG)
        usando la fórmula exacta de Marr & Hildreth (1988).

        ΔGσ(x,y) = (1 / (2πσ³)) * e^(-(x² + y²)/(2σ²)) * ((x² + y²)/σ² - 2)
        """
        # 1. Calcular el tamaño del kernel (mínimo n = 4σ + 1)
        k_size = int(2 * sigma + 1)
        if k_size % 2 == 0:
            k_size += 1
        r = k_size // 2

        # 2. Generar las coordenadas centradas
        x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        rsq = x**2 + y**2

        # 3. Construir la máscara LoG según la fórmula
        log_kernel = (1 / (2 * np.pi * (sigma ** 3))) * \
                     np.exp(-(rsq) / (2 * sigma ** 2)) * \
                     ((rsq / (sigma ** 2)) - 2)

        # 4. Aplicar la convolución directamente con esta máscara
        arr_log = Operaciones.aplicar_filtro_deslizante(img, log_kernel, normalizar=False)

        # 5. Detectar los cruces por cero usando el umbral ingresado por el usuario
        bordes = Operaciones._zero_crossing_detector(arr_log, threshold)

        # 6. Devolver la imagen de bordes
        return Imagen(Image.fromarray(bordes))



    @staticmethod
    def detector_log(img: 'Imagen', sigma: float, threshold: float) -> 'Imagen':
        """
        Implementación del LoG según Marr-Hildreth:
        ΔGσ(x,y) = (1 / (2πσ³)) * e^{-(x²+y²)/(2σ²)} * ((x²+y²)/σ² - 2)
        Tamaño mínimo de máscara n = 4*sigma + 1
        """
        # 1) tamaño del kernel (mínimo n = 4*sigma + 1)
        k_size = int(4 * sigma + 1)
        if k_size % 2 == 0:
            k_size += 1
        r = k_size // 2

        # 2) coordenadas centradas
        x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        rsq = x**2 + y**2

        # 3) máscara LoG (fórmula exacta)
        log_kernel = (1.0 / (2.0 * np.pi * (sigma ** 3))) * \
                     np.exp(-rsq / (2.0 * sigma ** 2)) * \
                     ((rsq / (sigma ** 2)) - 2.0)

        # 4) aplicar convolución (sin normalizar para mantener signos)
        arr_log = Operaciones.aplicar_filtro_deslizante(img, log_kernel, normalizar=False)

        # 5) detectar cruces por cero usando el umbral (asegurarse que threshold sea float)
        bordes = Operaciones._zero_crossing_detector(arr_log, float(threshold))

        return Imagen(Image.fromarray(bordes))

    @staticmethod
    def difusion_isotropica(img: Imagen, iteraciones: int) -> Imagen:
        sigma = np.sqrt(2 * iteraciones)
        # Usar un tamaño de kernel razonable para el sigma calculado
        k_size = int(2 * np.ceil(3 * sigma) + 1)
        return Operaciones.filtro_gaussiano(img, k_size=k_size, sigma=sigma)

    @staticmethod
    def difusion_anisotropica(img: Imagen, iteraciones: int, k: float) -> Imagen:
        arr = np.array(img.to_pil().convert('L')).astype(np.float32)
        for _ in range(iteraciones):
            grad_n = np.roll(arr, -1, axis=0) - arr
            grad_s = np.roll(arr, 1, axis=0) - arr
            grad_e = np.roll(arr, -1, axis=1) - arr
            grad_w = np.roll(arr, 1, axis=1) - arr
            cn, cs = np.exp(-(grad_n / k)**2), np.exp(-(grad_s / k)**2)
            ce, cw = np.exp(-(grad_e / k)**2), np.exp(-(grad_w / k)**2)
            arr += 0.25 * (cn*grad_n + cs*grad_s + ce*grad_e + cw*grad_w)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Imagen(Image.fromarray(arr))

    @staticmethod
    def filtro_bilateral(img: Imagen, k_size: int, sigma_espacial: float, sigma_rango: float) -> Imagen:
        arr = np.array(img.to_pil().convert('L')).astype(float)
        h, w = arr.shape
        ph = k_size // 2
        padded_arr = np.pad(arr, ((ph, ph), (ph, ph)), mode='edge')
        output = np.zeros_like(arr)
        
        ax = np.arange(-ph, ph + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel_espacial = np.exp(-(xx**2 + yy**2) / (2 * sigma_espacial**2))
        
        for y in range(h):
            for x in range(w):
                window = padded_arr[y:y+k_size, x:x+k_size]
                kernel_rango = np.exp(-((window - arr[y,x])**2) / (2 * sigma_rango**2))
                kernel_bilateral = kernel_espacial * kernel_rango
                output[y,x] = np.sum(kernel_bilateral * window) / np.sum(kernel_bilateral)
        
        return Imagen(Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)))

    @staticmethod
    def umbral_iterativo(img: Imagen) -> Imagen:
        arr = np.array(img.to_pil().convert('L'))
        T, T_prev = arr.mean(), 0
        while abs(T - T_prev) > 0.5:
            T_prev = T
            g1 = arr[arr >= T]; g2 = arr[arr < T]
            mu1 = g1.mean() if len(g1) > 0 else 0
            mu2 = g2.mean() if len(g2) > 0 else 0
            T = (mu1 + mu2) / 2
        return Operaciones.umbral(img, int(T))

    @staticmethod
    def umbral_otsu(img: Imagen) -> Imagen:
        arr = np.array(img.to_pil().convert('L'))
        hist = np.bincount(arr.ravel(), minlength=256)
        p = hist / arr.size
        max_var, mejor_umbral = 0, 0
        for t in range(1, 256):
            q1, q2 = np.sum(p[:t]), np.sum(p[t:])
            if q1 == 0 or q2 == 0: continue
            mu1 = np.sum(np.arange(t) * p[:t]) / q1
            mu2 = np.sum(np.arange(t, 256) * p[t:]) / q2
            var_entre = q1 * q2 * (mu1 - mu2)**2
            if var_entre > max_var:
                max_var, mejor_umbral = var_entre, t
        return Operaciones.umbral(img, mejor_umbral)

    @staticmethod
    def _get_otsu_threshold(channel: np.ndarray) -> int:
        """Calcula el umbral de Otsu para un único canal."""
        hist = np.bincount(channel.ravel(), minlength=256)
        if np.sum(hist) == 0: return 0
        p = hist / np.sum(hist)
        max_var, mejor_umbral = 0, 0
        for t in range(1, 256):
            q1, q2 = np.sum(p[:t]), np.sum(p[t:])
            if q1 == 0 or q2 == 0: continue
            mu1 = np.sum(np.arange(t) * p[:t]) / q1
            mu2 = np.sum(np.arange(t, 256) * p[t:]) / q2
            var_entre = q1 * q2 * (mu1 - mu2)**2
            if var_entre > max_var:
                max_var, mejor_umbral = var_entre, t
        return mejor_umbral

    @staticmethod
    def umbralizacion_por_bandas_rgb(img: Imagen) -> Imagen:
        """
        Calcula el umbral de Otsu para cada banda RGB y aplica la umbralización.
        Devuelve una imagen RGB con la combinación de las 3 umbralizaciones.
        """
        arr = img.to_numpy()
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("La imagen debe ser RGB.")
        
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        
        r_thresh = Operaciones._get_otsu_threshold(r)
        g_thresh = Operaciones._get_otsu_threshold(g)
        b_thresh = Operaciones._get_otsu_threshold(b)
        
        r_bin = np.where(r >= r_thresh, 255, 0).astype(np.uint8)
        g_bin = np.where(g >= g_thresh, 255, 0).astype(np.uint8)
        b_bin = np.where(b >= b_thresh, 255, 0).astype(np.uint8)
        
        output_arr = np.stack((r_bin, g_bin, b_bin), axis=-1)
        return Imagen(Image.fromarray(output_arr, 'RGB'))
    
    def _mediana_manual(arr):
        h, w = arr.shape
        out = arr.copy()
        for y in range(1, h-1):
            for x in range(1, w-1):
                vec = arr[y-1:y+2, x-1:x+2].flatten()
                out[y, x] = np.median(vec)
        return out

#------------ TP 3 --------------

    @staticmethod
    def detector_canny(img: Imagen, sigma=1.4, t1=20, t2=50):
        """
        Implementa Canny (sin scipy) con la ETAPA 4 MEJORADA (alineación de gradiente).
        """
        img_gris = Imagen(img.to_pil().convert("L"))
        arr_orig = img_gris.to_numpy().astype(np.float64)
        h, w = arr_orig.shape
        
        # --- ETAPA 1: Suavizamiento (Filtro Gaussiano) ---
        kernel_gauss = Operaciones._gauss_kernel(sigma)
        arr_suav = Operaciones._manual_convolve(arr_orig, kernel_gauss)

        # --- ETAPA 2: Cálculo de magnitud y dirección (Sobel) ---
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        
        Ix = Operaciones._manual_convolve(arr_suav, Kx)
        Iy = Operaciones._manual_convolve(arr_suav, Ky)
        
        mag = np.hypot(Ix, Iy)
        mag_max = np.max(mag)
        if mag_max > 0:
            mag = (mag / mag_max) * 255 # Normalizar
        
        theta = np.arctan2(Iy, Ix) # en radianes
        angle = np.rad2deg(theta) # Ángulo en grados para la ETAPA 4
        angle[angle < 0] += 180   # Mapear a [0, 180] para ETAPA 3

        # --- ETAPA 3: Supresión de no máximos (NMS) ---
        # (Esta etapa crea la variable 'nms' que usaremos)
        nms = np.zeros((h, w), dtype=np.float64)
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                q = 255
                r = 255
                ang_q = angle[y, x] # ángulo cuantizado

                if (0 <= ang_q < 22.5) or (157.5 <= ang_q <= 180): # 0°
                    q = mag[y, x + 1]
                    r = mag[y, x - 1]
                elif (22.5 <= ang_q < 67.5): # 45°
                    q = mag[y - 1, x + 1]
                    r = mag[y + 1, x - 1]
                elif (67.5 <= ang_q < 112.5): # 90°
                    q = mag[y - 1, x]
                    r = mag[y + 1, x]
                elif (112.5 <= ang_q < 157.5): # 135°
                    q = mag[y - 1, x - 1]
                    r = mag[y + 1, x + 1]

                if (mag[y, x] >= q) and (mag[y, x] >= r):
                    nms[y, x] = mag[y, x]
        
        # --- ETAPA 4: Umbralización con histéresis MEJORADA (y CORREGIDA) ---

        res = np.zeros((h, w), dtype=np.uint8)

        # 2) Identificar bordes fuertes (t > t2)
        #    CORRECCIÓN: Usar 'nms', NO 'mag'
        strong = nms >= t2
        res[strong] = 255

        # 3) Identificar candidatos débiles (t1 <= t <= t2)
        #    CORRECCIÓN: Usar 'nms', NO 'mag'
        weak = (nms >= t1) & (nms < t2)

        # 4) BFS (deque) para conectar débiles → fuertes **si están alineados**
        strong_y, strong_x = np.where(strong)
        queue = deque(list(zip(strong_y, strong_x)))

        # Vecindad 8
        vecinos = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        
        # Volver a mapear los ángulos de ETAPA 2 a [-180, 180] para 
        # que la resta 'abs(ang - angle[y, x])' funcione correctamente
        # (ej. 179° y -179° son similares, la resta debe ser 2°, no 358°)
        angle_check = np.rad2deg(theta) # Usamos el 'theta' original de arctan2

        while queue:
            y, x = queue.popleft()

            for dy, dx in vecinos:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    # Solo agregar píxeles débiles ('weak') conectados
                    if weak[yy, xx]:
                        
                        # --- Verificación de alineación ---
                        ang_fuerte = angle_check[y, x]
                        ang_debil = angle_check[yy, xx]
                        
                        # Comparamos la diferencia angular
                        diff = abs(ang_fuerte - ang_debil)
                        
                        # El 'wrap-around' (ej. 170° y -170° son cercanos)
                        if diff > 180:
                            diff = 360 - diff 

                        # permitimos variación de ±20°
                        if diff < 20:
                            res[yy, xx] = 255
                            weak[yy, xx] = False # Marcar como ya procesado
                            queue.append((yy, xx))

        return Imagen(Image.fromarray(res))




    @staticmethod
    def canny_exacto(img):
        import numpy as np
        arr = np.array(img.to_pil().convert("L"), dtype=float)

        # Derivadas exactas (sin suavizar)
        # dx[y,x] = I[y,x+1] - I[y,x]
        dx = np.zeros_like(arr)
        dx[:, :-1] = arr[:, 1:] - arr[:, :-1]

        dy = np.zeros_like(arr)
        dy[:-1, :] = arr[1:, :] - arr[:-1, :]

        # Magnitud exacta
        M = np.sqrt(dx**2 + dy**2)

        # Como la imagen es binaria, borde real = cambio fuerte = M > 0
        edge = (M > 0).astype(np.uint8) * 255

        return Imagen(Image.fromarray(edge))


    @staticmethod
    def _susan_core(arr, t=6):
        h, w = arr.shape
        R = 3
        mask = [(dy, dx) for dy in range(-R, R+1) for dx in range(-R, R+1)
                if dx*dx+dy*dy <= R*R]

        n = np.zeros_like(arr, dtype=float)

        for y in range(R, h-R):
            for x in range(R, w-R):
                centro = arr[y, x]
                cnt = 0
                for dy, dx in mask:
                    if abs(arr[y+dy, x+dx] - centro) < t:
                        cnt += 1
                n[y, x] = 1 - cnt/len(mask)

        return n

    #@staticmethod
    #def susan_bordes(img: Imagen, t=15, umbral=0.45):
    #    arr = np.array(img.to_pil().convert("L"), dtype=float)
    #    s = Operaciones._susan_core(arr, t)
    #    out = np.where(abs(s - 0.5) < umbral, 255, 0).astype(np.uint8)
    #    return Imagen(Image.fromarray(out))

    @staticmethod
    def susan_bordes(img: Imagen, umbral, grosor=1):
        import numpy as np
        tam_min=2
        I = np.array(img.to_pil().convert("L"), dtype=np.int32)
        I = ((I - I.min()) / (I.max() - I.min()) * 255).astype(np.int32)
        I = Operaciones._mediana_manual(I)

        h, w = I.shape

        mask = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if dy*dy + dx*dx <= 9]
        area_mask = len(mask)

        respuesta = np.zeros_like(I)

        for y in range(3, h-3):
            for x in range(3, w-3):
                p = I[y, x]
                N = 0
                for dy, dx in mask:
                    if abs(I[y+dy, x+dx] - p) < umbral:
                        N += 1
                respuesta[y, x] = N

        bordes = respuesta < (0.90 * area_mask)

        clean = bordes.copy()
        for y in range(1, h-1):
            for x in range(1, w-1):
                if clean[y, x] and np.sum(clean[y-1:y+2, x-1:x+2]) < tam_min:
                    clean[y, x] = False

        pil = img.to_pil().convert("RGB")

        # DIBUJAR CON GROSOR
        for (y, x) in zip(*np.where(clean)):
            for dy in range(-grosor, grosor+1):
                for dx in range(-grosor, grosor+1):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w:
                        pil.putpixel((xx, yy), (0, 255, 255))  # CYAN

        return Imagen(pil)



    #@staticmethod
    #def susan_esquinas(img: Imagen, t=15, umbral=0.20):
    #    arr = np.array(img.to_pil().convert("L"), dtype=float)
    #    s = Operaciones._susan_core(arr, t)
    #    out = np.where(s > (0.75 - umbral), 255, 0).astype(np.uint8)
    #    return Imagen(Image.fromarray(out))

    @staticmethod
    def susan_esquinas(img: Imagen, umbral, grosor=1):

        I = np.array(img.to_pil().convert("L"), dtype=np.int32)
        I = ((I - I.min()) / (I.max() - I.min()) * 255).astype(np.int32)
        I = Operaciones._mediana_manual(I)

        h, w = I.shape

        mask = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if dy*dy + dx*dx <= 9]
        area_mask = len(mask)

        respuesta = np.zeros_like(I)

        for y in range(3, h-3):
            for x in range(3, w-3):
                p = I[y, x]
                N = 0
                for dy, dx in mask:
                    if abs(I[y+dy, x+dx] - p) < umbral:
                        N += 1
                respuesta[y, x] = N

        # Más sensible a esquinas → aumentamos el umbral de selección
        esquinas = respuesta < (0.55 * area_mask)

        # Dibujar como CRUZ para que se vea bien
        pil = img.to_pil().convert("RGB")

        for (y, x) in zip(*np.where(esquinas)):
            for d in range(-grosor, grosor+1):
                if 0 <= y+d < h:
                    pil.putpixel((x, y+d), (0, 255, 255))
                if 0 <= x+d < w:
                    pil.putpixel((x+d, y), (0, 255, 255))

        return Imagen(pil)



    @staticmethod
    def hough_rectas(img_bordes: Imagen):
        """
        Calcula el acumulador de Hough.
        Recibe una IMAGEN DE BORDES (binaria).
        Devuelve (Acumulador, rhos, thetas)
        """
        bw = np.array(img_bordes.to_pil().convert("L"))
        # Asegurarse que los bordes (blanco) son 255 y el fondo 0
        bw = np.where(bw > 128, 255, 0) 

        y, x = np.nonzero(bw)
        h, w = bw.shape
        diag = int(np.hypot(h, w))

        # Rango de Thetas (-90 a 89)
        thetas = np.deg2rad(np.arange(-90, 90))
        # Rango de Rhos (-diag a diag)
        rhos = np.arange(-diag, diag + 1) # +1 para incluir 'diag'

        # Inicializar acumulador
        A = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

        # Pre-calcular cos y sin para eficiencia
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        # Votar en el acumulador
        for (yy, xx) in zip(y, x):
            for i in range(len(thetas)):
                # Calcular rho
                rho_val = xx * cos_thetas[i] + yy * sin_thetas[i]
                
                # Encontrar el índice más cercano en 'rhos'
                # (rho_val + diag) mapea [-diag, diag] a [0, 2*diag]
                rho_idx = int(round(rho_val)) + diag
                
                # Asegurar que el índice esté dentro de los límites de 'rhos'
                if 0 <= rho_idx < len(rhos):
                    A[rho_idx, i] += 1

        return A, rhos, thetas

    @staticmethod
    def hough_dibujar_rectas(img_original: Imagen, img_bordes: Imagen, umbral: int):
        """
        Dibuja las líneas detectadas por Hough sobre la imagen original.
        """
        # 1. Calcular el acumulador de Hough usando la imagen de bordES
        A, rhos, thetas = Operaciones.hough_rectas(img_bordes)

        # 2. Encontrar los picos en el acumulador (votos > umbral)
        #    np.nonzero devuelve (array_indices_y, array_indices_x)
        indices_y, indices_x = np.nonzero(A > umbral)

        # 3. Preparar la imagen original para dibujar
        #    Convertir a RGB para dibujar líneas de color (ej. rojo)
        out_pil = img_original.to_pil().convert("RGB")
        draw = ImageDraw.Draw(out_pil)
        w, h = out_pil.size

        # 4. Iterar sobre los picos y dibujar cada línea
        for i in range(len(indices_y)):
            # Obtener el rho y theta de los índices
            rho_idx = indices_y[i]
            theta_idx = indices_x[i]
            
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]
            
            # Convertir la línea (rho, theta) a formato (x1,y1), (x2,y2)
            a = np.cos(theta)
            b = np.sin(theta)
            
            # (x0, y0) es un punto en la línea
            x0 = a * rho
            y0 = b * rho

            # Calcular dos puntos (x1,y1) y (x2,y2) fuera de la imagen
            # para asegurarse de que la línea la cruce de lado a lado
            x1 = int(x0 + 1500 * (-b)) # 1500 es un nro grande
            y1 = int(y0 + 1500 * (a))
            x2 = int(x0 - 1500 * (-b))
            y2 = int(y0 - 1500 * (a))

            # Dibujar la línea
            draw.line(((x1, y1), (x2, y2)), fill="red", width=2)
        
        # 5. Devolver la imagen con las líneas dibujadas
        return Imagen(out_pil)
    

    @staticmethod
    def segmentacion_intercambio(img: Imagen, rect, max_iters=200):
        """
        Implementación del modelo de Intercambio de Píxeles (Shi & Karl).
        - img: Imagen original (Imagen)
        - rect: (x1, y1, x2, y2) rectángulo inicial (coordenadas en la imagen original)
        - max_iters: número máximo de iteraciones
        Devuelve: Imagen RGB original con L_in (rosa) y L_out (celeste) dibujadas.
        """
        import numpy as np
        from PIL import Image

        if rect is None:
            raise ValueError("Se requiere rect (x1,y1,x2,y2) como región inicial.")

        x1, y1, x2, y2 = rect
        # Normalizar caja
        x1, x2 = max(0, min(x1, x2)), max(0, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), max(0, max(y1, y2))

        I_rgb = np.array(img.to_pil().convert("RGB"))
        I = np.array(img.to_pil().convert("L"), dtype=float)  # trabajo en gris para Fd

        h, w = I.shape

        # --- Inicializar Phi: fondo = 3, objeto interior = -3 ---
        Phi = np.full((h, w), 3, dtype=int)
        Phi[y1:y2, x1:x2] = -3

        # Calcular theta0 (fondo) y theta1 (objeto) como medias iniciales
        # para estabilidad ignoramos la región inicial al estimar el fondo -> muestreo de borde
        fondo_mask = (Phi == 3)
        objeto_mask = (Phi == -3)
        if np.sum(fondo_mask) == 0 or np.sum(objeto_mask) == 0:
            raise ValueError("Región inicial inválida: objeto o fondo vacío.")
        theta0 = np.mean(I[fondo_mask])
        theta1 = np.mean(I[objeto_mask])

        eps = 1e-9
        def Fd_val(px):
            return np.log((abs(theta0 - px) + eps) / (abs(theta1 - px) + eps))

        # vecinos 4-conectados
        neigh = [(1,0),(-1,0),(0,1),(0,-1)]

        # Helper: construir L_in y L_out (contornos de 1-pixel)
        def compute_contours(P):
            L_in_set = set()
            L_out_set = set()
            for y in range(1, h-1):
                for x in range(1, w-1):
                    if P[y,x] < 0:
                        # si existe vecino positivo -> L_in
                        for dy,dx in neigh:
                            if P[y+dy, x+dx] > 0:
                                L_in_set.add((y,x)); break
                    elif P[y,x] > 0:
                        for dy,dx in neigh:
                            if P[y+dy, x+dx] < 0:
                                L_out_set.add((y,x)); break
            return L_in_set, L_out_set

        L_in, L_out = compute_contours(Phi)

        it = 0
        moved_any = True

        while it < max_iters and moved_any:
            it += 1
            moved_any = False

            # --- Paso 1: L_out -> posible convertirse en interior (avanza hacia objeto) ---
            to_in = []
            for (y,x) in list(L_out):
                fd = Fd_val(I[y,x])
                if fd > 0:
                    to_in.append((y,x))
            if to_in:
                moved_any = True
            # aplicar movimientos L_out -> -1 (temporal)
            for (y,x) in to_in:
                Phi[y,x] = -1  # marcador que avanza hacia objeto
                # sus vecinos exteriores se marcan como candidatos externos (1)
                for dy,dx in neigh:
                    yy, xx = y+dy, x+dx
                    if 0<=yy<h and 0<=xx<w and Phi[yy,xx] == 3:
                        Phi[yy,xx] = 1

            # convertir marcadores -1 que están rodeados por negativo en interior definitivo -3
            # (evita que la línea crezca en grosor)
            for (y,x) in list(zip(*np.where(Phi == -1))):
                # si tiene al menos un vecino < 0 (ya interior), hacemos interior definitivo
                if any(0 <= y+dy < h and 0 <= x+dx < w and Phi[y+dy, x+dx] < 0 for dy,dx in neigh):
                    Phi[y,x] = -3

            # --- Paso 2: L_in -> posible convertirse en exterior (retrocede hacia afuera) ---
            to_out = []
            for (y,x) in list(L_in):
                fd = Fd_val(I[y,x])
                if fd < 0:
                    to_out.append((y,x))
            if to_out:
                moved_any = True
            # aplicar movimientos L_in -> 1 (temporal)
            for (y,x) in to_out:
                Phi[y,x] = 1  # marcador que se mueve hacia el exterior
                for dy,dx in neigh:
                    yy, xx = y+dy, x+dx
                    if 0<=yy<h and 0<=xx<w and Phi[yy,xx] == -3:
                        Phi[yy,xx] = -1  # vecino interior se convierte en candidato interno

            # convertir marcadores 1 que están rodeados por positivos en exterior definitivo 3
            for (y,x) in list(zip(*np.where(Phi == 1))):
                if any(0 <= y+dy < h and 0 <= x+dx < w and Phi[y+dy, x+dx] > 0 for dy,dx in neigh):
                    Phi[y,x] = 3

            # Recalcular L_in y L_out (líneas de un píxel) después de los ajustes
            L_in, L_out = compute_contours(Phi)

            # Si no hubo mover_any en esta iteración, el algoritmo converge.
            # (moved_any fue marcado si hubo to_in o to_out)
            # bucle continuará hasta max_iters o hasta que no se muevan píxeles.

        # --- Construir imagen de salida: original RGB + dibujar L_in (rosa) y L_out (celeste) ---
        out_rgb = I_rgb.copy()
        # Asegurarse que las coordenadas son válidas
        for (y,x) in L_in:
            if 0 <= y < h and 0 <= x < w:
                out_rgb[y, x] = np.array([255, 0, 255], dtype=np.uint8)  # rosa
        for (y,x) in L_out:
            if 0 <= y < h and 0 <= x < w:
                out_rgb[y, x] = np.array([0, 255, 255], dtype=np.uint8)  # celeste

        return Imagen(Image.fromarray(out_rgb))
