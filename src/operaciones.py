import numpy as np
from PIL import Image
from imagen import Imagen

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
    

