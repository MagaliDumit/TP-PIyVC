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

#----------------------------- TP1 ----------------------------
    
    @staticmethod
    def gamma(img1: Imagen, y: float) -> Imagen:
        """
        Transforma img1 con gamma.
        γ, 0 < γ < 2 y γ ̸ = 1
        """
        a1 = img1.to_numpy().astype(int)

        b = ((a1 / 255) ** y)

        norm = (255 * b ).astype(np.uint8)    ##  (b ** y)

        # Asegurar rango [0,255] y convertir a 8 bits
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        return Imagen(Image.fromarray(norm))


    @staticmethod
    def negative(img1: Imagen) -> Imagen:
        """
        Convierte img1 a negativo.
        T(r) = 255 − r
        """
        a1 = img1.to_numpy().astype(int)
       
        norm = (255.0 -a1 ).astype(np.uint8)
        
        return Imagen(Image.fromarray(norm))

    @staticmethod
    def histograma(img1: Imagen) -> Imagen:
        a1 = img1.to_numpy().astype(int)
        """
        Calcula el histograma de una imagen en escala de grises.
        Devuelve un array de 256 frecuencias relativas.
        """
        total_pixeles = a1.size
        counts = np.bincount(a1.ravel(), minlength=256)
        hist = counts / total_pixeles
        return hist
    
    @staticmethod
    def umbral(img1: Imagen, u: int) -> Imagen:
        """
        Convierte img1 a binaria.
        T(r)= 255 si r ≥ u y 0 si r < u
        """
        a1 = img1.to_numpy().astype(int)
        
        norm = np.where(a1 >= u, 255, 0).astype(np.uint8)

        return Imagen(Image.fromarray(norm))
    
    @staticmethod
    def ecualizar_histograma(img1: Imagen) -> Imagen:
        """
        Ecualiza el histograma de una imagen en escala de grises.
        img: objeto Imagen (PIL -> numpy)
        return: Imagen ecualizada
        """
        # Convertir a numpy
        a1 = img1.to_numpy().astype(np.uint8).ravel()

        # 1. Histograma (frecuencias absolutas)
        hist = np.bincount(a1, minlength=256)

        # 2. CDF (función de distribución acumulada)
        cdf = hist.cumsum()

        # Normalizar (para que el mínimo no sea cero)
        cdf_min = cdf[cdf > 0].min()
        cdf_norm = (cdf - cdf_min) / (cdf[-1] - cdf_min)

        # 3. Mapear a [0,255]
        lookup = np.round(cdf_norm * 255).astype(np.uint8)

        # 4. Aplicar la transformación a la imagen
        result = lookup[a1].reshape(img1.to_numpy().shape)

        return Imagen(Image.fromarray(result))
        


    @staticmethod
    def generar_ruido_gaussiano(mu: float, sigma: float, porcentaje) -> np.ndarray:
        """Punto 7a: Genera ruido Gaussiano."""
        return np.random.normal(mu, sigma, porcentaje)


    @staticmethod
    def generar_ruido_rayleigh(xi: float, shape) -> np.ndarray:
        """Punto 7b: Genera ruido Rayleigh."""
        return np.random.rayleigh(xi, shape)


    @staticmethod
    def generar_ruido_exponencial(lambd: float, shape) -> np.ndarray:
        """Punto 7c: Genera ruido Exponencial."""
        return np.random.exponential(1/lambd, shape)


    @staticmethod
    def generar_ruido_sal_y_pimienta(density: float, shape) -> np.ndarray:
        """Punto 9: Genera ruido Sal y Pimienta."""
        h, w = shape
        output = np.zeros(shape, dtype=np.uint8)
        total_pixels = h * w
        num_salt = int(total_pixels * density / 2)
        num_pepper = int(total_pixels * density / 2)
        
        coords_salt = np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)
        output[coords_salt] = 255  # Sal
        
        coords_pepper = np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper)
        output[coords_pepper] = 0   # Pimienta
        
        return output


    @staticmethod
    def aplicar_ruido_gaussiano(img: Imagen, mu: float, sigma: float, porcentaje: float) -> Imagen:
        """Punto 8a: Contamina una imagen con ruido Gaussiano aditivo."""
        arr = img.to_numpy().astype(float)
        h, w = arr.shape
        total_pixels = h * w
        num_pixels_afectados = int(total_pixels * porcentaje)
        
        ruido = np.random.normal(mu, sigma, num_pixels_afectados)
        
        # Seleccionar píxeles aleatorios
        coords = np.random.randint(0, h, num_pixels_afectados), np.random.randint(0, w, num_pixels_afectados)
        
        arr[coords] += ruido
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Imagen(Image.fromarray(arr))


    @staticmethod
    def aplicar_ruido_multiplicativo(img: Imagen, generador_ruido, porcentaje: float) -> Imagen:
        """
        Helper para ruido multiplicativo (Rayleigh, Exponencial).
        """
        arr = img.to_numpy().astype(float)
        h, w = arr.shape
        total_pixels = h * w
        num_pixels_afectados = int(total_pixels * porcentaje)
        
        ruido = generador_ruido(num_pixels_afectados)
        
        # Seleccionar píxeles aleatorios
        coords = np.random.randint(0, h, num_pixels_afectados), np.random.randint(0, w, num_pixels_afectados)
        
        arr[coords] *= ruido
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Imagen(Image.fromarray(arr))
    

    @staticmethod
    def aplicar_ruido_rayleigh(img: Imagen, xi: float, porcentaje: float) -> Imagen:
        """Punto 8b: Contamina con ruido Rayleigh multiplicativo."""
        generador = lambda n: np.random.rayleigh(xi, n)
        return Operaciones.aplicar_ruido_multiplicativo(img, generador, porcentaje)
    

    @staticmethod
    def aplicar_ruido_exponencial(img: Imagen, lambd: float, porcentaje: float) -> Imagen:
        """Punto 8c: Contamina con ruido Exponencial multiplicativo."""
        generador = lambda n: np.random.exponential(1/lambd, n)
        return Operaciones.aplicar_ruido_multiplicativo(img, generador, porcentaje)


    @staticmethod
    def aplicar_ruido_sal_y_pimienta(img: Imagen, density: float) -> Imagen:
        """Punto 9: Aplica ruido Sal y Pimienta."""
        arr = img.to_numpy().astype(np.uint8).copy()
        h, w = arr.shape
        total_pixels = h * w
        num_pepper = int(total_pixels * density / 2)
        num_salt = int(total_pixels * density / 2)
        
        # Añadir sal (blanco)
        coords_salt = np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)
        arr[coords_salt] = 255
        
        # Añadir pimienta (negro)
        coords_pepper = np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper)
        arr[coords_pepper] = 0
        
        return Imagen(Image.fromarray(arr))


    @staticmethod
    def aplicar_filtro_deslizante(img: Imagen, kernel: np.ndarray, tipo_filtro: str) -> Imagen:
        """
        Helper para filtros espaciales.
        Aplica un filtro a una imagen utilizando una ventana deslizante.
        """
        arr = img.to_numpy().astype(float)
        h, w = arr.shape
        kh, kw = kernel.shape
        ph = kh // 2
        pw = kw // 2
        
        # Rellena el borde de la imagen para que el filtro se pueda aplicar
        padded_arr = np.pad(arr, ((ph, ph), (pw, pw)), mode='edge')
        output = np.zeros_like(arr)
        
        for y in range(h):
            for x in range(w):
                window = padded_arr[y:y+kh, x:x+kw]
                if tipo_filtro == 'media':
                    output[y, x] = np.mean(window)
                elif tipo_filtro == 'gaussiano':
                    output[y, x] = np.sum(window * kernel)
                elif tipo_filtro == 'realce_bordes':
                    output[y, x] = np.sum(window * kernel)
                elif tipo_filtro == 'prewitt horizontal':
                    output[y, x] = np.sum(window * kernel)
                elif tipo_filtro == 'prewitt vertical':
                    output[y, x] = np.sum(window * kernel)


        output = np.clip(output, 0, 255).astype(np.uint8)
        return Imagen(Image.fromarray(output))


    @staticmethod
    def filtro_media(img: Imagen, k_size: int = 3) -> Imagen:
        """Punto 10a: Filtro de la media."""
        kernel = np.ones((k_size, k_size)) / (k_size * k_size)
        return Operaciones.aplicar_filtro_deslizante(img, kernel, 'media')
    

    @staticmethod
    def filtro_mediana(img: Imagen, k_size: int = 3) -> Imagen:
        """Punto 10b: Filtro de la mediana."""
        arr = img.to_numpy().astype(float)
        h, w = arr.shape
        ph, pw = k_size // 2, k_size // 2
        padded_arr = np.pad(arr, ((ph, ph), (pw, pw)), mode='edge')
        output = np.zeros_like(arr)
        
        for y in range(h):
            for x in range(w):
                window = padded_arr[y:y+k_size, x:x+k_size]
                output[y, x] = np.median(window)

        output = np.clip(output, 0, 255).astype(np.uint8)
        return Imagen(Image.fromarray(output))
    

    @staticmethod
    def filtro_mediana_ponderada(img: Imagen, k_size: int = 3, weights=None) -> Imagen:
        """Punto 10c: Filtro de la mediana ponderada."""
        arr = img.to_numpy().astype(float)
        h, w = arr.shape
        ph, pw = k_size // 2, k_size // 2
        padded_arr = np.pad(arr, ((ph, ph), (pw, pw)), mode='edge')
        output = np.zeros_like(arr)
        
        if weights is None:
            # Ejemplo de matriz de pesos si no se proporciona
            if k_size == 3:
                weights = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
            else:
                weights = np.ones((k_size, k_size))
        
        for y in range(h):
            for x in range(w):
                window = padded_arr[y:y+k_size, x:x+k_size].flatten()
                
                # Crear la lista de píxeles ponderados
                weighted_pixels = []
                for i in range(k_size):
                    for j in range(k_size):
                        pixel_val = window[i*k_size + j]
                        weight = weights[i, j]
                        weighted_pixels.extend([pixel_val] * int(weight))
                
                # Calcular la mediana de la lista ponderada
                output[y, x] = np.median(np.sort(weighted_pixels))

        output = np.clip(output, 0, 255).astype(np.uint8)
        return Imagen(Image.fromarray(output))


    @staticmethod
    def filtro_gaussiano(img: Imagen, k_size: int = 3, sigma: float = 1.0) -> Imagen:
        """
        Punto 10d: Filtro de Gauss para diferentes valores de sigma.
        Genera el kernel y aplica el filtro.
        """
        ax = np.linspace(-(k_size - 1) / 2., (k_size - 1) / 2., k_size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        kernel = kernel / np.sum(kernel)
        
        return Operaciones.aplicar_filtro_deslizante(img, kernel, 'gaussiano')


    @staticmethod
    def realce_bordes(img: Imagen) -> Imagen:
        """
        Punto 10e: Realce de Bordes.
        Utiliza el kernel de Laplace.
        """
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        return Operaciones.aplicar_filtro_deslizante(img, kernel, 'realce_bordes')




#------------ TP 2 ---------------
    # detector_de_bordes
    def prewitt_horizontal(img: Imagen) -> Imagen:
        kernel = np.array([[-1, -1, -1], [0,0,0], [1,1,1]])
        return Operaciones.aplicar_filtro_deslizante(img, kernel, 'prewitt horizontal')
    
    def prewitt_vertical(img: Imagen) -> Imagen:
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1,0,1]])
        return Operaciones.aplicar_filtro_deslizante(img, kernel, 'prewitt vertical')
    
