import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel
from PIL import Image, ImageTk
from raw_reader import RAWReader
from imagen import Imagen
from operaciones import Operaciones


DISPLAY_SIZE = (500, 500)  # tamaño fijo de la vista (ancho, alto)

class ImageApp:
    def __init__(self, root, carpeta_imagenes):
        self.root = root
        self.carpeta = carpeta_imagenes
        self.root.title("TP - Editor de Imagenes")
        self.image = None            # Imagen activa (Imagen)
        self.image_result = None     # Resultado mostrado
        self.tk_img_original = None
        self.tk_img_result = None
        self.region_start = None     # en coordenadas originales (no en vista)
        self.region_end = None
        # mapa de README si existe
        readme_path = os.path.join(self.carpeta, "README.TXT")
        self.readme_map = RAWReader.leer_readme(readme_path)

        self._build_ui()
        self._build_menus()

    def _build_ui(self):
        frame_buttons = tk.Frame(self.root)
        frame_buttons.pack(side="left", fill="y")

        frame_original = tk.Frame(self.root, bd=2, relief="sunken")
        frame_original.pack(side="left", padx=10, pady=10)
        frame_result = tk.Frame(self.root, bd=2, relief="sunken")
        frame_result.pack(side="left", padx=10, pady=10)

        tk.Label(frame_original, text="Original").pack()
        self.canvas_original = tk.Canvas(frame_original, width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1], bg="black")
        self.canvas_original.pack()
        tk.Label(frame_result, text="Resultado").pack()
        self.canvas_result = tk.Canvas(frame_result, width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1], bg="black")
        self.canvas_result.pack()

        # botones
        tk.Button(frame_buttons, text="Cargar Imagen", width=20, command=self.cargar_imagen).pack(pady=4)
        tk.Button(frame_buttons, text="Guardar Resultado", width=20, command=self.guardar_imagen).pack(pady=4)
        tk.Button(frame_buttons, text="Obtener Pixel", width=20, command=self.get_pixel_dialog).pack(pady=4)
        tk.Button(frame_buttons, text="Modificar Pixel", width=20, command=self.modify_pixel_dialog).pack(pady=4)
        tk.Button(frame_buttons, text="Seleccionar Región (arrastre)", width=20, command=self.activate_region_selection).pack(pady=4)
        tk.Button(frame_buttons, text="Copiar Región", width=20, command=self.copy_region).pack(pady=4)
        tk.Button(frame_buttons, text="Resta de Imágenes", width=20, command=self.subtract_images).pack(pady=4)
        
        tk.Button(frame_buttons, text="Reiniciar Interfaz", width=20, command=self.reiniciar).pack(pady=12)


    def _build_menus(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # Menu Operaciones Básicas
        tp1_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Ejercicios del 1 al 5", menu=tp1_menu)
        tp1_menu.add_command(label="Gamma", command=self.gamma)        
        tp1_menu.add_command(label="Negativo", command=self.negative)
        tp1_menu.add_command(label="Mostrar Histograma Original", command=self.histograma_original)
        tp1_menu.add_command(label="Mostrar Histograma Resultado", command=self.histograma_resultado)
        tp1_menu.add_command(label="Umbral", command=self.umbral)
        tp1_menu.add_command(label="Ecualizar Histograma", command=self.ecualizar)
        
        # Menu Ruido
        ruido_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Ruido", menu=ruido_menu)
        ruido_menu.add_command(label="Ruido Gaussiano Aditivo", command=self.ruido_gaussiano)
        ruido_menu.add_command(label="Ruido Rayleigh Multiplicativo", command=self.ruido_rayleigh)
        ruido_menu.add_command(label="Ruido Exponencial Multiplicativo", command=self.ruido_exponencial)
        ruido_menu.add_command(label="Ruido Sal y Pimienta", command=self.ruido_sal_y_pimienta)

        # Menu Filtros
        filtros_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Filtros Espaciales", menu=filtros_menu)
        filtros_menu.add_command(label="Filtro de la Media", command=self.filtro_de_la_media)
        filtros_menu.add_command(label="Filtro de la Mediana", command=self.filtro_de_la_mediana)
        filtros_menu.add_command(label="Filtro de Mediana Ponderada", command=self.filtro_mediana_ponderada)
        filtros_menu.add_command(label="Filtro de Gaussiano", command=self.filtro_gaussiano)
        filtros_menu.add_command(label="Realce de Bordes (Laplace)", command=self.realce_de_bordes)

        # -------------- TP2 MENÚ -----------------
        tp2_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="TP2", menu=tp2_menu)

        # Detectores de bordes por gradiente
        bordes_grad_menu = tk.Menu(tp2_menu, tearoff=0)
        tp2_menu.add_cascade(label="Detectores de Borde (Gradiente)", menu=bordes_grad_menu)
        bordes_grad_menu.add_command(label="Detector de Prewitt", command=self.detector_prewitt)
        bordes_grad_menu.add_command(label="Detector de Sobel", command=self.detector_sobel)

        # Detectores de bordes por Laplaciano
        bordes_lap_menu = tk.Menu(tp2_menu, tearoff=0)
        tp2_menu.add_cascade(label="Detectores de Borde (Laplaciano)", menu=bordes_lap_menu)
        bordes_lap_menu.add_command(label="Detector Laplaciano", command=self.detector_laplaciano)
        bordes_lap_menu.add_command(label="Detector LoG (Marr-Hildreth)", command=self.detector_log)

        # Filtros avanzados
        filtros_av_menu = tk.Menu(tp2_menu, tearoff=0)
        tp2_menu.add_cascade(label="Filtros Avanzados", menu=filtros_av_menu)
        filtros_av_menu.add_command(label="Difusión Isotrópica", command=self.difusion_isotropica)
        filtros_av_menu.add_command(label="Difusión Anisotrópica", command=self.difusion_anisotropica)
        filtros_av_menu.add_command(label="Filtro Bilateral", command=self.filtro_bilateral)

        # Algoritmos de umbralización
        umbral_menu = tk.Menu(tp2_menu, tearoff=0)
        tp2_menu.add_cascade(label="Umbralización", menu=umbral_menu)
        umbral_menu.add_command(label="Umbral Óptimo Iterativo", command=self.umbral_iterativo)
        umbral_menu.add_command(label="Umbral de Otsu", command=self.umbral_otsu)
        umbral_menu.add_command(label="Umbralización por Bandas (RGB)", command=self.umbralizacion_por_bandas_rgb)


    def cargar_imagen_primero(self):
        if not self.image:
            messagebox.showwarning("Atención", "Cargue primero una imagen")
            return False
        return True

    # ---------- carga y visualización ----------
    def cargar_imagen(self):
        path = filedialog.askopenfilename(initialdir=self.carpeta, filetypes=[("Imagenes", "*.png *.jpg *.bmp *.RAW *.PGM")])
        
        # --- CORRECCIÓN: Manejar cancelación del diálogo ---
        if not path or not isinstance(path, str):
            return

        self.image_name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        
        try:
            if ext == ".raw":
                nombre = os.path.basename(path)
                if nombre in self.readme_map:
                    w,h = self.readme_map[nombre]
                else:
                    w = simpledialog.askinteger("Ancho RAW", "Ingrese ancho de la RAW:", minvalue=1)
                    h = simpledialog.askinteger("Alto RAW", "Ingrese alto de la RAW:", minvalue=1)
                    if not w or not h:
                        messagebox.showerror("Error","Dimensiones inválidas")
                        return
                raw_data = RAWReader.leer_raw(path, w, h)
                self.image = Imagen(raw_data)
            elif ext == ".pgm":
                pgm_data = RAWReader.leer_pgm(path)
                self.image = Imagen(pgm_data)
            else:
                pil = Image.open(path).convert("RGB")
                self.image = Imagen(pil)
        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))
            return

        self.show_image(self.image, self.canvas_original)
        self.canvas_result.delete("all")
        self.image_result = None

    def show_image(self, img_obj: Imagen, canvas: tk.Canvas):
        pil = img_obj.to_pil()
        disp = pil.resize(DISPLAY_SIZE)
        tk_img = ImageTk.PhotoImage(disp)
        canvas.delete("all")
        canvas.create_image(0,0, anchor="nw", image=tk_img)
        if canvas is self.canvas_original:
            self.tk_img_original = tk_img
            self.current_orig_size = img_obj.get_size()
            self.current_display_size = DISPLAY_SIZE
            self.current_image_obj = img_obj
        else:
            self.tk_img_result = tk_img
        if canvas is self.canvas_result:
            self.image_result = img_obj

    # ---------- get / set pixel ----------
    def get_pixel_dialog(self):
        if not self.cargar_imagen_primero(): return
        w,h = self.image.get_size()
        x = simpledialog.askinteger("x", f"X (0..{w-1}):", minvalue=0, maxvalue=w-1)
        y = simpledialog.askinteger("y", f"Y (0..{h-1}):", minvalue=0, maxvalue=h-1)
        if x is None or y is None: return
        val = self.image.get_pixel(x,y)
        messagebox.showinfo("Valor de pixel", f"Pixel ({x},{y}) = {val}")

    def modify_pixel_dialog(self):
        if not self.cargar_imagen_primero(): return
        w,h = self.image.get_size()
        x = simpledialog.askinteger("x", f"X (0..{w-1}):", minvalue=0, maxvalue=w-1)
        y = simpledialog.askinteger("y", f"Y (0..{h-1}):", minvalue=0, maxvalue=h-1)
        if x is None or y is None: return
        
        if self.image.to_pil().mode == 'L' or isinstance(self.image.data, list):
            v = simpledialog.askinteger("Valor gris", f"Valor (0..255):", minvalue=0, maxvalue=255)
            if v is None: return
            self.image.set_pixel(x,y,int(v))
        else: # RGB
            r,g,b = self.image.get_pixel(x,y)
            r_new = simpledialog.askinteger("R", "R (0..255):", initialvalue=r, minvalue=0, maxvalue=255)
            g_new = simpledialog.askinteger("G", "G (0..255):", initialvalue=g, minvalue=0, maxvalue=255)
            b_new = simpledialog.askinteger("B", "B (0..255):", initialvalue=b, minvalue=0, maxvalue=255)
            if None in (r_new, g_new, b_new): return
            self.image.set_pixel(x,y,(int(r_new),int(g_new),int(b_new)))
        
        self.show_image(self.image, self.canvas_result)

    # ---------- selección de región con mouse ----------
    def activate_region_selection(self):
        if not self.cargar_imagen_primero(): return
        self.canvas_original.bind("<Button-1>", self.on_click)
        self.canvas_original.bind("<B1-Motion>", self.on_drag)
        self.canvas_original.bind("<ButtonRelease-1>", self.on_release)
        messagebox.showinfo("Info", "Haga clic y arrastre en la imagen original para seleccionar región")

    def on_click(self, event):
        ox, oy = self._map_to_original(event.x, event.y)
        self.region_start = (ox, oy)
        self.canvas_original.delete("region_rect")

    def on_drag(self, event):
        self.canvas_original.delete("region_rect")
        x0,y0 = self._map_from_original(self.region_start[0], self.region_start[1])
        self.canvas_original.create_rectangle(x0, y0, event.x, event.y, outline="red", width=2, tags="region_rect")

    def on_release(self, event):
        ox, oy = self._map_to_original(event.x, event.y)
        self.region_end = (ox, oy)
        self.canvas_original.unbind("<Button-1>")
        self.canvas_original.unbind("<B1-Motion>")
        self.canvas_original.unbind("<ButtonRelease-1>")
        x1,y1,x2,y2 = self.region_start[0], self.region_start[1], self.region_end[0], self.region_end[1]
        if x2<=x1 or y2<=y1:
            messagebox.showwarning("Región inválida", "La región seleccionada es inválida")
            return
        region = self.image.copy_region((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))
        self.show_image(region, self.canvas_result)

    def _map_to_original(self, vx, vy):
        disp_w, disp_h = self.current_display_size
        orig_w, orig_h = self.current_orig_size
        ox = int(max(0, min(disp_w-1, vx)) * orig_w / disp_w)
        oy = int(max(0, min(disp_h-1, vy)) * orig_h / disp_h)
        return (ox, oy)

    def _map_from_original(self, ox, oy):
        disp_w, disp_h = self.current_display_size
        orig_w, orig_h = self.current_orig_size
        vx = int(ox * disp_w / orig_w)
        vy = int(oy * disp_h / orig_h)
        return (vx, vy)

    def copy_region(self):
        if not self.image or not self.region_start or not self.region_end:
            messagebox.showwarning("Atención", "Primero seleccione una región con el mouse")
            return
        x1,y1,x2,y2 = self.region_start[0], self.region_start[1], self.region_end[0], self.region_end[1]
        box = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
        region = self.image.copy_region(box)
        self.show_image(region, self.canvas_result)

    def subtract_images(self):
        if not self.cargar_imagen_primero(): return
        path2 = filedialog.askopenfilename(initialdir=self.carpeta, filetypes=[("Imagenes", "*.png *.jpg *.bmp *.RAW *.PGM")])
        if not path2: return
        try:
            img2_pil = Image.open(path2).convert("L")
            img2 = Imagen(img2_pil)
            result = Operaciones.subtract(self.image, img2)
        except Exception as e:
            messagebox.showerror("Error en resta", str(e))
            return
        self.show_image(result, self.canvas_result)

    def guardar_imagen(self):
        if not self.image_result:
            messagebox.showwarning("Atención", "No hay resultado para guardar")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPG","*.jpg")])
        if not path: return
        try:
            self.image_result.to_pil().save(path)
            messagebox.showinfo("Guardar", "Imagen guardada correctamente")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))

    def reiniciar(self):
        carpeta = self.carpeta
        self.root.destroy()
        nuevo_root = tk.Tk()
        ImageApp(nuevo_root, carpeta)
        nuevo_root.mainloop()

# -------------- TP1 -----------------
    def gamma(self):
        if not self.cargar_imagen_primero(): return
        y = simpledialog.askfloat("Transformación Gamma", "γ (0 < γ < 2, γ ≠ 1):", minvalue=0.01, maxvalue=1.99)
        if y is not None: self.show_image(Operaciones.gamma(self.image,y), self.canvas_result)

    def negative(self):
        if not self.cargar_imagen_primero(): return
        self.show_image(Operaciones.negative(self.image), self.canvas_result)
    
    def mostrar_histograma(self, img: Imagen, title: str):
        if img is None: return
        hist = Operaciones.histograma(img)
        win = Toplevel(self.root); win.title(title)
        canvas = tk.Canvas(win, width=512, height=300, bg="white"); canvas.pack(padx=10, pady=10)
        h_max = hist.max()
        if h_max > 0:
            for i, h in enumerate(hist):
                x0, x1 = 2 * i, 2 * i + 2
                y0 = 300 - int((h / h_max) * 280)
                canvas.create_rectangle(x0, y0, x1, 300, fill="skyblue", outline="black")
        
    def histograma_original(self):
        if not self.cargar_imagen_primero(): return
        self.mostrar_histograma(self.image, "Histograma Imagen Original")

    def histograma_resultado(self):
        if self.image_result is None:
            messagebox.showwarning("Advertencia", "No hay imagen de resultado.")
            return
        self.mostrar_histograma(self.image_result, "Histograma Imagen Resultado")

    def umbral(self):
        if not self.cargar_imagen_primero(): return
        u = simpledialog.askinteger("Umbralización", "Valor del umbral (0-255):", minvalue=0, maxvalue=255)
        if u is not None: self.show_image(Operaciones.umbral(self.image, u), self.canvas_result)
        
    def ecualizar(self):
        if not self.cargar_imagen_primero(): return
        self.show_image(Operaciones.ecualizar_histograma(self.image), self.canvas_result)
    
    def ruido_gaussiano(self):
        if not self.cargar_imagen_primero(): return
        sigma = simpledialog.askfloat("Ruido Gaussiano", "Sigma (desv. est.):")
        porcentaje = simpledialog.askfloat("Ruido Gaussiano", "Porcentaje de píxeles (0-1):", minvalue=0, maxvalue=1)
        if sigma is not None and porcentaje is not None:
            self.show_image(Operaciones.aplicar_ruido_gaussiano(self.image, 0, sigma, porcentaje), self.canvas_result)

    def ruido_rayleigh(self):
        if not self.cargar_imagen_primero(): return
        xi = simpledialog.askfloat("Ruido Rayleigh", "Xi:", minvalue=0.01)
        porcentaje = simpledialog.askfloat("Ruido Rayleigh", "Porcentaje (0-1):", minvalue=0, maxvalue=1)
        if xi is not None and porcentaje is not None:
            self.show_image(Operaciones.aplicar_ruido_rayleigh(self.image, xi, porcentaje), self.canvas_result)

    def ruido_exponencial(self):
        if not self.cargar_imagen_primero(): return
        lambd = simpledialog.askfloat("Ruido Exponencial", "Lambda:", minvalue=0.01)
        porcentaje = simpledialog.askfloat("Ruido Exponencial", "Porcentaje (0-1):", minvalue=0, maxvalue=1)
        if lambd is not None and porcentaje is not None:
            self.show_image(Operaciones.aplicar_ruido_exponencial(self.image, lambd, porcentaje), self.canvas_result)
        
    def ruido_sal_y_pimienta(self):
        if not self.cargar_imagen_primero(): return
        density = simpledialog.askfloat("Ruido Sal y Pimienta", "Densidad (0-1):", minvalue=0, maxvalue=1)
        if density is not None:
            self.show_image(Operaciones.aplicar_ruido_sal_y_pimienta(self.image, density), self.canvas_result)
        
    def filtro_de_la_media(self):
        if not self.cargar_imagen_primero(): return
        k_size = simpledialog.askinteger("Filtro de la Media", "Tamaño del kernel (impar):", minvalue=3, maxvalue=15)
        if k_size and k_size % 2 != 0:
            self.show_image(Operaciones.filtro_media(self.image, k_size), self.canvas_result)

    def filtro_de_la_mediana(self):
        if not self.cargar_imagen_primero(): return
        k_size = simpledialog.askinteger("Filtro de la Mediana", "Tamaño del kernel (impar):", minvalue=3, maxvalue=15)
        if k_size and k_size % 2 != 0:
            self.show_image(Operaciones.filtro_mediana(self.image, k_size), self.canvas_result)

    def filtro_mediana_ponderada(self):
        if not self.cargar_imagen_primero(): return
        self.show_image(Operaciones.filtro_mediana_ponderada(self.image, k_size=3), self.canvas_result)

    def filtro_gaussiano(self):
        if not self.cargar_imagen_primero(): return
        k_size = simpledialog.askinteger("Filtro Gaussiano", "Tamaño del kernel (impar):", minvalue=3, maxvalue=15)
        sigma = simpledialog.askfloat("Filtro Gaussiano", "Sigma:", minvalue=0.1)
        if k_size and k_size % 2 != 0 and sigma is not None:
            self.show_image(Operaciones.filtro_gaussiano(self.image, k_size, sigma), self.canvas_result)
         
    def realce_de_bordes(self):
        if not self.cargar_imagen_primero(): return
        self.show_image(Operaciones.realce_bordes(self.image), self.canvas_result)

    #------------ TP 2 Interfaz ---------------
    def detector_prewitt(self):
        if not self.cargar_imagen_primero(): return
        self.show_image(Operaciones.detector_prewitt(self.image), self.canvas_result)
    
    def detector_sobel(self):
        if not self.cargar_imagen_primero(): return
        self.show_image(Operaciones.detector_sobel(self.image), self.canvas_result)
        
    def detector_laplaciano(self):
        if not self.cargar_imagen_primero(): return
        self.show_image(Operaciones.detector_laplaciano(self.image), self.canvas_result)

    def detector_log(self):
        if not self.cargar_imagen_primero(): return
        k_size = simpledialog.askinteger("Detector LoG", "Tamaño del kernel (impar):", minvalue=3, maxvalue=21)
        sigma = simpledialog.askfloat("Detector LoG", "Sigma:", minvalue=0.1)
        if k_size and sigma and k_size % 2 != 0:
            self.show_image(Operaciones.detector_log(self.image, k_size, sigma), self.canvas_result)

    def difusion_isotropica(self):
        if not self.cargar_imagen_primero(): return
        iteraciones = simpledialog.askinteger("Difusión Isotrópica", "Nº de iteraciones:", minvalue=1)
        if iteraciones:
            self.show_image(Operaciones.difusion_isotropica(self.image, iteraciones), self.canvas_result)

    def difusion_anisotropica(self):
        if not self.cargar_imagen_primero(): return
        iteraciones = simpledialog.askinteger("Difusión Anisotrópica", "Nº de iteraciones:", minvalue=1)
        k = simpledialog.askfloat("Difusión Anisotrópica", "Parámetro K:", minvalue=1)
        if iteraciones and k:
            self.show_image(Operaciones.difusion_anisotropica(self.image, iteraciones, k), self.canvas_result)

    def filtro_bilateral(self):
        if not self.cargar_imagen_primero(): return
        k_size = simpledialog.askinteger("Filtro Bilateral", "Tamaño del kernel (impar):", minvalue=3)
        sigma_e = simpledialog.askfloat("Filtro Bilateral", "Sigma Espacial:", minvalue=0.1)
        sigma_r = simpledialog.askfloat("Filtro Bilateral", "Sigma de Rango:", minvalue=0.1)
        if k_size and sigma_e and sigma_r and k_size % 2 != 0:
            self.show_image(Operaciones.filtro_bilateral(self.image, k_size, sigma_e, sigma_r), self.canvas_result)

    def umbral_iterativo(self):
        if not self.cargar_imagen_primero(): return
        self.show_image(Operaciones.umbral_iterativo(self.image), self.canvas_result)

    def umbral_otsu(self):
        if not self.cargar_imagen_primero(): return
        self.show_image(Operaciones.umbral_otsu(self.image), self.canvas_result)
        
    def umbralizacion_por_bandas_rgb(self):
        if not self.cargar_imagen_primero(): return
        if self.image.to_pil().mode != 'RGB':
            messagebox.showerror("Error", "Esta operación requiere una imagen a color (RGB).")
            return

        win = Toplevel(self.root); win.title("Umbrales por Banda"); win.resizable(False, False)
        entries = {}
        for i, canal in enumerate(["Rojo", "Verde", "Azul"]):
            tk.Label(win, text=f"{canal}:").grid(row=i, column=0, sticky="w", padx=5, pady=5)
            min_e = tk.Entry(win, width=5); min_e.grid(row=i, column=1, pady=5); min_e.insert(0, "0")
            tk.Label(win, text="a").grid(row=i, column=2)
            max_e = tk.Entry(win, width=5); max_e.grid(row=i, column=3, pady=5); max_e.insert(0, "255")
            entries[canal] = (min_e, max_e)

        def aplicar():
            try:
                p = [int(e.get()) for k in entries for e in entries[k]]
                if not (0<=p[0]<=p[1]<=255 and 0<=p[2]<=p[3]<=255 and 0<=p[4]<=p[5]<=255):
                    messagebox.showerror("Error", "Valores fuera de rango (0-255) o min > max.", parent=win)
                    return
                self.show_image(Operaciones.umbralizacion_por_bandas_rgb(self.image, *p), self.canvas_result)
                win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Error de entrada: {e}", parent=win)

        tk.Button(win, text="Aplicar", command=aplicar).grid(row=3, columnspan=4, pady=10)
        win.transient(self.root); win.grab_set(); self.root.wait_window(win)