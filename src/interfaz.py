# src/interfaz.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from raw_reader import RAWReader
from imagen import Imagen
from operaciones import Operaciones

DISPLAY_SIZE = (400, 400)  # tamaño fijo de la vista (ancho, alto)

class ImageApp:
    def __init__(self, root, carpeta_imagenes):
        self.root = root
        self.carpeta = carpeta_imagenes
        self.root.title("TP0 - Visor RAW/PGM/Imagenes")
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
        tk.Button(frame_buttons, text="Salir", width=20, command=self.root.quit).pack(pady=4)

    # ---------- carga y visualización ----------
    def cargar_imagen(self):
        path = filedialog.askopenfilename(initialdir=self.carpeta, filetypes=[("Imagenes", "*.png *.jpg *.bmp *.RAW *.PGM")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".raw":
                nombre = os.path.basename(path)
                # si está en README, tomar dimensiones; si no, pedirlas
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

    def show_image(self, img_obj: Imagen, canvas: tk.Canvas):
        """Muestra imagen en el canvas (reescala a DISPLAY_SIZE). Guarda referencia PhotoImage."""
        pil = img_obj.to_pil()
        # reescalar exactamente a DISPLAY_SIZE (simplifica mapeo coordenadas)
        disp = pil.resize(DISPLAY_SIZE)
        tk_img = ImageTk.PhotoImage(disp)
        canvas.delete("all")
        canvas.create_image(0,0, anchor="nw", image=tk_img)
        # mantener referencias para que no las recoja el GC
        if canvas is self.canvas_original:
            self.tk_img_original = tk_img
            # guardar info para mapeo de clicks
            self.current_orig_size = img_obj.get_size()
            self.current_display_size = DISPLAY_SIZE
            self.current_image_obj = img_obj
        else:
            self.tk_img_result = tk_img
        # también guardar último resultado
        if canvas is self.canvas_result:
            self.image_result = img_obj

    # ---------- get / set pixel ----------
    def get_pixel_dialog(self):
        if not self.image:
            messagebox.showwarning("Atención", "Cargue primero una imagen")
            return
        # pedir coordenadas en coordenadas de la imagen original
        w,h = self.image.get_size()
        x = simpledialog.askinteger("x", f"X (0..{w-1}):", minvalue=0, maxvalue=w-1)
        y = simpledialog.askinteger("y", f"Y (0..{h-1}):", minvalue=0, maxvalue=h-1)
        if x is None or y is None:
            return
        val = self.image.get_pixel(x,y)
        messagebox.showinfo("Valor de pixel", f"Pixel ({x},{y}) = {val}")

    def modify_pixel_dialog(self):
        if not self.image:
            messagebox.showwarning("Atención", "Cargue primero una imagen")
            return
        w,h = self.image.get_size()
        x = simpledialog.askinteger("x", f"X (0..{w-1}):", minvalue=0, maxvalue=w-1)
        y = simpledialog.askinteger("y", f"Y (0..{h-1}):", minvalue=0, maxvalue=h-1)
        if x is None or y is None:
            return
        # dependiendo si es grayscale o color pedimos valor
        if isinstance(self.image.data, list):
            v = simpledialog.askinteger("Valor gris", f"Valor (0..255):", minvalue=0, maxvalue=255)
            if v is None: return
            self.image.set_pixel(x,y,int(v))
        else:
            # color
            r = simpledialog.askinteger("R", "R (0..255):", minvalue=0, maxvalue=255)
            g = simpledialog.askinteger("G", "G (0..255):", minvalue=0, maxvalue=255)
            b = simpledialog.askinteger("B", "B (0..255):", minvalue=0, maxvalue=255)
            if None in (r,g,b): return
            self.image.set_pixel(x,y,(int(r),int(g),int(b)))
        # mostrar cambio en panel resultado
        self.show_image(self.image, self.canvas_result)

    # ---------- selección de región con mouse ----------
    def activate_region_selection(self):
        if not self.image:
            messagebox.showwarning("Atención", "Cargue primero una imagen")
            return
        # bind a canvas original
        self.canvas_original.bind("<Button-1>", self.on_click)
        self.canvas_original.bind("<ButtonRelease-1>", self.on_release)
        messagebox.showinfo("Info", "Haga clic y arrastre en la imagen original para seleccionar región")

    def on_click(self, event):
        # event.x/y en coordenadas de la vista (DISPLAY_SIZE)
        ox, oy = self._map_to_original(event.x, event.y)
        self.region_start = (ox, oy)

    def on_release(self, event):
        ox, oy = self._map_to_original(event.x, event.y)
        self.region_end = (ox, oy)
        # mostrar info de región
        x1,y1 = self.region_start
        x2,y2 = self.region_end
        x1,x2 = min(x1,x2), max(x1,x2)
        y1,y2 = min(y1,y2), max(y1,y2)
        if x2<=x1 or y2<=y1:
            messagebox.showwarning("Región inválida", "La región seleccionada es inválida")
            return
        # calcular y mostrar promedio
        region = self.image.copy_region((x1,y1,x2,y2))
        arr = region.to_numpy()
        import numpy as np
        total_pixels = arr.shape[0] * arr.shape[1]
        promedio = arr.mean(axis=(0,1))
        messagebox.showinfo("Región",
                            f"Píxeles: {total_pixels}\nPromedio: {promedio}")
        # mostrar región en panel resultado
        self.show_image(region, self.canvas_result)

    def _map_to_original(self, vx, vy):
        """Mapea coordenada del view (vx,vy) al original (ox,oy)."""
        disp_w, disp_h = self.current_display_size
        orig_w, orig_h = self.current_orig_size
        # límite dentro del display
        vx = max(0, min(disp_w-1, vx))
        vy = max(0, min(disp_h-1, vy))
        ox = int(vx * orig_w / disp_w)
        oy = int(vy * orig_h / disp_h)
        ox = max(0, min(orig_w-1, ox))
        oy = max(0, min(orig_h-1, oy))
        return (ox, oy)

    # ---------- copiar región ----------
    def copy_region(self):
        if not self.image or not self.region_start or not self.region_end:
            messagebox.showwarning("Atención", "Primero seleccione una región con el mouse")
            return
        x1,y1 = self.region_start
        x2,y2 = self.region_end
        box = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
        region = self.image.copy_region(box)
        self.show_image(region, self.canvas_result)

    # ---------- resta de imágenes ----------
    def subtract_images(self):
        if not self.image:
            messagebox.showwarning("Atención", "Cargue primero la imagen base")
            return
        path2 = filedialog.askopenfilename(initialdir=self.carpeta, filetypes=[("Imagenes", "*.png *.jpg *.bmp *.RAW *.PGM")])
        if not path2:
            return
        ext = os.path.splitext(path2)[1].lower()
        try:
            if ext == ".raw":
                nombre = os.path.basename(path2)
                if nombre in self.readme_map:
                    w,h = self.readme_map[nombre]
                else:
                    # intentar usar tamaño de la imagen base
                    w,h = self.image.get_size()
                    # si no existe mapping pedimos confirmación (o pedir explícitamente)
                    # si no coincide, pedir dimensiones
                    # pedimos dimensiones para el raw
                    w = simpledialog.askinteger("Ancho RAW", "Ingrese ancho de la RAW:", initialvalue=w, minvalue=1)
                    h = simpledialog.askinteger("Alto RAW", "Ingrese alto de la RAW:", initialvalue=h, minvalue=1)
                    if not w or not h:
                        messagebox.showerror("Error","Dimensiones inválidas")
                        return
                raw2 = RAWReader.leer_raw(path2, w, h)
                img2 = Imagen(raw2)
            elif ext == ".pgm":
                pgm2 = RAWReader.leer_pgm(path2)
                img2 = Imagen(pgm2)
            else:
                img2_pil = Image.open(path2).convert("RGB")
                # redimensionar a tamaño de la imagen base
                target_size = self.image.get_size()
                img2_pil = img2_pil.resize(target_size)
                img2 = Imagen(img2_pil)
        except Exception as e:
            messagebox.showerror("Error al leer segunda imagen", str(e))
            return

        try:
            result = Operaciones.subtract(self.image, img2)
        except Exception as e:
            messagebox.showerror("Error en resta", str(e))
            return
        self.show_image(result, self.canvas_result)

    # ---------- guardar resultado ----------
    def guardar_imagen(self):
        if not self.image_result:
            messagebox.showwarning("Atención", "No hay resultado para guardar")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("PGM","*.pgm")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pgm":
                # guardar en ASCII P2 para simplicidad
                pil = self.image_result.to_pil().convert("L")
                w,h = pil.size
                data = list(pil.getdata())
                with open(path, "w") as f:
                    f.write("P2\n# guardado por TP0\n")
                    f.write(f"{w} {h}\n255\n")
                    for i in range(h):
                        row = data[i*w:(i+1)*w]
                        f.write(" ".join(map(str,row)) + "\n")
            else:
                self.image_result.to_pil().save(path)
            messagebox.showinfo("Guardar", "Imagen guardada correctamente")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))

    # ---------- reiniciar ventana ----------
    def reiniciar(self):
        # destruye la ventana y crea una nueva
        carpeta = self.carpeta
        self.root.destroy()
        nuevo_root = tk.Tk()
        app = ImageApp(nuevo_root, carpeta)
        nuevo_root.mainloop()

    # ---------- negativo ----------
    def negative(self):
        if not self.image:
            messagebox.showwarning("Atención", "Cargue primero la imagen base")
            return

        result = Operaciones.negative(self.image)
        return self.show_image(result, self.canvas_result)
    



