[readme.md](https://github.com/user-attachments/files/22018414/readme.md)
TP0-20250821/
 ├─ Imágenes/
 │   ├─ BARCO.RAW
 │   ├─ GIRL.RAW
 │   ├─ LENA.RAW
 │   ├─ GIRL2.RAW
 │   ├─ TESTpgm.PGM
 │   └─ README.TXT
 └─ src/
     ├─ raw_reader.py
     ├─ imagen.py
     ├─ operaciones.py
     ├─ interfaz.py
     └─ main.py


sudo apt update
sudo apt install python3-tk

pip install --upgrade pillow

##Operaciones disponibles##

###Copiar región
Permite seleccionar una porción rectangular de la imagen y duplicarla dentro de la misma ventana.
(Se selecciona manualmente el área desde la interfaz).

###Modificar un píxel
Permite cambiar manualmente la intensidad (0–255) de un píxel específico en la imagen.

###Resta de imágenes###
Realiza la operación 
Iresultado=I1−I2 sin truncamiento.
Los valores negativos se mantienen y se normalizan para mostrar la imagen correctamente.

###Reiniciar ventana###
Cierra todos los elementos cargados y restaura la interfaz a su estado inicial, sin necesidad de reiniciar el programa.

##Interfaz gráfica (Tkinter)

Cargar imágenes: Se puede abrir .pgm y .raw desde un cuadro de diálogo.

Visualización: La imagen cargada se muestra en un Canvas.

Botones de acción: Cada operación está vinculada a un botón de Tkinter.

Reinicio: El botón "Reiniciar" borra todas las imágenes y restaura la ventana.
