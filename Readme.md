Implemenetación del efecto borroso de una imagen utilizando Posix, OpenMP y Cuda.

En cada una de las carpetas se encuentran los archivos necesarios para la ejecución del programa así como un readme con las instrucciones de compilación y ejecución.
Para cada caso, el usuario puede enviar como parámetro la imagen a procesar en cualquier formato y de cualquier tamaño, así como la cantidad de bloques  y de hilos por bloque con la que se desea ejecutar el programa.
De esta forma, puede obtener el efecto borroso en la imagen deseada, así como conocer el tempo de ejecución para cada caso. Para esto, en cada carpeta se agregó además, un script de ejecución del programa, en el que se procesan 3 imágenes de 720, 1080 y 4k, para un kernel de 1 a 15 y con un número de hilos de 2 a 16.
Cada una de esatas ejecuiones se realiza un total de 10 veces, para finalmente tomar el promedio del tiempo de las 10 ieraciones y realizar las respectivas comparaciones.
