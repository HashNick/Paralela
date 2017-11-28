Practica 3

para compilar solo es necesario usar la siguiente linea en la linea de comandos

nvcc `pkg-config --cflags opencv` -o `basename blur-effect.cu .cu` blur-effect.cu `pkg-config --libs opencv` -lm'

para correr el programa con todas las combinaciones posibles y compilarlo

python script_ejecutar_todo.py

si se quiere solo ejecutar para una combinacion especifica seria

./blur-effect "nombre de la imagen de entrada" "nombre de la imagen de salida" #kernel #hilos #bloques

Ej: ./blur-effect 720.jpg salida.jpg 3 16 10
