/*tomado de https://medium.com/@jsdario/cuda-opencv-en-c-8d1020a04fbb*/
#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>
// Declaramos la funcion que invoca al kernel
void rgba_to_grey(uchar4 * const d_rgbaImage,
                  unsigned char* const d_greyImage, 
                  size_t numRows, size_t numCols, int kernel, int n_threads, int n_blocks);
// Incluye las definiciones del fichero de mas arriba
#include "preprocess.cpp"
int main(int argc, char **argv) {
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;
  std::string input_file;
  std::string output_file;
  int n_threads, n_blocks, kernel;
switch (argc)
  {
 case 5:
   input_file = std::string(argv[1]);
   kernel = atoi(argv[2]);
   n_threads = atoi(argv[3]);
   n_blocks = atoi(argv[4]);
   output_file = "output.png";
   break;
 default:
      std::cerr << "Usage: ./to_bw input_file [output_filename]" << std::endl;
      exit(1);
  }
  // Carga la imagen y nos prepara los punteros para la entrada y  
  // salida de datos
  preProcess(&h_rgbaImage, &h_greyImage, 
        &d_rgbaImage, &d_greyImage, input_file);
  // Invoca al kernel
  rgba_to_grey(d_rgbaImage, d_greyImage, numRows(), numCols(), n_blocks, n_threads, kernel);
  size_t numPixels = numRows()*numCols();
  checkCudaErrors( 
    cudaMemcpy(h_greyImage, d_greyImage, 
      sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
/* Saca la imagen en escala de grises */
  cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)h_greyImage);
  // Abre la ventana
  cv::namedWindow("to_bw");
  // Pasa la imagen a la ventana anterior
  cv::imshow("to_bw", output);
  cvWaitKey (0);
  cvDestroyWindow ("to_bw");
  // Imprime a fichero
  cv::imwrite(output_file.c_str(), output);
/* Libera memoria*/
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
return 0;
}