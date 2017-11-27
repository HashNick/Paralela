#include "utils.h"
#include <stdio.h>
#include <math.h>       /* ceil */


#define PI 3.141592653589793238462643383
#define e  2.718281828459045235360287471
#define sigma  1.9

float **filtro;

void filtro_matriz(int kernel){
  float sum = 0, v=sigma*sigma;
    for(int i=0; i < kernel; i++){        
        for(int j=0; j < kernel; j++){ 

           filtro[i][j] = (1/(2*PI*v)) * pow(e, -(pow(floor(kernel/2),2)+ pow(floor(kernel/2),2)))/(2*v);    
           sum += filtro[i][j];   
        }        
    }
    
    for(int i=0; i<kernel; i++){
        for(int j=0; j<kernel; j++){        
           filtro[i][j] /= sum;
        }
    }
}

__global__
void rgba_to_grey_kernel(
const uchar4* const rgbaImage, 
unsigned char* const greyImage, 
int numRows, int numCols, int kernel, int n_blocks/*, float **filtro*/) {
// El mapeo de los componentes uchar4 aRGBA es:
// .x -> R ; .y -> G ; .z -> B ; .w -> A
//La salida debe ser resultado de aplicar la siguiente formula //resultado = .299f * R + .587f * G + .114f * B;
//Nota: Ignoramos el canal alfa
  long long int total_px = numRows * numCols;  // total pixels
  int tam = total_px / n_blocks;
  int dimension = tam / blockDim.x;
  int k = kernel/2;            
  int i, pos_x, pos_y, pixel;
  uchar4 s, p;
  double blue, red, green;
  for(int j = 0, fila, columna; j < dimension; j++){/*
    blue = 0.0;
    red = 0.0;
    green = 0.0;
    i = ( blockIdx.x * tam ) + (threadIdx.x * dimension) + j;
    fila = i / numCols;
    columna = i % numCols;
    for(int x=-k; x<=k; x++){
      for(int y=-k; y<=k; y++){

        //multiplicar lo del filtro 
        if(fila+x<0) pos_x = fila+x*-1;
        else if(fila+x>=numRows) pos_x = fila-x;
        else pos_x = fila+x;
        if(columna+y<0) pos_y = columna+y*-1;
        else if(columna+y>=numCols) pos_y = columna-y;
        else pos_y = columna+y;

        pixel = (pos_x*numCols) + pos_y;
        s = rgbaImage[ pixel ]; // pixel que procesa este hilo
        blue += s.x * filtro[y+k][x+k];
        green += s.y * filtro[y+k][x+k];
        red += s.z * filtro[y+k][x+k];
      }
    }*/
    /*
    //Asignacion del pixel despues del blur
    p = rgbaImage[i];
    greyImage[i] = blue * p.x +
                   green * p.y +
                   red * p.z;
                   */
/*
    struct Params *param;
    param = arguments;
    //Recorremos la matriz de la imagen original
    for(int i=param->begin; i<param->end; i++){       
        for (int j=0; j<img->width; j++){
            CvScalar p, s;           
            double blue=0.0, red=0.0, green=0.0;
            for(int x=-k; x<=k; x++){
                for(int y=-k; y<=k; y++){             
                    int pos_x, pos_y;
                    if(i+x<0) pos_x = i+x*-1;
                    else if(i+x>=img->height) pos_x = i-x;
                    else pos_x = i+x;
                    if(j+y<0) pos_y = j+y*-1;
                    else if(j+y>=img->width) pos_y = j-y;
                    else pos_y = j+y;       



                    //Obtenemos la posicion del pixel                                                            
                    s = cvGet2D(img,pos_x,pos_y);                    
                    blue += s.val[0]*filtro[y+k][x+k];
                    green += s.val[1]*filtro[y+k][x+k];
                    red += s.val[2]*filtro[y+k][x+k];
                }      
            }
            //Modificamos la posicion del pixel de la imagen clonada
            p = cvGet2D(result,i,j);
            p.val[0] = blue;
            p.val[1] = green;
            p.val[2] = red;
            
            cvSet2D(result,i,j,p);
        }   
}
    }*/
    if( threadIdx.x % 15 != 0 ){
      int i = ( blockIdx.x * tam ) + (threadIdx.x * dimension) + j;
      uchar4 px = rgbaImage[i]; // pixel que procesa este hilo
      greyImage[i] = .299f * px.x +
                     .587f * px.y +
                     .114f * px.z;
    }
  }
}
void rgba_to_grey(uchar4 * const d_rgbaImage, unsigned char* const d_greyImage, size_t numRows, size_t numCols, int n_blocks, int n_threads, int kernel)
{
  // Dado que no importa la posicion relativa de los pixels
  // en este algoritmo, la estrategia para asignar hilos a
  // bloques y rejillas sera sencillamente la de cubrir
  // a todos los pixeles con hebras en el eje X
  //long int grids_n = ceil(total_px / TxB); // grids numer
/*  const dim3 blockSize(TxB, 1, 1);
  const dim3 gridSize(grids_n, 1, 1);*/
  int gridSize = n_blocks;
  int blockSize = n_threads;/*
  filtro = (float**) malloc(kernel * sizeof(float *) ); 
  if(filtro == NULL){
    printf("No memory space");
  }
  filtro_matriz( kernel );*/
  rgba_to_grey_kernel<<<gridSize, blockSize>>>(
                      d_rgbaImage, d_greyImage, numRows, numCols, kernel, n_blocks/*, filtro*/);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
}