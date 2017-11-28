#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <pthread.h>

#define PI 3.141592653589793238462643383
#define e  2.718281828459045235360287471
#define sigma  1.9
#define MAX_THREADS 16

IplImage *result, *img;
int kernel;
float **filtro;
pthread_t th_id[MAX_THREADS];

//Estructura para definir inicio y fin de cada hilo
struct Params{
    int begin;
    int end;  
}parameters[MAX_THREADS];

//Con esta función se aplica la función gausiana a la matriz
// de la imagen y  luego se normaliza
void filtro_matriz(){
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

//Para aplicar el efecto gaussiano se aplica una transformación, de tal forma que cada pixel de la imagen 
//tomé el valor promedio de los pixeles que lo rodean. 

void* blur(void *arguments){
    struct Params *param;
    param = arguments;
    int k = kernel/2;
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
    pthread_exit(NULL);
}

int main(int args, char *argv[]){
    char *imagen = argv[1];
    kernel = atoi(argv[2]);
    int n_threads = atoi(argv[3]);
    if(kernel%2==0){
        printf("kernel must be odd\n");
        return 0;
    }
    img = cvLoadImage(imagen,CV_LOAD_IMAGE_COLOR);
    //Creamos el espacio en memoria de la matriz gaussiana
    filtro = (float**) malloc(kernel * sizeof(float *) );    
    for(int i=0; i< kernel; i++){
        filtro[i] = (float *) malloc(kernel * sizeof(float));
    }
    if(filtro == NULL){
        printf("No memory space");
    }
    filtro_matriz();
    //Clonamos la imagen del resultado
    result =  cvCloneImage(img);
    
    //Creamos los hilos
    for(int i=0; i<n_threads; i++){    
        parameters[i].begin = i * (img->height/n_threads);
        parameters[i].end = parameters[i].begin + (img->height/n_threads)-1;      
    }
    
    for(int i=0; i<n_threads; i++){
        if(pthread_create(&th_id[i], NULL, blur, (void *)&parameters[i])!=0)
            perror("Thread could not be created");        
    }
    
    for(int i=0; i<n_threads; i++){
        if( pthread_join(th_id[i], NULL) != 0)
            perror("Thread could not end");
    }
    //Mostramos la imagen con el filtro
    //cvNamedWindow("Gaussian Blur",CV_WINDOW_NORMAL);
    //cvShowImage("Gaussian Blur", result);
    cvWaitKey(0);   
    //Liberamos memoria de la matriz gaussiana
    for(int i=0; i<kernel;i++){
        float *cur = filtro[i];
        free(cur);
    }
    free(filtro); 
    
    return 0;
}
