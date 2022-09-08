#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>

#include <openacc.h>

void ISRA(double *image_vector, double *endmembers, double *abundanceVector, int it, int lines, int samples, int bands, int targets)
{
  	double *h_Num;
  	double *h_aux;
  	double *h_Den;
  	int lines_samples = lines*samples;
    int i,j;
    
    h_Num = (double*) malloc(lines_samples * targets * sizeof(double));
	  h_aux = (double*) malloc(lines_samples * bands * sizeof(double));
	  h_Den = (double*) malloc(lines_samples * targets * sizeof(double));

  	for(i=0; i<lines_samples*targets; i++)
  		abundanceVector[i]=1;

  	double alpha = 1, beta = 0;

    dgemm_("N", "T", &lines_samples, &targets, &bands, &alpha, image_vector, &lines_samples, endmembers, &targets, &beta, h_Num, &lines_samples);
    
    for(i=0; i<it; i++)
    {

        dgemm_("N", "N", &lines_samples, &bands, &targets, &alpha, abundanceVector, &lines_samples, endmembers, &targets, &beta, h_aux, &lines_samples);
        
        dgemm_("N", "T", &lines_samples, &targets, &bands, &alpha, h_aux, &lines_samples, endmembers, &targets, &beta, h_Den, &lines_samples);
        
        #pragma acc parallel loop
        for(j=0; j<lines_samples*targets; j++)
            abundanceVector[j]=abundanceVector[j]*(h_Num[j]/h_Den[j]);
    }

    free(h_Den);
  	free(h_Num);
  	free(h_aux);
}
