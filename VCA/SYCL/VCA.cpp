#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "oneapi/mkl.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <sys/time.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define MAXLINE 200
#define MAXCAD 200
#define EPSILON 1.11e-16


float time_diff(struct timeval *start2, struct timeval *end2)
{
  return (end2->tv_sec - start2->tv_sec) + 1e-6*(end2->tv_usec - start2->tv_usec);
}

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 *
 * Modified and parallelized by: Adrian Real and Oscar Ruiz
 * Centre: Universidad Complutense de Madrid
 * */
void cleanString(char *cadena, char *out)
{
    int i,j;
    for( i = j = 0; cadena[i] != 0;++i)
    {
        if(isalnum(cadena[i])||cadena[i]=='{'||cadena[i]=='.'||cadena[i]==',')
        {
            out[j]=cadena[i];
            j++;
        }
    }
    for( i = j; out[i] != 0;++i)
        out[j]=0;
}

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
int readHeader1(char* filename, int *lines, int *samples, int *bands, int *dataType,
		char* interleave, int *byteOrder, char* waveUnit)
{
    FILE *fp;
    char line[MAXLINE] ="";
    char value [MAXLINE] = "";

    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
        while(fgets(line, MAXLINE, fp)!=NULL)
        {
            //Samples
            if(strstr(line, "samples")!=NULL && samples !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *samples = atoi(value);
            }

            //Lines
            if(strstr(line, "lines")!=NULL && lines !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *lines = atoi(value);
            }

            //Bands
            if(strstr(line, "bands")!=NULL && bands !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *bands = atoi(value);
            }

            //Interleave
            if(strstr(line, "interleave")!=NULL && interleave !=NULL)
            {
                cleanString(strstr(line, "="),value);
                strcpy(interleave,value);
            }

            //Data Type
            if(strstr(line, "data type")!=NULL && dataType !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *dataType = atoi(value);
            }

            //Byte Order
            if(strstr(line, "byte order")!=NULL && byteOrder !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *byteOrder = atoi(value);
            }

            //Wavelength Unit
            if(strstr(line, "wavelength unit")!=NULL && waveUnit !=NULL)
            {
                cleanString(strstr(line, "="),value);
                strcpy(waveUnit,value);
            }

        }
        fclose(fp);
        return 0;
    }
    else
    	return -2; //No file found
}

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
int readHeader2(char* filename, double* wavelength)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char value [MAXLINE] = "";

    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
        while(fgets(line, MAXLINE, fp)!=NULL)
        {
            //Wavelength
            if(strstr(line, "wavelength =")!=NULL && wavelength !=NULL)
            {
                char strAll[100000]=" ";
                char *pch;
                int cont = 0;
                do
                {
                    fgets(line, 200, fp);
                    cleanString(line,value);
                    strcat(strAll,value);
                } while(strstr(line, "}")==NULL);

                pch = strtok(strAll,",");

                while (pch != NULL)
                {
                    wavelength[cont]= atof(pch);
                    pch = strtok (NULL, ",");
                    cont++;
                }
            }

		}
		fclose(fp);
		return 0;
	}
	else
		return -2; //No file found
}


/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
int loadImage(char* filename, double* image, int lines, int samples, int bands, int dataType, char* interleave)
{

    FILE *fp;
    short int *tipo_short_int;
    float *tipo_float;
    double * tipo_double;
    unsigned int *tipo_uint;
    int i, j, k, op;
    long int lines_samples = lines*samples;


    if ((fp=fopen(filename,"rb"))!=NULL)
    {

        fseek(fp,0L,SEEK_SET);
        tipo_float = (float*)malloc(lines_samples*bands*sizeof(float));
        switch(dataType)
        {
            case 2:
                tipo_short_int = (short int*)malloc(lines_samples*bands*sizeof(short int));
                fread(tipo_short_int,1,(sizeof(short int)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                    tipo_float[i]=(float)tipo_short_int[i];
                free(tipo_short_int);
                break;

            case 4:
                fread(tipo_float,1,(sizeof(float)*lines_samples*bands),fp);
                break;

            case 5:
                tipo_double = (double*)malloc(lines_samples*bands*sizeof(double));
                fread(tipo_double,1,(sizeof(double)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                    tipo_float[i]=(float)tipo_double[i];
                free(tipo_double);
                break;

            case 12:
                tipo_uint = (unsigned int*)malloc(lines_samples*bands*sizeof(unsigned int));
                fread(tipo_uint,1,(sizeof(unsigned int)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                    tipo_float[i]=(float)tipo_uint[i];
                free(tipo_uint);
                break;

        }
        fclose(fp);

        if(interleave == NULL)
            op = 0;
        else
        {
            if(strcmp(interleave, "bsq") == 0) op = 0;
            if(strcmp(interleave, "bip") == 0) op = 1;
            if(strcmp(interleave, "bil") == 0) op = 2;
        }


        switch(op)
        {
            case 0:
                for(i=0; i<lines*samples*bands; i++)
                    image[i] = tipo_float[i];
                break;

            case 1:
                for(i=0; i<bands; i++)
                    for(j=0; j<lines*samples; j++)
                        image[i*lines*samples + j] = tipo_float[j*bands + i];
                break;

            case 2:
                for(i=0; i<lines; i++)
                    for(j=0; j<bands; j++)
                        for(k=0; k<samples; k++)
                            image[j*lines*samples + (i*samples+k)] = tipo_float[k+samples*(i*bands+j)];
                break;
        }
        free(tipo_float);
        return 0;
    }
    return -2;
}

/*
 * Author: Luis Ignacio Jimenez Gil
 * Centre: Universidad de Extremadura
 * */
int readValueResults(char* filename)
{
    FILE *fp;
    int in, error;
    int value;
    if ((fp=fopen(filename,"rb"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
        error = fread(&value,1,sizeof(int),fp);
        if(error != sizeof(int)) in = -1;
        else in = (int)value;
        fclose(fp);
    }
    else return -1;

    return in;
}

/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
int writeResult(double *image, const char* filename, int lines, int samples, int bands, int dataType, char* interleave)
{

	short int *imageSI;
	float *imageF;
	double *imageD;

	int i,j,k,op;

	if(interleave == NULL)
		op = 0;
	else
	{
		if(strcmp(interleave, "bsq") == 0) op = 0;
		if(strcmp(interleave, "bip") == 0) op = 1;
		if(strcmp(interleave, "bil") == 0) op = 2;
	}

	if(dataType == 2)
	{
		imageSI = (short int*)malloc(lines*samples*bands*sizeof(short int));

        switch(op)
        {
			case 0:
				for(i=0; i<lines*samples*bands; i++)
					imageSI[i] = (short int)image[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						imageSI[j*bands + i] = (short int)image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageSI[i*bands*samples + (j*samples + k)] = (short int)image[j*lines*samples + (i*samples + k)];
				break;
        }

	}

	if(dataType == 4)
	{
		imageF = (float*)malloc(lines*samples*bands*sizeof(float));
        switch(op)
        {
			case 0:
				for(i=0; i<lines*samples*bands; i++)
					imageF[i] = (float)image[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						imageF[j*bands + i] = (float)image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageF[i*bands*samples + (j*samples + k)] = (float)image[j*lines*samples + (i*samples + k)];
				break;
        }
	}

	if(dataType == 5)
	{
		imageD = (double*)malloc(lines*samples*bands*sizeof(double));
        switch(op)
        {
			case 0:
				for(i=0; i<lines*samples*bands; i++)
					imageD[i] = image[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						imageD[j*bands + i] = image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageD[i*bands*samples + (j*samples + k)] = image[j*lines*samples + (i*samples + k)];
				break;
        }
	}

    FILE *fp;
    if ((fp=fopen(filename,"wb"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
	    switch(dataType)
	    {
	    case 2: fwrite(imageSI,1,(lines*samples*bands * sizeof(short int)),fp); free(imageSI); break;
	    case 4: fwrite(imageF,1,(lines*samples*bands * sizeof(float)),fp); free(imageF); break;
	    case 5: fwrite(imageD,1,(lines*samples*bands * sizeof(double)),fp); free(imageD); break;
	    }
	    fclose(fp);


	    return 0;
    }

    return -3;
}

/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
int writeHeader(char* filename, int lines, int samples, int bands, int dataType,
		char* interleave, int byteOrder, char* waveUnit, double* wavelength)
{
    FILE *fp;
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
		fseek(fp,0L,SEEK_SET);
		fprintf(fp,"ENVI\ndescription = {\nExported from MATLAB}\n");
		if(samples != 0) fprintf(fp,"samples = %d", samples);
		if(lines != 0) fprintf(fp,"\nlines   = %d", lines);
		if(bands != 0) fprintf(fp,"\nbands   = %d", bands);
		if(dataType != 0) fprintf(fp,"\ndata type = %d", dataType);
		if(interleave != NULL) fprintf(fp,"\ninterleave = %s", interleave);
		if(byteOrder != 0) fprintf(fp,"\nbyte order = %d", byteOrder);
		if(waveUnit != NULL) fprintf(fp,"\nwavelength units = %s", waveUnit);
		if(waveUnit != NULL)
		{
			fprintf(fp,"\nwavelength = {\n");
			for(int i=0; i<bands; i++)
			{
				if(i==0) fprintf(fp, "%f", wavelength[i]);
				else
					if(i%3 == 0) fprintf(fp, ", %f\n", wavelength[i]);
					else fprintf(fp, ", %f", wavelength[i]);
			}
			fprintf(fp,"}");
		}
		fclose(fp);
		return 0;
    }
    return -3;
}


/*
 * Author: Luis Ignacio Jimenez Gil
 * Centre: Universidad de Extremadura
 * */
void mean(double* matrix, int rows, int cols, int dim, double* out)
{
	int i,j;

	if(dim == 1)
	{
		for(i=0; i<cols; i++) out[i] = 0;

		for(i=0; i<cols; i++)
			for(j=0; j<rows; j++)
				out[i] += matrix[j*cols + i];

		for(i=0; i<cols; i++) out[i] = out[i]/cols;
	}
	else
	{
		for(i=0; i<rows; i++) out[i] = 0;

		for(i=0; i<rows; i++)
			for(j=0; j<cols; j++)
				out[i] += matrix[i*cols + j];

		for(i=0; i<rows; i++) out[i] = out[i]/rows;

	}
}

/* Siconos-Numerics, Copyright INRIA 2005-2011.
 * Siconos is a program dedicated to modeling, simulation and control
 * of non smooth dynamical systems.
 * Siconos is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * Siconos is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Siconos; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Contact: Vincent ACARY, siconos-team@lists.gforge.inria.fr
*/
/*
void pinv(double * A, int n, int m)
{
	int dimS = MIN(n,m);
	int LDU = n;
	int LDVT = m;
	int i,j;
	double alpha = 1, beta = 0;
	int lwork  = MAX(1,MAX(3*MIN(m, n)+MAX(m,n),5*MIN(m,n))) , info;
	double *work  = (double*)malloc(lwork*sizeof(double));

	double *S = (double*) malloc(dimS * sizeof(double));//eigenvalues
	double *U = (double*) malloc(LDU * n * sizeof(double));//eigenvectors
	double *VT = (double*) malloc(LDVT * m * sizeof(double));//eigenvectors

	dgesvd_("S", "S", &n, &m, A, &n, S, U, &LDU, VT, &LDVT, work, &lwork, &info);

	double maxi = S[0];
	for(i=1; i<dimS; i++)
		if(maxi<S[i]) maxi = S[i];

	double tolerance = EPSILON*MAX(n,m)*maxi;

	int rank = 0;
	for (i=0; i<dimS; i++)
	{
		if (S[i] > tolerance)
		{
			rank ++;
			S[i] = 1.0 / S[i];
		}
	}

	// Compute the pseudo inverse
	// Costly version with full DGEMM
	double * Utranstmp = (double*)malloc(n * m * sizeof(double));
	for (i=0; i<dimS; i++)
		for (j=0;  j<n; j++) Utranstmp[i + j * m] = S[i] * U[j + i * n];

	for (i=dimS;  i<m; i++)
		for (j=0;  j<n; j++) Utranstmp[i + j * m] = 0.0;

	dgemm_("T", "N", &m, &n, &m, &alpha, VT, &m, Utranstmp, &m, &beta, A, &m);

	free(work);
	free(U);
	free(VT);
	free(Utranstmp);
	free(S);
}*/

/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
int main(int argc, char *argv[])//(double *image, int lines, int samples, int bands, int targets, double SNR, double *endmembers)
{

	/*
	 * PARAMETERS
	 * argv[1]: Input image file
	 * argv[2]: number of endmembers to be extracted
	 * argv[3]: Signal noise ratio (SNR)
	 * argv[4]: estimated endmember signatures obtained
	 * */

	if(argc != 5)
	{
		printf("EXECUTION ERROR VCA Iterative: Parameters are not correct.");
		printf("./VCA [Image Filename] [Number of endmembers] [SNR] [Output endmembers]");
		fflush(stdout);
		exit(-1);
	}
	// Input parameters:
	char image_filename[MAXCAD];
	char header_filename[MAXCAD];

	strcpy(image_filename,argv[1]);
	strcpy(header_filename,argv[1]);
	strcat(header_filename,".hdr");

  // Cola
  sycl::queue my_queue{sycl::cpu_selector{}};

  std::cout << "Device: "
            << my_queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;


	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char *interleave, *waveUnit;
	interleave = (char*)malloc(MAXCAD*sizeof(char));
	waveUnit = (char*)malloc(MAXCAD*sizeof(char));

	// Load image
	int error = readHeader1(header_filename, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("\nEXECUTION ERROR VCA Iterative: Error 1 reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double* wavelength = (double*)malloc(bands*sizeof(double));

	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("\nEXECUTION ERROR VCA Iterative: Error 2 reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

  double* image_vector   = sycl::malloc_device<double>(lines*samples*bands, my_queue);
  double* image_vector_h = sycl::malloc_host<double>(lines*samples*bands, my_queue);

	error = loadImage(argv[1],image_vector_h, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("\nEXECUTION ERROR VCA Iterative: Error reading image file: %s.", argv[1]);
		fflush(stdout);
		exit(-1);
	}

  my_queue.submit([&] (sycl::handler& h) {
      h.memcpy(image_vector, image_vector_h, sizeof(double)*lines*samples*bands);
  }).wait();

	//***TARGETS VALUE OR LOAD VALUE****
	int targets;
	if(strstr(argv[2], "/") == NULL)
		targets = atoi(argv[2]);
	else
	{
		targets = readValueResults(argv[2]);
		fflush(stdout);
		if(targets == -1)
		{
			printf("EXECUTION ERROR IEA Iterative: Targets was not set correctly file: %s.", argv[2]);
			fflush(stdout);
			exit(-1);
		}
	}
	//**********************************

	//START CLOCK***************************************
  struct timeval start;
  struct timeval end;
  gettimeofday(&start, NULL);
	//**************************************************
	int i, j, lines_samples = lines*samples;
	double alpha =1, beta = 0;

  double* endmembers   = sycl::malloc_device<double>(targets*bands, my_queue);
  double* r_m          = sycl::malloc_device<double>(bands, my_queue);
  double* R_o          = sycl::malloc_device<double>(bands*lines_samples, my_queue);
  double* svdMat       = sycl::malloc_device<double>(bands*bands, my_queue);
  double* VT           = sycl::malloc_device<double>(bands*bands, my_queue);
  double* D            = sycl::malloc_device<double>(bands, my_queue);
  double* U            = sycl::malloc_device<double>(bands*bands, my_queue);
  double* Ud           = sycl::malloc_device<double>(bands*targets, my_queue);
  double* x_p          = sycl::malloc_device<double>(lines_samples * targets, my_queue);
  double* u            = sycl::malloc_device<double>(targets, my_queue);
  double* Rp           = sycl::malloc_device<double>(bands*lines_samples, my_queue);
  double* y            = sycl::malloc_device<double>(lines_samples * targets, my_queue);
  double* sumxu        = sycl::malloc_device<double>(lines_samples, my_queue);
  double* w            = sycl::malloc_device<double>(targets, my_queue);
  double* A            = sycl::malloc_device<double>(targets*targets, my_queue);
  double* A2           = sycl::malloc_device<double>(targets*targets, my_queue);
  double* aux          = sycl::malloc_device<double>(targets*targets, my_queue);
  double* f            = sycl::malloc_device<double>(targets, my_queue);
  int64_t* index       = sycl::malloc_device<int64_t>(targets, my_queue);
  
  double* x_p_h        = sycl::malloc_host<double>(lines_samples * targets, my_queue);
  double* r_m_h        = sycl::malloc_host<double>(bands, my_queue);
  double* w_h          = sycl::malloc_host<double>(targets, my_queue);
  double* f_h          = sycl::malloc_host<double>(targets, my_queue);
  double* sumxu_h      = sycl::malloc_host<double>(lines_samples, my_queue);
  int64_t* index_h     = sycl::malloc_host<int64_t>(targets, my_queue);
  double* endmembers_h = sycl::malloc_host<double>(targets*bands, my_queue);
  double* u_h          = sycl::malloc_host<double>(targets, my_queue);
  double* A_h          = sycl::malloc_host<double>(targets*targets, my_queue);
  
  // Inicializacion a 0
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(endmembers, 1, targets*bands*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(r_m, 0, bands*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(R_o, 0, lines_samples*bands*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(svdMat, 0, bands*bands*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(VT, 0, bands*bands*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(D, 0, bands*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(U, 0, bands*bands*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(Ud, 0, targets*bands*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(x_p, 0, targets*lines_samples*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(u_h, 0, targets*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(Rp, 0, bands*lines_samples*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(y, 0, targets*lines_samples*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(sumxu, 0, lines_samples*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(w, 0, targets*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(A_h, 0, targets*targets*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(A2, 0, targets*targets*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(aux, 0, targets*targets*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(f, 0, targets*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(index_h, 0, targets*sizeof(int64_t));
  }).wait();
  
  // Declaraciones de SVD  
	int lwork  = MAX(1,MAX(3*MIN(bands, bands)+MAX(bands,bands),5*MIN(bands,bands)));

  int64_t* info        = sycl::malloc_device<int64_t>(1, my_queue);

  double maxi;
  double tolerance;
  int rank;

	double SNR = atof(argv[3]), SNR_es;
	double sum1 = 0.0, sum2 = 0.0, powery, powerx, mult = 0.0;

	// Variables de la funcion pinv
	int lwork_pinv  = MAX(1,MAX(3*MIN(targets, targets)+MAX(targets,targets),5*MIN(targets,targets)));

	double *work_pinv = NULL;

  double* S_pinv       = sycl::malloc_device<double>(targets, my_queue);
  double* U_pinv       = sycl::malloc_device<double>(targets*targets, my_queue);
  double* VT_pinv      = sycl::malloc_device<double>(targets*targets, my_queue);

  double* Utranstmp    = sycl::malloc_device<double>(targets*targets, my_queue);

  my_queue.submit([&] (sycl::handler& h) {
      h.memset(S_pinv, 0, targets*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(U_pinv, 0, targets*targets*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(VT_pinv, 0, targets*targets*sizeof(double));
  });
  my_queue.submit([&] (sycl::handler& h) {
      h.memset(Utranstmp, 0, targets*targets*sizeof(double));
  }).wait();
  
  double* S_pinv_h = sycl::malloc_host<double>(targets, my_queue);

  // Constantes
  auto nonTrans = oneapi::mkl::transpose::nontrans;
  auto yesTrans = oneapi::mkl::transpose::trans;  

  cl::sycl::context context = my_queue.get_context();
  cl::sycl::device device = my_queue.get_device();
  
  my_queue.submit([&](auto &h) {
    h.parallel_for(sycl::range(bands), [=](auto i) {
      int j;
    
  		for(j=0; j<lines_samples; j++)
  			r_m[i] += image_vector[i*lines_samples+j];
     
  		r_m[i] /= lines_samples;

  		for(j=0; j<lines_samples; j++)
  			R_o[i*lines_samples+j] = image_vector[i*lines_samples+j] - r_m[i];
    });
  }).wait();

  try {
    auto event1 = oneapi::mkl::blas::column_major::gemm(my_queue, yesTrans, nonTrans, bands, bands, lines_samples, alpha, R_o, lines_samples, R_o, lines_samples, beta, svdMat, bands);
    event1.wait_and_throw();
  }
  catch(oneapi::mkl::lapack::exception const& e) {
    std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
    return *info;
  }
  
  my_queue.wait();

  my_queue.submit([&](auto &h) {
    h.parallel_for(sycl::range(bands*bands), [=](auto i) {
      svdMat[i] /= lines_samples;
    });
  }).wait();

  try {
    lwork = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(my_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, bands, bands, bands, bands, bands);
    my_queue.wait_and_throw();
  }
  catch(oneapi::mkl::lapack::exception const& e) {
    printf("Error in gesvd_scratchpad_size()...\nExiting...\n");
    return -1;
  }

  double* work = static_cast<double*>(sycl::malloc_device(lwork * sizeof(double), device, context));
  if (lwork != 0 && !work) {
    printf("Error allocating scratchpad in the device memory...\nExiting...\n");
    return -1;
  }
  
  try {
    auto event2 = oneapi::mkl::lapack::gesvd(my_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, bands, bands, svdMat, bands, D, U, bands, VT, bands, work, lwork);
    event2.wait_and_throw();
  }
  catch(oneapi::mkl::lapack::exception const& e) {
    std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\ninfo: " << e.info() << std::endl;
    return *info;
  }
  
  my_queue.wait();

  my_queue.submit([&](auto &h) {
     h.parallel_for(sycl::range<2>(bands,targets), [=](auto index) {
       int i = index[0];
       int j = index[1];
       Ud[i*targets +j] = VT[i*bands +j];
     });
  }).wait();
 
  try {
    auto event3 = oneapi::mkl::blas::column_major::gemm(my_queue, nonTrans, yesTrans, targets, lines_samples, bands, alpha, Ud, targets, R_o, lines_samples, beta, x_p, targets);
    event3.wait_and_throw();
  }
  catch(oneapi::mkl::lapack::exception const& e) {
    std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
    return *info;
  }
  
  my_queue.wait();

  my_queue.submit([&] (sycl::handler& h) {
      h.memcpy(x_p_h, x_p, sizeof(double)*lines_samples*targets);
  }).wait();
  
  // r_m
  my_queue.submit([&] (sycl::handler& h) {
      h.memcpy(r_m_h, r_m, sizeof(double)*bands);
  }).wait();
  
	for(i=0; i<lines_samples*bands; i++)
	{
		sum1 += pow(image_vector_h[i],2);
		if(i < lines_samples*targets) sum2 += pow(x_p_h[i],2);
		if(i < bands) mult += pow(r_m_h[i],2);
	}

	powery = sum1 / lines_samples; 
	powerx = sum2 / lines_samples + mult;
  
	SNR_es = 10 * log10((powerx - targets / bands * powery) / (powery - powerx));
  
	if(SNR == 0) SNR = SNR_es;
	double SNR_th = 15 + 10*log10(targets), c;

  if(SNR < SNR_th)
	{
    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range<2>(bands,targets), [=](auto index) {
        int i = index[0];
        int j = index[1];
        if(j<targets-1) Ud[i*targets + j] = VT[i*bands +j];
        else Ud[i*targets +j] = 0;
      });
    }).wait();
    
		sum1 = 0;
		for(i=0; i<targets; i++)
		{
  			for(j=0; j<lines_samples; j++)
  			{
  				if(i == (targets-1)) x_p_h[i*lines_samples+j] = 0;
  				u_h[i] += pow(x_p_h[i*lines_samples+j], 2);
  			}
  
  			if(sum1 < u_h[i]) sum1 = u_h[i]; 
		}

		c = sqrt(sum1);

    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(x_p, x_p_h, sizeof(double)*lines_samples*targets);
    }).wait();
     
    try {
      auto event4 = oneapi::mkl::blas::column_major::gemm(my_queue, yesTrans, nonTrans, bands, lines_samples, targets, alpha, Ud, targets, x_p, targets, beta, Rp, bands);
      event4.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
    
    my_queue.wait();
    
    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range(bands), [=](auto i) {
        int j;
        
  			for(j=0; j<lines_samples; j++)
  				Rp[i*lines_samples+j] += r_m[i];
      });
    }).wait();

    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range<2>(targets,lines_samples), [=](auto index) {
        int i = index[0];
        int j = index[1];
				if(i<targets-1) y[i*lines_samples+j] = x_p[i*lines_samples+j];
				else y[i*lines_samples+j] = c;
      });
    }).wait();
	}
	else
	{
    try {
      auto event5 = oneapi::mkl::blas::column_major::gemm(my_queue, yesTrans, nonTrans, bands, bands, lines_samples, alpha, image_vector, lines_samples, image_vector, lines_samples, beta, svdMat, bands);
      event5.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
    
    my_queue.wait();
        
    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range(bands*bands), [=](auto i) {
        svdMat[i] /= lines_samples;
      });
    }).wait();

    try {
      auto event6 = oneapi::mkl::lapack::gesvd(my_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, bands, bands, svdMat, bands, D, U, bands, VT, bands, work, lwork);
      event6.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
  
    my_queue.wait();
      
    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range<2>(bands,targets), [=](auto index) {
        int i = index[0];
        int j = index[1];
        Ud[i*targets + j] = VT[i*bands + j];
      });
    }).wait();

    try {
      auto event7 = oneapi::mkl::blas::column_major::gemm(my_queue, nonTrans, yesTrans, targets, lines_samples, bands, alpha, Ud, targets, image_vector, lines_samples, beta, x_p, targets);
      event7.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
    
    my_queue.wait();

    try {
      auto event8 = oneapi::mkl::blas::column_major::gemm(my_queue, yesTrans, nonTrans, bands, lines_samples, targets, alpha, Ud, targets, x_p, targets, beta, Rp, bands);
      event8.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
    
    my_queue.wait();

    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(u, u_h, sizeof(double)*targets);
    }).wait();

    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range(targets), [=](auto i) {
        int j;

  			for(j=0; j<lines_samples; j++)
  				u[i] += x_p[i*lines_samples+j];

  			for(j=0; j<lines_samples; j++)
  				y[i*lines_samples+j] = x_p[i*lines_samples+j] * u[i];
      });
    }).wait();

    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range(lines_samples), [=](auto i) {
        int j;
        
  			for(j=0; j<targets; j++)
  				sumxu[i] += y[j*lines_samples+i];
      });
    }).wait();

    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range(targets), [=](auto i) {
        int j;
        
  			for(j=0; j<lines_samples; j++)
  				y[i*lines_samples+j] /= sumxu[j];
      });
    }).wait();

	}
	int lmax, one;

  {
	  lmax = 2147483647; //INT_MAX;
	  one = 1;
  }  

  A_h[(targets-1)*targets] = 1; 

  my_queue.submit([&] (sycl::handler& h) {
      h.memcpy(A, A_h, sizeof(double)*targets*targets);
  }).wait();

	for(i=0; i<targets; i++)
	{

    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range(targets), [=](auto j) {
      	w[j] = 16000 % lmax; // Cambiamos el valor rand() por un valor fijo 16000
      	w[j] /= lmax;
      });
    }).wait();

    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range(targets*targets), [=](auto j) {
        A2[j] = A[j];
      });
    }).wait();

		/*
		*
		*
		*
		* Implementacion directa de PINV
		*
		*
		*
		*/
   
		int i_pinv,j_pinv;

    try {
      lwork_pinv = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(my_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, targets, targets, targets, targets, targets);
      my_queue.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      printf("Error in gesvd_scratchpad_size()...\nExiting...\n");
      return -1;
    }

    double* work_pinv = static_cast<double*>(sycl::malloc_device(lwork_pinv * sizeof(double), device, context));
    if (lwork_pinv != 0 && !work_pinv) {
      printf("Error allocating scratchpad in the device memory...\nExiting...\n");
      return -1;
    }
  
    try {
      auto event9 = oneapi::mkl::lapack::gesvd(my_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, targets, targets, A2, targets, S_pinv, U_pinv, targets, VT_pinv, targets, work_pinv, lwork_pinv);
      event9.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
    my_queue.wait();

    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(S_pinv_h, S_pinv, sizeof(double)*targets);
    }).wait();
  
    maxi = S_pinv_h[0];
  
		for(i_pinv=1; i_pinv<targets; i_pinv++)
    {
			if(maxi<S_pinv_h[i_pinv]) maxi = S_pinv_h[i_pinv];
    }

    tolerance = EPSILON*MAX(targets,targets)*maxi;
    rank = 0;		

		for (i_pinv=0; i_pinv<targets; i_pinv++)
		{
			if (S_pinv_h[i_pinv] > tolerance)
			{
				rank += 1;
				S_pinv_h[i_pinv] = 1.0 / S_pinv_h[i_pinv];
			}
		}

    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(S_pinv, S_pinv_h, sizeof(double)*targets);
    }).wait();

    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range<2>(targets,targets), [=](auto index) {
        int i_pinv = index[0];
        int j_pinv = index[1];
        Utranstmp[i_pinv + j_pinv * targets] = S_pinv[i_pinv] * U_pinv[j_pinv + i_pinv * targets];
      });
    }).wait();

    try {
      auto event10 = oneapi::mkl::blas::column_major::gemm(my_queue, yesTrans, nonTrans, targets, targets, targets, alpha, VT_pinv, targets, Utranstmp, targets, beta, A2, targets);
      event10.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
    
    my_queue.wait();

    try {
      auto event11 = oneapi::mkl::blas::column_major::gemm(my_queue, nonTrans, nonTrans, targets, targets, targets, alpha, A2, targets, A, targets, beta, aux, targets);
      event11.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
    my_queue.wait();

    try {
      auto event12 = oneapi::mkl::blas::column_major::gemm(my_queue, nonTrans, nonTrans, targets, one, targets, alpha, aux, targets, w, targets, beta, f, targets);
      event12.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
    
    my_queue.wait();

    sum1 = 0;

    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(w_h, w, sizeof(double)*targets);
    }).wait();
    
    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(f_h, f, sizeof(double)*targets);
    }).wait();

    for(j=0; j<targets; j++)
    {
    	f_h[j] = w_h[j] - f_h[j];
    	sum1 += pow(f_h[j],2);
    }

    for(j=0; j<targets; j++) f_h[j] /= sqrt(sum1); 

    // f to device
    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(f, f_h, sizeof(double)*targets);
    }).wait();

    try {
      auto event13 = oneapi::mkl::blas::column_major::gemm(my_queue, nonTrans, yesTrans, one, lines_samples, targets, alpha, f, one, y, lines_samples, beta, sumxu, one);
      event13.wait_and_throw();
    }
    catch(oneapi::mkl::lapack::exception const& e) {
      std::cout << "Unexpected exception caught during synchronous call to BLAS API:\ninfo: " << e.info() << std::endl;
      return *info;
    }
    my_queue.wait();
    sum2 = 0;
    
    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(sumxu_h, sumxu, sizeof(double)*lines_samples);
    }).wait();
    
    for(j=0; j<lines_samples; j++)
    {
    	if(sumxu_h[j] < 0) sumxu_h[j] *= -1;
    	if(sum2 < sumxu_h[j])
    	{
    		sum2 = sumxu_h[j];
    		index_h[i] = j;
    	}
    }

    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(index, index_h, sizeof(int64_t)*targets);
    }).wait();
     
    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range(targets), [=](auto j) {
        A[j*targets + i] = y[j*lines_samples+index[i]];
      });
    }).wait();

    my_queue.submit([&](auto &h) {
      h.parallel_for(sycl::range(bands), [=](auto j) {
        endmembers[j*targets+ i] = Rp[j+bands * index[i]];
      });
    }).wait();
    
    // endmembers to host
    my_queue.submit([&] (sycl::handler& h) {
        h.memcpy(endmembers_h, endmembers, sizeof(double)*targets*bands);
    }).wait();

  }

	//END CLOCK*****************************************
  gettimeofday(&end, NULL);
  printf("Time spent: %0.8f sec\n",time_diff(&start, &end));
	fflush(stdout);
	//**************************************************
	strcpy(image_filename, argv[4]);
	strcpy(header_filename, image_filename);
	strcat(header_filename, ".hdr");
	error = writeHeader(header_filename, targets, 1, bands, dataType, interleave, byteOrder, waveUnit, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR VCA SYCL: Error writing header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	error = writeResult(endmembers_h, argv[4], targets, 1, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR VCA SYCL: Error writing image file: %s.", argv[3]);
		fflush(stdout);
		exit(-1);
	}

	sycl::free(image_vector, context);
	sycl::free(endmembers, context);
	sycl::free(sumxu, context);
	sycl::free(aux, context);
	sycl::free(f, context);
	sycl::free(A, context);
	sycl::free(A2, context);
	sycl::free(w, context);
	sycl::free(index, context);
	sycl::free(y, context);
	sycl::free(u, context);
	sycl::free(Rp, context);
	sycl::free(x_p, context);
	sycl::free(R_o, context);
	sycl::free(svdMat, context);
	sycl::free(D, context);
	sycl::free(U, context);
	sycl::free(VT, context);
	sycl::free(r_m, context);
	sycl::free(Ud, context);
	sycl::free(U_pinv, context);
	sycl::free(VT_pinv, context);
	sycl::free(Utranstmp, context);
	sycl::free(S_pinv, context);
  sycl::free(info, context);
  sycl::free(S_pinv_h, context);
  sycl::free(x_p_h, context);
  sycl::free(r_m_h, context);
  sycl::free(w_h, context);
  sycl::free(f_h, context);
  sycl::free(sumxu_h, context);
  sycl::free(index_h, context);
  sycl::free(endmembers_h, context);
  sycl::free(u_h, context);
  sycl::free(A_h, context);
  sycl::free(image_vector_h, context);

	return 0;
}
