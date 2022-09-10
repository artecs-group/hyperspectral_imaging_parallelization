#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <openacc.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <assert.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define MAXLINE 200
#define MAXCAD 200
#define EPSILON 1.11e-16

#ifdef __cplusplus
extern "C" {
#endif
 
int dgemm_(char *transa, char *transb, int *m, int *
		n, int *k, double *alpha, double *a, int *lda,
		double *b, int *ldb, double *beta, double *c, int
		*ldc);
   
#ifdef __cplusplus
}
#endif


void dgesvd_( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info );

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

	double *image_vector = (double *) malloc (sizeof(double)*(lines*samples*bands));

	error = loadImage(argv[1],image_vector, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("\nEXECUTION ERROR VCA Iterative: Error reading image file: %s.", argv[1]);
		fflush(stdout);
		exit(-1);
	}

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

	double *Ud = (double*) calloc(bands * targets , sizeof(double));
	double *x_p = (double*) calloc(lines_samples * targets , sizeof(double));
	double *y = (double*) calloc(lines_samples * targets , sizeof(double));
	double *R_o = (double*)calloc(bands*lines_samples,sizeof(double));
	double *r_m = (double*)calloc(bands,sizeof(double));
	double *svdMat = (double*)calloc(bands*bands,sizeof(double));
	double *D = (double*) calloc(bands , sizeof(double));
	double *U = (double*) calloc(bands * bands , sizeof(double));
	double *VT = (double*) calloc(bands * bands , sizeof(double));
	double *endmembers = (double*) calloc(targets * bands , sizeof(double));
	double *Rp = (double*)calloc(bands*lines_samples,sizeof(double));
	double *u = (double*)calloc(targets,sizeof(double));
	double *sumxu = (double*)calloc(lines_samples,sizeof(double));
	int* index = (int*)calloc(targets,sizeof(int));
	double *w = (double*)calloc(targets,sizeof(double));
	double *A = (double*)calloc(targets*targets,sizeof(double));
	double *A2 = (double*)calloc(targets*targets,sizeof(double));
	double *aux = (double*)calloc(targets*targets,sizeof(double));
	double *f = (double*)calloc(targets,sizeof(double));
    
  	// Stream CUDA
  	cudaStream_t stream;
  	cudaStreamCreate(&stream);
  
  	// Handler cuBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
 
	// Status cuSOLVER
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  
	// Handler cuSOLVER
	cusolverDnHandle_t cusolverHandle = NULL;
	cusolver_status = cusolverDnCreate(&cusolverHandle);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  
	// Stream independiente para cuSOLVER?
	// Stream-Link cuSOLVER v2
	cudaStream_t streamCusolver;
	cudaStreamCreate(&streamCusolver);
	cusolver_status = cusolverDnSetStream(cusolverHandle, streamCusolver);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  
	// Declaraciones de SVD
	int lwork  = MAX(1,MAX(3*MIN(bands, bands)+MAX(bands,bands),5*MIN(bands,bands)));
	int *info;
	double *work = NULL;
	
	double maxi;
	double tolerance;
	int rank;
 
	//rwork
	double *rwork  = (double*)malloc(lwork*sizeof(double));
	
	// Control de errores de cuBLAS
	int cublas_error;

	double SNR = atof(argv[3]), SNR_es;
	double sum1, sum2, powery, powerx, mult = 0;

	// Variables de la funicon pinv
 
	int lwork_pinv  = MAX(1,MAX(3*MIN(targets, targets)+MAX(targets,targets),5*MIN(targets,targets)));
	int *info_pinv;
	double *work_pinv = NULL;
	
	double *rwork_pinv = (double*)calloc(lwork_pinv,sizeof(double));
	double *S_pinv = (double*) calloc(targets,sizeof(double)); 
	double *U_pinv = (double*) calloc(targets * targets,sizeof(double)); 
	double *VT_pinv = (double*) calloc(targets * targets,sizeof(double)); 
	double * Utranstmp = (double*)calloc(targets * targets,sizeof(double)); 

  int limit = sizeof(int);

	//Inicializacion endmembers
	for(i=0; i<bands*targets; i++)
	endmembers[i]=1;


#pragma acc data copy(endmembers[0:targets * bands]) \
            	 copyin(image_vector[0:(lines*samples*bands)], r_m[0:bands], R_o[0:bands*lines_samples], \
						svdMat[0:bands*bands], VT[0:bands * bands],D[0:bands],U[0:bands * bands],	Ud[0:bands * targets], \
						x_p[0:lines_samples * targets], u[0:targets], Rp[0:bands*lines_samples], \
						y[0:lines_samples * targets], sumxu[0:lines_samples], w[0:targets], A[0:targets*targets], \
						A2[0:targets*targets],aux[0:targets*targets], f[0:targets], index[0:targets],  \
						S_pinv[0:targets], U_pinv[0:targets * targets], VT_pinv[0:targets * targets], \
						Utranstmp[0:targets * targets], info[0:limit], rwork[0:lwork],rwork_pinv[0:lwork_pinv], info_pinv[0:limit],  \
           				SNR, powery, powerx, mult,maxi,sum1,sum2, tolerance,rank) 
{
        
	#pragma acc parallel loop
	for(i=0; i<bands; i++)
	{
	  #pragma acc loop seq 
		for(j=0; j<lines_samples; j++)
			r_m[i] += image_vector[i*lines_samples+j];
   
		r_m[i] /= lines_samples;
  
		#pragma acc loop
		for(j=0; j<lines_samples; j++)
			R_o[i*lines_samples+j] = image_vector[i*lines_samples+j] - r_m[i];
  	}
    
	#pragma acc host_data use_device(R_o,svdMat)
	{
		cublas_error = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, bands, bands, lines_samples, &alpha, R_o, lines_samples, R_o, lines_samples, &beta, svdMat, bands);
		
		if( cublas_error != CUBLAS_STATUS_SUCCESS )
		{
		printf( "failed cuBLAS execution %d\n", cublas_error );
		exit(1);
		}   
		
	}
  
	cublasGetStream(handle, &stream);
	cudaStreamSynchronize(stream);
	
	#pragma acc parallel loop
	for(i=0; i<bands*bands; i++) svdMat[i] /= lines_samples;

	cusolver_status = cusolverDnDgesvd_bufferSize(
						cusolverHandle,
						bands,
						bands,
						&lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaMalloc((void**)&work , sizeof(double)*lwork);
  
	#pragma acc host_data use_device(svdMat,VT,D,U, info, rwork)
	{
		cusolver_status = cusolverDnDgesvd(cusolverHandle,'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, work, lwork, rwork, info);
		
		if( cusolver_status != CUSOLVER_STATUS_SUCCESS )
		{
		printf( "failed cuSOLVER 1 execution %d\n", cusolver_status );
		exit(1);
		}
	}
    
	#pragma acc parallel loop
	for(i=0; i<bands; i++){
		  #pragma acc loop
		  for(j=0; j<targets; j++)
			  Ud[i*targets +j] = VT[i*bands +j];
  	}
    
	#pragma acc host_data use_device(R_o,Ud,x_p)
	{
		cublas_error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, targets, lines_samples, bands, &alpha, Ud, targets, R_o, lines_samples, &beta, x_p, targets);
		
		if( cublas_error != CUBLAS_STATUS_SUCCESS )
		{
		printf( "failed cuBLAS execution %d\n", cublas_error );
		exit(1);
		}   
	}

	cublasGetStream(handle, &stream);
	cudaStreamSynchronize(stream);
  
 
   	sum1 =0;
   	sum2 = 0;
  	mult = 0;
 
	#pragma acc parallel loop reduction(+:sum1, sum2, mult)
	for(i=0; i<lines_samples*bands; i++)
	{
		sum1 += pow(image_vector[i],2);
		if(i < lines_samples*targets) sum2 += pow(x_p[i],2);
		if(i < bands) mult += pow(r_m[i],2);
	}

	powery = sum1 / lines_samples; 
	powerx = sum2 / lines_samples + mult;
  
	SNR_es = 10 * log10((powerx - targets / bands * powery) / (powery - powerx));
  
	if(SNR == 0) SNR = SNR_es;
	double SNR_th = 15 + 10*log10(targets), c;
 
  	if(SNR < SNR_th)
	{
		#pragma acc parallel loop 
		for(i=0; i<bands; i++) 
		{
			#pragma acc loop
			for(j=0; j<targets; j++)
				if(j<targets-1) Ud[i*targets + j] = VT[i*bands +j];
				else Ud[i*targets +j] = 0;
		} 
		
		sum1 = 0;
		
			
		#pragma acc parallel loop
		for(i=0; i<targets; i++)
		{
			#pragma acc loop 
			for(j=0; j<lines_samples; j++)
			{
				if(i == (targets-1)) x_p[i*lines_samples+j] = 0;
				u[i] += pow(x_p[i*lines_samples+j], 2);
			}

			if(sum1 < u[i]) sum1 = u[i];
		}

		c = sqrt(sum1);
		
		#pragma acc host_data use_device(Ud,x_p,Rp)
		{
		
			cublas_error = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, bands, lines_samples, targets, &alpha, Ud, targets, x_p, targets, &beta, Rp, bands);
		
			if( cublas_error != CUBLAS_STATUS_SUCCESS )
			{
				printf( "failed cuBLAS execution %d\n", cublas_error );
				exit(1);
			}  
		}    
		
		cublasGetStream(handle, &stream);
		cudaStreamSynchronize(stream);
	
		#pragma acc parallel loop
		for(i=0; i<bands; i++) 
		{
			#pragma acc loop
			for(j=0; j<lines_samples; j++)
				Rp[i*lines_samples+j] += r_m[i];
		}
	
		#pragma acc parallel loop
		for(i=0; i<targets; i++) 
		{
			#pragma acc loop
			for(j=0; j<lines_samples; j++)
				if(i<targets-1) y[i*lines_samples+j] = x_p[i*lines_samples+j];
				else y[i*lines_samples+j] = c;
		}
	}
	else
	{
		#pragma acc host_data use_device(image_vector,svdMat)
		{
		cublas_error = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, bands, bands, lines_samples, &alpha, image_vector, lines_samples, image_vector, lines_samples, &beta, svdMat, bands);
		
		if( cublas_error != CUBLAS_STATUS_SUCCESS )
		{
			printf( "failed cuBLAS execution %d\n", cublas_error );
			exit(1);
		}  
		}
		
		cublasGetStream(handle, &stream);
		cudaStreamSynchronize(stream);	
			
			#pragma acc parallel loop
			for(i=0; i<bands*bands; i++) svdMat[i] /= lines_samples;

		#pragma acc host_data use_device(svdMat,VT,U,D, info, rwork)
		{
		cusolver_status = cusolverDnDgesvd(cusolverHandle,'S','S', bands, bands, svdMat, bands, D, U, bands, VT, bands, work, lwork, rwork, info);

		if( cusolver_status != CUSOLVER_STATUS_SUCCESS )
		{
			printf( "failed cuSOLVER 2 execution %d\n", cusolver_status );
			exit(1);
		}
		}
		
			#pragma acc parallel loop
			for(i=0; i<bands; i++) {
				#pragma acc loop
				for(j=0; j<targets; j++)
					Ud[i*targets +j] = VT[i*bands +j];
		}
		
		#pragma acc host_data use_device(image_vector,Ud,x_p)
		{
			cublas_error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, targets, lines_samples,bands, &alpha, Ud, targets, image_vector, lines_samples, &beta, x_p, targets);
		
		if( cublas_error != CUBLAS_STATUS_SUCCESS )
		{
			printf( "failed cuBLAS execution %d\n", cublas_error );
			exit(1);
		}  
		}
		
		cublasGetStream(handle, &stream);
		cudaStreamSynchronize(stream);
		
		#pragma acc host_data use_device(Ud,x_p,Rp)
			{
		cublas_error = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, bands, lines_samples, targets, &alpha, Ud, targets, x_p, targets, &beta, Rp, bands);
		
		if( cublas_error != CUBLAS_STATUS_SUCCESS )
		{
			printf( "failed cuBLAS execution %d\n", cublas_error );
			exit(1);
		}  
		}

		cublasGetStream(handle, &stream);
		cudaStreamSynchronize(stream);

		#pragma acc parallel loop
		for(i=0; i<targets; i++)
		{
			#pragma acc loop seq
			for(j=0; j<lines_samples; j++)
				u[i] += x_p[i*lines_samples+j];

			#pragma acc loop
			for(j=0; j<lines_samples; j++)
				y[i*lines_samples+j] = x_p[i*lines_samples+j] * u[i];
		}

		#pragma acc parallel loop
		for(i=0; i<lines_samples; i++) {
			#pragma acc loop
			for(j=0; j<targets; j++)
				sumxu[i] += y[j*lines_samples+i];
		}
	
		#pragma acc parallel loop
		for(i=0; i<targets; i++) {
 
			for(j=0; j<lines_samples; j++)
				y[i*lines_samples+j] /= sumxu[j];
		}
	}
 
	//srand(time(NULL));
	int lmax, one;

	lmax = 2147483647; //INT_MAX;
	one = 1;  
  
	#pragma acc kernels
	A[(targets-1)*targets] = 1;
  
	for(i=0; i<targets; i++)
	{
		#pragma acc parallel loop
		for(j=0; j<targets; j++)
		{ 
			w[j] = 16000 % lmax; // Cambiamos el valor rand() por un valor fijo 16000
			w[j] /= lmax;
		}
    
		#pragma acc parallel loop
		for(j=0; j<targets*targets; j++) A2[j] = A[j];
    
		//pinv(A2, targets, targets);
		//void pinv(double * A, int n, int m)
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

		cusolver_status = cusolverDnDgesvd_bufferSize(
						cusolverHandle,
						targets,
						targets,
						&lwork_pinv);
		assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
		cudaMalloc((void**)&work_pinv , sizeof(double)*lwork_pinv);
        
    
		#pragma acc host_data use_device(A2,S_pinv,U_pinv,VT_pinv, info_pinv, rwork_pinv)
		{

		cusolver_status = cusolverDnDgesvd(cusolverHandle,'S','S', targets, targets, A2, targets, S_pinv, U_pinv, targets, VT_pinv, targets, work_pinv, lwork_pinv, rwork_pinv, info_pinv);
		
		if( cusolver_status != CUSOLVER_STATUS_SUCCESS )
		{
			printf( "failed cuSOLVER 3 execution %d\n", cusolver_status );
			exit(1);
		}
		}
  
    
   
      	maxi = S_pinv[0];
  	
    
		#pragma acc kernels
		#pragma acc loop seq
			for(i_pinv=1; i_pinv<targets; i_pinv++)
		{
			if(maxi<S_pinv[i_pinv]) maxi = S_pinv[i_pinv];
		}
    

    
		#pragma acc kernels
		{		
		tolerance = EPSILON*MAX(targets,targets)*maxi;
		rank = 0;
		}  

    	#pragma acc parallel loop reduction(+:rank)
		for (i_pinv=0; i_pinv<targets; i_pinv++)
		{
			if (S_pinv[i_pinv] > tolerance)
			{
				rank += 1;
				S_pinv[i_pinv] = 1.0 / S_pinv[i_pinv];
			}
		}
            
		#pragma acc parallel loop
		for (i_pinv=0; i_pinv<targets; i_pinv++) {
			#pragma acc loop
			for (j_pinv=0; j_pinv<targets; j_pinv++)
				Utranstmp[i_pinv + j_pinv * targets] = S_pinv[i_pinv] * U_pinv[j_pinv + i_pinv * targets];
		} 

		#pragma acc host_data use_device(A2,VT_pinv,Utranstmp)
		{

			cublas_error = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,targets, targets, targets, &alpha, VT_pinv, targets, Utranstmp, targets, &beta, A2, targets);
			
			if( cublas_error != CUBLAS_STATUS_SUCCESS )
			{
			printf( "failed cuBLAS execution %d\n", cublas_error );
			exit(1);
			} 
		}
    
		cublasGetStream(handle, &stream);
		cudaStreamSynchronize(stream);
		/*
		*
		*
		*
		*
		*
		*
		*
		*/
    
		#pragma acc host_data use_device(A2,A,aux)
		{
			cublas_error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, targets, targets, targets, &alpha, A2, targets, A, targets, &beta, aux, targets);
			
			if( cublas_error != CUBLAS_STATUS_SUCCESS )
			{
				printf( "failed cuBLAS execution %d\n", cublas_error );
				exit(1);
			} 
		}
    
		cublasGetStream(handle, &stream);
		cudaStreamSynchronize(stream);    

		#pragma acc host_data use_device(aux,w,f)
		{
			cublas_error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, targets, one, targets, &alpha, aux, targets, w, targets, &beta, f, targets);
			
			if( cublas_error != CUBLAS_STATUS_SUCCESS )
			{
				printf( "failed cuBLAS execution %d\n", cublas_error );
				exit(1);
			} 
    	}
    
		cublasGetStream(handle, &stream);
		cudaStreamSynchronize(stream);
   
   
    
   		sum1 = 0;
	
		#pragma acc parallel loop reduction(+:sum1)
		for(j=0; j<targets; j++)
		{
			f[j] = w[j] - f[j];
			sum1 += pow(f[j],2);
		}
    
		#pragma acc parallel loop
		for(j=0; j<targets; j++) f[j] /= sqrt(sum1);
    
    
		#pragma acc host_data use_device(f,y,sumxu,index)
		{
			cublas_error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, one, lines_samples, targets, &alpha, f, one, y, lines_samples, &beta, sumxu, one);
			
			if( cublas_error != CUBLAS_STATUS_SUCCESS )
			{
				printf( "failed cuBLAS execution %d\n", cublas_error );
				exit(1);
			} 
		}
    
		cublasGetStream(handle, &stream);
		cudaStreamSynchronize(stream);
   
		#pragma acc kernels
		sum2 = 0;

		#pragma acc kernels
		#pragma acc loop seq
		for(j=0; j<lines_samples; j++)
		{
			if(sumxu[j] < 0) sumxu[j] *= -1;
			if(sum2 < sumxu[j])
			{
				sum2 = sumxu[j];
				index[i] = j;
			}
		}

		#pragma acc parallel loop
		for(j=0; j<targets; j++)
			A[j*targets + i] = y[j*lines_samples+index[i]];
                    
		#pragma acc parallel loop
		for(j=0; j<bands; j++)
    		endmembers[j*targets+ i] = Rp[j+bands * index[i]];

	}  
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
		printf("EXECUTION ERROR VCA Iterative: Error writing header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	error = writeResult(endmembers, argv[4], targets, 1, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR VCA Iterative: Error writing image file: %s.", argv[3]);
		fflush(stdout);
		exit(-1);
	}
 

	free(sumxu);
	free(aux);
	free(f);
	free(A);
	free(A2);
	free(w);
	free(index);
	free(y);
	free(u);
	free(Rp);
	free(x_p);
	free(R_o);
	free(svdMat);
	free(D);
	free(U);
	free(VT);
	free(r_m);
	free(Ud);
	free(image_vector);
	free(endmembers);
	free(U_pinv);
	free(VT_pinv);
	free(Utranstmp);
	free(S_pinv);

	cublasDestroy(handle);
	cusolverDnDestroy(cusolverHandle);
 


	return 0;
}
