#include <iostream>
#include <stdio.h>
#include <string.h>

#if defined(SYCL)
#include "VD/sycl/vd.hpp"
#include "VCA/sycl/vca.hpp"
#elif defined(OPENMP)
#include "VD/openmp/vd.hpp"
#include "VCA/openmp/vca.hpp"
#else
#include "VD/sequential/vd.hpp"
#include "VCA/sequential/vca.hpp"
#endif

#define MAXLINE 200
#define MAXCAD 200

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura 
 */
void cleanString(char* str, char* out) {
    int i,j;
    for( i = j = 0; str[i] != 0; i++) {
        if(isalnum(str[i]) || str[i] == '{' || str[i]== '.' || str[i] == ',') {
            out[j] = str[i];
            j++;
        }
    }
    for( i = j; out[i] != 0; i++)
        out[j] = 0;
}


/*
 * Author: Jorge Sevilla Cedillo & Youssef El Faqir El Rhazoui
 */
int readHeader1(char* filename, int* lines, int* samples, int* bands, int* dataType,
		char* interleave, int* byteOrder, char* waveUnit)
{
    FILE* fp;
    char line[MAXLINE] = "";
    char value [MAXLINE] = "";

    if ((fp=fopen(filename, "rt")) != NULL)
    {
        fseek(fp, 0L, SEEK_SET);
        while(fgets(line, MAXLINE, fp) != NULL) {
            //Samples
            if(strstr(line, "samples") != NULL && samples != NULL)
            {
                cleanString(strstr(line, "="),value);
                *samples = atoi(value);
            }

            //Lines
            else if(strstr(line, "lines") != NULL && lines != NULL)
            {
                cleanString(strstr(line, "="),value);
                *lines = atoi(value);
            }

            //Bands
            else if(strstr(line, "bands") != NULL && bands != NULL)
            {
                cleanString(strstr(line, "="),value);
                *bands = atoi(value);
            }

            //Interleave
            else if(strstr(line, "interleave") != NULL && interleave != NULL)
            {
                cleanString(strstr(line, "="),value);
                strcpy(interleave,value);
            }

            //Data Type
            else if(strstr(line, "data type") != NULL && dataType != NULL)
            {
                cleanString(strstr(line, "="),value);
                *dataType = atoi(value);
            }

            //Byte Order
            else if(strstr(line, "byte order") != NULL && byteOrder != NULL)
            {
                cleanString(strstr(line, "="),value);
                *byteOrder = atoi(value);
            }

            //Wavelength Unit
            else if(strstr(line, "wavelength unit") != NULL && waveUnit != NULL)
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


int readHeader2(char* filename, double* wavelength) {
    FILE* fp;
    char line[MAXLINE] = "";
    char value [MAXLINE] = "";

    if ((fp=fopen(filename,"rt")) != NULL)
    {
        fseek(fp,0L,SEEK_SET);
        while(fgets(line, MAXLINE, fp) != NULL)
        {
            //Wavelength
            if(strstr(line, "wavelength =") != NULL && wavelength != NULL) {
                char strAll[100000] = "";
                char* pch;
                int cont = 0;

                do {
                    fgets(line, MAXLINE, fp);
                    cleanString(line,value);
                    strcat(strAll,value);
                } while (strstr(line, "}") == NULL);

                pch = strtok(strAll,",");

                while (pch != NULL) {
                    wavelength[cont] = atof(pch);
                    pch = strtok(NULL, ",");
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


template <typename T>
void convertToFloat(long int lines_samples, int bands, FILE *fp, float* type_float){
    T* typeValue = new T[lines_samples*bands];
    fread(typeValue, 1, (sizeof(T) * lines_samples * bands), fp);
    for(int i = 0; i < lines_samples * bands; i++)
        type_float[i] = (float) typeValue[i];
    delete[] typeValue;
}


/*
 * Author: Jorge Sevilla Cedillo & Youssef El Faqir El Rhazoui
 * Centre: Universidad de Extremadura
 * */
int loadImage(char* filename, double* image, int lines, int samples, int bands, int dataType, char* interleave) {
    FILE *fp;
    float *type_float;
    int op;
    long int lines_samples = lines*samples;
    short int* tv;

    if ((fp=fopen(filename,"rb"))!=NULL) {

        fseek(fp, 0L, SEEK_SET);
        type_float = new float[lines_samples*bands];
        
        switch(dataType) {
            case 2:
                convertToFloat<short int>(lines_samples, bands, fp, type_float);
                break;
            case 4:
                fread(type_float, 1, (sizeof(float)*lines_samples*bands), fp);
                break;
            case 5:
                convertToFloat<double>(lines_samples, bands, fp, type_float);
                break;
            case 12:
                convertToFloat<unsigned int>(lines_samples, bands, fp, type_float);
                break;
            default:
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

        switch(op) {
            case 0:
                for(int i = 0; i < lines*samples*bands; i++)
                    image[i] = type_float[i];
                break;

            case 1:
                for(int i = 0; i < bands; i++)
                    for(int j = 0; j < lines*samples; j++)
                        image[i*lines*samples + j] = type_float[j*bands + i];
                break;

            case 2:
                for(int i = 0; i < lines; i++)
                    for(int j = 0; j < bands; j++)
                        for(int k = 0; k < samples; k++)
                            image[j*lines*samples + (i*samples+k)] = type_float[k+samples*(i*bands+j)];
                break;
        }
        delete[] type_float;
        return 0;
    }
    return -2;
}


int main(int argc, char* argv[]) {
	if(argc != 4) {
        std::cout << "Parameters are not correct." << std::endl
                  << "./main <Image Filename> <Approximation value> <Signal noise ratio (SNR)>" << std::endl;
		exit(-1);
	}

	// Read image
	char cad[MAXCAD];
	char interleave[MAXCAD];
	char waveUnit[MAXCAD];
	int lines{0}, samples{0}, bands{0}, dataType{0}, byteOrder{0};

    // Read header first parameters
	strcpy(cad, argv[1]);
	strcat(cad, ".hdr");
	int error = readHeader1(cad, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0) {
        std::cout << "Error reading header file: " << cad << std::endl; 
		exit(-1);
	}

    // Read header wavelenght, which requires bands from previous read
    double* wavelength = new double[bands];
    error = readHeader2(cad, wavelength);
	if(error != 0) {
        std::cout << "Error reading wavelength from header file: " << cad << std::endl; 
		exit(-1);
	}

	double *image = new double[lines*samples*bands];
	error = loadImage(argv[1], image, lines, samples, bands, dataType, interleave);
	if(error != 0) {
        std::cout << "Error reading image file: " << argv[1] << std::endl;
		exit(-1);
	}

    int approxVal = atoi(argv[2]);
    float SNR     = atof(argv[3]);

#if defined(SYCL)
SYCL_VD vd = SYCL_VD(lines, samples, bands);
#elif defined(OPENMP)
OpenMP_VD vd = OpenMP_VD(lines, samples, bands);
#else
SequentialVD vd = SequentialVD(lines, samples, bands);
#endif

    std::cout << "---------------- VD -----------------" << std::endl;
    vd.run(approxVal, image);
    std::cout << "-------------------------------------" << std::endl << std::endl;

#if defined(SYCL)
SYCL_VCA vca = SYCL_VCA(lines, samples, bands, vd.getNumberEndmembers());
#elif defined(OPENMP)
OpenMP_VCA vca = OpenMP_VCA(lines, samples, bands, vd.getNumberEndmembers());
#else
SequentialVCA vca = SequentialVCA(lines, samples, bands, vd.getNumberEndmembers());
#endif

    std::cout << "---------------- VCA ----------------" << std::endl;
    vca.run(SNR, image);
    std::cout << "-------------------------------------" << std::endl << std::endl;

	delete[] image;
	delete[] wavelength;
    
	return 0;
}
