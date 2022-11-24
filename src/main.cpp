#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#if defined(SYCL)
#include "VD/sycl/vd.hpp"
#include "VCA/sycl/vca.hpp"
#include "ISRA/sycl/isra.hpp"
#elif defined(OPENMP)
#include "VD/openmp/vd.hpp"
#include "VCA/openmp/vca.hpp"
#include "ISRA/openmp/isra.hpp"
#else
#include "VD/sequential/vd.hpp"
#include "VCA/sequential/vca.hpp"
#include "ISRA/sequential/isra.hpp"
#endif

/*
 * Author: Jorge Sevilla Cedillo & Youssef El Faqir El Rhazoui
 */
void cleanString(const std::string& str, std::string* out) {
    for(int i{0}; i < str.length(); i++) {
        if(isalnum(str[i]) || str[i] == '{' || str[i]== '.' || str[i] == ',')
            out->push_back(str[i]);
    }
}


/*
 * Author: Jorge Sevilla Cedillo & Youssef El Faqir El Rhazoui
 */
int readHeader1(const std::string& filename, int* lines, int* samples, int* bands, int* dataType,
		std::string* interleave, int* byteOrder, std::string* waveUnit)
{
    std::string line;
    std::string value;
    std::ifstream inFile;
    inFile.open(filename, std::ios::in);

    if(!inFile.is_open())
        return -2; //No file found

    while(std::getline(inFile, line)) {
        size_t s_pos = line.find("=");
        if(s_pos != std::string::npos)
            cleanString(line.substr(s_pos, line.length()-1-s_pos), &value);
        
        if(line.find("samples") != std::string::npos && samples != NULL)
            *samples = std::stoi(value);
        else if(line.find("lines") != std::string::npos && lines != NULL)
            *lines = std::stoi(value);
        else if(line.find("bands") != std::string::npos && bands != NULL)
            *bands = std::stoi(value);
        else if(line.find("interleave") != std::string::npos && interleave != NULL)
            *interleave = value;
        else if(line.find("data type") != std::string::npos && dataType != NULL)
            *dataType = std::stoi(value);
        else if(line.find("byte order") != std::string::npos && byteOrder != NULL)
            *byteOrder = std::stoi(value);
        else if(line.find("wavelength unit") != std::string::npos && waveUnit != NULL)
            *waveUnit = value;
    }

    inFile.close();
    return 0;
}


int readHeader2(std::string filename, double* wavelength) {
    std::cout << "readHeader2" << std::endl;
    std::string line;
    std::string value;
    std::string strAll;
    std::string pch;
    std::string delimiter{","};
    std::ifstream inFile;
    inFile.open(filename, std::ios::in);

    if(!inFile.is_open())
        return -2; //No file found

    while(std::getline(inFile, line)) {
        if(line.find("wavelength =") != std::string::npos && wavelength != NULL) {
            int cont = 0;
            do {
                std::getline(inFile, line);
                cleanString(line, &value);
                strAll += value;
            } while (line.find("}") != std::string::npos);

            int dPos{0};
            while ((dPos = strAll.find(delimiter)) != std::string::npos) {
                pch = strAll.substr(0, dPos);
                wavelength[cont] = std::stof(pch);
                strAll.erase(0, dPos + delimiter.length());
                cont++;
            }
        }
    }
    inFile.close();
    return 0;
}


template <typename T>
void convertToFloat(unsigned int lines_samples, int bands, std::ifstream& inFile, float* type_float){
    std::cout << "convertToFloat" << std::endl;
    T* typeValue = new T[lines_samples*bands];
    for(int i = 0; i < lines_samples * bands; i++) {
        inFile >> typeValue[i];
        type_float[i] = (float) typeValue[i];
    }
    delete[] typeValue;
}


/*
 * Author: Jorge Sevilla Cedillo & Youssef El Faqir El Rhazoui
 * */
int loadImage(const std::string& filename, double* image, int lines, int samples, 
    int bands, int dataType, std::string* interleave) {
    
    std::cout << "loadImage" << std::endl;
    float *type_float;
    int op{0};
    unsigned int lines_samples = lines*samples;
    short int* tv;
    std::ifstream inFile;
    inFile.open(filename, std::ios::in);

    if(!inFile.is_open())
        return -2; //No file found

    type_float = new float[lines_samples*bands];
    
    switch(dataType) {
        case 2:
            convertToFloat<short>(lines_samples, bands, inFile, type_float);
            break;
        case 4:
            convertToFloat<float>(lines_samples, bands, inFile, type_float);
            break;
        case 5:
            convertToFloat<double>(lines_samples, bands, inFile, type_float);
            break;
        case 12:
            convertToFloat<unsigned int>(lines_samples, bands, inFile, type_float);
            break;
        default:
            break;
    }
    inFile.close();

    if(*interleave == "bsq") op = 0;
    else if(*interleave == "bip") op = 1;
    else if(*interleave == "bil") op = 2;

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


// /*
//  * Author: Luis Ignacio Jimenez
//  * Centre: Universidad de Extremadura
//  * */
// int writeResult(double *image, const std::string* filename, int lines, int samples, int bands, int dataType, std::string* interleave)
// {

// 	short int *imageSI;
// 	float *imageF;
// 	double *imageD;

// 	int i,j,k,op;

// 	if(interleave == NULL)
// 		op = 0;
// 	else
// 	{
// 		if(strcmp(interleave, "bsq") == 0) op = 0;
// 		if(strcmp(interleave, "bip") == 0) op = 1;
// 		if(strcmp(interleave, "bil") == 0) op = 2;
// 	}

// 	if(dataType == 2)
// 	{
// 		imageSI = (short int*)malloc(lines*samples*bands*sizeof(short int));

//         switch(op)
//         {
// 			case 0:
// 				for(i=0; i<lines*samples*bands; i++)
// 					imageSI[i] = (short int)image[i];
// 				break;

// 			case 1:
// 				for(i=0; i<bands; i++)
// 					for(j=0; j<lines*samples; j++)
// 						imageSI[j*bands + i] = (short int)image[i*lines*samples + j];
// 				break;

// 			case 2:
// 				for(i=0; i<lines; i++)
// 					for(j=0; j<bands; j++)
// 						for(k=0; k<samples; k++)
// 							imageSI[i*bands*samples + (j*samples + k)] = (short int)image[j*lines*samples + (i*samples + k)];
// 				break;
//         }

// 	}

// 	if(dataType == 4)
// 	{
// 		imageF = (float*)malloc(lines*samples*bands*sizeof(float));
//         switch(op)
//         {
// 			case 0:
// 				for(i=0; i<lines*samples*bands; i++)
// 					imageF[i] = (float)image[i];
// 				break;

// 			case 1:
// 				for(i=0; i<bands; i++)
// 					for(j=0; j<lines*samples; j++)
// 						imageF[j*bands + i] = (float)image[i*lines*samples + j];
// 				break;

// 			case 2:
// 				for(i=0; i<lines; i++)
// 					for(j=0; j<bands; j++)
// 						for(k=0; k<samples; k++)
// 							imageF[i*bands*samples + (j*samples + k)] = (float)image[j*lines*samples + (i*samples + k)];
// 				break;
//         }
// 	}

// 	if(dataType == 5)
// 	{
// 		imageD = (double*)malloc(lines*samples*bands*sizeof(double));
//         switch(op)
//         {
// 			case 0:
// 				for(i=0; i<lines*samples*bands; i++)
// 					imageD[i] = image[i];
// 				break;

// 			case 1:
// 				for(i=0; i<bands; i++)
// 					for(j=0; j<lines*samples; j++)
// 						imageD[j*bands + i] = image[i*lines*samples + j];
// 				break;

// 			case 2:
// 				for(i=0; i<lines; i++)
// 					for(j=0; j<bands; j++)
// 						for(k=0; k<samples; k++)
// 							imageD[i*bands*samples + (j*samples + k)] = image[j*lines*samples + (i*samples + k)];
// 				break;
//         }
// 	}

//     FILE *fp;
//     if ((fp=fopen(filename,"wb"))!=NULL)
//     {
//         fseek(fp,0L,SEEK_SET);
// 	    switch(dataType)
// 	    {
// 	    case 2: fwrite(imageSI,1,(lines*samples*bands * sizeof(short int)),fp); free(imageSI); break;
// 	    case 4: fwrite(imageF,1,(lines*samples*bands * sizeof(float)),fp); free(imageF); break;
// 	    case 5: fwrite(imageD,1,(lines*samples*bands * sizeof(double)),fp); free(imageD); break;
// 	    }
// 	    fclose(fp);


// 	    return 0;
//     }

//     return -3;
// }

// /*
//  * Author: Luis Ignacio Jimenez
//  * Centre: Universidad de Extremadura
//  * */
// int writeHeader(std::string* filename, int lines, int samples, int bands, int dataType,
// 		std::string* interleave, int byteOrder, std::string* waveUnit, double* wavelength)
// {
//     FILE *fp;
//     if ((fp=fopen(filename,"wt"))!=NULL)
//     {
// 		fseek(fp,0L,SEEK_SET);
// 		fprintf(fp,"ENVI\ndescription = {\nExported from MATLAB}\n");
// 		if(samples != 0) fprintf(fp,"samples = %d", samples);
// 		if(lines != 0) fprintf(fp,"\nlines   = %d", lines);
// 		if(bands != 0) fprintf(fp,"\nbands   = %d", bands);
// 		if(dataType != 0) fprintf(fp,"\ndata type = %d", dataType);
// 		if(interleave != NULL) fprintf(fp,"\ninterleave = %s", interleave);
// 		if(byteOrder != 0) fprintf(fp,"\nbyte order = %d", byteOrder);
// 		if(waveUnit != NULL) fprintf(fp,"\nwavelength units = %s", waveUnit);
// 		if(waveUnit != NULL)
// 		{
// 			fprintf(fp,"\nwavelength = {\n");
// 			for(int i=0; i<bands; i++)
// 			{
// 				if(i==0) fprintf(fp, "%f", wavelength[i]);
// 				else
// 					if(i%3 == 0) fprintf(fp, ", %f\n", wavelength[i]);
// 					else fprintf(fp, ", %f", wavelength[i]);
// 			}
// 			fprintf(fp,"}");
// 		}
// 		fclose(fp);
// 		return 0;
//     }
//     return -3;
// }


int main(int argc, char* argv[]) {
	if(argc != 5) {
        std::cout << "Parameters are not correct." << std::endl
                  << "./main <Image Filename> <Approximation value> <Signal noise ratio (SNR)> <Max iterations> " << std::endl;
		exit(-1);
	}

	// Read image
	std::string filename;
	std::string interleave;
	std::string waveUnit;
	int lines{0}, samples{0}, bands{0}, dataType{0}, byteOrder{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float appTime{0.f};

    // Read header first parameters
    filename = argv[1];
    filename += ".hdr";
	int error = readHeader1(filename, &lines, &samples, &bands, &dataType, &interleave, &byteOrder, &waveUnit);
	if(error != 0) {
        std::cerr << "Error reading header file: " << filename << std::endl; 
		exit(-1);
	}

    // Read header wavelenght, which requires bands from previous read
    double* wavelength = new double[bands];
    error = readHeader2(filename, wavelength);
	if(error != 0) {
        std::cerr << "Error reading wavelength from header file: " << filename << std::endl; 
		exit(-1);
	}

	double *image = new double[lines*samples*bands];
    filename = argv[1];
	error = loadImage(filename, image, lines, samples, bands, dataType, &interleave);
	if(error != 0) {
        std::cerr << "Error reading image file: " << argv[1] << std::endl;
		exit(-1);
	}

    int approxVal = atoi(argv[2]);
    float SNR     = atof(argv[3]);
    int maxIter   = atoi(argv[4]);

    std::cout << std::endl << "Parameters:" << std::endl
                    << "    -> Lines                    = " << lines << std::endl
                    << "    -> Samples                  = " << samples << std::endl
                    << "    -> Bands                    = " << bands << std::endl
                    << "    -> Approximation value (VD) = " << approxVal << std::endl
                    << "    -> SNR (VCA)                = " << SNR << std::endl
                    << "    -> Max iterations (ISRA)    = " << maxIter << std::endl;
    std::cout << std::endl << "Starting image processing ";
    start = std::chrono::high_resolution_clock::now();

#if defined(SYCL)
    std::cout << "with SYCL implementation." << std::endl;
    SYCL_VD vd = SYCL_VD(lines, samples, bands);
#elif defined(OPENMP)
    std::cout << "with OpenMP implementation." << std::endl << std::endl;
    OpenMP_VD vd = OpenMP_VD(lines, samples, bands);
#else
    std::cout << "with sequential implementation." << std::endl << std::endl;
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

#if defined(SYCL)
SYCL_ISRA isra = SYCL_ISRA(lines, samples, bands, vd.getNumberEndmembers());
#elif defined(OPENMP)
OpenMP_ISRA isra = OpenMP_ISRA(lines, samples, bands, vd.getNumberEndmembers());
#else
SequentialISRA isra = SequentialISRA(lines, samples, bands, vd.getNumberEndmembers());
#endif

    std::cout << "---------------- ISRA ----------------" << std::endl;
    isra.run(maxIter, image, vca.getEndmembers());
    std::cout << "-------------------------------------" << std::endl << std::endl;

    end = std::chrono::high_resolution_clock::now();
    appTime += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    std::cout << "Image processing took = " << appTime << " (s)" << std::endl;
    std::cout << "Writing results on: " << std::endl;

    // strcpy(filename, argv[1]);
	// strcat(filename, "_processed.hdr");
	// error = writeHeader(filename, samples, lines, vd.getNumberEndmembers(), 4, interleave, 0, NULL, NULL);
	// if(error != 0) {
	// 	printf("Error writing endmembers header file: %s.", filename);
    //     std::cerr << "Error writing image processed on: " << filename << std::cout;
	// 	fflush(stdout);
	// 	return error;
	// }

	// error = writeResult(isra.getAbundanceMatrix(), argv[4],samples,lines, vd.getNumberEndmembers(), 4, interleave);
	// if(error != 0)
	// {
	// 	printf("EXECUTION ERROR ISRA Iterative: Error writing endmembers file: %s.", argv[4]);
	// 	fflush(stdout);
	// 	return error;
	// }

	delete[] image;
	delete[] wavelength;
	return 0;
}
