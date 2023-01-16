#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <numeric>

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
            cleanString(line.substr(s_pos, line.length()-s_pos), &value);
        
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
        
        value = "";
    }

    inFile.close();
    return 0;
}


int readHeader2(std::string filename, double* wavelength) {
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
                value = "";
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
    T* typeValue = new T[lines_samples*bands];
    for(int i = 0; i < lines_samples * bands; i++) {
        inFile.read(reinterpret_cast<char*>(&typeValue[i]), sizeof(T));
        type_float[i] = (float) typeValue[i];
    }
    delete[] typeValue;
}


/*
 * Author: Jorge Sevilla Cedillo & Youssef El Faqir El Rhazoui
 * */
int loadImage(const std::string& filename, double* image, int lines, int samples, 
    int bands, int dataType, std::string* interleave) {
    
    float *type_float;
    int op{0};
    unsigned int lines_samples = lines*samples;
    short int* tv;
    std::ifstream inFile;
    inFile.open(filename, std::ifstream::binary);

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
    }
    inFile.close();

    if(*interleave == "bsq") op = 0;
    else if(*interleave == "bip") op = 1;
    else if(*interleave == "bil") op = 2;

    switch(op) {
        case 0:
            for (size_t i = 0; i < bands*lines*samples; i++)
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

template <typename T>
int parseAndWriteImage(int lines, int samples, int bands, int op, const std::string& filename, const double* image, T* imageT){
    std::ofstream outFile;
    outFile.open(filename, std::ofstream::binary);

    if(!outFile.is_open())
        return -3; //No file found

    switch(op) {
        case 0:
            for(int i = 0; i < lines*samples*bands; i++){
                imageT[i] = (T)image[i];
                outFile.write(reinterpret_cast<char*>(&imageT[i]), sizeof(T));
            }
            break;

        case 1:
            for(int i = 0; i < bands; i++)
                for(int j = 0; j < lines*samples; j++){
                    imageT[j*bands + i] = (T)image[i*lines*samples + j];
                    outFile.write(reinterpret_cast<char*>(&imageT[j*bands + i]), sizeof(T));
                }
            break;

        case 2:
            for(int i = 0; i < lines; i++)
                for(int j = 0; j < bands; j++)
                    for(int k = 0; k < samples; k++){
                        imageT[i*bands*samples + (j*samples + k)] = (T)image[j*lines*samples + (i*samples + k)];
                        outFile.write(reinterpret_cast<char*>(&imageT[i*bands*samples + (j*samples + k)]), sizeof(T));
                    }
            break;
    }
    
    outFile.close();
    delete[] imageT;
    return 0;
}


/*
 * Author: Luis Ignacio Jimenez & Youssef El Faqir El Rhazoui
 * */
int writeResult(double* image, const std::string& filename, int lines, int samples,
    int bands, int dataType, const std::string& interleave)
{
	short int* imageSI;
	float* imageF;
	double* imageD;
	int op{0};

    if(interleave == "bsq") op = 0;
    else if(interleave == "bip") op = 1;
    else if(interleave == "bil") op = 2;

	if(dataType == 2) {
		imageSI = new short int[lines*samples*bands];
        return parseAndWriteImage(lines, samples, bands, op, filename, image, imageSI);
	}

	else if(dataType == 4) {
		imageF = new float[lines*samples*bands];
        return parseAndWriteImage(lines, samples, bands, op, filename, image, imageF);
	}

	else if(dataType == 5) {
		imageD = new double[lines*samples*bands];
        return parseAndWriteImage(lines, samples, bands, op, filename, image, imageD);
	}
    return 0;
}

/*
 * Author: Luis Ignacio Jimenez & Youssef El Faqir El Rhazoui
 * */
int writeHeader(const std::string& filename, int lines, int samples, int bands, int dataType,
		const std::string& interleave, int byteOrder, const std::string& waveUnit, double* wavelength)
{

    std::ofstream outFile;
    outFile.open(filename, std::ofstream::out);

    if(!outFile.is_open())
        return -3; //No file found

    outFile << "ENVI" << std::endl 
            << "description = {" << std::endl 
            << "Exported from MATLAB}" << std::endl;

    if(samples != 0)  outFile << "samples = " << samples << std::endl;
    if(lines != 0)    outFile << "lines   = " << lines << std::endl;
    if(bands != 0)    outFile << "bands   = " << bands << std::endl;
    if(dataType != 0) outFile << "data type = " << dataType << std::endl;
    if(interleave.length() == 0) outFile << "interleave = " << interleave << std::endl;
    if(byteOrder != 0) outFile << "byte order = " << byteOrder << std::endl;
    if(waveUnit.length() == 0) {
        outFile << "wavelength units = " << waveUnit << std::endl
                << "wavelength = {" << std::endl;
        for(int i = 0; i < bands; i++) {
            if(i == 0) outFile << wavelength[i];
            else
                if(i % 3 == 0) outFile << ", " << wavelength[i] << std::endl;
                else outFile << ", " << wavelength[i];
        }
        outFile << "}";
    }
    outFile.close();
    return 0;
}


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
    double* wavelength = new double[bands]();
    error = readHeader2(filename, wavelength);
	if(error != 0) {
        std::cerr << "Error reading wavelength from header file: " << filename << std::endl; 
		exit(-1);
	}

	double *image = new double[lines*samples*bands]();
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

    filename = argv[1];
    filename += "_processed.hdr";
    std::cout << "Writing results on: " << filename << std::endl;
	error = writeHeader(filename, samples, lines, vd.getNumberEndmembers(), dataType, interleave, byteOrder, waveUnit, wavelength);
	if(error != 0) {
        std::cerr << "Error writing endmembers header file on: " << filename << std::endl;
		return error;
	}

    filename = argv[1];
    filename += "_processed";
	error = writeResult(isra.getAbundanceMatrix(), filename, samples, lines, vd.getNumberEndmembers(), dataType, interleave);
	if(error != 0) {
        std::cerr << "Error writing endmembers file on: " << filename << std::endl;
		return error;
	}

	delete[] image;
	delete[] wavelength;
	return 0;
}
