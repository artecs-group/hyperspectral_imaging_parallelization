#ifndef _FILE_UTILS_
#define _FILE_UTILS_

#include <fstream>
#include <sstream>
#include <string>

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
            for(int i = 0; i < lines*samples*bands; i++)
                imageT[i] = (T)image[i];
            break;

        case 1:
            for(int i = 0; i < bands; i++)
                for(int j = 0; j < lines*samples; j++)
                    imageT[j*bands + i] = (T)image[i*lines*samples + j];
            break;

        case 2:
            for(int i = 0; i < lines; i++)
                for(int j = 0; j < bands; j++)
                    for(int k = 0; k < samples; k++)
                        imageT[i*bands*samples + (j*samples + k)] = (T)image[j*lines*samples + (i*samples + k)];
            break;
        default:
            break;
    }

    outFile.write(reinterpret_cast<char*>(imageT), lines * samples * bands * sizeof(T));
    outFile.close();
    return 0;
}


/*
 * Author: Luis Ignacio Jimenez & Youssef El Faqir El Rhazoui
 * */
int writeResult(double* image, const std::string& filename, int lines, int samples,
    int bands, int dataType, const std::string& interleave)
{
    int info{0};
	int op{0};

    if(interleave == "bsq") op = 0;
    else if(interleave == "bip") op = 1;
    else if(interleave == "bil") op = 2;

	if(dataType == 2) {
		short int* imageSI = new short int[lines*samples*bands];
        info = parseAndWriteImage(lines, samples, bands, op, filename, image, imageSI);
        delete[] imageSI;
	}

	else if(dataType == 4) {
		float* imageF = new float[lines*samples*bands];
        info = parseAndWriteImage(lines, samples, bands, op, filename, image, imageF);
        delete[] imageF;
	}

	else if(dataType == 5) {
		double* imageD = new double[lines*samples*bands];
        info = parseAndWriteImage(lines, samples, bands, op, filename, image, imageD);
        delete[] imageD;
	}
    return info;
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
    if(interleave.length() != 0) outFile << "interleave = " << interleave << std::endl;
    if(byteOrder != 0) outFile << "byte order = " << byteOrder << std::endl;
    if(waveUnit.length() != 0) {
        outFile << "wavelength units = " << waveUnit << std::endl;
        if(waveUnit != "Unknown") {
            outFile << "wavelength = {" << std::endl;
            for(int i = 0; i < bands; i++) {
                if(i == 0) outFile << wavelength[i];
                else
                    if(i % 3 == 0) outFile << ", " << wavelength[i] << std::endl;
                    else outFile << ", " << wavelength[i];
            }
            outFile << "}";
        }
    }
    outFile.close();
    return 0;
}


/*
 * Author: Youssef El Faqir El Rhazoui
 * */
int writeEndmemberSignatures(const std::string& filename, int bands, int endmembers, const double* signatures) {
    std::ofstream outFile;
    outFile.open(filename, std::ofstream::out);

    if(!outFile.is_open())
        return -3; //No file found

    for (size_t j = 0; j < endmembers; j++) {
        outFile << "== " << j << std::endl;
        for (size_t i = 0; i < bands; i++)
            outFile << signatures[j*bands + i] << std::endl;
    }

    outFile.close();
    return 0;
}


/*
 * Author: Youssef El Faqir El Rhazoui
 * */
double* loadEndmemberSignatures(const std::string& filename, int* rows, int* cols) {
    std::string line;
    std::ifstream inFile;
    inFile.open(filename, std::ifstream::in);

    if(!inFile.is_open())
        return nullptr;

    while(std::getline(inFile, line)) {
        size_t s_pos = line.find("=");
        if(s_pos == std::string::npos)
            (*cols)++;
        else {
            (*rows)++;
            *cols = 0;
        }
    }
    inFile.clear();
    inFile.seekg(0, std::ios::beg);

    double* signatures = new double[*rows * *cols];
    int i{0};
    while(std::getline(inFile, line)) {
        size_t s_pos = line.find("=");
        if(s_pos == std::string::npos)
            signatures[i++] = std::stod(line);
    }
    inFile.close();
    return signatures;
}

#endif