#include <iostream>
#include <string>
#include <chrono>
#include <numeric>

#include "utils/file_utils.hpp"
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
    vd.clearMemory();
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
    vca.clearMemory();
    writeEndmemberSignatures("End-Cupriteb-c-02.txt", bands, vd.getNumberEndmembers(), vca.getEndmembers());
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
    isra.clearMemory();
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
