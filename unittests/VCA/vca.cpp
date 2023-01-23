#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "../../src/utils/file_utils.hpp"
#include "../../src/VCA/sequential/vca.hpp"

/**
 * Tests sequential vca with Cuprite file. 
*/
TEST(VCA_Test, Cuprite) {
	std::string binFile{"../../data/Cuprite"};
    std::string headerFile{"../../data/Cuprite.hdr"};
	std::string interleave;
	std::string waveUnit;
	int lines{0}, samples{0}, bands{0}, dataType{0}, byteOrder{0};

    // Load Cuprite file
	int error = readHeader1(headerFile, &lines, &samples, &bands, &dataType, &interleave, &byteOrder, &waveUnit);
	if(error != 0) {
        std::cerr << "Error reading header file: " << headerFile << std::endl; 
		exit(-1);
	}

    // Read header wavelenght, which requires bands from previous read
    double* wavelength = new double[bands]();
    error = readHeader2(headerFile, wavelength);
	if(error != 0) {
        std::cerr << "Error reading wavelength from header file: " << headerFile << std::endl; 
		exit(-1);
	}

	double *image = new double[lines*samples*bands]();
	error = loadImage(binFile, image, lines, samples, bands, dataType, &interleave);
	if(error != 0) {
        std::cerr << "Error reading image file: " << binFile << std::endl;
		exit(-1);
	}

    // Run VCA to get endmember signatures
    const int n_endmembers = 19;
    SequentialVCA vca = SequentialVCA(lines, samples, bands, n_endmembers);
    vca.run(0, image);
    double* computedSignatures = vca.getEndmembers();

    //Load true signatures from its file
    std::string signFile{"../../data/test/cuprite_real_signatures.txt"};
    int sbands{0}, sendm{0};
    double* trueSignatures = loadEndmemberSignatures(signFile, &sendm, &sbands);
	if(trueSignatures == nullptr) {
        std::cerr << "Error reading signatures file: " << signFile << std::endl;
		exit(-1);
	}

    // Check values
    double* angles = new double[bands * sendm]();
    auto SAD = [] (double *A, double *B, int size) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < size; i++) {
            dot += A[i] * B[i];
            normA += A[i] * A[i];
            normB += B[i] * B[i];
        }
        double s = std::acos(dot / std::sqrt(normA * normB));
        return std::abs(s);
    };

    double* signs_x = new double[bands];
    double* signs_y = new double[bands];
    for (int x = 0; x < n_endmembers; x++) {
        for (int y = 0; y < sendm; y++) {
            for(int i = 0; i < bands; i++) {
                signs_x[i] = computedSignatures[x*n_endmembers + i];
                signs_y[i] = trueSignatures[y*sendm + i];
                //std::cout << signs_x[i] << ", ";
            }
            return;
            angles[x*sendm + y] = SAD(signs_x, signs_y, bands);
            //std::cout << angles[x*sendm + y] << "    ";
        }
        //std::cout << std::endl;
    }

    double* result = new double[sendm];
    for (size_t k = 0; k < sendm; k++) {
        double minimo_ik = std::numeric_limits<double>::max();
        int index;
        double *begin = &angles[0], *end = begin + bands*sendm;
        auto min_element = std::min_element(begin, end);
        minimo_ik = *min_element;
        index = std::distance(begin, min_element);
        int xmin = (index / bands) + 1;
        result[xmin] = minimo_ik * 180 / M_PI; // transform from radians to degrees
    }

    // get the mean
    double res_mean = std::accumulate(result, result + sendm, 0.0);
    res_mean /= sendm;

    std::cout << res_mean << std::endl;

    delete[] angles;
    delete[] image;
    delete[] trueSignatures;
    delete[] result;
    delete[] signs_x;
    delete[] signs_y;
    delete[] wavelength;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}