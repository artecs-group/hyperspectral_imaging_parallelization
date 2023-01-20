#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "mkl.h"
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

	double *image = new double[lines*samples*bands]();
	error = loadImage(binFile, image, lines, samples, bands, dataType, &interleave);
	if(error != 0) {
        std::cerr << "Error reading image file: " << binFile << std::endl;
		exit(-1);
	}

    // Run VCA to get endmember signatures
    int n_endmembers = 19;
    SequentialVCA vca = SequentialVCA(lines, samples, bands, n_endmembers);
    vca.run(0, image);
    double* endmembers = vca.getEndmembers();

    //Load true signatures from its file
    std::string signFile{"../../data/test/cuprite_real_signatures.txt"};
    int sbands{0}, sendm{0};
    double* signatures = loadEndmemberSignatures(signFile, &sbands, &sendm);
	if(signatures == nullptr) {
        std::cerr << "Error reading signatures file: " << signFile << std::endl;
		exit(-1);
	}

    // Check values
    double* angles = new double[bands * sendm]();
    auto SAD = [] (double *A, double *B, int size) {
        double a_dot_a{0}, b_dot_b{0}, dot{0};
        for (int i = 0; i < size; i++) {
            a_dot_a += A[i] * A[i];
            b_dot_b += B[i] * B[i];
            dot += A[i] * B[i];
        }
        double s = std::acos(dot / (std::sqrt(a_dot_a) * std::sqrt(b_dot_b)));
        return std::abs(s);
    };

    for (int x = 0; x < bands; x++) {
        for (int y = 0; y < sendm; y++) {
            double* endm_x = new double[n_endmembers];
            double* signature_y = new double[sendm];

            for (int i = 0; i < n_endmembers; i++) endm_x[i] = endmembers[i*bands + x];
            for (int i = 0; i < sendm; i++) signature_y[i] = signatures[i*sendm + y];

            angles[x*sendm + y] = SAD(endm_x, signature_y, n_endmembers);

            delete[] endm_x;
            delete[] signature_y;
        }
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
    delete[] signatures;
    delete[] result;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}