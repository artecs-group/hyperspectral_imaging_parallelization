#ifndef _VCA_SEQUENTIAL_
#define _VCA_SEQUENTIAL_

#include "../vca_interface.hpp"

class SequentialVCA: I_VCA {
    public:
        SequentialVCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~SequentialVCA();
        void run(float SNR, const double* image);
        double* getEndmembers() { return endmembers; };
};

#endif