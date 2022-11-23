#ifndef _ISRA_SEQUENTIAL_
#define _ISRA_SEQUENTIAL_

#include "../isra_interface.hpp"

class SequentialISRA: I_ISRA {
    public:
        SequentialISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~SequentialISRA();
        void run(int maxIter, const double* image, const double* endmembers);
        double* getAbundanceMatrix() { return abundanceMatrix; };
};

#endif