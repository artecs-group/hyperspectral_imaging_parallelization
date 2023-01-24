#ifndef _VD_SEQUENTIAL_
#define _VD_SEQUENTIAL_

#include "../../common/interfaces/vd_interface.hpp"

class SequentialVD: I_VD {
    public:
        SequentialVD(int _lines, int _samples, int _bands);
        ~SequentialVD();
        void run(const int approxVal, const double* image);
        unsigned int getNumberEndmembers() { return endmembers; };
        void clearMemory();
};

#endif