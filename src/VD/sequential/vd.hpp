#ifndef _VD_SEQUENTIAL_
#define _VD_SEQUENTIAL_

#include "../vd_interface.hpp"

class SequentialVD: I_VD {
    public:
        SequentialVD(int _lines, int _samples, int _bands);
        ~SequentialVD();
        void run(const int approxVal, double* image);
};

#endif