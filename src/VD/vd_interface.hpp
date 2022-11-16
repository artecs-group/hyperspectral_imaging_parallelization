#ifndef _VD_INTERFACE_
#define _VD_INTERFACE_

class I_VD {
    public:
        virtual void run(const int approxVal, const double* image) = 0;
        unsigned int result;

    protected:
        static constexpr int FPS{5};
        unsigned int lines, samples, bands;
        double *meanSpect, *Cov, *Corr, *CovEigVal, *CorrEigVal, *U, *VT, *estimation, *meanImage;
        unsigned int* count;
};
#endif