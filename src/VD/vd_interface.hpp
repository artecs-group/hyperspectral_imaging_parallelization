#ifndef _VD_INTERFACE_
#define _VD_INTERFACE_

class I_VD {
    public:
        virtual void run(int approxVal, double* image) = 0;
        int result;

    protected:
        static constexpr int FPS{5};
        int lines, samples, bands;
        double sigmaSquareTest, sigmaTest, TaoTest;
        double *meanSpect, *Cov, *Corr, *CovEigVal, *CorrEigVal, *U, *VT, *estimation;
};
#endif