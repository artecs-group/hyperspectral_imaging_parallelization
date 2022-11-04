#ifndef _VD_
#define _VD_

class VD {
    public:
        virtual void run(int approxVal, double* image) = 0;
        int result;

    protected:
        static constexpr int FPS{5};
        int lines, samples, bands;
        double sigmaSquareTest, sigmaTest, TaoTest;
        double *meanSpect, *Cov, *Corr, *CovEigVal, *CorrEigVal, *U, *VT, *estimation;
};


class SequentialVD: VD {
    public:
        SequentialVD(int _lines, int _samples, int _bands);
        ~SequentialVD();
        void run(int approxVal, double* image);
};

#endif