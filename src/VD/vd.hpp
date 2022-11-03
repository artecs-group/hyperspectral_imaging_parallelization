#ifndef _VD_
#define _VD_

class VD {
    public:
        virtual void run(int, double*);
        virtual void free();

        int result;

    protected:
        static constexpr int FPS{5};

        int bands;
        int N;

        double sigmaSquareTest;
        double sigmaTest;
        double TaoTest;

        double *meanSpect;
        double *Cov;
        double *Corr;
        double *CovEigVal;
        double *CorrEigVal;
        double *U;
        double *VT;
        double *estimation;
};


class SequentialVD: VD {
    public:
        SequentialVD(int bands, int N);
        ~SequentialVD();
        void run(int, double*);
};


#endif