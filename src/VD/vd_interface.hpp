#ifndef _VD_INTERFACE_
#define _VD_INTERFACE_

class I_VD {
    public:
        virtual void run(const int approxVal, const double* image) = 0;
        virtual unsigned int getNumberEndmembers() = 0;
    protected:
        static constexpr int FPS{5};
        unsigned int lines, samples, bands, endmembers;
        double *meanSpect{nullptr}, 
               *Cov{nullptr}, 
               *Corr{nullptr}, 
               *CovEigVal{nullptr}, 
               *CorrEigVal{nullptr}, 
               *U{nullptr}, 
               *VT{nullptr}, 
               *estimation{nullptr}, 
               *meanImage{nullptr};
        unsigned int* count{nullptr};
};
#endif