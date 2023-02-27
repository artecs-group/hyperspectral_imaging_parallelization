#ifndef _ISRA_INTERFACE_
#define _ISRA_INTERFACE_

class I_ISRA {
    public:
        virtual void run(int maxIter, const double* image, const double* endmembers) = 0;
        virtual double* getAbundanceMatrix() = 0;
        virtual void clearMemory() = 0;
    protected:
        virtual void preProcessAbundance(const double* image, double* Ab, const double* e, int targetEndmembers, int lines, int samples, int bands) = 0;
        virtual void invTR(double* A, int p) = 0;
        unsigned int lines, samples, bands, targetEndmembers;
        double *abundanceMatrix{nullptr}, 
               *numerator{nullptr}, 
               *denominator{nullptr}, 
               *aux{nullptr},
               *Et_E{nullptr},
               *comput{nullptr};
        long long* ipiv{nullptr};
};
#endif