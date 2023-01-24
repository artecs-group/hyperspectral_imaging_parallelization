#ifndef _ISRA_INTERFACE_
#define _ISRA_INTERFACE_

class I_ISRA {
    public:
        virtual void run(int maxIter, const double* image, const double* endmembers) = 0;
        virtual double* getAbundanceMatrix() = 0;
        virtual void clearMemory() = 0;
    protected:
        unsigned int lines, samples, bands, targetEndmembers;
        double *abundanceMatrix{nullptr}, 
               *numerator{nullptr}, 
               *denominator{nullptr}, 
               *aux{nullptr};
};
#endif