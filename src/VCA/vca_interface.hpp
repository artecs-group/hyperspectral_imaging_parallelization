#ifndef _VCA_INTERFACE_
#define _VCA_INTERFACE_

#define EPSILON 1.11e-16

class I_VCA {
    public:
        virtual void run(float SNR, const double* image) = 0;
        virtual double* getEndmembers() = 0;

    protected:
        unsigned int lines, samples, bands, targetEndmembers;
        unsigned int *index;
        double *Ud, *x_p, *y, *meanImage, *mean, *svdMat, *D, *U, *VT, 
               *Rp, *u, *sumxu, *w, *A, *A2, *aux, *f, *endmembers,
               *pinvS, *pinvU, *pinvVT, *Utranstmp;
};
#endif