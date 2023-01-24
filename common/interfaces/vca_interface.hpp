#ifndef _VCA_INTERFACE_
#define _VCA_INTERFACE_

class I_VCA {
    public:
        virtual void run(float SNR, const double* image) = 0;
        virtual double* getEndmembers() = 0;
        virtual void clearMemory() = 0;

    protected:
        int pinv_lwork, lwork;
        unsigned int lines, samples, bands, targetEndmembers;
        unsigned int *index{nullptr};
        double *Ud{nullptr}, 
               *x_p{nullptr}, 
               *y{nullptr}, 
               *meanImage{nullptr}, 
               *mean{nullptr}, 
               *svdMat{nullptr}, 
               *D{nullptr}, 
               *U{nullptr}, 
               *VT{nullptr}, 
               *Rp{nullptr}, 
               *u{nullptr}, 
               *sumxu{nullptr}, 
               *w{nullptr}, 
               *A{nullptr}, 
               *A_copy{nullptr},
               *pinvA{nullptr}, 
               *aux{nullptr}, 
               *f{nullptr}, 
               *endmembers{nullptr},
               *pinvS{nullptr}, 
               *pinvU{nullptr}, 
               *pinvVT{nullptr},
               *pinv_work{nullptr},
               *work{nullptr};
};
#endif