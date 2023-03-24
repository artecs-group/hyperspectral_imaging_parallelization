#ifndef _VCA_INTERFACE_
#define _VCA_INTERFACE_

class I_VCA {
public:
    virtual void run(float SNR, const double* image) = 0;
    virtual double* getEndmembers() = 0;
    virtual void clearMemory() = 0;

protected:
    unsigned int lines, samples, bands, targetEndmembers;
};
#endif