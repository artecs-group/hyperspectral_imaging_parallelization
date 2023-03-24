#ifndef _VD_INTERFACE_
#define _VD_INTERFACE_

class I_VD {
    public:
        virtual void run(const int approxVal, const double* image) = 0;
        virtual unsigned int getNumberEndmembers() = 0;
        virtual void clearMemory() = 0;
    protected:
        static constexpr int FPS{5};
        unsigned int lines, samples, bands, endmembers;
};
#endif