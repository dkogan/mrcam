#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

extern "C" {
#include "clahe.h"
}



extern "C"
__attribute__((visibility("hidden")))
void clahe( uint8_t* buffer, // input AND output
            int width, int height,
            double clipLimit)
{
    static bool inited = false;
    static Ptr<CLAHE> clahe;
    if(!inited)
    {
        inited = true;
        clahe = createCLAHE(clipLimit);
    }

    Mat cvmat_src(height, width,
                  CV_8UC1,
                  buffer, width);
    Mat cvmat_dst(height, width,
                  CV_8UC1,
                  buffer, width);
    clahe->apply(cvmat_src, cvmat_dst);
}
