#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

extern "C" {
#include "clahe.h"
}


static Ptr<CLAHE> _clahe = NULL;

extern "C"
__attribute__((visibility("hidden")))
void clahe( uint8_t* buffer, // input AND output
            int width, int height,
            double clipLimit)
{
    if(_clahe == NULL)
        _clahe = createCLAHE(clipLimit);

    Mat cvmat_src(height, width,
                  CV_8UC1,
                  buffer, width);
    Mat cvmat_dst(height, width,
                  CV_8UC1,
                  buffer, width);
    _clahe->apply(cvmat_src, cvmat_dst);
}
