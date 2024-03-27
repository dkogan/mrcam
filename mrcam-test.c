#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <mrcal/mrcal.h>

#include "mrcam.h"



static const char* camera_name = "192.168.0.2";
static const int Nframes = 10;


#define MSG(fmt, ...) fprintf(stderr, "%s(%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define try(expr, ...) do {                             \
        if(!(expr))                                     \
        {                                                       \
            MSG("Failure!!! '" #expr "' is false" __VA_ARGS__); \
            goto done;                                  \
        }                                               \
    } while(0)



int main(int argc, char **argv)
{
    int result = 1;

    mrcam_t ctx = mrcam_init(camera_name);
    if(!mrcam_is_inited(&ctx))
        return 1;

    mrcal_image_bgr_t image_colormap = {};

    for(int i=0; i<Nframes; i++)
    {

        const char* outdir = "/tmp";
        const char* fmt = "%s/frame-%05d-cam%d.png"; // in variable to not confuse MSG()
        char filename[1024];
        try( snprintf(filename, sizeof(filename),
                      fmt,
                      outdir, i, 0) < (int)sizeof(filename) );


#if 0
        // 16-bit images; make heatmap

        mrcal_image_uint16_t image;
        if(!mrcam_get_frame_uint16(&image, 0, &ctx))
            return 1;

        if(image_colormap.data == NULL)
        {
            image_colormap.width  = image.width;
            image_colormap.height = image.height;
            image_colormap.stride = image_colormap.width * 3;

            try(NULL != (image_colormap.data = (mrcal_bgr_t*)malloc(image_colormap.height*image_colormap.stride)));
        }

        try(mrcal_apply_color_map_uint16(&image_colormap,
                                         &image,
                                         true, // auto_min
                                         true, // auto_max
                                         true,
                                         0, // in_min
                                         0, // in_max
                                         0,0,0));

        try(mrcal_image_bgr_save(filename, &image_colormap),
            ": writing to '%s'",
            filename);
#else
        // 8-bit images; write out grayscale
        mrcal_image_uint8_t image;
        if(!mrcam_get_frame_uint8(&image, 0, &ctx))
            return 1;

        try(mrcal_image_uint8_save(filename, &image),
            ": writing to '%s'",
            filename);
#endif


        MSG("Wrote '%s'", filename);
        sleep(1);
    }

    result = 0;

 done:
    mrcam_free(&ctx);

    return result;
}
