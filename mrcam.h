#pragma once

#include <mrcal/mrcal-image.h>

// I define this myself, in order to avoid requiring the user to #include
// <arv.h>. This is a few common formats. MRCAM_PIXFMT_... has an equivalent
// ARV_PIXEL_FORMAT
#define LIST_MRCAM_PIXFMT(_)                    \
  _(MONO_8,  1)                                 \
  /* Each pixel takes up 16 bits. NOT packed */ \
  _(MONO_10, 2)                                 \
  _(MONO_12, 2)                                 \
  _(MONO_14, 2)                                 \
  _(MONO_16, 2)

typedef enum
{
#define MRCAM_PIXFMT_ENUM(name, bytes_per_pixel) MRCAM_PIXFMT_ ## name,
    LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_ENUM) MRCAM_PIXFMT_COUNT
#undef MRCAM_PIXFMT_ENUM
} mrcam_pixfmt_t;

typedef struct
{
    // void to not require including arv.h
    void* camera;
    void* buffer;
    void* stream;

    bool buffer_is_pushed_to_stream : 1;

    // Maybe I don't NEED to store these here, but it makes life easier
    mrcam_pixfmt_t pixfmt;
    int bytes_per_pixel;

} mrcam_t;

// camera_name = NULL means "first available camera"
bool mrcam_init(// out
                mrcam_t* ctx,
                // in
                const char* camera_name,
                const mrcam_pixfmt_t pixfmt,
                // if either is <=0, we try to autodetect by asking the camera
                // for WidthMax and HeightMax. Some cameras report the native
                // resolution of the imager there, but some others report bugus
                // values, and the user then MUST provide the correct
                // dimensions
                int width,
                int height);

// deallocates everything, and sets all the pointers in ctx to NULL
void mrcam_free(mrcam_t* ctx);

bool mrcam_is_inited(mrcam_t* ctx);

void mrcam_set_verbose(void);

// timeout_us=0 means "wait forever"
//
// The image structure should exist in memory, but the data buffer doesn't need
// to be preallocated or freed. Usage:
//   mrcal_image_uintX_t image;
//   mrcam_get_frame_uintX(&image, timeout_us, &ctx);
//   // no free(image.data)
bool mrcam_get_frame_uint8( // out
                            mrcal_image_uint8_t* image,
                            // in
                            const uint64_t timeout_us,
                            mrcam_t* ctx);
bool mrcam_get_frame_uint16(// out
                            mrcal_image_uint16_t* image,
                            // in
                            const uint64_t timeout_us,
                            mrcam_t* ctx);

