#pragma once

#include <mrcal/mrcal-image.h>

// I define this myself, in order to avoid requiring the user to #include
// <arv.h>. This is a few common formats. MRCAM_PIXFMT_... has an equivalent
// ARV_PIXEL_FORMAT
#define LIST_MRCAM_PIXFMT(_)                    \
  _(MONO_8,        1, false)                    \
  /* Each pixel takes up 16 bits. NOT packed */ \
  _(MONO_10,       2, false)                    \
  _(MONO_12,       2, false)                    \
  _(MONO_14,       2, false)                    \
  _(MONO_16,       2, false)                    \
  _(BAYER_GR_8,    1, true)                     \
  _(BAYER_RG_8,    1, true)                     \
  _(BAYER_GB_8,    1, true)                     \
  _(BAYER_BG_8,    1, true)                     \
  _(BAYER_GR_10,   2, true)                     \
  _(BAYER_RG_10,   2, true)                     \
  _(BAYER_GB_10,   2, true)                     \
  _(BAYER_BG_10,   2, true)                     \
  _(BAYER_GR_12,   2, true)                     \
  _(BAYER_RG_12,   2, true)                     \
  _(BAYER_GB_12,   2, true)                     \
  _(BAYER_BG_12,   2, true)                     \
  _(BAYER_GR_16,   2, true)                     \
  _(BAYER_RG_16,   2, true)                     \
  _(BAYER_GB_16,   2, true)                     \
  _(BAYER_BG_16,   2, true)                     \
  _(RGB_8_PACKED,  3, true)                     \
  _(BGR_8_PACKED,  3, true)                     \
  _(RGBA_8_PACKED, 4, true)                     \
  _(BGRA_8_PACKED, 4, true)                     \
  _(RGB_10_PACKED, 6, true)                     \
  _(BGR_10_PACKED, 6, true)                     \
  _(RGB_12_PACKED, 6, true)                     \
  _(BGR_12_PACKED, 6, true)


typedef enum
{
#define ENUM(name, bytes_per_pixel, is_color) MRCAM_PIXFMT_ ## name,
    LIST_MRCAM_PIXFMT(ENUM) MRCAM_PIXFMT_COUNT
#undef ENUM
} mrcam_pixfmt_t;

typedef struct
{
    // void to not require including arv.h
    void* camera;
    void* buffer;

    // Maybe I don't NEED to store these here, but it makes life easier
    mrcam_pixfmt_t pixfmt;
    int            bytes_per_pixel;
    bool           is_color : 1;
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
bool mrcam_get_frame_bgr(   // out
                            mrcal_image_bgr_t* image,
                            // in
                            const uint64_t timeout_us,
                            mrcam_t* ctx);
