#pragma once

#include <mrcal/mrcal-image.h>

// I define this myself, in order to avoid requiring the user to #include
// <arv.h>. This is a few common formats. MRCAM_PIXFMT_... has an equivalent
// ARV_PIXEL_FORMAT
#define LIST_MRCAM_PIXFMT(_)                                            \
  _(MONO_8,        uint8)                                               \
  /* Each pixel takes up 16 bits. NOT packed */                         \
  _(MONO_10,       uint16)                                              \
  _(MONO_12,       uint16)                                              \
  _(MONO_14,       uint16)                                              \
  _(MONO_16,       uint16)                                              \
  _(BAYER_GR_8,    bgr)                                                 \
  _(BAYER_RG_8,    bgr)                                                 \
  _(BAYER_GB_8,    bgr)                                                 \
  _(BAYER_BG_8,    bgr)                                                 \
  /* Higher-depth color images are converted down to 8 bits/pixel for now*/ \
  _(BAYER_GR_10,   bgr)                                                 \
  _(BAYER_RG_10,   bgr)                                                 \
  _(BAYER_GB_10,   bgr)                                                 \
  _(BAYER_BG_10,   bgr)                                                 \
  _(BAYER_GR_12,   bgr)                                                 \
  _(BAYER_RG_12,   bgr)                                                 \
  _(BAYER_GB_12,   bgr)                                                 \
  _(BAYER_BG_12,   bgr)                                                 \
  _(BAYER_GR_16,   bgr)                                                 \
  _(BAYER_RG_16,   bgr)                                                 \
  _(BAYER_GB_16,   bgr)                                                 \
  _(BAYER_BG_16,   bgr)                                                 \
  _(RGB_8_PACKED,  bgr)                                                 \
  _(BGR_8_PACKED,  bgr)                                                 \
  _(RGBA_8_PACKED, bgr)                                                 \
  _(BGRA_8_PACKED, bgr)                                                 \
  _(RGB_10_PACKED, bgr)                                                 \
  _(BGR_10_PACKED, bgr)                                                 \
  _(RGB_12_PACKED, bgr)                                                 \
  _(BGR_12_PACKED, bgr)


typedef enum
{
#define ENUM(name, ...) MRCAM_PIXFMT_ ## name,
    LIST_MRCAM_PIXFMT(ENUM) MRCAM_PIXFMT_COUNT
#undef ENUM
} mrcam_pixfmt_t;

typedef enum {MRCAM_UNKNOWN = -1,
              MRCAM_uint8, MRCAM_uint16, MRCAM_bgr } mrcam_output_type_t;
mrcam_output_type_t mrcam_output_type(mrcam_pixfmt_t pixfmt);


typedef struct
{
    // arv stuff; void to not require including arv.h
    void* camera;
    void* buffer;

    // Details about the requested pixel format, that I'm using to talk to the
    // camera. I don't NEED to store all these here, but it makes life easier
    mrcam_pixfmt_t pixfmt;

    // Used to convert from non-trivial pixel formats coming out of the camera
    // to unpacked bits mrcam reports. Needed for bayered or packed formats.
    // Unused for all others
    struct SwsContext* sws_context;
    uint8_t*           output_image_buffer;

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
