#pragma once

#include <mrcal/mrcal-image.h>

// I define this myself, in order to avoid requiring the user to #include
// <arv.h>. This is a few common formats. MRCAM_PIXFMT_... has an equivalent
// ARV_PIXEL_FORMAT.
//
// An ffmpeg pixel format of AV_PIX_FMT_NONE means "the input is already packed,
// and there's no need to invoke ffmmpeg at all"
#define LIST_MRCAM_PIXFMT(_)                            \
  _(MONO_8,        uint8,  AV_PIX_FMT_NONE)             \
  /* Each pixel takes up 16 bits. NOT packed */         \
  _(MONO_10,       uint16, AV_PIX_FMT_NONE)             \
  _(MONO_12,       uint16, AV_PIX_FMT_NONE)             \
  _(MONO_14,       uint16, AV_PIX_FMT_NONE)             \
  _(MONO_16,       uint16, AV_PIX_FMT_NONE)             \
  _(BAYER_GR_8,    bgr,    AV_PIX_FMT_BAYER_GRBG8)      \
  _(BAYER_RG_8,    bgr,    AV_PIX_FMT_BAYER_RGGB8)      \
  _(BAYER_GB_8,    bgr,    AV_PIX_FMT_BAYER_GBRG8)      \
  _(BAYER_BG_8,    bgr,    AV_PIX_FMT_BAYER_BGGR8)      \
  _(RGB_8_PACKED,  bgr,    AV_PIX_FMT_NONE)             \
  _(BGR_8_PACKED,  bgr,    AV_PIX_FMT_NONE)


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
