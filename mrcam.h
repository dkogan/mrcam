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
} mrcam_t;

// camera_name = NULL means "first available camera"
bool mrcam_init(// out
                mrcam_t* ctx,
                // in
                const char* camera_name,
                const mrcam_pixfmt_t pixfmt);

// deallocates everything, and sets all the pointers in ctx to NULL
void mrcam_free(mrcam_t* ctx);

bool mrcam_is_inited(mrcam_t* ctx);

void mrcam_set_verbose(void);

// timeout_us=0 means "wait forever"
bool mrcam_get_frame_uint8( // out
                            mrcal_image_uint8_t* image,
                            // in
                            const uint64_t timeout_us,
                            mrcam_t* ctx);

// timeout_us=0 means "wait forever"
bool mrcam_get_frame_uint16(// out
                            mrcal_image_uint16_t* image,
                            // in
                            const uint64_t timeout_us,
                            mrcam_t* ctx);

