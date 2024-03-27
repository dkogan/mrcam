#pragma once

#include <mrcal/mrcal-image.h>

typedef enum
{
    MRCAM_PIXFMT_UINT8,

    // Each pixel takes up 16 bits. NOT packed
    MRCAM_PIXFMT_UINT10,
    MRCAM_PIXFMT_UINT12,
    MRCAM_PIXFMT_UINT14,
    MRCAM_PIXFMT_UINT16
} mrcal_pixfmt_t;

typedef struct
{
    const int width, height;
    const mrcal_pixfmt_t pixfmt;
} mrcam_settings_t;



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
                const char* camera_name);

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

