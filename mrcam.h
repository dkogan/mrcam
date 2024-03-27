#pragma once

#include <mrcal/mrcal-image.h>


typedef struct
{
    // void to not require including arv.h
    void* camera;
    void* buffer;
    void* stream;

    bool buffer_is_pushed_to_stream;
} mrcam_t;

// camera_name = NULL means "first available camera"
mrcam_t mrcam_init(const char* camera_name);

// deallocates everything, and sets all the pointers in ctx to NULL
void mrcam_free(mrcam_t* ctx);

bool mrcam_is_inited(mrcam_t* ctx);

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

