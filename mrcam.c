#include <stdlib.h>
#include <stdio.h>
#include <arv.h>

#include "mrcam.h"
#include "util.h"


// This is meant for debugging, so making it a global is fine
static bool verbose = false;


static const int     WIDTH                     = 1280;
static const int     HEIGHT                    = 1024;
static const char*   PIXEL_FORMAT              = "Mono8";
static const int     BPP                       = 1;
static const int     PAYLOAD_SIZE_EXPECTED_MAX = WIDTH*HEIGHT*BPP;




#define try(expr, ...) do {                                     \
        if(verbose)                                             \
            MSG("Evaluating   '" #expr "'");                    \
        if(!(expr))                                             \
        {                                                       \
            MSG("Failure!!! '" #expr "' is false" __VA_ARGS__); \
            goto done;                                          \
        }                                                       \
    } while(0)

#define try_arv(expr) do {                              \
        if(verbose)                                     \
            MSG("Calling   '" #expr "'");               \
        expr;                                           \
        if(error != NULL)                               \
        {                                               \
            MSG("Failure!!! '" #expr "' produced '%s'", \
                error->message);                        \
            g_error_free(error);                        \
            error = NULL;                               \
            goto done;                                  \
        }                                               \
    } while(0)

#define try_arv_with_extra_condition(expr, condition) do {              \
        if(verbose)                                                     \
            MSG("Calling   '" #expr "'");                               \
        expr;                                                           \
        if(error != NULL)                                               \
        {                                                               \
            MSG("Failure!!! '" #expr "' produced '%s'",                 \
                error->message);                                        \
            g_error_free(error);                                        \
            error = NULL;                                               \
            goto done;                                                  \
        }                                                               \
        if(!(condition))                                                \
        {                                                               \
            MSG("Failure!!! '" #expr "' produced no error, but the extra condition '" #condition "' failed"); \
            goto done;                                                  \
        }                                                               \
    } while(0)



#define DEFINE_INTERNALS(ctx)                                           \
    ArvCamera** camera __attribute__((unused)) = (ArvCamera**)(&(ctx)->camera); \
    ArvBuffer** buffer __attribute__((unused)) = (ArvBuffer**)(&(ctx)->buffer); \
    ArvStream** stream __attribute__((unused)) = (ArvStream**)(&(ctx)->stream)

bool mrcam_init(// out
                mrcam_t* ctx,
                // in
                const char* camera_name)
{
    bool result = false;

    GError *error  = NULL;

    *ctx = (mrcam_t){};

    DEFINE_INTERNALS(ctx);

    try_arv_with_extra_condition( *camera = arv_camera_new (camera_name,
                                                            &error),
                                  ARV_IS_CAMERA(*camera) );

    try_arv(arv_camera_set_integer(*camera, "Width",       WIDTH,        &error));
    try_arv(arv_camera_set_integer(*camera, "Height",      HEIGHT,       &error));
    try_arv(arv_camera_set_string (*camera, "PixelFormat", PIXEL_FORMAT, &error));

    // Some cameras start up with the test-pattern enabled. So I turn it off
    // unconditionally. This setting doesn't exist on all cameras. And if it
    // doesn't, this will fail, and I ignore the failure
    arv_camera_set_string (*camera, "TestPattern", "Off", &error);
    if(error != NULL)
        g_error_free(error);
    error = NULL;

    gint payload_size;
    try_arv(payload_size = arv_camera_get_payload(*camera, &error));

    try(payload_size <= PAYLOAD_SIZE_EXPECTED_MAX);

    try(*buffer = arv_buffer_new(payload_size, NULL));

    try_arv_with_extra_condition( *stream = arv_camera_create_stream(*camera, NULL, NULL, &error),
                                  ARV_IS_STREAM(*stream) );
    try_arv( arv_camera_set_acquisition_mode(*camera, ARV_ACQUISITION_MODE_SINGLE_FRAME, &error) );


    result = true;

 done:
    if(!result)
    {
        mrcam_free(ctx);
        // ctx is {} now
    }
    return result;
}

// deallocates everything, and sets all the pointers in ctx to NULL
void mrcam_free(mrcam_t* ctx)
{
    DEFINE_INTERNALS(ctx);

    g_clear_object(stream);
    g_clear_object(buffer);
    g_clear_object(camera);
}

bool mrcam_is_inited(mrcam_t* ctx)
{
    DEFINE_INTERNALS(ctx);

    return *camera != NULL;
}

void mrcam_set_verbose(void)
{
    verbose = true;
}


// Fill in the image. Assumes that the buffer has valid data
static
bool fill_image_generic(// out
                        mrcal_image_uint8_t* image, // type doesn't matter in this function
                        // in
                        ArvBuffer* buffer,
                        const size_t sizeof_pixel)
{
    bool result = false;
    size_t size;

    image->width  = arv_buffer_get_image_width (buffer);
    image->height = arv_buffer_get_image_height(buffer);
    image->data   = (uint8_t*)arv_buffer_get_image_data(buffer, &size);

    image->stride = image->width * sizeof_pixel;
    if((size_t)(image->height*image->stride) != size)
    {
        MSG("Unexpected image dimensions. Expected buffersize = height*stride = height*width*bytes_per_pixel = %zd * %zd * %zd = %zd, but got %zd",
            (size_t)(image->height), (size_t)(image->width), sizeof_pixel,
            (size_t)(image->height*image->stride),
            size);
        goto done;
    }

    result = true;

 done:
    return result;
}

static
bool fill_image_uint8(// out
                      mrcal_image_uint8_t* image,
                      // in
                      ArvBuffer* buffer)
{
    ArvPixelFormat pixfmt = arv_buffer_get_image_pixel_format(buffer);
    if(!(pixfmt == ARV_PIXEL_FORMAT_MONO_8))
        return false;
    return fill_image_generic((mrcal_image_uint8_t*)image, buffer, sizeof(uint8_t));
}

static
bool fill_image_uint16(// out
                       mrcal_image_uint16_t* image,
                       // in
                       ArvBuffer* buffer)
{
    ArvPixelFormat pixfmt = arv_buffer_get_image_pixel_format(buffer);
    if(!(pixfmt == ARV_PIXEL_FORMAT_MONO_10 ||
         pixfmt == ARV_PIXEL_FORMAT_MONO_12 ||
         pixfmt == ARV_PIXEL_FORMAT_MONO_14 ||
         pixfmt == ARV_PIXEL_FORMAT_MONO_16))
        return false;
    return fill_image_generic((mrcal_image_uint8_t*)image, buffer, sizeof(uint16_t));
}

static
bool get_frame__internal(mrcam_t* ctx,
                         const uint64_t timeout_us)
{
    DEFINE_INTERNALS(ctx);

    bool result = false;

    GError     *error       = NULL;
    ArvBuffer*  buffer_here = NULL;
    bool        acquiring   = false;

    if(!ctx->buffer_is_pushed_to_stream)
    {
        arv_stream_push_buffer(*stream, *buffer);
        ctx->buffer_is_pushed_to_stream = true;
    }

    try_arv( arv_camera_start_acquisition(*camera, &error));
    acquiring = true;

    if (timeout_us > 0) buffer_here = arv_stream_timeout_pop_buffer(*stream, timeout_us);
    else                buffer_here = arv_stream_pop_buffer        (*stream);

    try_arv(arv_camera_stop_acquisition(*camera, &error));
    acquiring = false;

    try(buffer_here == *buffer);
    try(ARV_IS_BUFFER(*buffer));

    // We have our buffer back. It may or may not contain valid data, but it
    // isn't in the stream anymore
    ctx->buffer_is_pushed_to_stream = false;

    ArvBufferStatus status = arv_buffer_get_status(*buffer);
    // All the statuses from arvbuffer.h, as of aravis 0.8.30
#define LIST_STATUS(_)                          \
  _(ARV_BUFFER_STATUS_UNKNOWN)                  \
  _(ARV_BUFFER_STATUS_SUCCESS)                  \
  _(ARV_BUFFER_STATUS_CLEARED)                  \
  _(ARV_BUFFER_STATUS_TIMEOUT)                  \
  _(ARV_BUFFER_STATUS_MISSING_PACKETS)          \
  _(ARV_BUFFER_STATUS_WRONG_PACKET_ID)          \
  _(ARV_BUFFER_STATUS_SIZE_MISMATCH)            \
  _(ARV_BUFFER_STATUS_FILLING)                  \
  _(ARV_BUFFER_STATUS_ABORTED)                  \
  _(ARV_BUFFER_STATUS_PAYLOAD_NOT_SUPPORTED)
#define CHECK(s) else if(status == s) { MSG("ERROR: arv_stream_pop_buffer() says " #s); goto done; }
    if(status == ARV_BUFFER_STATUS_SUCCESS) ;
    LIST_STATUS(CHECK)
    else
    {
        MSG("ERROR: arv_stream_pop_buffer() returned unknown status %d", status);
        goto done;
    }
#undef LIST_STATUS
#undef CHECK


    ArvBufferPayloadType payload_type = arv_buffer_get_payload_type(*buffer);
    // All the payload_types from arvbuffer.h, as of aravis 0.8.30
#define LIST_PAYLOAD_TYPE(_)                            \
  _(ARV_BUFFER_PAYLOAD_TYPE_UNKNOWN)                    \
  _(ARV_BUFFER_PAYLOAD_TYPE_NO_DATA)                    \
  _(ARV_BUFFER_PAYLOAD_TYPE_IMAGE)                      \
  _(ARV_BUFFER_PAYLOAD_TYPE_RAWDATA)                    \
  _(ARV_BUFFER_PAYLOAD_TYPE_FILE)                       \
  _(ARV_BUFFER_PAYLOAD_TYPE_CHUNK_DATA)                 \
  _(ARV_BUFFER_PAYLOAD_TYPE_EXTENDED_CHUNK_DATA)        \
  _(ARV_BUFFER_PAYLOAD_TYPE_JPEG)                       \
  _(ARV_BUFFER_PAYLOAD_TYPE_JPEG2000)                   \
  _(ARV_BUFFER_PAYLOAD_TYPE_H264)                       \
  _(ARV_BUFFER_PAYLOAD_TYPE_MULTIZONE_IMAGE)            \
  _(ARV_BUFFER_PAYLOAD_TYPE_MULTIPART)
#define CHECK(s) else if(payload_type == s) { MSG("ERROR: arv_stream_pop_buffer() says " #s "; I only know about ARV_BUFFER_PAYLOAD_TYPE_IMAGE"); goto done; }
    if(payload_type == ARV_BUFFER_PAYLOAD_TYPE_IMAGE) ;
    LIST_PAYLOAD_TYPE(CHECK)
    else
    {
        MSG("ERROR: arv_stream_pop_buffer() returned unknown payload_type %d", payload_type);
        goto done;
    }
#undef LIST_PAYLOAD_TYPE
#undef CHECK

    result = true;

 done:

    if(acquiring)
        // if still aquiring for some reason, stop that, with no error checking
        arv_camera_stop_acquisition(*camera, &error);

    if(!result)
        return NULL;
    return buffer;
}

// timeout_us=0 means "wait forever"
bool mrcam_get_frame_uint8(// out
                           mrcal_image_uint8_t* image,
                           // in
                           const uint64_t timeout_us,
                           mrcam_t* ctx)
{
    DEFINE_INTERNALS(ctx);

    if(!get_frame__internal(ctx, timeout_us))
        return false;

    return fill_image_uint8(image, *buffer);
}

// timeout_us=0 means "wait forever"
bool mrcam_get_frame_uint16(// out
                            mrcal_image_uint16_t* image,
                            // in
                            const uint64_t timeout_us,
                            mrcam_t* ctx)
{
    DEFINE_INTERNALS(ctx);

    if(!get_frame__internal(ctx, timeout_us))
        return false;

    return fill_image_uint16(image, *buffer);
}
