#include <stdlib.h>
#include <stdio.h>
#include <arv.h>

#include <libswscale/swscale.h>
#include <libavutil/pixfmt.h>
#include <libavutil/imgutils.h>


#include "mrcam.h"
#include "util.h"


// This is meant for debugging, so making it a global is fine
static bool verbose = false;

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
    ArvBuffer** buffer __attribute__((unused)) = (ArvBuffer**)(&(ctx)->buffer)







static
ArvPixelFormat get_ArvPixelFormat(mrcam_pixfmt_t pixfmt)
{
    switch(pixfmt)
    {
#define MRCAM_PIXFMT_CHOOSE(name, ...) case MRCAM_PIXFMT_ ## name: return ARV_PIXEL_FORMAT_ ## name;
    LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_CHOOSE)
    default:
        MSG("ERROR: unknown mrcam_pixfmt_t = %d", (int)pixfmt);
        return 0;
    }
#undef MRCAM_PIXFMT_CHOOSE

}

static
int get_bytes_per_pixel(mrcam_pixfmt_t pixfmt)
{
    switch(pixfmt)
    {
#define MRCAM_PIXFMT_CHOOSE(name, bytes_per_pixel, ...) case MRCAM_PIXFMT_ ## name: return bytes_per_pixel;
    LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_CHOOSE)
    default:
        MSG("ERROR: unknown mrcam_pixfmt_t = %d", (int)pixfmt);
        return 0;
    }
#undef MRCAM_PIXFMT_CHOOSE
}

static
bool get_is_color(mrcam_pixfmt_t pixfmt)
{
    switch(pixfmt)
    {
#define MRCAM_PIXFMT_CHOOSE(name, bytes_per_pixel, is_color) case MRCAM_PIXFMT_ ## name: return is_color;
    LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_CHOOSE)
    default:
        MSG("ERROR: unknown mrcam_pixfmt_t = %d", (int)pixfmt);
        return 0;
    }
#undef MRCAM_PIXFMT_CHOOSE
}

static
const char* get_pixel_format_string(mrcam_pixfmt_t pixfmt)
{
    switch(pixfmt)
    {
#define MRCAM_PIXFMT_CHOOSE(name, ...) case MRCAM_PIXFMT_ ## name: return #name;
    LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_CHOOSE)
    default:
        MSG("ERROR: unknown mrcam_pixfmt_t = %d", (int)pixfmt);
        return "UNKNOWN";
    }
#undef MRCAM_PIXFMT_CHOOSE
}


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
                int height)
{
    bool result = false;

    GError *error  = NULL;

    *ctx = (mrcam_t){};

    DEFINE_INTERNALS(ctx);

    try_arv_with_extra_condition( *camera = arv_camera_new (camera_name,
                                                            &error),
                                  ARV_IS_CAMERA(*camera) );

    if(width <= 0 || height <= 0)
    {
        // Use WidthMax and HeightMax
        gint dummy,_width,_height;
        try_arv( arv_camera_get_width_bounds( *camera, &dummy, &_width,  &error) );
        try_arv( arv_camera_get_height_bounds(*camera, &dummy, &_height, &error) );
        width  = (int)_width;
        height = (int)_height;
    }

    try_arv(arv_camera_set_integer(*camera, "Width",  width,  &error));
    try_arv(arv_camera_set_integer(*camera, "Height", height, &error));

    ctx->pixfmt          = pixfmt;
    ctx->is_color        = get_is_color(pixfmt);
    ctx->bytes_per_pixel = get_bytes_per_pixel(pixfmt);
    if(ctx->bytes_per_pixel == 0)
        goto done;

    ArvPixelFormat arv_pixfmt = get_ArvPixelFormat(pixfmt);
    if(arv_pixfmt == 0)
        goto done;

    arv_camera_set_pixel_format(*camera, arv_pixfmt, &error);
    if(error != NULL)
    {
        MSG("Failure!!! Couldn't set the requested pixel format: '%s': '%s'",
            get_pixel_format_string(pixfmt),
            error->message);
        g_error_free(error);
        error = NULL;


        MSG("Available pixel formats supported by the camera:");
        const char** available_pixel_formats;
        guint n_pixel_formats;
        try_arv( available_pixel_formats =
                 arv_camera_dup_available_pixel_formats_as_strings(*camera, &n_pixel_formats, &error) );
        for(guint i=0; i<n_pixel_formats; i++)
            MSG("  %s",
                available_pixel_formats[i]);
        g_free(available_pixel_formats);

        goto done;
    }

    // Some cameras start up with the test-pattern enabled. So I turn it off
    // unconditionally. This setting doesn't exist on all cameras. And if it
    // doesn't, this will fail, and I ignore the failure
    arv_camera_set_string (*camera, "TestPattern", "Off", &error);
    if(error != NULL)
        g_error_free(error);
    error = NULL;

    gint payload_size;
    try_arv(payload_size = arv_camera_get_payload(*camera, &error));

    // The payload is sometimes larger than expected, if the camera wants to pad
    // the data. In that case I can simply ignore the trailing bits. If it's
    // smaller than expected, however, then I don't know what to do, and I give
    // up
    if(payload_size < width*height*ctx->bytes_per_pixel)
    {
        MSG("Error! Requested pixel format '%s' says it wants payload_size=%d. But this is smaller than expected width*height*bytes_per_pixel = %d*%d*%d = %d",
            get_pixel_format_string(pixfmt),
            payload_size,
            width,height,ctx->bytes_per_pixel,
            width*height*ctx->bytes_per_pixel);
        goto done;
    }

    try(*buffer = arv_buffer_new(payload_size, NULL));

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

    // The payload is sometimes larger than expected, if the camera wants to pad
    // the data. In that case I can simply ignore the trailing bits. If it's
    // smaller than expected, however, then I don't know what to do, and I give
    // up
    if((size_t)(image->height*image->stride) > size)
    {
        MSG("Unexpected image dimensions: insufficient data returned. Expected buffersize = height*stride = height*width*bytes_per_pixel = %zd * %zd * %zd = %zd, but got %zd",
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

#define MRCAM_PIXFMT_CHECK(name, bytes_per_pixel, is_color) || (bytes_per_pixel==1 && !is_color && pixfmt == ARV_PIXEL_FORMAT_ ## name)
    const bool acceptable_pixfmt =
        false LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_CHECK);
#undef MRCAM_PIXFMT_CHECK
    try(acceptable_pixfmt);

    return fill_image_generic((mrcal_image_uint8_t*)image, buffer, sizeof(uint8_t));

 done:
    return false;
}

static
bool fill_image_uint16(// out
                       mrcal_image_uint16_t* image,
                       // in
                       ArvBuffer* buffer)
{
    ArvPixelFormat pixfmt = arv_buffer_get_image_pixel_format(buffer);

#define MRCAM_PIXFMT_CHECK(name, bytes_per_pixel, is_color) || (bytes_per_pixel==2 && !is_color && pixfmt == ARV_PIXEL_FORMAT_ ## name)
    const bool acceptable_pixfmt =
        false LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_CHECK);
#undef MRCAM_PIXFMT_CHECK
    try(acceptable_pixfmt);

    return fill_image_generic((mrcal_image_uint8_t*)image, buffer, sizeof(uint16_t));


 done:
    return false;
}

static
bool fill_image_bgr(// out
                    mrcal_image_bgr_t* image,
                    // in
                    ArvBuffer* buffer)
{
    ArvPixelFormat pixfmt = arv_buffer_get_image_pixel_format(buffer);

#define MRCAM_PIXFMT_CHECK(name, bytes_per_pixel, is_color) || (is_color && pixfmt == ARV_PIXEL_FORMAT_ ## name)
    const bool acceptable_pixfmt =
        false LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_CHECK);
#undef MRCAM_PIXFMT_CHECK
    try(acceptable_pixfmt);

    bool result = false;

    // I have SOME color format. Today I don't actually support them all, so I
    // put the known logic here
    if(pixfmt == ARV_PIXEL_FORMAT_BGR_8_PACKED)
        return fill_image_generic((mrcal_image_uint8_t*)image, buffer, sizeof(mrcal_bgr_t));

    const size_t sizeof_pixel = 3;

    if(pixfmt == ARV_PIXEL_FORMAT_BAYER_RG_8)
    {
        const unsigned int width  = (unsigned int)arv_buffer_get_image_width (buffer);
        const unsigned int height = (unsigned int)arv_buffer_get_image_height(buffer);

        const int output_stride = width*3;
        const int input_stride  = width*1;

        static struct SwsContext* sws_context;
        static uint8_t*           image_buffer;

        if(sws_context == NULL)
        {
            try(NULL !=
                (sws_context =
                 sws_getContext(// source
                                width,height,
                                AV_PIX_FMT_BAYER_RGGB8,

                                // destination
                                width,height,
                                AV_PIX_FMT_BGR24,

                                // misc stuff
                                SWS_POINT, NULL, NULL, NULL)));

            // If I were to dealloc, I'd do this:
            //   if(sws_context)
            //   {
            //       sws_freeContext(sws_context);
            //       sws_context = NULL;
            //   }

            try(NULL != (image_buffer = malloc(output_stride * height)));
        }

        size_t size;
        const uint8_t* bytes_frame = (uint8_t*)arv_buffer_get_image_data(buffer, &size);
        // The payload is sometimes larger than expected, if the camera wants to pad
        // the data. In that case I can simply ignore the trailing bits. If it's
        // smaller than expected, however, then I don't know what to do, and I give
        // up
        if((size_t)(height*input_stride) > size)
        {
            MSG("Unexpected image dimensions: insufficient data returned. Expected buffersize = height*stride = height*width*bytes_per_pixel = %zd * %zd * %zd = %zd, but got %zd",
                (size_t)(height), (size_t)(width), sizeof_pixel,
                (size_t)(height*output_stride),
                size);
            goto done;
        }

        // This is overly-complex because ffmpeg supports lots of weird
        // pixel formats, including planar ones
        int      scale_stride_value[4];
        uint8_t* scale_source_value[4];
        try(0 < av_image_fill_arrays(scale_source_value, scale_stride_value,
                                     bytes_frame,
                                     AV_PIX_FMT_BAYER_RGGB8,
                                     width,height,
                                     1) );

        sws_scale(sws_context,
                  // source
                  (const uint8_t*const*)scale_source_value,
                  scale_stride_value,
                  0, height,
                  // destination buffer, stride
                  (uint8_t*const*)&image_buffer,
                  &output_stride);

        image->width  = width;
        image->height = height;
        image->data   = (mrcal_bgr_t*)image_buffer;
        image->stride = output_stride;

        result = true;
        goto done;
    }

    MSG("I don't know how to handle ARV_PIXEL_FORMAT %d", pixfmt);
    goto done;

 done:
    return result;
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


    ArvStream* stream;

    try_arv_with_extra_condition( stream = arv_camera_create_stream(*camera, NULL, NULL, &error),
                                  ARV_IS_STREAM(stream) );
    try_arv( arv_camera_set_acquisition_mode(*camera, ARV_ACQUISITION_MODE_SINGLE_FRAME, &error) );

    arv_stream_push_buffer(stream, *buffer);

    try_arv( arv_camera_start_acquisition(*camera, &error));
    acquiring = true;

    if (timeout_us > 0) buffer_here = arv_stream_timeout_pop_buffer(stream, timeout_us);
    else                buffer_here = arv_stream_pop_buffer        (stream);

    try_arv(arv_camera_stop_acquisition(*camera, &error));
    acquiring = false;

    try(buffer_here == *buffer);
    try(ARV_IS_BUFFER(*buffer));

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

    g_clear_object(&stream);

    return result;
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

// timeout_us=0 means "wait forever"
bool mrcam_get_frame_bgr(// out
                         mrcal_image_bgr_t* image,
                         // in
                         const uint64_t timeout_us,
                         mrcam_t* ctx)
{
    DEFINE_INTERNALS(ctx);

    if(!get_frame__internal(ctx, timeout_us))
        return false;

    return fill_image_bgr(image, *buffer);
}
