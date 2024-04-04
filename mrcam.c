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
            g_clear_error(&error);                      \
            goto done;                                  \
        }                                               \
    } while(0)

#define try_arv_ok_if(expr, ok_if_cond) do {            \
        if(verbose)                                     \
            MSG("Calling   '" #expr "'");               \
        expr;                                           \
        if(error != NULL && !(ok_if_cond))              \
        {                                               \
            MSG("Failure!!! '" #expr "' produced '%s'", \
                error->message);                        \
            g_clear_error(&error);                      \
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
            g_clear_error(&error);                                      \
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






mrcam_output_type_t mrcam_output_type(mrcam_pixfmt_t pixfmt)
{
    switch(pixfmt)
    {
#define CHOOSE(name, T, ...) case MRCAM_PIXFMT_ ## name: return MRCAM_ ## T;
        LIST_MRCAM_PIXFMT(CHOOSE)
    default: break;
#undef CHOOSE
    }

    MSG("Unknown pixfmt. This is a bug");
    return MRCAM_UNKNOWN;
}


static
ArvPixelFormat pixfmt__ArvPixelFormat(mrcam_pixfmt_t pixfmt)
{
    switch(pixfmt)
    {
#define CHOOSE(name, ...) case MRCAM_PIXFMT_ ## name: return ARV_PIXEL_FORMAT_ ## name;
    LIST_MRCAM_PIXFMT(CHOOSE)
    default:
        MSG("ERROR: unknown mrcam_pixfmt_t = %d", (int)pixfmt);
        return 0;
    }
#undef CHOOSE

}

static
const char* pixfmt__name(mrcam_pixfmt_t pixfmt)
{
    switch(pixfmt)
    {
#define CHOOSE(name, ...) case MRCAM_PIXFMT_ ## name: return #name;
    LIST_MRCAM_PIXFMT(CHOOSE)
    default:
        MSG("ERROR: unknown mrcam_pixfmt_t = %d", (int)pixfmt);
        return "UNKNOWN";
    }
#undef CHOOSE
}

static
bool is_little_endian(void)
{
    union
    {
        uint16_t u16;
        uint8_t  u8[2];
    } u = {.u16 = 1};
    return u.u8[0] == 1;
}

// Returns the input/output pixel formats, as denoted by ffmpeg. AV_PIX_FMT_NONE
// means "the data already comes in fully unpacked; no ffmpeg processing is
// needed"
static
bool pixfmt__av_pixfmt(// input
                       enum AVPixelFormat* av_pixfmt_input,
                       enum AVPixelFormat* av_pixfmt_output,
                       // output
                       mrcam_pixfmt_t pixfmt)
{
    // output
    switch(mrcam_output_type(pixfmt))
    {
    case MRCAM_uint8:
        *av_pixfmt_output = AV_PIX_FMT_GRAY8;
        break;
    case MRCAM_uint16:
        *av_pixfmt_output = is_little_endian() ?
            AV_PIX_FMT_GRAY16LE :
            AV_PIX_FMT_GRAY16BE;
        break;
    case MRCAM_bgr:
        *av_pixfmt_output = AV_PIX_FMT_BGR24;
        break;
    default:
        MSG("Unknown pixfmt. This is a bug");
        return false;
    }

    switch(pixfmt)
    {

#define CHOOSE(name, T, av_pixfmt)              \
        case MRCAM_PIXFMT_ ## name:             \
            *av_pixfmt_input = av_pixfmt;       \
            break;

        LIST_MRCAM_PIXFMT(CHOOSE)
    default:
        MSG("ERROR: unknown mrcam_pixfmt_t = '%s'",
            pixfmt__name(pixfmt));
        return false;
#undef CHOOSE
    }

    return true;
}

static
int pixfmt__output_bytes_per_pixel(mrcam_pixfmt_t pixfmt)
{
    switch(mrcam_output_type(pixfmt))
    {
    case MRCAM_uint8:  return 1;
    case MRCAM_uint16: return 2;
    case MRCAM_bgr:    return 3;
    default: ;
    }
    return 0;
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

    ctx->pixfmt = pixfmt;

    ArvPixelFormat arv_pixfmt = pixfmt__ArvPixelFormat(pixfmt);
    if(arv_pixfmt == 0)
        goto done;

    arv_camera_set_pixel_format(*camera, arv_pixfmt, &error);
    if(error != NULL)
    {
        MSG("Failure!!! Couldn't set the requested pixel format: '%s': '%s'",
            pixfmt__name(pixfmt),
            error->message);
        g_clear_error(&error);


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
    // unconditionally. This setting doesn't exist on all cameras; if it
    // doesn't, I ignore that failure


    try_arv_ok_if(arv_camera_set_string (*camera, "TestPattern", "Off", &error),
                  error->code == ARV_DEVICE_ERROR_FEATURE_NOT_FOUND);
    if(error != NULL)
    {
        // No TestPattern setting exists. I ignore the error
        g_clear_error(&error);
    }

    gint payload_size;
    try_arv(payload_size = arv_camera_get_payload(*camera, &error));
    try(*buffer = arv_buffer_new(payload_size, NULL));

    enum AVPixelFormat av_pixfmt_input, av_pixfmt_output;
    try(pixfmt__av_pixfmt(&av_pixfmt_input, &av_pixfmt_output,
                          pixfmt));

    if(av_pixfmt_input == AV_PIX_FMT_NONE)
    {
        // The pixel format is already unpacked. We don't need to do any
        // conversions
    }
    else
    {
        // We need to convert stuff. Use libswscale (ffmpeg) to set up the
        // converter
        try(NULL !=
            (ctx->sws_context =
             sws_getContext(// source
                            width,height,
                            av_pixfmt_input,

                            // destination
                            width,height,
                            av_pixfmt_output,

                            // misc stuff
                            SWS_POINT, NULL, NULL, NULL)));
        try(NULL != (ctx->output_image_buffer =
                     malloc(pixfmt__output_bytes_per_pixel(pixfmt) *
                            width * height)));
    }

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

    if(ctx->sws_context)
    {
        sws_freeContext(ctx->sws_context);
        ctx->sws_context = NULL;
    }

    free(ctx->output_image_buffer);
    ctx->output_image_buffer = NULL;
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
bool fill_image_unpacked(// out
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
bool fill_image_swscale(// out
                        mrcal_image_uint8_t* image, // any type
                        // in
                        ArvBuffer* buffer,
                        mrcam_t* ctx)
{
    bool result = false;

    const unsigned int width  = (unsigned int)arv_buffer_get_image_width (buffer);
    const unsigned int height = (unsigned int)arv_buffer_get_image_height(buffer);

    const int output_stride = width*pixfmt__output_bytes_per_pixel(ctx->pixfmt);

    enum AVPixelFormat av_pixfmt_input, av_pixfmt_output;
    try(pixfmt__av_pixfmt(&av_pixfmt_input, &av_pixfmt_output,
                          ctx->pixfmt));

    size_t size;
    const uint8_t* bytes_frame = (uint8_t*)arv_buffer_get_image_data(buffer, &size);

    // This is overly-complex because ffmpeg supports lots of weird
    // pixel formats, including planar ones
    int      scale_stride_value[4];
    uint8_t* scale_source_value[4];
    try(0 < av_image_fill_arrays(scale_source_value, scale_stride_value,
                                 bytes_frame,
                                 av_pixfmt_input,
                                 width,height,
                                 1) );

    const int input_stride = scale_stride_value[0];
    // The payload is sometimes larger than expected, if the camera wants to pad
    // the data. In that case I can simply ignore the trailing bits. If it's
    // smaller than expected, however, then I don't know what to do, and I give
    // up
    if((size_t)(height*input_stride) > size)
    {
        MSG("Unexpected image dimensions: insufficient data returned. Expected buffersize = height*stride = %zd * %zd = %zd, but got %zd",
            (size_t)(height), (size_t)(input_stride),
            (size_t)(height)* (size_t)(input_stride),
            size);
        goto done;
    }

    sws_scale(ctx->sws_context,
              // source
              (const uint8_t*const*)scale_source_value,
              scale_stride_value,
              0, height,
              // destination buffer, stride
              (uint8_t*const*)&ctx->output_image_buffer,
              &output_stride);

    image->width  = width;
    image->height = height;
    image->data   = (uint8_t*)ctx->output_image_buffer;
    image->stride = output_stride;

    result = true;

 done:

    return result;
}

static bool is_pixfmt_matching(ArvPixelFormat pixfmt,
                               mrcam_pixfmt_t mrcam_pixfmt)
{
#define CHECK(name, T, ...) if(pixfmt == ARV_PIXEL_FORMAT_ ## name && mrcam_pixfmt == MRCAM_PIXFMT_ ## name) return true;
    LIST_MRCAM_PIXFMT(CHECK);
#undef CHECK
    MSG("Mismatched pixel format! I asked for '%s', but got ArvPixelFormat 0x%x",
        pixfmt__name(mrcam_pixfmt), (unsigned int)pixfmt);
    return false;
}

static
bool fill_image_uint8(// out
                      mrcal_image_uint8_t* image,
                      // in
                      ArvBuffer* buffer,
                      mrcam_t* ctx)
{
    ArvPixelFormat pixfmt = arv_buffer_get_image_pixel_format(buffer);

    if(!is_pixfmt_matching(pixfmt, ctx->pixfmt))
        return false;

    if(mrcam_output_type(ctx->pixfmt) != MRCAM_uint8)
    {
        MSG("%s() unexpected image type", __func__);
        return false;
    }

    return fill_image_unpacked((mrcal_image_uint8_t*)image, buffer, sizeof(uint8_t));
}

static
bool fill_image_uint16(// out
                       mrcal_image_uint16_t* image,
                       // in
                       ArvBuffer* buffer,
                       mrcam_t* ctx)
{
    ArvPixelFormat pixfmt = arv_buffer_get_image_pixel_format(buffer);

    if(!is_pixfmt_matching(pixfmt, ctx->pixfmt))
        return false;

    if(mrcam_output_type(ctx->pixfmt) != MRCAM_uint16)
    {
        MSG("%s() unexpected image type", __func__);
        return false;
    }

    return fill_image_unpacked((mrcal_image_uint8_t*)image, buffer, sizeof(uint16_t));
}

static
bool fill_image_bgr(// out
                    mrcal_image_bgr_t* image,
                    // in
                    ArvBuffer* buffer,
                    mrcam_t* ctx)
{
    ArvPixelFormat pixfmt = arv_buffer_get_image_pixel_format(buffer);

    if(!is_pixfmt_matching(pixfmt, ctx->pixfmt))
        return false;

    if(mrcam_output_type(ctx->pixfmt) != MRCAM_bgr)
    {
        MSG("%s() unexpected image type", __func__);
        return false;
    }


    const size_t sizeof_pixel = 3;

    bool result = false;

    // I have SOME color format. Today I don't actually support them all, and I
    // put what I have so far
    if(pixfmt == ARV_PIXEL_FORMAT_BGR_8_PACKED)
        return fill_image_unpacked((mrcal_image_uint8_t*)image, buffer, sizeof(mrcal_bgr_t));

    if(pixfmt == ARV_PIXEL_FORMAT_BAYER_RG_8)
    {
        if(!fill_image_swscale((mrcal_image_uint8_t*)image,
                               buffer, ctx))
            goto done;

        result = true;
        goto done;
    }

    MSG("I don't yet know how to handle ARV_PIXEL_FORMAT 0x%x", pixfmt);
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


    const ArvAcquisitionMode modes_to_try[] =
        { ARV_ACQUISITION_MODE_SINGLE_FRAME,
          ARV_ACQUISITION_MODE_MULTI_FRAME,
          ARV_ACQUISITION_MODE_CONTINUOUS };
    int imode = 0;
    const int Nmodes = (int)(sizeof(modes_to_try)/sizeof(modes_to_try[0]));
    for(; imode < Nmodes; imode++)
    {
        try_arv_ok_if( arv_camera_set_acquisition_mode(*camera, modes_to_try[imode], &error),
                       error->code == ARV_GC_ERROR_ENUM_ENTRY_NOT_FOUND );
        if(error == NULL) break; // success; done
        g_clear_error(&error);
    }
    if(imode == Nmodes)
    {
        MSG("Failure!!! arv_camera_set_acquisition_mode() couldn't set any common mode. All were rejected");
        goto done;
    }


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

    return fill_image_uint8(image, *buffer, ctx);
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

    return fill_image_uint16(image, *buffer, ctx);
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

    return fill_image_bgr(image, *buffer, ctx);
}
