#include <stdlib.h>
#include <stdio.h>
#include <arv.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/time.h>

#include <libswscale/swscale.h>
#include <libavutil/pixfmt.h>
#include <libavutil/imgutils.h>

#include "mrcam.h"
#include "util.h"

// debugging code; disabled by default
#if 1
  #define TIMESTAMP() do { } while(0)
#else
  #define TIMESTAMP() do { printf("TIME %d %lld\n", __LINE__, (long long)gettimeofday_uint64()); } while(0)
#endif


#define DEFINE_INTERNALS(ctx)                                           \
    ArvCamera** camera  __attribute__((unused)) = (ArvCamera**)(&(ctx)->camera); \
    ArvStream** stream  __attribute__((unused)) = (ArvStream**)(&(ctx)->stream); \
    ArvBuffer** buffers __attribute__((unused)) = (ArvBuffer**)((ctx)->buffers);





mrcam_output_type_t mrcam_output_type(mrcam_pixfmt_t pixfmt)
{
    switch(pixfmt)
    {
        // I just check one; MRCAM_PIXFMT_name and ..._name_genicam are identical
#define CHOOSE(name, name_genicam, T, ...) case MRCAM_PIXFMT_ ## name: return MRCAM_ ## T;
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
        // I just check one; MRCAM_PIXFMT_name and ..._name_genicam are identical
#define CHOOSE(name, name_genicam, ...) case MRCAM_PIXFMT_ ## name: return ARV_PIXEL_FORMAT_ ## name;
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
#define CHOOSE(name, name_genicam, ...) case MRCAM_PIXFMT_ ## name: return #name;
    LIST_MRCAM_PIXFMT(CHOOSE)
    default:
        MSG("ERROR: unknown mrcam_pixfmt_t = %d", (int)pixfmt);
        return "UNKNOWN";
    }
#undef CHOOSE
}

static
const char* pixfmt__name_from_ArvPixelFormat(ArvPixelFormat pixfmt)
{
    switch(pixfmt)
    {
#define CHOOSE(name, name_genicam, ...) case ARV_PIXEL_FORMAT_ ## name: return #name;
    LIST_MRCAM_PIXFMT(CHOOSE)
    default:
        MSG("ERROR: unknown ArvPixelFormat = %d", (int)pixfmt);
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
        // I just check one; MRCAM_PIXFMT_name and ..._name_genicam are identical
#define CHOOSE(name, name_genicam, T, av_pixfmt)        \
        case MRCAM_PIXFMT_ ## name:                     \
            *av_pixfmt_input = av_pixfmt;               \
            break;

        LIST_MRCAM_PIXFMT(CHOOSE)
    default:
        MSG("ERROR: unknown mrcam_pixfmt_t = %d; this is a bug!",
            (int)pixfmt);
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

static bool open_serial_device(mrcam_t* ctx)
{
    bool result = false;

    // open the serial device, and make it as raw as possible
    const char* device = "/dev/ttyS0";
    const speed_t baud = B2400;
    try( (ctx->fd_tty_trigger = open(device, O_WRONLY|O_NOCTTY)) >= 0 );
    try( tcflush(ctx->fd_tty_trigger, TCIOFLUSH) == 0 );

    struct termios options = {.c_iflag = IGNBRK,
                              .c_cflag = CS8 | CREAD | CLOCAL};
    try(cfsetspeed(&options, baud) == 0);
    try(tcsetattr(ctx->fd_tty_trigger, TCSANOW, &options) == 0);

    result = true;
 done:
    return result;
}




static void
callback_arv(void* cookie, ArvStreamCallbackType type, ArvBuffer* buffer);

static void report_available_pixel_formats(ArvCamera* camera,
                                           mrcam_t* ctx)
{
    GError* error  = NULL;
    const char** available_pixel_formats = NULL;
    guint n_pixel_formats;

    MSG("Available pixel formats supported by the camera:");
    try_arv( available_pixel_formats =
             arv_camera_dup_available_pixel_formats_as_strings(camera, &n_pixel_formats, &error) );
    for(guint i=0; i<n_pixel_formats; i++)
        MSG("  %s",
            available_pixel_formats[i]);

 done:
    g_free(available_pixel_formats);
}

void mrcam_push_buffer(void**   buffer, // ArvBuffer**, without requiring #include arv.h
                       mrcam_t* ctx)
{
    if(*buffer != NULL)
    {
        if(ctx->verbose)
            MSG("arv_stream_push_buffer(%p)", *buffer);
        arv_stream_push_buffer((ArvStream*)(ctx->stream), (ArvBuffer*)(*buffer));
        *buffer = NULL;
    }
}

static bool
init_stream(mrcam_t* ctx)
{
    bool    result = false;
    GError* error  = NULL;

    DEFINE_INTERNALS(ctx);

    // The frame de-init happens in receive_image() after I captured the image.
    // If I'm asynchronous, receive_image() happens from the callback_arv(),
    // which is called from the stream thread, which means I cannot stop the
    // stream thread in receive_image(). So I just let the stream thread run,
    // and restart it when I need a new frame
    g_clear_object(stream);

    try_arv_and( *stream = arv_camera_create_stream(*camera,
                                                    callback_arv, ctx,
                                                    &error),
                 ARV_IS_STREAM(*stream) );

    /*
    For the Emergent HR-20000 cameras. I get lost packets if I don't do this.
    Running as root is another workaround: it enables the "Packet socket
    method". Either of these are needed in addition to the GevSCPSPacketSize
    setting

    Ultimately we need to set the SO_RCVBUF setting on the socket. Described
    like this in socket(7):

       SO_RCVBUF
              Sets or gets the maximum socket receive buffer in bytes. The
              kernel doubles this value (to allow space for bookkeeping
              overhead) when it is set using setsockopt(2), and this doubled
              value is returned by getsockopt(2). The default value is set by
              the /proc/sys/net/core/rmem_default file, and the maximum allowed
              value is set by the /proc/sys/net/core/rmem_max file. The minimum
              (doubled) value for this option is 256.

    The setting ends up in sk_rcvbuf in the kernel:

      https://lxr.linux.no/#linux+v6.7.1/net/core/sock.c#L1250
      https://lxr.linux.no/#linux+v6.7.1/net/core/sock.c#L960

    Which is used in multiple places in

      https://lxr.linux.no/#linux+v6.7.1/net/ipv4/udp.c

    If the data comes in faster than it can be read, it's stored in this buffer;
    if the buffer is too small, the packets are thrown away. The logic and
    available diagnostics are in the udp.c file linked above. /proc/net/snmp
    records the dropped packets and there's a "udp_fail_queue_rcv_skb"
    tracepoint to catch some paths

    In aravis it is sufficient to ask for an "auto" buffer size, and it will be
    large-enough to hold a single buffer
    */
    if(ARV_IS_GV_STREAM(*stream))
        g_object_set (*stream,
                      "socket-buffer", ARV_GV_STREAM_SOCKET_BUFFER_AUTO,
                      NULL);


    // If ARV_ACQUISITION_MODE_MULTI_FRAME, I ask for just one frame
    if(ctx->acquisition_mode == MRCAM_ACQUISITION_MODE_MULTI_FRAME)
        try_arv(arv_camera_set_integer(*camera, "AcquisitionFrameCount", 1, &error));
    if(error != NULL)
        g_clear_error(&error);

#define SET(name, ...)                                                  \
    else if(ctx->acquisition_mode == MRCAM_ACQUISITION_MODE_ ## name)   \
        try_arv(arv_camera_set_acquisition_mode(*camera, ARV_ACQUISITION_MODE_ ## name, &error));
    if(0); LIST_MRCAM_ACQUISITION_MODE(SET)
    else
    {
        MSG("Unknown acquisition_mode mode '%d'; I know about:", (int)(ctx->acquisition_mode));
#define SAY(name, ...) MSG("  " #name);
        LIST_MRCAM_ACQUISITION_MODE(SAY);
#undef SAY
        goto done;
    }
#undef SET

    result = true;

    if(ctx->verbose)
        MSG("arv_stream_push_buffer() ALL %d buffers:", ctx->Nbuffers);
    for(int i=0; i<ctx->Nbuffers; i++)
    {
        arv_stream_push_buffer(*stream, buffers[i]);
        if(ctx->verbose)
            MSG("  %p", buffers[i]);
    }

 done:
    if(!result)
        g_clear_object(stream);

    return result;
}

// camera_name = NULL means "first available camera"
bool mrcam_init(// out
                mrcam_t* ctx,
                // in
                const char* camera_name,
                const mrcam_options_t* options)
{
    bool result = false;
    GError* error  = NULL;
    *ctx = (mrcam_t){ .pixfmt                          = options->pixfmt,
                      .trigger                         = options->trigger,
                      .acquisition_mode                = options->acquisition_mode,
                      .verbose                         = options->verbose,
                      .time_decimation_factor          = options->time_decimation_factor,
                      .Nbuffers                        = options->Nbuffers,
                      .fd_tty_trigger                  = -1};

    DEFINE_INTERNALS(ctx);


    if(ctx->trigger == MRCAM_TRIGGER_HARDWARE_TTYS0)
        try(open_serial_device(ctx));

    try_arv_and( *camera = arv_camera_new (camera_name,
                                           &error),
                 ARV_IS_CAMERA(*camera) );

    int width  = options->width;
    int height = options->height;
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

    ArvPixelFormat arv_pixfmt = pixfmt__ArvPixelFormat(ctx->pixfmt);
    if(arv_pixfmt == 0)
        goto done;

    try_arv_extra_reporting( arv_camera_set_pixel_format(*camera, arv_pixfmt, &error),
                             {
                                 MSG("  Setting pixel format: '%s'", pixfmt__name(ctx->pixfmt));
                             },
                             {},
                             {
                                 MSG("Couldn't set the requested pixel format: '%s'",
                                     pixfmt__name(ctx->pixfmt));

                                 report_available_pixel_formats(*camera, ctx);
                             });

    // Some cameras start up with the test-pattern enabled. So I turn it off
    // unconditionally. This setting doesn't exist on all cameras; if it
    // doesn't, I ignore that failure


    try_arv_or(arv_camera_set_string (*camera, "TestPattern", "Off", &error),
               error->code == ARV_DEVICE_ERROR_FEATURE_NOT_FOUND);
    if(error != NULL)
    {
        // No TestPattern setting exists. I ignore the error
        g_clear_error(&error);
    }

    ctx->buffers = malloc(ctx->Nbuffers * sizeof(ctx->buffers[0]));
    if(ctx->buffers == NULL)
    {
        MSG("Couldn't alloc %d buffer pointers'", ctx->Nbuffers);
        goto done;
    }

    gint payload_size;
    try_arv(payload_size = arv_camera_get_payload(*camera, &error));
    for(int i=0; i<ctx->Nbuffers; i++)
        try(ctx->buffers[i] = arv_buffer_new(payload_size, NULL));

    enum AVPixelFormat av_pixfmt_input, av_pixfmt_output;
    try(pixfmt__av_pixfmt(&av_pixfmt_input, &av_pixfmt_output,
                          ctx->pixfmt));

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
        try(NULL != (ctx->sws_output_buffer =
                     malloc(pixfmt__output_bytes_per_pixel(ctx->pixfmt) *
                            width * height)));
    }

    // Set the triggering strategy
    if(ctx->trigger != MRCAM_TRIGGER_NONE)
    {
        try_arv_or( arv_camera_set_string(*camera, "TriggerMode", "On", &error),
                    error->code == ARV_DEVICE_ERROR_FEATURE_NOT_FOUND );
        if(error != NULL)
        {
            // No TriggerMode is available at all. I ignore the error. If there WAS
            // a TriggerMode, and I couldn't set it to "On", then I DO flag an error
            g_clear_error(&error);
        }

        // If either the feature or the requested enum don't exist, I let it go
        try_arv_or( arv_camera_set_string(*camera, "TriggerSelector", "FrameStart", &error),
                    error->code == ARV_DEVICE_ERROR_FEATURE_NOT_FOUND ||
                    error->code == ARV_GC_ERROR_ENUM_ENTRY_NOT_FOUND );
        if(error != NULL)
            g_clear_error(&error);
    }
    if(ctx->trigger == MRCAM_TRIGGER_SOFTWARE)
    {
        try_arv_or( arv_camera_set_string(*camera, "TriggerSource",   "Software", &error),
                    error->code == ARV_DEVICE_ERROR_FEATURE_NOT_FOUND ||
                    error->code == ARV_GC_ERROR_ENUM_ENTRY_NOT_FOUND );
        if(error != NULL)
            g_clear_error(&error);
    }
    else if(ctx->trigger == MRCAM_TRIGGER_HARDWARE_TTYS0 ||
            ctx->trigger == MRCAM_TRIGGER_HARDWARE_EXTERNAL)
    {
        try_arv_or( arv_camera_set_string(*camera, "TriggerSource",   "Line0", &error),
                    error->code == ARV_DEVICE_ERROR_FEATURE_NOT_FOUND ||
                    error->code == ARV_GC_ERROR_ENUM_ENTRY_NOT_FOUND );
        if(error != NULL)
            g_clear_error(&error);
    }
    else if(ctx->trigger == MRCAM_TRIGGER_NONE)
    {}
    else
    {
        MSG("Unknown trigger enum value '%d'", ctx->trigger);
        goto done;
    }

    // High-res cameras need BIG packets to maintain the signal integrity. Here
    // I ask for packets 9kB in size; that's about the biggest we can have
    try_arv_or( arv_camera_set_integer(*camera, "GevSCPSPacketSize", 9000, &error),
                error->code == ARV_DEVICE_ERROR_FEATURE_NOT_FOUND );
    if(error != NULL)
        g_clear_error(&error);

    if(!init_stream(ctx))
        goto done;

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

    // I do not clear the buffers. It looks like g_clear_object(stream) does
    // that for me, and if I do it myself, there's a double-free

    // for(int i=0; i<ctx->Nbuffers; i++)
    //     g_clear_object(&buffers[i]);

    g_clear_object(stream);
    g_clear_object(camera);

    if(ctx->sws_context)
    {
        sws_freeContext(ctx->sws_context);
        ctx->sws_context = NULL;
    }

    free(ctx->sws_output_buffer);
    ctx->sws_output_buffer = NULL;

    if(ctx->fd_tty_trigger >= 0)
    {
        close(ctx->fd_tty_trigger);
        ctx->fd_tty_trigger = -1;
    }

    free(ctx->buffers);
    ctx->buffers = NULL;
}

bool mrcam_is_inited(mrcam_t* ctx)
{
    DEFINE_INTERNALS(ctx);

    return *camera != NULL;
}

static uint64_t gettimeofday_uint64()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t) tv.tv_sec * 1000000ULL + (uint64_t) tv.tv_usec;
}

// Fill in the image. Assumes that the buffer has valid data
// On error returns false and sets *image = {}
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
    if(!result)
        *image = (mrcal_image_uint8_t){}; // indicate error
    return result;
}

// On error returns false and sets *image = {}
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
              (uint8_t*const*)&ctx->sws_output_buffer,
              &output_stride);

    image->width  = width;
    image->height = height;
    image->data   = (uint8_t*)ctx->sws_output_buffer;
    image->stride = output_stride;

    result = true;

 done:
    if(!result)
        *image = (mrcal_image_uint8_t){}; // indicate error
    return result;
}

static bool is_pixfmt_matching(ArvPixelFormat pixfmt,
                               mrcam_pixfmt_t mrcam_pixfmt)
{
#define CHECK(name, name_genicam, T, ...) if(pixfmt == ARV_PIXEL_FORMAT_ ## name && mrcam_pixfmt == MRCAM_PIXFMT_ ## name) return true;
    LIST_MRCAM_PIXFMT(CHECK);
#undef CHECK
    MSG("Mismatched pixel format! I asked for '%s', but got ArvPixelFormat 0x%x",
        pixfmt__name(mrcam_pixfmt), (unsigned int)pixfmt);
    return false;
}

// On error returns false and sets *image = {}
static
bool fill_image(// out
                mrcal_image_uint8_t* image, // type may not be 100% right; layout is the same
                // in
                ArvBuffer* buffer,
                mrcam_t* ctx)
{
    if(ctx->verbose) MSG("");

    ArvPixelFormat pixfmt = arv_buffer_get_image_pixel_format(buffer);
    if(!is_pixfmt_matching(pixfmt, ctx->pixfmt))
        return false;

    if(mrcam_output_type(ctx->pixfmt) == MRCAM_uint8)
        return fill_image_unpacked(image, buffer, sizeof(uint8_t));
    if(mrcam_output_type(ctx->pixfmt) == MRCAM_uint16)
        return fill_image_unpacked(image, buffer, sizeof(uint16_t));
    if(mrcam_output_type(ctx->pixfmt) == MRCAM_bgr)
    {
        // I have SOME color format. Today I don't actually support them all, and I
        // put what I have so far
        if(pixfmt == ARV_PIXEL_FORMAT_BGR_8_PACKED)
            return fill_image_unpacked(image, buffer, sizeof(mrcal_bgr_t));

        if(ctx->sws_context != NULL)
            return fill_image_swscale((mrcal_image_uint8_t*)image,
                                      buffer, ctx);

        MSG("doesn't yet know how to handle pixfmt '%s'",
            pixfmt__name_from_ArvPixelFormat(pixfmt));
        *image = (mrcal_image_uint8_t){}; // indicate error
        return false;
    }

    MSG("Unknown pixfmt. This is a bug");
    *image = (mrcal_image_uint8_t){}; // indicate error
    return false;
}

// meant to be called after request()
//
// on exit:
// if(*buffer != NULL) { *buffer contain the just-popped buffer, even if this function
//                       failed. The image points to this buffer. When we're
//                       done with the image, push the buffer back by calling
//                       mrcam_push_buffer(), to let aravis use it
//                       for new incoming frames }
//
// else                { no buffer was popped }
static
bool receive_image(// out
                   uint64_t* timestamp_us, // may be NULL
                   ArvBuffer** buffer,

                   // fill this image with the data, on success. On failure set to {}
                   // May be NULL
                   // specific image type may not be right; it doesn't matter
                   mrcal_image_uint8_t* image,

                   // in
                   const uint64_t timeout_us,
                   mrcam_t* ctx)
{
    if(ctx->verbose) MSG("%s(timeout_us=%"PRIu64")", __func__, timeout_us);

    DEFINE_INTERNALS(ctx);
    bool        result      = false;
    GError*     error       = NULL;

    *buffer = NULL;

    if(!ctx->acquiring)
    {
        MSG("No acquisition in progress. Nothing to receive");
        goto done;
    }

    // May block. If we don't want to block, it's our job to make sure to have
    // called receive_image() when an output image has already been buffered
    if (timeout_us > 0)
    {
        if(ctx->verbose)
            MSG("Calling   arv_stream_timeout_pop_buffer(timeout_us = %"PRIu64") ...", timeout_us);
        *buffer = arv_stream_timeout_pop_buffer(*stream, timeout_us);
    }
    else
    {
        if(ctx->verbose)
            MSG("Calling   arv_stream_pop_buffer() ...");
        *buffer = arv_stream_pop_buffer(*stream);
    }
    if(ctx->verbose)
        MSG("... received buffer %p", *buffer);

    // This MUST have been done by this function, regardless of things failing
    if(ctx->acquisition_mode != MRCAM_ACQUISITION_MODE_CONTINUOUS)
    {
        try_arv(arv_camera_stop_acquisition(*camera, &error));

        // if it failed, ctx->acquiring will remain at true. Probably there's no way
        // to recover anyway
        ctx->acquiring = false;
    }

    ArvBufferStatus status = arv_buffer_get_status(*buffer);

    // All the statuses from arvbuffer.h, as of aravis 0.8.29
#define LIST_STATUS(_)                          \
  _(ARV_BUFFER_STATUS_UNKNOWN)                  \
  _(ARV_BUFFER_STATUS_SUCCESS)                  \
  _(ARV_BUFFER_STATUS_CLEARED)                  \
  _(ARV_BUFFER_STATUS_TIMEOUT)                  \
  _(ARV_BUFFER_STATUS_MISSING_PACKETS)          \
  _(ARV_BUFFER_STATUS_WRONG_PACKET_ID)          \
  _(ARV_BUFFER_STATUS_SIZE_MISMATCH)            \
  _(ARV_BUFFER_STATUS_FILLING)                  \
  _(ARV_BUFFER_STATUS_ABORTED)
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

    if(timestamp_us != NULL)
        *timestamp_us = gettimeofday_uint64();
    result = true;

 done:

    if(image != NULL)
    {
        if(result)
            result = fill_image(image, *buffer, ctx);

        if(!result)
            *image = (mrcal_image_uint8_t){}; // error
    }
    // I want to make sure that arv_camera_stop_acquisition() was called by this
    // function. That happened on top. If it failed, there isn't anything I can
    // do about it.
    return result;
}

static void
callback_arv(void* cookie, ArvStreamCallbackType type, ArvBuffer* buffer)
{
    mrcam_t* ctx = (mrcam_t*)cookie;

    // This is going to be called from a different thread than the rest of the
    // application. The sequence SHOULD be:
    //
    // - request()
    // - internal machinery causes this callback_arv() to be called
    // - this callback does its thing to call receive_image(), which should
    //   disable future callbacks until the next request
    //
    // Something COULD break this though. Extra calls of callback_arv() should
    // be benign: ctx->active_callback == NULL will be true. Insufficient calls
    // of callback_arv() will cause us to wait forever for the callback, and
    // will permanently break stuff. We'll see if this happens.
    //
    // Furthermore, while this function is called from another thread, it was
    // set up to work synchronously, so no locking of any sort should be needed.
    // We'll see if that happens also

    if(ctx->active_callback == NULL)
    {
        // Nothing to do. This CAN happen if we're capturing continuously: the
        // frames are coming in without us explicitly asking for each one, so we
        // might not be ready to do anything with the frame yet
        if(ctx->verbose)
            MSG("Nothing to do: user active_callback==NULL");
        return;
    }


    switch (type)
    {
    case ARV_STREAM_CALLBACK_TYPE_INIT:
        TIMESTAMP();
        if(ctx->verbose)
            MSG("ARV_STREAM_CALLBACK_TYPE_INIT: Stream thread started");
        /* Here you may want to change the thread priority arv_make_thread_realtime() or
         * arv_make_thread_high_priority() */
        break;
    case ARV_STREAM_CALLBACK_TYPE_START_BUFFER:
        TIMESTAMP();
        if(ctx->verbose)
            MSG("ARV_STREAM_CALLBACK_TYPE_START_BUFFER: The first packet of a new frame was received");
        ctx->timestamp_start_buffer_us = gettimeofday_uint64();
        break;
    case ARV_STREAM_CALLBACK_TYPE_BUFFER_DONE:
        TIMESTAMP();
        /* The buffer is received, successfully or not. It is already pushed in
         * the output FIFO.
         *
         * You could here signal the new buffer to another thread than the main
         * one, and pull/push the buffer from this another thread.
         *
         * Or use the buffer here. We need to pull it, process it, then push it
         * back for reuse by the stream receiving thread */
        {
            if(ctx->verbose)
                MSG("ARV_STREAM_CALLBACK_TYPE_BUFFER_DONE");

            mrcal_image_uint8_t image; // specific image type may not be right; it doesn't matter

            const bool on_decimation =
                (ctx->time_decimation_factor <= 1) ||
                (++ctx->time_decimation_index == ctx->time_decimation_factor);

            ArvBuffer* buffer_popped;
            receive_image(NULL, &buffer_popped,
                          on_decimation ? &image : NULL,
                          0, ctx);

            // On error image is {0}, which indicates an error. We invoke the
            // callback regardless. I want to make sure that the caller can be
            // sure to expect ONE callback with each request
            if(on_decimation)
            {
                // It's the callback's responsibility to push the buffer back
                // when it's done. I don't arv_stream_push_buffer() right after
                // the active_callback() here because active_callback() may want
                // to do something with the buffer asynchronously, and the
                // buffer may still be used then
                if(ctx->verbose)
                    MSG("calling the active_callback()");
                ctx->active_callback(image,
                                     buffer_popped,
                                     ctx->timestamp_start_buffer_us,
                                     ctx->active_callback_cookie);
                ctx->time_decimation_index = 0;

                // If we're not persistent, we got the frame, the callback was
                // invoked, and we're done. To get another frame, we must
                // request() it, with another active_callback
                if(ctx->acquisition_mode != MRCAM_ACQUISITION_MODE_CONTINUOUS)
                    ctx->active_callback = NULL;

                // the buffer remains popped until the active callback is done
                // with it. The callback MUST push it back
            }
            else
            {
                if(ctx->verbose)
                    MSG("off-decimation. NOT calling the active_callback()");
                mrcam_push_buffer((void**)&buffer_popped, ctx);

                // External triggering or whatever else. New frame requests
                // happen here
                if(ctx->active_callback_off_decimation != NULL)
                    ctx->active_callback_off_decimation((mrcal_image_uint8_t){},NULL,0,ctx->active_callback_cookie);
            }
        }

        break;
    case ARV_STREAM_CALLBACK_TYPE_EXIT:
        if(ctx->verbose)
            MSG("ARV_STREAM_CALLBACK_TYPE_EXIT");
        break;
    }
}

bool mrcam_request( // in
                    mrcam_callback_t* callback,
                    mrcam_callback_t* callback_off_decimation,
                    void* cookie,
                    mrcam_t* ctx)
{
    if(ctx->verbose) MSG("");

    DEFINE_INTERNALS(ctx);
    bool    result = false;
    GError* error  = NULL;

    ctx->timestamp_request_us = gettimeofday_uint64();

    if( (ctx->trigger == MRCAM_TRIGGER_NONE ||
         ctx->trigger == MRCAM_TRIGGER_HARDWARE_EXTERNAL) &&
        ctx->acquiring &&
        ctx->acquisition_mode == MRCAM_ACQUISITION_MODE_CONTINUOUS)

    {
        // Nothing to do. We're not triggering, so we will get frames as they
        // come, not as we ask for them. I thus don't check for requesting an
        // already-requested frame
        ctx->active_callback                = callback;
        ctx->active_callback_off_decimation = callback_off_decimation;
        ctx->active_callback_cookie         = cookie;

        return true;
    }

    if( ctx->acquisition_mode != MRCAM_ACQUISITION_MODE_CONTINUOUS &&
        (ctx->acquiring || ctx->active_callback != NULL))
    {
        MSG("Acquisition already in progress: acquisition_persistent=%d, acquiring=%d, active_callback_exists=%d. If mrcam_request() was called, wait for the callback or call mrcam_cancel_request()",
            ctx->acquisition_mode == MRCAM_ACQUISITION_MODE_CONTINUOUS, ctx->acquiring, !!ctx->active_callback);
        goto done;
    }

    ctx->active_callback                = callback;
    ctx->active_callback_off_decimation = callback_off_decimation;
    ctx->active_callback_cookie         = cookie;

    if(ctx->verbose)
    {
        gint n_input_buffers;
        gint n_output_buffers;
        arv_stream_get_n_buffers (*stream,
                                  &n_input_buffers,
                                  &n_output_buffers);
        MSG("n_input_buffers,n_output_buffers = %d,%d",
            n_input_buffers,n_output_buffers);
    }

    if(!ctx->acquiring)
        try_arv( arv_camera_start_acquisition(*camera, &error));
    ctx->acquiring = true;

    if(ctx->trigger == MRCAM_TRIGGER_SOFTWARE)
    {
        // For the Emergent HR-20000 cameras; the others should be able to
        // free-run. If the feature doesn't exist, I let it go
        try_arv_or(arv_camera_execute_command(*camera, "TriggerSoftware", &error),
                   error->code == ARV_DEVICE_ERROR_FEATURE_NOT_FOUND );
        if(error != NULL)
            g_clear_error(&error);
    }
    else if(ctx->trigger == MRCAM_TRIGGER_HARDWARE_TTYS0)
    {
        // If the previous trigger pulse is still high for some reason, wait for
        // it to drop
        TIMESTAMP();
        try(0 == tcdrain(ctx->fd_tty_trigger));
        try(1 == write(ctx->fd_tty_trigger, &((char){'\xff'}), 1));
    }
    else if(ctx->trigger == MRCAM_TRIGGER_HARDWARE_EXTERNAL)
    {
        // A trigger signal will magically come from somewhere. I don't worry
        // about it here; nothing to do
    }
    else if(ctx->trigger == MRCAM_TRIGGER_NONE)
    {}
    else
    {
        MSG("Unknown trigger enum value '%d'", ctx->trigger);
        goto done;
    }

    result = true;

 done:
    if(!result)
    {
        if(ctx->acquisition_mode != MRCAM_ACQUISITION_MODE_CONTINUOUS &&
           ctx->acquiring)
        {
            // if still acquiring for some reason, stop that, with limited error checking
            arv_camera_stop_acquisition(*camera, &error);
            if(error != NULL)
            {
                MSG("Failure!!! Couldn't arv_camera_stop_acquisition() in the error handler: '%s'",
                    error->message);
                g_clear_error(&error);
            }

            // I set the no-acquiring flag even if we failed to stop. Unlikely
            // we can recover from this.
            ctx->acquiring = false;
        }

        g_clear_object(stream);
    }

    return result;
}


/* timeout_us=0 means "wait forever" */
bool mrcam_pull(/* out */
                mrcal_image_uint8_t* image,
                // the buffer. Call mrcam_push_buffer(buffer) when done with the
                // image
                void** buffer,
                uint64_t* timestamp_us,
                /* in */
                // The frame period. Used only if time_decimation_factor > 1;
                // sets the delay of the off-decimation frame requests. Pass 0
                // to ignore, and to request the frames immediately
                uint64_t period_us,
                const uint64_t timeout_us,
                mrcam_t* ctx)
{
    if(ctx->verbose) MSG("%s()", __func__);

    int N = ctx->time_decimation_factor;
    if(N < 1) N = 1;

    *buffer = NULL;

    for(; N > 0; N--)
    {
        bool result =
            mrcam_request(NULL, NULL, NULL, ctx) &&
            // may block
            receive_image(timestamp_us,
                          (ArvBuffer**)buffer,
                          N == 1 ? image : NULL, // fill in the image on the
                                                 // last round, which will be
                                                 // returned to the user
                          timeout_us, ctx);
        if(!result)
        {
            mrcam_push_buffer(buffer, ctx);
            return false;
        }
        if(N > 1)
        {
            // Off-decimation. We ignore this image, and keep going
            mrcam_push_buffer(buffer, ctx);
            mrcam_sleep_until_next_request(period_us, ctx);
        }
    }

    // The caller MUST mrcal_push_buffer(buffer) when done
    return true;
}

bool mrcam_cancel_request(mrcam_t* ctx)
{
    if(ctx->verbose) MSG("");

#warning finish this
    return false;
}

void mrcam_sleep_until_next_request(// The frame period. Used only if
                                    // time_decimation_factor > 1; sets the
                                    // delay of the off-decimation frame
                                    // requests. Pass 0 to ignore, and to
                                    // request the frames immediately
                                    uint64_t period_us,
                                    mrcam_t* ctx)
{
    if(period_us == 0)
        return;

    const uint64_t time_now_us = gettimeofday_uint64();
    int64_t time_sleep_us;
    if(ctx->timestamp_request_us == 0)
        time_sleep_us = period_us;
    else
        time_sleep_us = (ctx->timestamp_request_us + period_us) - time_now_us;
    if(time_sleep_us > 0)
        usleep(time_sleep_us);
}
