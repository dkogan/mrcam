#pragma once

#include <mrcal/mrcal-image.h>

// I define this myself, in order to avoid requiring the user to #include
// <arv.h>. This is a few common formats. The fields are:
//
// - MRCAM_PIXFMT_.../ARV_PIXEL_FORMAT_... (these are equivalent)
//
// - The formats returned by arv_camera_dup_available_pixel_formats_as_strings()
//   or by running "arv-tool-0.8 features PixelFormat"; these are similar to the
//   above, but spelled differently
//
// - The output dense type we use after we unpack the data
//
// - The ffmpeg conversion type, if needed
//
// An ffmpeg pixel format of AV_PIX_FMT_NONE means "the input is already packed,
// and there's no need to invoke ffmpeg at all"
//
// Today I don't yet support the high-deptch color formats because there's no
// corresponding mrcal_image_XXXX_t
//
// The packed formats are not yet supported because they're weird. I just tried
// to support ARV_PIXEL_FORMAT_MONO_10_PACKED. I'm seeing that my camera is
// packing each pixel into 12 bits instead of 10. And I see that libswscale
// doesn't support this: AV_PIX_FMT_GRAY10BE and AV_PIX_FMT_GRAY12BE use 16 bits
// per pixel, not 10 or 12
#define LIST_MRCAM_PIXFMT(_)                                    \
  _(MONO_8,       Mono8,    uint8,  AV_PIX_FMT_NONE)            \
  /* Each pixel takes up 16 bits. NOT packed */                 \
  _(MONO_10,      Mono10,   uint16, AV_PIX_FMT_NONE)            \
  _(MONO_12,      Mono12,   uint16, AV_PIX_FMT_NONE)            \
  _(MONO_14,      Mono14,   uint16, AV_PIX_FMT_NONE)            \
  _(MONO_16,      Mono16,   uint16, AV_PIX_FMT_NONE)            \
  _(BAYER_GR_8,   BayerGR8, bgr,    AV_PIX_FMT_BAYER_GRBG8)     \
  _(BAYER_RG_8,   BayerRG8, bgr,    AV_PIX_FMT_BAYER_RGGB8)     \
  _(BAYER_GB_8,   BayerGB8, bgr,    AV_PIX_FMT_BAYER_GBRG8)     \
  _(BAYER_BG_8,   BayerBG8, bgr,    AV_PIX_FMT_BAYER_BGGR8)     \
  _(RGB_8_PACKED, RGB8,     bgr,    AV_PIX_FMT_NONE)            \
  _(BGR_8_PACKED, BGR8,     bgr,    AV_PIX_FMT_NONE)


typedef enum
{
#define ENUM(name, name_genicam,...)                           \
    MRCAM_PIXFMT_ ## name,                                     \
    MRCAM_PIXFMT_ ## name_genicam = MRCAM_PIXFMT_ ## name,

    LIST_MRCAM_PIXFMT(ENUM) MRCAM_PIXFMT_COUNT
#undef ENUM
} mrcam_pixfmt_t;


/*
The trigger modes:

NONE: no triggering at all. The camera will decide to send us frames all by
  itself.

SOFTWARE: software "triggering". We request a frame by sending a
  "TriggerSoftware" command. The image will be captured "soon" after it is
  requested. This is useful for testing, but if any kind of camera
  synchronization is required, this mode does not work

HARDWARE_EXTERNAL: "hardware" triggering. The implementation details are
  specific to each camera. Usually there's a physical pin feeding into the
  camera; an electrical pulse on this pin initiates the image capture. In this
  'HARDWARE_EXTERNAL' mode we tell the camera to wait for a pulse to begin
  capture, but we don't actually supply this pulse. Some external process has to
  do that

HARDWARE_TTYS0: like 'HARDWARE_EXTERNAL', but we produce the trigger pulse
  as well: by sending \xFF to /dev/ttyS0. The start bit in each character is
  the pulse. It is assumed that the Tx pin in the RS-232 port is connected
  (usually through some level shifters and/or buffers) to the trigger pin in
  the camera
*/
#define LIST_MRCAM_TRIGGER(_)                   \
    _(NONE)                                     \
    _(SOFTWARE)                                 \
    _(HARDWARE_TTYS0)                           \
    _(HARDWARE_EXTERNAL)
typedef enum {
#define ENUM(name, ...) MRCAM_TRIGGER_ ## name,
    LIST_MRCAM_TRIGGER(ENUM) MRCAM_TRIGGER_COUNT
#undef ENUM
} mrcam_trigger_t;

/* These have the same meaning as ARV_ACQUISITION_MODE_... The numerical values
may not be the same because the mrcam.h header does NOT #include arv.h

The SINGLE_FRAME and MULTI_FRAME modes are frame-by-frame modes: we request and
capture one frame, then we request and capture the next one, and so on. The
MULTI_FRAME mode also asks for ONE frame. These usually should start and stop
the acquisition for each frame, and should set acquisition_persistent to false.
Some cameras require the opposite, however

By contrast, the CONTINUOUS mode usually should start the acquisition at the
beginning of the capture, and keeps it going for all the subsequent frames, and
should set acquisition_persistent to true. Some cameras require the opposite,
however
*/
#define LIST_MRCAM_ACQUISITION_MODE(_)          \
    _(SINGLE_FRAME)                             \
    _(MULTI_FRAME)                              \
    _(CONTINUOUS)
typedef enum {
#define ENUM(name, ...) MRCAM_ACQUISITION_MODE_ ## name,
    LIST_MRCAM_ACQUISITION_MODE(ENUM) MRCAM_ACQUISITION_MODE_COUNT
#undef ENUM
} mrcam_acquisition_mode_t;

typedef struct
{
    void* ctx;    // mrcam_t
    void* buffer; // This is ArvBuffer*, but without requiring #including arv.h
} mrcam_buffer_t;



typedef enum {MRCAM_UNKNOWN = -1,
              MRCAM_uint8, MRCAM_uint16, MRCAM_bgr } mrcam_output_type_t;
mrcam_output_type_t mrcam_output_type(mrcam_pixfmt_t pixfmt);

typedef void (mrcam_callback_image_uint8_t )(mrcal_image_uint8_t image,
                                             mrcam_buffer_t* buffer,
                                             uint64_t timestamp_us,
                                             void* cookie);
typedef void (mrcam_callback_image_uint16_t)(mrcal_image_uint16_t image,
                                             mrcam_buffer_t* buffer,
                                             uint64_t timestamp_us,
                                             void* cookie);
typedef void (mrcam_callback_image_bgr_t)(   mrcal_image_bgr_t image,
                                             mrcam_buffer_t* buffer,
                                             uint64_t timestamp_us,
                                             void* cookie);
typedef void (mrcam_callback_t )(void* cookie);

typedef struct
{
    // arv stuff; void to not require #including arv.h
    void* camera;
    void* buffers[5];
    void* stream;

    // Details about the requested pixel format, that I'm using to talk to the
    // camera. I don't NEED to store all these here, but it makes life easier
    mrcam_pixfmt_t pixfmt;

    mrcam_trigger_t          trigger;
    mrcam_acquisition_mode_t acquisition_mode;

    // Used to convert from non-trivial pixel formats coming out of the camera
    // to unpacked bits mrcam reports. Needed for bayered or packed formats.
    // Unused for all others
    struct SwsContext* sws_context;
    uint8_t*           output_image_buffer;


    // current active callback. Type may not be 100% right (may be uint8 or
    // uint16, or bgr, ...), but the data layout is the same
    mrcam_callback_image_uint8_t* active_callback;
    // Callback used if time_decimation_factor > 1. Called after each frame
    // capture that was NOT aligned with the decimation cycle. Can be NULL.
    // Useful for things like external triggering, which must happen with EVERY
    // captured frame, not just the decimated ones
    mrcam_callback_t*             active_callback_off_decimation;
    void*                         active_callback_cookie;
                                                \
    // used if MRCAM_TRIGGER_HARDWARE_TTYS0
    int fd_tty_trigger;

    // If time_decimation_factor > 1, we report every Nth frame to the user.
    // This applies to mrcam_pull_...() and mrcam_request(). The
    // time_decimation_index is the internal counter that's updated to respect
    // the factor when using the asynchronous request() function
    int time_decimation_factor;
    int time_decimation_index;

    bool acquiring                       : 1; // we're acquiring RIGHT NOW

    // Options. These don't change after init
    bool acquisition_persistent          : 1; // never stop the acquisition
    bool recreate_stream_with_each_frame : 1;
    bool verbose                         : 1;
} mrcam_t;

typedef struct
{
    const mrcam_pixfmt_t           pixfmt;
    const mrcam_trigger_t          trigger;
    const mrcam_acquisition_mode_t acquisition_mode;

    // if either is <=0, we try to autodetect by asking the camera
    // for WidthMax and HeightMax. Some cameras report the native
    // resolution of the imager there, but some others report bugus
    // values, and the user then MUST provide the correct
    // dimensions
    int width;
    int height;

    // If time_decimation_factor > 1, we report every Nth frame to the user.
    int time_decimation_factor;
    bool acquisition_persistent; // never stop the acquisition
    // Shouldn't be needed, but I can't get data from some cameras
    // without it
    bool recreate_stream_with_each_frame;
    bool verbose;
} mrcam_options_t;

// camera_name = NULL means "first available camera"
bool mrcam_init(// out
                mrcam_t* ctx,
                // in
                const char* camera_name,
                const mrcam_options_t* options);

// deallocates everything, and sets all the pointers in ctx to NULL
void mrcam_free(mrcam_t* ctx);

bool mrcam_is_inited(mrcam_t* ctx);




// Synchronous get-image functions.
//
// timeout_us=0 means "wait forever"
//
// The image structure should exist in memory, but the data buffer doesn't need
// to be preallocated or freed
//
// Usage:
//
//   {
//     ...
//     mrcal_image_uint8_t image;
//     mrcam_pull_uint8(&image, timeout_us, &ctx);
//     // no free(image.data); image structure valid until next
//     // mrcam_pull... call.
//     //
//     // do stuff with image
//     ...
//   }
bool mrcam_pull_uint8( // out
                       mrcal_image_uint8_t* image,
                       uint64_t* timestamp_us,
                       // in
                       const uint64_t timeout_us,
                       mrcam_t* ctx);
bool mrcam_pull_uint16(// out
                       mrcal_image_uint16_t* image,
                       uint64_t* timestamp_us,
                       // in
                       const uint64_t timeout_us,
                       mrcam_t* ctx);
bool mrcam_pull_bgr(   // out
                       mrcal_image_bgr_t* image,
                       uint64_t* timestamp_us,
                       // in
                       const uint64_t timeout_us,
                       mrcam_t* ctx);

// Asynchronous get-image functions
//
// Asynchronous usage:
//
//   void callback(mrcal_image_uint8_t image,
//                 mrcam_buffer_t* buffer,
//                 uint64_t timestamp_us,
//                 void* cookie)
//   {
//       // we may or may not be in the same thread where the image
//       // was requested
//
//       // Do stuff with image. When we are done using the image, you must
//       // call:
//       mrcam_callback_done_with_buffer(buffer);
//
//       // The data inside the image is now no-longer usable
//   }
//   ....
//   some_other_function
//   {
//     ...
//     mrcam_request_uint8(&callback, &ctx);
//     // mrcam_request_uint8() returns immediately. we do other
//     // unrelated stuff now. When the image comes in, the callback will
//     // be called
//     ...
//   }
//
// If mrcam_request_...() failed, there will be no callback call.
//
// If mrcam_request_...() succeeded, there will be exactly ONE callback
// call. If there was a problem, the callback will have image.data==NULL
//
// To give up waiting on a callback, call mrcam_cancel_request()
// Requesting an image before the previous one was processed is an error
bool mrcam_request_uint8( // in
                          mrcam_callback_image_uint8_t* callback,
                          mrcam_callback_t*             callback_off_decimation,
                          void* cookie,
                          mrcam_t* ctx);
bool mrcam_request_uint16(// in
                          mrcam_callback_image_uint16_t* callback,
                          mrcam_callback_t*              callback_off_decimation,
                          void* cookie,
                          mrcam_t* ctx);
bool mrcam_request_bgr(   // in
                          mrcam_callback_image_bgr_t* callback,
                          mrcam_callback_t*           callback_off_decimation,
                          void* cookie,
                          mrcam_t* ctx);
bool mrcam_cancel_request(mrcam_t* ctx);

void mrcam_callback_done_with_buffer(mrcam_buffer_t* buffer);
