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
MULTI_FRAME mode also asks for ONE frame. These start and stop the acquisition
for each frame.

By contrast, the CONTINUOUS mode starts the acquisition at the beginning of the
capture, and keeps it going for all the subsequent frames
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



typedef enum {MRCAM_UNKNOWN = -1,
              MRCAM_uint8, MRCAM_uint16, MRCAM_bgr } mrcam_output_type_t;
mrcam_output_type_t mrcam_output_type(mrcam_pixfmt_t pixfmt);

typedef void (mrcam_callback_t )(mrcal_image_uint8_t image, // type may not be exact
                                 void* buffer,              // ArvBuffer*, without requiring #include arv.h
                                 uint64_t timestamp_us,
                                 void* cookie);

typedef struct
{
    // arv stuff; void to not require #including arv.h
    void* camera;
    void** buffers;
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
    uint8_t*           sws_output_buffer;


    // current active callback
    mrcam_callback_t* active_callback;
    // Callback used if time_decimation_factor > 1. Called after each frame
    // capture that was NOT aligned with the decimation cycle. Can be NULL.
    // Useful for things like external triggering, which must happen with EVERY
    // captured frame, not just the decimated ones
    mrcam_callback_t* active_callback_off_decimation;
    void*             active_callback_cookie;
                                                \
    // used if MRCAM_TRIGGER_HARDWARE_TTYS0
    int fd_tty_trigger;

    // If time_decimation_factor > 1, we report every Nth frame to the user.
    // This applies to mrcam_pull() and mrcam_request(). The
    // time_decimation_index is the internal counter that's updated to respect
    // the factor when using the asynchronous request() function
    int time_decimation_factor;
    int time_decimation_index;

    int Nbuffers;

    uint64_t timestamp_request_us;
    uint64_t timestamp_start_buffer_us;

    bool acquiring                       : 1; // we're acquiring RIGHT NOW

    // Options. These don't change after init
    bool verbose                         : 1;
} mrcam_t;

typedef struct
{
    mrcam_pixfmt_t           pixfmt;
    mrcam_trigger_t          trigger;
    mrcam_acquisition_mode_t acquisition_mode;

    // if either is <=0, we try to autodetect by asking the camera
    // for WidthMax and HeightMax. Some cameras report the native
    // resolution of the imager there, but some others report bugus
    // values, and the user then MUST provide the correct
    // dimensions
    int width;
    int height;

    int Nbuffers;

    // If time_decimation_factor > 1, we report every Nth frame to the user.
    int time_decimation_factor;
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




// Synchronous capture
// Usage:
//
//   capture()
//   {
//     ...
//     mrcal_image_uint8_t image;
//     void* buffer;
//     mrcam_pull(&image, &buffer, timeout_us, &ctx);
//     // do stuff with image
//     mrcam_push_buffer(buffer);
//     ...
//   }
//
// timeout_us=0 means "wait forever"
//
// Note: with multiple cameras and especially time_decimation_factor>1
// mrcam_pull() might not be what you want. You might want to mrcam_request() to
// ask for data for all the cameras at the same time, instead of waiting for all
// of one camera's data to come in before requesting frames from the next camera
//
// The image structure should exist in memory, but the data buffer doesn't need
// to be preallocated or freed
//
// You MUST call mrcam_push_buffer(buffer) when it is done processing the image,
// to make the buffer available to future captures.
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
                mrcam_t* ctx);

// Asynchronous capture
// Usage:
//
//   void callback(mrcal_image_uint8_t image,
//                 void* buffer,
//                 uint64_t timestamp_us,
//                 void* cookie)
//   {
//       // we may or may not be in the same thread where the image
//       // was requested
//
//       // Do stuff with image. When we are done using the image, we MUST do
//       mrcam_push_buffer(buffer);
//
//       // The data inside the image is now no-longer usable
//   }
//   ....
//   capture()
//   {
//     ...
//     mrcam_request(&callback, callback_off_decimation, &ctx);
//     // mrcam_request() returns immediately. we do other
//     // unrelated stuff now. When the image comes in, the callback will
//     // be called
//     ...
//   }
//
// timeout_us=0 means "wait forever"
//
// If mrcam_request() failed and returned false, there will be no callback call.
//
// If mrcam_request() succeeded, there will be usually be ONE callback call; if
// there was a problem, the callback will have image.data==NULL
//
// The exception to the one-request-one-callback rule is free-running captures:
//
//   ctx->acquisition_mode == MRCAM_ACQUISITION_MODE_CONTINUOUS &&
//   (ctx->trigger == MRCAM_TRIGGER_NONE ||
//    ctx->trigger == MRCAM_TRIGGER_HARDWARE_EXTERNAL)
//
// In that scenario mrcam_request() only does anything on the first call (it
// initiates the capture). In subsequent calls, mrcam_request() doesn't do
// anything, and the frames come in on their own whenever the camera wants to
// send them. It is still recommended to call mrcam_request() even during this
// scenario to, at the very least, be able to restart the capture if something
// goes wrong
//
// The callback MUST call mrcam_push_buffer(buffer) when it is done processing
// the image, to make the buffer available to future captures. Even if
// image==NULL (some error occurred), you must mrcam_push_buffer(buffer)
//
// To give up waiting on a callback, call mrcam_cancel_request()
// Requesting an image before the previous one was processed is an error
bool mrcam_request( // in
                    mrcam_callback_t* callback,
                    mrcam_callback_t* callback_off_decimation,
                    void* cookie,
                    mrcam_t* ctx);
bool mrcam_cancel_request(mrcam_t* ctx);

void mrcam_push_buffer(void**   buffer, // the buffer, from mrcam_pull() or a mrcam_callback_t
                       mrcam_t* ctx);

void mrcam_sleep_until_next_request(// The frame period. Used only if
                                    // time_decimation_factor > 1; sets the
                                    // delay of the off-decimation frame
                                    // requests. Pass 0 to ignore, and to
                                    // request the frames immediately
                                    uint64_t period_us,
                                    mrcam_t* ctx);

/*
An equalization routine meant to boost the local contrast in 16-bit images.
Based on:

  Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared
  Image Hyeonjae Gil, Myeon-Hwan Jeon, and Ayoung Kim

  @article{gil2024fieldscale,
    title={Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared Image},
    author={Gil, Hyeonjae and Jeon, Myung-Hwan and Kim, Ayoung},
    journal={IEEE Robotics and Automation Letters},
    year={2024},
    publisher={IEEE}
  }

  Original author: Hyeonjae Gil
  Author email: h.gil@snu.ac.kr
*/
bool mrcam_equalize_fieldscale(// out
                               mrcal_image_uint8_t*        image_out,
                               // in
                               const mrcal_image_uint16_t* image_in);
