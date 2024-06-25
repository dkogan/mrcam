#pragma once

#include <mrcal/mrcal-image.h>

// I define this myself, in order to avoid requiring the user to #include
// <arv.h>. This is a few common formats. MRCAM_PIXFMT_... has an equivalent
// ARV_PIXEL_FORMAT.
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


#define LIST_MRCAM_TRIGGER(_) \
    _(SOFTWARE) \
    _(TTYS0)    \
    _(HARDWARE_EXTERNAL)

typedef enum {
#define ENUM(name, ...) MRCAM_TRIGGER_ ## name,
    LIST_MRCAM_TRIGGER(ENUM) MRCAM_TRIGGER_COUNT
#undef ENUM
} mrcam_trigger_t;



typedef enum {MRCAM_UNKNOWN = -1,
              MRCAM_uint8, MRCAM_uint16, MRCAM_bgr } mrcam_output_type_t;
mrcam_output_type_t mrcam_output_type(mrcam_pixfmt_t pixfmt);

typedef void (mrcam_callback_image_uint8_t )(mrcal_image_uint8_t image,
                                             uint64_t timestamp_us,
                                             void* cookie);
typedef void (mrcam_callback_image_uint16_t)(mrcal_image_uint16_t image,
                                             uint64_t timestamp_us,
                                             void* cookie);
typedef void (mrcam_callback_image_bgr_t)(   mrcal_image_bgr_t image,
                                             uint64_t timestamp_us,
                                             void* cookie);



typedef struct
{
    // arv stuff; void to not require including arv.h
    void* camera;
    void* buffer;
    void* stream;

    // Details about the requested pixel format, that I'm using to talk to the
    // camera. I don't NEED to store all these here, but it makes life easier
    mrcam_pixfmt_t pixfmt;

    // The trigger mode
    mrcam_trigger_t trigger;

    // Used to convert from non-trivial pixel formats coming out of the camera
    // to unpacked bits mrcam reports. Needed for bayered or packed formats.
    // Unused for all others
    struct SwsContext* sws_context;
    uint8_t*           output_image_buffer;


    // current active callback. Type may not be 100% right (may be uint8 or
    // uint16, or bgr, ...), but the data layout is the same
    mrcam_callback_image_uint8_t* active_callback;
    void*                         active_callback_cookie;
                                                \
    // used if MRCAM_TRIGGER_TTYS0
    int fd_tty_trigger;

    bool acquiring : 1;
    bool recreate_stream_with_each_frame : 1;
    bool verbose : 1;

} mrcam_t;

typedef struct
{
    const mrcam_pixfmt_t pixfmt;
    mrcam_trigger_t trigger;
    // if either is <=0, we try to autodetect by asking the camera
    // for WidthMax and HeightMax. Some cameras report the native
    // resolution of the imager there, but some others report bugus
    // values, and the user then MUST provide the correct
    // dimensions
    int width;
    int height;

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
//   void cb(mrcal_image_uint8_t image,
//           uint64_t timestamp_us)
//   {
//       // no free(image.data); image structure valid until next
//       // mrcam_request... call.
//       // we may or may not be in the same thread where the image
//       // was requested
//   }
//   ....
//   some_other_function
//   {
//     ...
//     mrcam_request_uint8(&cb, &ctx);
//     // no free(image.data)
//     // mrcam_request_uint8() returned immediately. we do other
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
                          mrcam_callback_image_uint8_t* cb,
                          void* cookie,
                          mrcam_t* ctx);
bool mrcam_request_uint16(// in
                          mrcam_callback_image_uint16_t* cb,
                          void* cookie,
                          mrcam_t* ctx);
bool mrcam_request_bgr(   // in
                          mrcam_callback_image_bgr_t* cb,
                          void* cookie,
                          mrcam_t* ctx);
bool mrcam_cancel_request(mrcam_t* ctx);
