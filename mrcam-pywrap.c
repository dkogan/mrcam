#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <poll.h>
#include <assert.h>
#include <pthread.h>
#include <mrcal/mrcal-image.h>
#include <arv.h>

#include "mrcam.h"

#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}

#define IS_NULL(x) ((x) == NULL || (PyObject*)(x) == Py_None)

#define BARF(fmt, ...) PyErr_Format(PyExc_RuntimeError, "%s:%d %s(): "fmt, __FILE__, __LINE__, __func__, ## __VA_ARGS__)

// the try...() macros in util.h will produce Python errors
#define ERR(fmt, ...) BARF(fmt, ##__VA_ARGS__)
#include "util.h"



// Python is silly. There's some nuance about signal handling where it sets a
// SIGINT (ctrl-c) handler to just set a flag, and the python layer then reads
// this flag and does the thing. Here I'm running C code, so SIGINT would set a
// flag, but not quit, so I can't interrupt the capture. Thus I reset the SIGINT
// handler to the default, and put it back to the python-specific version when
// I'm done
#define SET_SIGINT() struct sigaction sigaction_old;                    \
do {                                                                    \
    if( 0 != sigaction(SIGINT,                                          \
                       &(struct sigaction){ .sa_handler = SIG_DFL },    \
                       &sigaction_old) )                                \
    {                                                                   \
        BARF("sigaction() failed");                                     \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        BARF("sigaction-restore failed"); \
} while(0)


#define PIPE_FD_READ  0
#define PIPE_FD_WRITE 1
typedef struct {
    PyObject_HEAD

    mrcam_t ctx;

    // Descriptors returned from pipe()

    // Used by the asynchronous frame grabbing using the request() function.
    // When a frame is available, a image_ready_t structure is sent over this
    // pipe. The Python main thread can then read the pipe to process the frame
    int pipe_capture[2];

    // Used to asynchronously save images to disk by
    // async_save_image_and_push_buffer_t()
    int pipe_save[2];

    // Used to asynchronously save images to disk by
    // async_save_image_and_push_buffer_t()
    pthread_t thread_save;
    bool      thread_save_active : 1;

} camera;

// The structure being sent over the pipe when an image is received
typedef struct
{
    mrcal_image_uint8_t mrcal_image; // type might not be exact
    uint64_t            timestamp_us;
    ArvBuffer*          buffer;
    bool                off_decimation;
} image_ready_t;


typedef bool (mrcal_image_uint8_save_t)(const char* filename, const mrcal_image_uint8_t* image);
// The structure being sent over the pipe when we want to write an image to disk
// asynchronously (writing might be slow, and we want to parallelize it)
typedef struct
{
    mrcal_image_uint8_t mrcal_image; // type might not be exact
    ArvBuffer*          buffer;
    char                path[1024 - sizeof(ArvBuffer*) - sizeof(mrcal_image_uint8_t)];
    mrcal_image_uint8_save_t* mrcal_image_generic_save; // type-specific function
} async_save_image_and_push_buffer_t;



static bool int_from_sequence_element(// out
                                      int* x,
                                      // in
                                      PyObject* py_sequence,
                                      int i)
{
    bool      result   = false;
    PyObject* py_value = NULL;

    py_value = PySequence_GetItem(py_sequence,i);
    if(py_value == NULL || !PyLong_Check(py_value))
        goto done;

    *x = (int)PyLong_AsLong(py_value);
    result = true;

 done:
    Py_XDECREF(py_value);
    return result;
}

static bool string_from_pystring__leave_if_null_or_none(// out
                                                        const char** string,
                                                        // in
                                                        PyObject* py_string)
{
    if(IS_NULL(py_string))
        return true;

    if(!PyUnicode_Check(py_string))
        return false;

    *string = PyUnicode_AsUTF8(py_string);
    if(string == NULL)
    {
        // error is set
        return false;
    }
    return true;
}

static void* thread_save(void* cookie);

static int
camera_init(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    int result = -1;

    SET_SIGINT();

    // These must match camera_param_keys in _parse_args_postprocess() in
    // mrcam.py
    char* keywords[] = {"name",
                        "pixfmt",
                        "acquisition_mode",
                        "trigger",
                        "time_decimation_factor",
                        "dims",
                        "Nbuffers",
                        "verbose",
                        NULL};

    // The default pixel format is "MONO_8". Should match the one in the
    // LIST_OPTIONS macro in mrcam-test.c
    const char* camera_name             = NULL;
    const char* pixfmt_string           = "MONO_8";
    const char* acquisition_mode_string = "SINGLE_FRAME";
    const char* trigger_string          = "SOFTWARE";
    PyObject* py_pixfmt_string           = NULL;
    PyObject* py_acquisition_mode_string = NULL;
    PyObject* py_trigger_string          = NULL;

    int         time_decimation_factor  = 1;
    PyObject* py_width_height = NULL;
    int width   = 0; // by default, auto-detect the dimensions
    int height  = 0;
    int verbose = 0;
    int Nbuffers = 10; // default buffer count

    mrcam_pixfmt_t           pixfmt;
    mrcam_acquisition_mode_t acquisition_mode;
    mrcam_trigger_t          trigger;

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|z$OOOiOip:mrcam.__init__", keywords,
                                     &camera_name,
                                     &py_pixfmt_string,
                                     &py_acquisition_mode_string,
                                     &py_trigger_string,
                                     &time_decimation_factor,
                                     &py_width_height,
                                     &Nbuffers,
                                     &verbose))
        goto done;

    if(0 != pipe(self->pipe_capture))
    {
        BARF("Couldn't init pipe");
        goto done;
    }
    if(0 != pipe(self->pipe_save))
    {
        BARF("Couldn't init pipe");
        goto done;
    }

    if(!string_from_pystring__leave_if_null_or_none(&pixfmt_string, py_pixfmt_string))
    {
        BARF("if pixfmt is given and non-None, it must be a string");
        goto done;
    }
    if(!string_from_pystring__leave_if_null_or_none(&acquisition_mode_string, py_acquisition_mode_string))
    {
        BARF("if acquisition_mode is given and non-None, it must be a string");
        goto done;
    }
    if(!string_from_pystring__leave_if_null_or_none(&trigger_string, py_trigger_string))
    {
        BARF("if trigger is given and non-None, it must be a string");
        goto done;
    }

    if(0) ;
#define PARSE(name, name_genicam, ...)           \
    else if(0 == strcasecmp(pixfmt_string, #name) || \
            0 == strcasecmp(pixfmt_string, #name_genicam))  \
        pixfmt = MRCAM_PIXFMT_ ## name;
    LIST_MRCAM_PIXFMT(PARSE)
    else
    {
#define SAY(name, name_genicam, ...) "'" #name "', " #name_genicam "', "
        BARF("Unknown pixel format '%s'; mrcam knows about: ("
             LIST_MRCAM_PIXFMT(SAY)
             ")\n"
             "These aren't all supported by each camera:\n"
             "run 'arv-tool-0.8 features PixelFormat' to query the hardware",
             pixfmt_string);
        goto done;
#undef SAY
    }
#undef PARSE

    if(0) ;
#define PARSE(name, ...)                        \
    else if(0 == strcasecmp(acquisition_mode_string, #name))  \
        acquisition_mode = MRCAM_ACQUISITION_MODE_ ## name;
    LIST_MRCAM_ACQUISITION_MODE(PARSE)
    else
    {
#define SAY(name, ...) "'" #name "', "
        BARF("Unknown acquisition_mode mode '%s'; I know about: ("
             LIST_MRCAM_ACQUISITION_MODE(SAY)
             ")",
             acquisition_mode_string);
        goto done;
#undef SAY
    }
#undef PARSE

    if(0) ;
#define PARSE(name, ...)                        \
    else if(0 == strcasecmp(trigger_string, #name))  \
        trigger = MRCAM_TRIGGER_ ## name;
    LIST_MRCAM_TRIGGER(PARSE)
    else
    {
#define SAY(name, ...) "'" #name "', "
        BARF("Unknown trigger mode '%s'; I know about: ("
             LIST_MRCAM_TRIGGER(SAY)
             ")",
             trigger_string);
        goto done;
#undef SAY
    }
#undef PARSE

    if(!IS_NULL(py_width_height))
    {
        if(!(PySequence_Check(py_width_height) &&
             PySequence_Size (py_width_height) == 2))
        {
            BARF("'dims' is given; it must be an iterable with exactly 2 integers");
            goto done;
        }
        if(!int_from_sequence_element(&width, py_width_height,0))
        {
            BARF("'dims' is given; it must be an iterable with exactly 2 integers; item 0 coulnd't be interpreted as an int");
            goto done;
        }
        if(!int_from_sequence_element(&height, py_width_height,1))
        {
            BARF("'dims' is given; it must be an iterable with exactly 2 integers; item 0 coulnd't be interpreted as an int");
            goto done;
        }
    }

    const mrcam_options_t mrcam_options =
        {
            .pixfmt                          = pixfmt,
            .acquisition_mode                = acquisition_mode,
            .trigger                         = trigger,
            .time_decimation_factor          = time_decimation_factor,
            .width                           = width,
            .height                          = height,
            .Nbuffers                        = Nbuffers,
            .verbose                         = verbose
        };
    if(!mrcam_init(&self->ctx,
                   camera_name,
                   &mrcam_options))
    {
        BARF("Couldn't init mrcam camera");
        goto done;
    }

    if(0 != pthread_create(&self->thread_save,
                           NULL,
                           thread_save, self))
    {
        BARF("Couldn't start image-save thread");
        goto done;
    }
    self->thread_save_active = true;

    result = 0;

 done:
    RESET_SIGINT();
    return result;
}

static void camera_dealloc(camera* self)
{
    if(self->thread_save_active)
    {
        if(0 != pthread_cancel(self->thread_save))
            BARF("Couldn't pthread_cancel(thread_save)... Continuing dealloc anyway");
        else
        {
            if(0 != pthread_join(self->thread_save, NULL))
                BARF("Couldn't pthread_join(thread_save)... Continuing dealloc anyway");

        }
        self->thread_save_active = false;
    }

    mrcam_free(&self->ctx);

    close(self->pipe_capture[PIPE_FD_READ]);
    close(self->pipe_capture[PIPE_FD_WRITE]);
    close(self->pipe_save   [PIPE_FD_READ]);
    close(self->pipe_save   [PIPE_FD_WRITE]);

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
numpy_image_from_mrcal_image(// type not exact
                             const mrcal_image_uint8_t* mrcal_image, // type might not be exact,
                             mrcam_output_type_t type)
{
    switch(type)
    {
    case MRCAM_uint8:
        {
            const int bytes_per_pixel__output = 1;
            if(mrcal_image->stride != mrcal_image->width * bytes_per_pixel__output)
            {
                BARF("Image returned by mrcam_pull() is not contiguous");
                return NULL;
            }
            return
                PyArray_SimpleNewFromData(2,
                                          ((npy_intp[]){mrcal_image->height,
                                                        mrcal_image->width}),
                                          NPY_UINT8,
                                          mrcal_image->data);
        }
        break;

    case MRCAM_uint16:
        {
            const int bytes_per_pixel__output = 2;
            if(mrcal_image->stride != mrcal_image->width * bytes_per_pixel__output)
            {
                BARF("Image returned by mrcam_pull() is not contiguous");
                return NULL;
            }
            return
                PyArray_SimpleNewFromData(2,
                                          ((npy_intp[]){mrcal_image->height,
                                                        mrcal_image->width}),
                                          NPY_UINT16,
                                          mrcal_image->data);
        }
        break;

    case MRCAM_bgr:
        {
            const int bytes_per_pixel__output = 3;
            if(mrcal_image->stride != mrcal_image->width * bytes_per_pixel__output)
            {
                BARF("Image returned by mrcam_pull() is not contiguous");
                return NULL;
            }
            return
                PyArray_SimpleNewFromData(3,
                                          ((npy_intp[]){mrcal_image->height,
                                                        mrcal_image->width,
                                                        3}),
                                          NPY_UINT8,
                                          mrcal_image->data);
        }
        break;

    default:
        BARF("Getting here is a bug");
        return NULL;
    }
}

static
bool mrcal_image_from_numpy_image(mrcal_image_uint8_t* image, // type not exact
                                  mrcal_image_uint8_save_t** generic_save,
                                  PyArrayObject*       py_image )
{
    if(!PyArray_Check(py_image))
    {
        BARF("Input not a numpy array");
        return false;
    }

    if( PyArray_TYPE(py_image) == NPY_UINT8 &&
        PyArray_NDIM(py_image) == 2 )
    {
        if(PyArray_STRIDES(py_image)[1] != (int)sizeof(uint8_t))
        {
            BARF("8-bit monochrome image expected to have densely-stored pixels");
            return false;
        }
        if(generic_save != NULL) *generic_save = &mrcal_image_uint8_save;
    }
    else if( PyArray_TYPE(py_image) == NPY_UINT16 &&
             PyArray_NDIM(py_image) == 2 )
    {
        if(PyArray_STRIDES(py_image)[1] != (int)sizeof(uint16_t))
        {
            BARF("16-bit monochrome image expected to have densely-stored pixels");
            return false;
        }
        if(generic_save != NULL) *generic_save = (mrcal_image_uint8_save_t*)&mrcal_image_uint16_save;
    }
    else if( PyArray_TYPE(py_image) == NPY_UINT8 &&
             PyArray_NDIM(py_image) == 3 )
    {
        if(PyArray_DIMS(py_image)[2] != 3)
        {
            BARF("image.ndim==3, but image.shape[-1]!=3: not a 3-channel color image");
            return false;
        }
        if(PyArray_STRIDES(py_image)[2] != (int)sizeof(uint8_t))
        {
            BARF("24-bit color image expected to have densely-stored channels");
            return false;
        }
        if(PyArray_STRIDES(py_image)[1] != 3*(int)sizeof(uint8_t))
        {
            BARF("24-bit color image expected to have densely-stored channels");
            return false;
        }
        if(generic_save != NULL) *generic_save = (mrcal_image_uint8_save_t*)&mrcal_image_bgr_save;
    }
    else
    {
        BARF("Unexpected image type. I know about 8-bit-1-channel, 16-bit-1-channel, 8-bit-3-channel images only");
        return false;
    }

    image->width  = PyArray_DIMS   (py_image)[1];
    image->height = PyArray_DIMS   (py_image)[0];
    image->stride = PyArray_STRIDES(py_image)[0];
    image->data   = PyArray_DATA   (py_image);

    return true;
}


static bool fd_has_data(int fd)
{
    struct pollfd fds = {.fd     = fd,
                         .events = POLLIN};
    int result = poll(&fds, 1, 0);
    if(result < 0)
    {
        MSG("poll() failed... reporting that no data is available. Recovery is likely impossible");
        return false;
    }

    return result > 0;
}

static bool currently_processing_image(const camera* self)
{
#warning need to know if request() but not yet ready
    return
        fd_has_data(self->pipe_capture[PIPE_FD_READ]);
}

// Synchronous frame processing. May block. No pipe
static PyObject*
pull(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject* result = NULL;
    PyObject* image  = NULL;
    void*     buffer = NULL;

    char* keywords[] = {"timeout",
                        "period",
                        NULL};

    double timeout_sec = 0.0;
    double period_sec  = 0.0;

    if(currently_processing_image(self))
    {
        BARF("An image is already being processed...");
        goto done;
    }


    SET_SIGINT();

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|$dd:mrcam.pull", keywords,
                                     &timeout_sec,
                                     &period_sec))
        goto done;

    if(timeout_sec < 0) timeout_sec = 0;

    // generic type
    mrcal_image_uint8_t mrcal_image;
    uint64_t            timestamp_us;
    if(!mrcam_pull( &mrcal_image,
                    &buffer,
                    &timestamp_us,
                    (uint64_t)(period_sec  * 1e6),
                    (uint64_t)(timeout_sec * 1e6),
                    &self->ctx))
    {
        BARF("mrcam_pull() failed");
        goto done;
    }

    image = numpy_image_from_mrcal_image(&mrcal_image, mrcam_output_type(self->ctx.pixfmt));
    if(image == NULL)
        // BARF() already called
        goto done;

    result = Py_BuildValue("{sOsNsd}",
                           "image",     image,
                           "buffer",    PyLong_FromVoidPtr(buffer),
                           "timestamp", (double)timestamp_us / 1e6);
    if(result == NULL)
    {
        BARF("Couldn't build %s() result", __func__);
        goto done;
    }

 done:

    Py_XDECREF(image);
    if(result == NULL)
        mrcam_push_buffer(&buffer, &self->ctx);
    RESET_SIGINT();
    return result;
}

static
ssize_t read_persistent(int fd, uint8_t* buf, size_t count)
{
    ssize_t N      = 0;
    ssize_t Nremaining = count;
    ssize_t Nhere;
    do
    {
        Nhere = read(fd, buf, Nremaining);
        if(Nhere < 0)
        {
            if(errno==EINTR)
                continue;
            return Nhere;
        }
        if(Nhere == 0)
            return N;
        N          += Nhere;
        Nremaining -= Nhere;
        buf        = &buf[Nhere];
    } while(Nremaining);
    return N;
}
static
ssize_t write_persistent(int fd, const uint8_t* buf, size_t count)
{
    ssize_t N      = 0;
    ssize_t Nremaining = count;
    ssize_t Nhere;
    do
    {
        Nhere = write(fd, buf, Nremaining);
        if(Nhere < 0)
        {
            if(errno==EINTR)
                continue;
            return Nhere;
        }
        if(Nhere == 0)
            return N;
        N          += Nhere;
        Nremaining -= Nhere;
        buf        = &buf[Nhere];
    } while(Nremaining);
    return N;
}

static
void
callback(mrcal_image_uint8_t mrcal_image, // type might not be exact
         void*    buffer, // ArvBuffer*, without requiring #include arv.h
         uint64_t timestamp_us,
         void* cookie)
{
    camera* self = (camera*)cookie;

    image_ready_t s = {.mrcal_image  = mrcal_image,
                       .timestamp_us = timestamp_us,
                       .buffer       = buffer};

    if(sizeof(s) != write_persistent(self->pipe_capture[PIPE_FD_WRITE], (uint8_t*)&s, sizeof(s)))
    {
        MSG("Couldn't write image metadata to pipe!");
        return;
    }
}

static
void
callback_off_decimation(__attribute__((unused)) mrcal_image_uint8_t mrcal_image, // type might not be exact
                        __attribute__((unused)) void*    buffer, // ArvBuffer*, without requiring #include arv.h
                        __attribute__((unused)) uint64_t timestamp_us,
                        void* cookie)
{
    camera* self = (camera*)cookie;

    image_ready_t s = {.off_decimation = true};

    if(sizeof(s) != write_persistent(self->pipe_capture[PIPE_FD_WRITE], (uint8_t*)&s, sizeof(s)))
    {
        MSG("Couldn't write image metadata to pipe!");
        return;
    }
}


static PyObject*
request(camera* self, PyObject* args)
{
    if(!mrcam_request( &callback,
                       &callback_off_decimation,
                       self,
                       &self->ctx))
    {
        BARF("mrcam_request() failed");
        goto done;
    }

    Py_RETURN_NONE;

 done:

    // already called BARF()

    return NULL;
}

static PyObject*
requested_image(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject* result = NULL;
    PyObject* image  = NULL;

    char* keywords[] = {"block",
                        NULL};

    int block = 0;

    SET_SIGINT();

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|$p:mrcam.requested_image", keywords,
                                     &block))
        goto done;

    if(!block && !fd_has_data(self->pipe_capture[PIPE_FD_READ]))
    {
        BARF("Non-blocking mode requested, but no data is available to be read");
        goto done;
    }

    image_ready_t s;
    if(sizeof(s) != read_persistent(self->pipe_capture[PIPE_FD_READ], (uint8_t*)&s, sizeof(s)))
    {
        BARF("Couldn't read image metadata from pipe!");
        goto done;
    }

    if(!s.off_decimation)
    {
        if(s.mrcal_image.data != NULL)
        {
            image = numpy_image_from_mrcal_image(&s.mrcal_image, mrcam_output_type(self->ctx.pixfmt));
            if(image == NULL)
                // BARF() already called
                goto done;
        }
        else
        {
            // Error occurred. I return None as the image
            image = Py_None;
            Py_INCREF(image);
        }

        result = Py_BuildValue("{sOsdsNsi}",
                               "image",          image,
                               "timestamp",      (double)s.timestamp_us / 1e6,
                               "buffer",         PyLong_FromVoidPtr(s.buffer),
                               "off_decimation", 0);
    }
    else
        result = Py_BuildValue("{sOsOsOsi}",
                               "image",          Py_None,
                               "timestamp",      Py_None,
                               "buffer",         Py_None,
                               "off_decimation", 1);
    if(result == NULL)
    {
        BARF("Couldn't build %s() result", __func__);
        goto done;
    }

 done:
    Py_XDECREF(image);

    RESET_SIGINT();
    return result;
}


// PyDict_GetItemString() with full error checking. The documentation:
//   https://docs.python.org/3/c-api/dict.html
// suggests to not use PyDict_GetItemString() but to do more complete error
// checking ourselves. This function does that
static PyObject* GetItemString(PyObject* dict, const char* key_string)
{
    PyObject* key    = NULL;
    PyObject* result = NULL;

    key = PyUnicode_FromString(key_string);
    if(key == NULL)
    {
        BARF("Couldn't create PyObject from short string. Giving up");
        goto done;
    }
    result = PyDict_GetItemWithError(dict, key);
    if(result == NULL)
    {
        if(!PyErr_Occurred())
            BARF("No expected key '%s' found in dict", key_string);
        goto done;
    }

 done:
    Py_XDECREF(key);
    return result;
}


// Recursively accumulate all implemented feature nodes into the set. Include
// only nodes whose name matches the regex. If regex == NULL, we accumulate
// EVERYTHING. If a node name matches a category name, we accumulate everything
// in that category
//
// Based on arv_tool_list_features() in aravis/src/arvtool.c
static bool
accumulate_feature(// output; we accumulate here
                   PyObject* set,
                   // input
                   ArvGc* genicam,
                   const char* feature,
                   GRegex* regex)
{
    ArvGcNode* node = arv_gc_get_node(genicam, feature);

    if(! (ARV_IS_GC_FEATURE_NODE (node) &&
          arv_gc_feature_node_is_implemented (ARV_GC_FEATURE_NODE (node), NULL) &&
          arv_gc_feature_node_is_available(   ARV_GC_FEATURE_NODE(node),  NULL)) )
        return true;

    const char* name = arv_gc_feature_node_get_name (ARV_GC_FEATURE_NODE (node));
    gboolean match = true;
    if(regex)
        match = g_regex_match(regex,
                              name,
                              0, NULL);

    if (ARV_IS_GC_CATEGORY (node))
    {
        // Recurve even if the category name didn't match
        const GSList* features = arv_gc_category_get_features (ARV_GC_CATEGORY (node));
        for (const GSList* iter = features; iter != NULL; iter = iter->next)
            if(!accumulate_feature(set,
                                   genicam,
                                   iter->data,
                                   // If the category name DID match, select
                                   // all the children nodes
                                   match ? NULL : regex))
                return false;
    }
    else if (match)
    {
        PyObject* pyname = PyUnicode_FromString(name);
        if(pyname == NULL)
        {
            BARF("Couldn't create name object");
            return false;
        }
        int result = PySet_Add(set, pyname);
        Py_DECREF(pyname);
        // Exception already set if PySet_Add() failed
        return (0 == result);
    }

    return true;
}


static PyObject*
features(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject*   result       = NULL;
    GError*     error        = NULL;
    const char* regex_string = NULL;

    ArvGc*    genicam = NULL;
    GRegex*   regex   = NULL;
    PyObject* set     = NULL;

    mrcam_t* ctx = &self->ctx;
    ArvDevice* device = arv_camera_get_device(ARV_CAMERA(self->ctx.camera));
    if(device == NULL)
    {
        BARF("Couldn't arv_camera_get_device()");
        goto done;
    }

    char* keywords[] = {"regex",
                        NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|s:mrcam.features", keywords,
                                     &regex_string))
        goto done;

    set = PySet_New(NULL);
    if(set == NULL)
    {
        BARF("Couldn't create output set");
        goto done;
    }

    genicam = arv_device_get_genicam (device);
    if(regex_string != NULL)
    {
        regex   = g_regex_new(regex_string, 0,0, &error);
        if(error != NULL || regex == NULL)
        {
            BARF("Couldn't compile regex from string '%s'", regex_string);
            g_clear_error(&error);
            goto done;
        }
    }
    if(!accumulate_feature(set,
                           genicam, "Root", regex))
        goto done;

    result = set;

 done:
    if(regex != NULL)
        g_regex_unref(regex);
    if(result == NULL)
        Py_XDECREF(set);
    return result;
}


static PyObject*
stream_stats(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject* result = NULL;
    gint      n_input_buffers;
    gint      n_output_buffers;

    if(self->ctx.stream == NULL)
    {
        BARF("The stream is NULL. Is the camera initialized?");
        goto done;
    }

    arv_stream_get_n_buffers ((ArvStream*)self->ctx.stream,
                              &n_input_buffers,
                              &n_output_buffers);

    result = Py_BuildValue("{sisi}",
                           "n_input_buffers",  n_input_buffers,
                           "n_output_buffers", n_output_buffers);
    if(result == NULL)
    {
        BARF("Couldn't build %s() result", __func__);
        goto done;
    }

 done:
    return result;
}


static PyObject*
push_buffer(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject* result = NULL;
    PyObject* py_buffer = NULL;

    char* keywords[] = {"buffer",
                        NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|O:mrcam.push_buffer", keywords,
                                     &py_buffer))
        goto done;

    if(!IS_NULL(py_buffer))
    {
        void* buffer = PyLong_AsVoidPtr(py_buffer);
        mrcam_push_buffer(&buffer, &self->ctx);
    }

    result = Py_None;
    Py_INCREF(result);

 done:
    return result;
}

static PyObject*
async_save_image_and_push_buffer(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject* result = NULL;

    const char* path      = NULL;
    PyObject*   py_image  = NULL;
    PyObject*   py_buffer = NULL;

    async_save_image_and_push_buffer_t s;

    char* keywords[] = {"path",
                        "image",
                        "buffer",
                        NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "sOO:mrcam.async_save_image_and_push_buffer",
                                     keywords,
                                     &path,
                                     &py_image,
                                     &py_buffer))
        goto done;

    if(IS_NULL(py_buffer))
    {
        BARF("buffer is NULL or None. This is a bug");
        goto done;
    }
    s.buffer = (ArvBuffer*)PyLong_AsVoidPtr(py_buffer);

    if(strlen(path) > sizeof(s.path)-1)
    {
        BARF("The path is too large to fit into the statically allocated async_save_image_and_push_buffer_t.path. Boost its size");
        goto done;
    }
    strcpy(s.path, path);

    if(!mrcal_image_from_numpy_image(&s.mrcal_image,
                                     &s.mrcal_image_generic_save,
                                     (PyArrayObject*)py_image))
        // BARF() already called
        goto done;

    if(sizeof(s) != write_persistent(self->pipe_save[PIPE_FD_WRITE], (uint8_t*)&s, sizeof(s)))
    {
        MSG("Couldn't write image-save data to pipe!");
        goto done;
    }

    result = Py_None;
    Py_INCREF(result);

 done:
    return result;
}

static
void* thread_save(void* cookie)
{
    camera* self = (camera*)cookie;
    async_save_image_and_push_buffer_t s;

    while(true)
    {
        if(sizeof(s) != read_persistent(self->pipe_save[PIPE_FD_READ], (uint8_t*)&s, sizeof(s)))
        {
            fprintf(stderr, "Couldn't read image-save data from pipe.... Giving up on the thread\n");
            return NULL;
        }

        if( !(*s.mrcal_image_generic_save)(s.path,&s.mrcal_image) )
        {
            fprintf(stderr,
                    "## ERROR: couldn't save image to path='%s'\n",
                    s.path);
            continue;
        }

        mrcam_push_buffer((void**)&s.buffer, &self->ctx);
    }

    return NULL;
}

// The feature must be of any of the types in what[]. The list is ended with a
// NULL. what_all_string is for error reporting
static bool feature_Check(PyObject* feature,
                          const char** what,
                          const char* what_all_string)
{
    if(!PyDict_Check(feature))
    {
        BARF("Expected to be passed a dict. The one returned by mrcam.feature_descriptor()");
        return false;
    }

    PyObject* type = GetItemString(feature, "type");
    if(type == NULL) return false;

    if(!PyUnicode_Check(type))
    {
        BARF("The 'type' must be a string");
        return false;
    }
    if(*what == NULL)
    {
        BARF("BUG: nothing in the what[] list");
        return false;
    }
    while(*what)
    {
        if(0 == PyUnicode_CompareWithASCIIString(type, *what))
            return true;
        what = &what[1];
    }

    if(what_all_string)
        BARF("The 'type' must be any of (%s), but here it is '%S'",
             what_all_string,
             type);
    else
        BARF("The 'type' must be '%s', but here it is '%S'",
             what,
             type);
    return false;
}

static void* feature_GetNodeOrAddress(PyObject* feature,
                                      bool* is_node)
{
    PyObject* node;
    const char* what;

    what = "node";
    node = GetItemString(feature, what);
    if(node == NULL) return NULL;

    if(node != Py_None)
        *is_node = true;
    else
    {
        what = "address";
        node = GetItemString(feature, what);
        if(node == NULL) return NULL;

        *is_node = false;
    }

    if(!PyLong_Check(node))
    {
        BARF("The '%s' must be a long", what);
        return NULL;
    }
    static_assert(sizeof(void*) >= sizeof(long), "I'm using a PyLong to store a pointer; it must be large-enough");

    typedef union
    {
        void* node;
        long  long_value;
    } node_and_long_t;

    node_and_long_t u = {.long_value = PyLong_AsLong(node)};
    if(PyErr_Occurred())
    {
        BARF("The '%s' couldn't be interpreted numerically", what);
        return NULL;
    }
    if(u.node == NULL)
    {
        BARF("The '%s' was NULL", what);
        return NULL;
    }
    return u.node;
}

static PyObject*
feature_descriptor(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject*   result       = NULL;
    ArvGcNode*  feature_node = NULL;
    ArvDevice*  device       = NULL;
    GError*     error        = NULL;
    const char* feature      = NULL;
    const char** available_enum_entries = NULL;

    mrcam_t* ctx = &self->ctx; // for the try_arv...() macros

    char* keywords[] = {"feature",
                        NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "s:mrcam.feature_descriptor", keywords,
                                     &feature))
        goto done;

    long address;
    if( feature[0] == 'R' &&
        feature[1] == '[' &&
        feature[strlen(feature)-1] == ']' )
    {
        errno = 0;
        address = strtol(&feature[2], NULL, 0);
        if(errno != 0)
        {
            BARF("Feature '%s' starts with R[ but doesn't have the expected R[ADDRESS] format", feature);
            goto done;
        }

        result = Py_BuildValue("{sssOsls(ii)sisOsO}",
                               "type",           "integer",
                               "node",           Py_None,
                               "address",        address,
#warning these numbers are made-up
                               "bounds",         200, 300,
                               "increment",      1,
                               "representation", Py_None,
                               "unit",           Py_None);
        if(result == NULL)
        {
            BARF("Couldn't build %s() result", __func__);
            goto done;
        }

        goto done;
    }


    device = arv_camera_get_device(ARV_CAMERA(self->ctx.camera));
    if(device == NULL)
    {
        BARF("Couldn't arv_camera_get_device()");
        goto done;
    }

    feature_node = arv_device_get_feature(device, feature);
    if(feature_node == NULL)
    {
        BARF("Couldn't arv_device_get_feature(\"%s\")", feature);
        goto done;
    }

    bool is_available, is_implemented;
    try_arv(is_available   = arv_gc_feature_node_is_available(  ARV_GC_FEATURE_NODE(feature_node), &error));
    try_arv(is_implemented = arv_gc_feature_node_is_implemented(ARV_GC_FEATURE_NODE(feature_node), &error));
    if(!(is_available && is_implemented))
    {
        BARF("feature '%s' must be available,implemented; I have (%d,%d)",
             feature,
             is_available, is_implemented);
        goto done;
    }

    // Check enumeration before integer. enumerations provide the integer
    // interface also, but I want the special enum one
    if( ARV_IS_GC_ENUMERATION(feature_node) )
    {
        guint n_values;
        try_arv(available_enum_entries =
                arv_gc_enumeration_dup_available_string_values(ARV_GC_ENUMERATION(feature_node),
                                                               &n_values,
                                                               &error));
        PyObject* entries = PyTuple_New(n_values);
        if(entries == NULL)
        {
            BARF("Couldn't create new tuple of enum entries of %d values", n_values);
            goto done;
        }
        for(guint i=0; i<n_values; i++)
        {
            PyObject* s = PyUnicode_FromString(available_enum_entries[i]);
            if(s == NULL)
            {
                BARF("Couldn't create enum entry string");
                goto done;
            }
            PyTuple_SET_ITEM(entries, i, s);
        }

        result = Py_BuildValue("{ssslsN}",
                               "type",           "enumeration",
                               "node",           feature_node,
                               "entries",        entries);
        if(result == NULL)
        {
            BARF("Couldn't build %s() result", __func__);
            goto done;
        }

        goto done;
    }
    else if( ARV_IS_GC_INTEGER(feature_node) )
    {
        gint64 min,max,increment;
        try_arv(min       = arv_gc_integer_get_min(ARV_GC_INTEGER(feature_node), &error));
        try_arv(max       = arv_gc_integer_get_max(ARV_GC_INTEGER(feature_node), &error));
        try_arv(increment = arv_gc_integer_get_inc(ARV_GC_INTEGER(feature_node), &error));

        const char* representation;
        switch(arv_gc_integer_get_representation(ARV_GC_INTEGER(feature_node)))
        {
        case ARV_GC_REPRESENTATION_LINEAR:      representation = "LINEAR";      break;
        case ARV_GC_REPRESENTATION_LOGARITHMIC: representation = "LOGARITHMIC"; break;
        case ARV_GC_REPRESENTATION_BOOLEAN:     representation = "BOOLEAN";     break;
        default:                                representation = "OTHER";       break;
        }

        const char* unit = arv_gc_integer_get_unit(ARV_GC_INTEGER(feature_node));


        result = Py_BuildValue("{sssls(dd)sdssss}",
                               "type",           "integer",
                               "node",           feature_node,
                               "bounds",         (double)min, (double)max,
                               "increment",      (double)increment,
                               "representation", representation,
                               "unit",           unit);
        if(result == NULL)
        {
            BARF("Couldn't build %s() result", __func__);
            goto done;
        }
    }
    else if( ARV_IS_GC_FLOAT(feature_node) )
    {
        double min,max,increment;
        try_arv(min       = arv_gc_float_get_min(ARV_GC_FLOAT(feature_node), &error));
        try_arv(max       = arv_gc_float_get_max(ARV_GC_FLOAT(feature_node), &error));
        try_arv(increment = arv_gc_float_get_inc(ARV_GC_FLOAT(feature_node), &error));

        const char* representation;
        switch(arv_gc_float_get_representation(ARV_GC_FLOAT(feature_node)))
        {
        case ARV_GC_REPRESENTATION_LINEAR:      representation = "LINEAR";      break;
        case ARV_GC_REPRESENTATION_LOGARITHMIC: representation = "LOGARITHMIC"; break;
        case ARV_GC_REPRESENTATION_BOOLEAN:     representation = "BOOLEAN";     break;
        default:                                representation = "OTHER";       break;
        }

        const char* unit = arv_gc_float_get_unit(ARV_GC_FLOAT(feature_node));

        gint64 precision;
        try_arv(precision = arv_gc_float_get_display_notation(ARV_GC_FLOAT(feature_node)));

        result = Py_BuildValue("{sssls(dd)sdsssssl}",
                               "type",           "float",
                               "node",           feature_node,
                               "bounds",         (double)min, (double)max,
                               "increment",      (double)increment,
                               "representation", representation,
                               "unit",           unit,
                               "precision",      (long)precision);
        if(result == NULL)
        {
            BARF("Couldn't build %s() result", __func__);
            goto done;
        }
    }
    else if( ARV_IS_GC_BOOLEAN(feature_node) )
    {
        result = Py_BuildValue("{sssl}",
                               "type", "boolean",
                               "node", feature_node);
        if(result == NULL)
        {
            BARF("Couldn't build %s() result", __func__);
            goto done;
        }
    }
    else if( ARV_IS_GC_COMMAND(feature_node) )
    {
        result = Py_BuildValue("{sssl}",
                               "type", "command",
                               "node", feature_node);
        if(result == NULL)
        {
            BARF("Couldn't build %s() result", __func__);
            goto done;
        }

    }
    else
    {
        BARF("unsupported feature type. I only support (integer,float,boolean,command,enumeration)");
        goto done;
    }

 done:
    g_free(available_enum_entries);
    return result;
}

static PyObject*
feature_value(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject* result       = NULL;
    PyObject* feature      = NULL;
    PyObject* value        = NULL;
    void*     feature_node = NULL;
    bool      is_node      = true;
    GError*   error        = NULL;

    mrcam_t* ctx = &self->ctx; // for the try_arv...() macros
    char* keywords[] = {"feature",
                        "value",
                        NULL};

    ArvDevice* device = arv_camera_get_device(ARV_CAMERA(self->ctx.camera));
    if(device == NULL)
    {
        BARF("Couldn't arv_camera_get_device()");
        goto done;
    }

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "O|O:mrcam.feature_value", keywords,
                                     &feature, &value))
        goto done;

    if(!feature_Check(feature,
                      (const char*[]){"integer",
                                      "float",
                                      "boolean",
                                      "command",
                                      "enumeration",
                                      NULL},
                      "'integer','float','boolean','command','enumeration'"))
        goto done;

    feature_node = feature_GetNodeOrAddress(feature, &is_node);
    if(feature_node == NULL) goto done;

    if(value == NULL)
    {
        // getter
        if(!is_node)
        {
            guint32 value;
            try_arv(arv_device_read_register(device, (guint64)feature_node, &value, &error));
            result = Py_BuildValue("(l{sO})",
                                   (long)value,
                                   "locked", Py_False);
            goto done;
        }


        bool is_locked;
        try_arv(is_locked = arv_gc_feature_node_is_locked(ARV_GC_FEATURE_NODE(feature_node), &error));

        // Check enumeration before integer. enumerations provide the integer
        // interface also, but I want the special enum one
        if( ARV_IS_GC_ENUMERATION(feature_node) )
        {
            const char* s;
            try_arv(s = arv_gc_enumeration_get_string_value(ARV_GC_ENUMERATION(feature_node),
                                                            &error));
            result = Py_BuildValue("(s{sO})",
                                   s,
                                   "locked", is_locked ? Py_True : Py_False);
        }
        else if( ARV_IS_GC_INTEGER(feature_node) )
        {
            gint64 value_here;
            try_arv(value_here = arv_gc_integer_get_value(ARV_GC_INTEGER(feature_node),
                                                          &error));
            result = Py_BuildValue("(l{sO})",
                                   (long)value_here,
                                   "locked", is_locked ? Py_True : Py_False);
        }
        else if( ARV_IS_GC_FLOAT(feature_node) )
        {
            double value_here;
            try_arv(value_here = arv_gc_float_get_value(ARV_GC_FLOAT(feature_node),
                                                        &error));
            result = Py_BuildValue("(d{sO})",
                                   value_here,
                                   "locked", is_locked ? Py_True : Py_False);
        }
        else if( ARV_IS_GC_BOOLEAN(feature_node) )
        {
            gboolean value_here;
            try_arv(value_here = arv_gc_boolean_get_value(ARV_GC_BOOLEAN(feature_node),
                                                          &error));
            result = Py_BuildValue("(O{sO})",
                                   value_here ? Py_True : Py_False,
                                   "locked", is_locked ? Py_True : Py_False);
        }
        else if( ARV_IS_GC_COMMAND(feature_node) )
        {
            result = Py_BuildValue("(O{sO})",
                                   Py_None,
                                   "locked", is_locked ? Py_True : Py_False);
        }
        else
        {
            BARF("BUG: the node must be one of (integer,float,boolean,command,enumeration)");
            goto done;
        }
    }
    else
    {
        // setter

        if(!is_node)
        {
            gint64 value_here;
            if(     PyLong_Check(value))  value_here = PyLong_AsLong(value);
            else if(PyFloat_Check(value)) value_here = (gint64)round(PyFloat_AsDouble(value));
            else
            {
                BARF("Given value not interpretable as an integer or a float");
                goto done;
            }
            if(PyErr_Occurred())
            {
                BARF("Given value not interpretable");
                goto done;
            }

            try_arv( arv_device_write_register(device, (guint64)feature_node, (guint32)value_here, &error) );

            Py_INCREF(Py_None);
            result = Py_None;
            goto done;
        }


        // Check enumeration before integer. enumerations provide the integer
        // interface also, but I want the special enum one
        if( ARV_IS_GC_ENUMERATION(feature_node) )
        {
            if(!PyUnicode_Check(value))
            {
                BARF("Expected string for the enumeration entry");
                goto done;
            }
            const char* s = PyUnicode_AsUTF8(value);
            try_arv(arv_gc_enumeration_set_string_value(ARV_GC_ENUMERATION(feature_node),
                                                        s,
                                                        &error));
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else if( ARV_IS_GC_INTEGER(feature_node) )
        {
            gint64 value_here;
            if(     PyLong_Check(value))  value_here = PyLong_AsLong(value);
            else if(PyFloat_Check(value)) value_here = (gint64)round(PyFloat_AsDouble(value));
            else
            {
                BARF("Given value not interpretable as an integer or a float");
                goto done;
            }
            if(PyErr_Occurred())
            {
                BARF("Given value not interpretable");
                goto done;
            }

            try_arv(arv_gc_integer_set_value(ARV_GC_INTEGER(feature_node),
                                             value_here,
                                             &error));
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else if( ARV_IS_GC_FLOAT(feature_node) )
        {
            double value_here;

            if(     PyFloat_Check(value)) value_here = PyFloat_AsDouble(value);
            else if(PyLong_Check(value))  value_here = (double)round(PyLong_AsLong(value));
            else
            {
                BARF("Given value not interpretable as an integer or a float");
                goto done;
            }
            if(PyErr_Occurred())
            {
                BARF("Given value not interpretable as a float");
                goto done;
            }

            try_arv(arv_gc_float_set_value(ARV_GC_FLOAT(feature_node),
                                           value_here,
                                           &error));
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else if( ARV_IS_GC_BOOLEAN(feature_node) )
        {
            gboolean value_here = PyObject_IsTrue(value);
            if(PyErr_Occurred())
            {
                BARF("Given value not interpretable as a boolean");
                goto done;
            }
            try_arv(arv_gc_boolean_set_value(ARV_GC_BOOLEAN(feature_node),
                                             value_here,
                                             &error));
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else if( ARV_IS_GC_COMMAND(feature_node) )
        {
            gboolean value_here = PyObject_IsTrue(value);
            if(PyErr_Occurred())
            {
                BARF("Given value not interpretable as a boolean");
                goto done;
            }
            if(value_here)
                try_arv(arv_gc_command_execute(ARV_GC_COMMAND(feature_node),
                                               &error));
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else
        {
            BARF("BUG: the node must be one of (integer,float,boolean,command,enumeration)");
            goto done;
        }
    }

 done:
    return result;
}

static
PyObject* equalize_fieldscale(PyObject* NPY_UNUSED(self),
                              PyObject* args,
                              PyObject* kwargs)
{
    PyObject*      result       = NULL;
    PyArrayObject* py_image     = NULL;
    PyArrayObject* py_image_out = NULL;

    char* keywords[] = { "image",
                         NULL};
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O:mrcam.equalize_fieldscale",
                                     keywords,
                                     &py_image))
        goto done;

    if(! (PyArray_Check(py_image) &&
          PyArray_TYPE(py_image) == NPY_UINT16 &&
          PyArray_NDIM(py_image) == 2 &&
          PyArray_STRIDES(py_image)[1] == (int)sizeof(uint16_t) &&
          PyArray_STRIDES(py_image)[0] == (int)sizeof(uint16_t)*PyArray_DIM(py_image,1)) )
    {
        PyErr_SetString(PyExc_RuntimeError, "'image' must be a densely-stored image with uint16 data");
        return false;
    }

    const int H = PyArray_DIM(py_image,0);
    const int W = PyArray_DIM(py_image,1);

    py_image_out =
        (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){H,W}), NPY_UINT8);
    if(py_image_out == NULL)
        goto done;

    if(!mrcam_equalize_fieldscale(// out
                 &(mrcal_image_uint8_t ){.width  = W,
                                         .height = H,
                                         .stride = PyArray_STRIDE(py_image_out,0),
                                         .data   = PyArray_DATA  (py_image_out) },
                 // in
                 &(mrcal_image_uint16_t){.width  = W,
                                         .height = H,
                                         .stride = PyArray_STRIDE(py_image,0),
                                         .data   = PyArray_DATA  (py_image) } ))
    {
        BARF("mrcam_equalize_fieldscale() failed");
        goto done;
    }

    result = (PyObject*)py_image_out;

 done:
    if(result == NULL)
        Py_XDECREF(py_image_out);

    return result;
}


static const char camera_docstring[] =
#include "camera.docstring.h"
    ;
static const char pull_docstring[] =
#include "pull.docstring.h"
    ;
static const char request_docstring[] =
#include "request.docstring.h"
    ;
static const char requested_image_docstring[] =
#include "requested_image.docstring.h"
    ;
static const char fd_image_ready_docstring[] =
#include "fd_image_ready.docstring.h"
    ;
static const char timestamp_request_us_docstring[] =
#include "timestamp_request_us.docstring.h"
    ;
static const char feature_descriptor_docstring[] =
#include "feature_descriptor.docstring.h"
    ;
static const char feature_value_docstring[] =
#include "feature_value.docstring.h"
    ;
static const char features_docstring[] =
#include "features.docstring.h"
    ;
static const char stream_stats_docstring[] =
#include "stream_stats.docstring.h"
    ;
static const char push_buffer_docstring[] =
#include "push_buffer.docstring.h"
    ;
static const char async_save_image_and_push_buffer_docstring[] =
#include "async_save_image_and_push_buffer.docstring.h"
    ;
static const char equalize_fieldscale_docstring[] =
#include "equalize_fieldscale.docstring.h"
    ;


static PyMethodDef camera_methods[] =
    {
        PYMETHODDEF_ENTRY(, pull,           METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, request,        METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, requested_image,METH_VARARGS | METH_KEYWORDS),

        PYMETHODDEF_ENTRY(, feature_descriptor, METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, feature_value,      METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, features,           METH_VARARGS | METH_KEYWORDS),

        PYMETHODDEF_ENTRY(, stream_stats,       METH_NOARGS),
        PYMETHODDEF_ENTRY(, push_buffer, METH_VARARGS | METH_KEYWORDS),

        PYMETHODDEF_ENTRY(, async_save_image_and_push_buffer, METH_VARARGS | METH_KEYWORDS),
        {}
    };

#define MRCAM_T_UINT64 Py_T_ULONG
static_assert(sizeof(uint64_t) == sizeof(unsigned long),
              "Here I'm assuming that uint64_t == unsigned long");
static PyMemberDef camera_members[] =
    {
        {"fd_image_ready",       Py_T_INT,       offsetof(camera,pipe_capture[PIPE_FD_READ]),                  READONLY, fd_image_ready_docstring},
        {"timestamp_request_us", MRCAM_T_UINT64, offsetof(camera,ctx.timestamp_request_us), READONLY, timestamp_request_us_docstring},
        {}
    };


static
PyObject* GenericNew_and_zero(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyObject* obj = PyType_GenericNew(type, args, kwds);

    // Reset all the non-PyObject stuff
    memset( (uint8_t*)obj + sizeof(PyObject),
            0,
            sizeof(camera) - sizeof(PyObject));

    return obj;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
// PyObject_HEAD_INIT throws
//   warning: missing braces around initializer []
// This isn't mine to fix, so I'm ignoring it
static PyTypeObject camera_type =
{
     PyObject_HEAD_INIT(NULL)
    .tp_name      = "mrcam.camera",
    .tp_basicsize = sizeof(camera),
    .tp_new       = GenericNew_and_zero,
    .tp_init      = (initproc)camera_init,
    .tp_dealloc   = (destructor)camera_dealloc,
    .tp_methods   = camera_methods,
    .tp_members   = camera_members,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = camera_docstring,
};
#pragma GCC diagnostic pop


static PyMethodDef functions[] =
    {
        PYMETHODDEF_ENTRY(, equalize_fieldscale, METH_VARARGS | METH_KEYWORDS),
        {}
    };

#define MODULE_DOCSTRING \
    "mrcam: an aravis frontend to genicam cameras\n"

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "mrcam",
     MODULE_DOCSTRING,
     -1,
     functions,
    };

PyMODINIT_FUNC PyInit_mrcam(void)
{
    if (PyType_Ready(&camera_type) < 0)
        return NULL;

    PyObject* module = PyModule_Create(&module_def);

    Py_INCREF(&camera_type);
    PyModule_AddObject(module, "camera", (PyObject *)&camera_type);

    import_array();

    return module;
}
