#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <poll.h>
#include <assert.h>

#include <arv.h>

#include "mrcam.h"

#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}

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



typedef struct {
    PyObject_HEAD

    mrcam_t ctx;

    // Returned from pipe(). Used by the asynchronous frame grabbing using the
    // request() function. When a frame is available, a image_ready_t structure
    // is sent over this pipe. The Python main thread can then read the pipe to
    // process the frame
    union
    {
        int pipefd[2];
        struct
        {
            int fd_read, fd_write;
        };
    };

} camera;

// The structure being sent over the pipe
typedef struct
{
    mrcal_image_uint8_t mrcal_image; // type might not be exact
    uint64_t            timestamp_us;
} image_ready_t;


static int
camera_init(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    int result = -1;

    char* keywords[] = {"name",
                        "pixfmt",
                        "trigger",
                        "width",
                        "height",
                        "recreate_stream_with_each_frame",
                        "verbose",
                        NULL};

    // The default pixel format is "MONO_8". Should match the one in the
    // LIST_OPTIONS macro in mrcam-test.c
    const char* camera_name    = NULL;
    const char* pixfmt_string  = "MONO_8";
    const char* trigger_string = "SOFTWARE";
    int width   = 0; // by default, auto-detect the dimensions
    int height  = 0;
    int recreate_stream_with_each_frame = 0;
    int verbose = 0;

    mrcam_pixfmt_t  pixfmt;
    mrcam_trigger_t trigger;

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|z$ssiipp:mrcam.__init__", keywords,
                                     &camera_name,
                                     &pixfmt_string,
                                     &trigger_string,
                                     &width, &height,
                                     &recreate_stream_with_each_frame,
                                     &verbose))
        goto done;

    if(0 != pipe(self->pipefd))
    {
        BARF("Couldn't init pipe");
        goto done;
    }



    if(0) ;
#define PARSE(name, ...)                        \
    else if(0 == strcmp(pixfmt_string, #name))  \
        pixfmt = MRCAM_PIXFMT_ ## name;
    LIST_MRCAM_PIXFMT(PARSE)
    else
    {
#define SAY(name, ...) "'" #name "', "
        BARF("Unknown pixel format '%s'; I know about: ("
             LIST_MRCAM_PIXFMT(SAY)
             ")",
             pixfmt_string);
        goto done;
#undef SAY
    }
#undef PARSE

    if(0) ;
#define PARSE(name, ...)                        \
    else if(0 == strcmp(trigger_string, #name))  \
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


    const mrcam_options_t mrcam_options =
        {
            .pixfmt                          = pixfmt,
            .trigger                         = trigger,
            .width                           = width,
            .height                          = height,
            .recreate_stream_with_each_frame = recreate_stream_with_each_frame,
            .verbose                         = verbose
        };
    if(!mrcam_init(&self->ctx,
                   camera_name,
                   &mrcam_options))
    {
        BARF("Couldn't init mrcam camera");
        goto done;
    }

    result = 0;

 done:
    return result;
}

static void camera_dealloc(camera* self)
{
    mrcam_free(&self->ctx);

    close(self->fd_read);
    close(self->fd_write);

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
numpy_image_from_mrcal_image(// type not exact
                             const mrcal_image_uint8_t* mrcal_image,
                             mrcam_output_type_t type)
{
    switch(type)
    {
    case MRCAM_uint8:
        {
            const int bytes_per_pixel__output = 1;
            if(mrcal_image->stride != mrcal_image->width * bytes_per_pixel__output)
            {
                BARF("Image returned by mrcam_pull_...() is not contiguous");
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
                BARF("Image returned by mrcam_pull_...() is not contiguous");
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
                BARF("Image returned by mrcam_pull_...() is not contiguous");
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
        fd_has_data(self->fd_read);
}

// Synchronous frame processing. May block. No pipe
static PyObject*
pull(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject* result = NULL;
    PyObject* image  = NULL;

    char* keywords[] = {"timeout",
                        NULL};

    double timeout_sec = 0.0;

    if(currently_processing_image(self))
    {
        BARF("An image is already being processed...");
        goto done;
    }


    SET_SIGINT();

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|$d:mrcam.pull", keywords,
                                     &timeout_sec))
        goto done;

    // generic type
    mrcal_image_uint8_t mrcal_image;
    uint64_t timestamp_us;

    switch(mrcam_output_type(self->ctx.pixfmt))
    {
    case MRCAM_uint8:
        {
            if(!mrcam_pull_uint8( (mrcal_image_uint8_t*)&mrcal_image,
                                  &timestamp_us,
                                  (uint64_t)(timeout_sec * 1e6),
                                  &self->ctx))
            {
                BARF("mrcam_pull...() failed");
                goto done;
            }
        }
        break;

    case MRCAM_uint16:
        {
            if(!mrcam_pull_uint16( (mrcal_image_uint16_t*)&mrcal_image,
                                   &timestamp_us,
                                   (uint64_t)(timeout_sec * 1e6),
                                   &self->ctx))
            {
                BARF("mrcam_pull...() failed");
                goto done;
            }
        }
        break;

    case MRCAM_bgr:
        {
            if(!mrcam_pull_bgr( (mrcal_image_bgr_t*)&mrcal_image,
                                &timestamp_us,
                                (uint64_t)(timeout_sec * 1e6),
                                &self->ctx))
            {
                BARF("mrcam_pull...() failed");
                goto done;
            }
        }
        break;

    default:
        goto done;
    }

    image = numpy_image_from_mrcal_image(&mrcal_image, mrcam_output_type(self->ctx.pixfmt));
    if(image == NULL)
        // BARF() already called
        goto done;

    result = Py_BuildValue("{sOsk}",
                           "image",        image,
                           "timestamp_us", timestamp_us);
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
callback_generic(mrcal_image_uint8_t mrcal_image, // type might not be exact
                 uint64_t timestamp_us,
                 void* cookie)
{
    camera* self = (camera*)cookie;

    image_ready_t s = {.mrcal_image  = mrcal_image,
                       .timestamp_us = timestamp_us};

    if(sizeof(s) != write_persistent(self->fd_write, (uint8_t*)&s, sizeof(s)))
    {
        MSG("Couldn't write image metadata to pipe!");
        return;
    }
}

static PyObject*
request(camera* self, PyObject* args)
{
    if(currently_processing_image(self))
    {
        BARF("An image is already being processed...");
        goto done;
    }

    switch(mrcam_output_type(self->ctx.pixfmt))
    {
    case MRCAM_uint8:
        if(!mrcam_request_uint8( (mrcam_callback_image_uint8_t* )&callback_generic,
                                 self,
                                 &self->ctx))
        {
            BARF("mrcam_request...() failed");
            goto done;
        }
        break;

    case MRCAM_uint16:
        if(!mrcam_request_uint16((mrcam_callback_image_uint16_t*)&callback_generic,
                                 self,
                                 &self->ctx))
        {
            BARF("mrcam_request...() failed");
            goto done;
        }
        break;

    case MRCAM_bgr:
        if(!mrcam_request_bgr(   (mrcam_callback_image_bgr_t*   )&callback_generic,
                                 self,
                                 &self->ctx))
        {
            BARF("mrcam_request...() failed");
            goto done;
        }
        break;

    default:
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

    if(!block && !fd_has_data(self->fd_read))
    {
        BARF("Non-blocking mode requested, but no data is available to be read");
        goto done;
    }

    image_ready_t s;
    if(sizeof(s) != read_persistent(self->fd_read, (uint8_t*)&s, sizeof(s)))
    {
        BARF("Couldn't read image metadata from pipe!");
        goto done;
    }

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

    result = Py_BuildValue("{sOsk}",
                           "image",        image,
                           "timestamp_us", s.timestamp_us);
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
          arv_gc_feature_node_is_implemented (ARV_GC_FEATURE_NODE (node), NULL)))
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
feature_set(camera* self, PyObject* args, PyObject* kwargs)
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
                                     "|s:mrcam.feature_set", keywords,
                                     &regex_string))
        goto done;

    set = PySet_New(NULL);
    if(set == NULL)
    {
        BARF("Couldn't create output set");
        goto done;
    }

    genicam = arv_device_get_genicam (device);
    regex   = g_regex_new(regex_string, 0,0, &error);
    if(error != NULL || regex == NULL)
    {
        BARF("Couldn't compile regex from string '%s'", regex_string);
        g_clear_error(&error);
        goto done;
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

static void* feature_GetNode(PyObject* feature)
{
    PyObject* node = GetItemString(feature, "node");
    if(node == NULL) return NULL;

    if(!PyLong_Check(node))
    {
        BARF("The 'node' must be a long");
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
        BARF("The 'node' couldn't be interpreted numerically");
        return NULL;
    }
    if(u.node == NULL)
    {
        BARF("The 'node' was NULL");
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
    GError*   error        = NULL;

    mrcam_t* ctx = &self->ctx; // for the try_arv...() macros

    char* keywords[] = {"feature",
                        "value",
                        NULL};

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

    feature_node = feature_GetNode(feature);
    if(feature_node == NULL) goto done;

    if(value == NULL)
    {
        // getter

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
static const char feature_descriptor_docstring[] =
#include "feature_descriptor.docstring.h"
    ;
static const char feature_value_docstring[] =
#include "feature_value.docstring.h"
    ;
static const char feature_set_docstring[] =
#include "feature_set.docstring.h"
    ;


static PyMethodDef camera_methods[] =
    {
        PYMETHODDEF_ENTRY(, pull,           METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, request,        METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, requested_image,METH_VARARGS | METH_KEYWORDS),

        PYMETHODDEF_ENTRY(, feature_descriptor, METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, feature_value,      METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, feature_set,        METH_VARARGS | METH_KEYWORDS),
        {}
    };

static PyMemberDef camera_members[] =
    {
        {"fd_image_ready", T_INT, offsetof(camera,pipefd[0]), READONLY, fd_image_ready_docstring},
        {}
    };


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
    .tp_new       = PyType_GenericNew,
    .tp_init      = (initproc)camera_init,
    .tp_dealloc   = (destructor)camera_dealloc,
    .tp_methods   = camera_methods,
    .tp_members   = camera_members,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = camera_docstring,
};
#pragma GCC diagnostic pop



#define MODULE_DOCSTRING \
    "Python-wrapper around the mrcam aravis wrapper library\n"

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "mrcam",
     MODULE_DOCSTRING,
     -1,
     NULL,
    };

PyMODINIT_FUNC PyInit__mrcam(void)
{
    if (PyType_Ready(&camera_type) < 0)
        return NULL;

    PyObject* module = PyModule_Create(&module_def);

    Py_INCREF(&camera_type);
    PyModule_AddObject(module, "camera", (PyObject *)&camera_type);

    import_array();

    return module;
}
