#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <poll.h>
#include <assert.h>

#include "mrcam.h"
#include "util.h"

#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}

#define BARF(fmt, ...) PyErr_Format(PyExc_RuntimeError, "%s:%d %s(): "fmt, __FILE__, __LINE__, __func__, ## __VA_ARGS__)


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
                        "width",
                        "height",
                        "verbose",
                        NULL};

    // The default pixel format is "MONO_8". Should match the one in the
    // LIST_OPTIONS macro in mrcam-test.c
    const char* camera_name   = NULL;
    const char* pixfmt_string = "MONO_8";
    int width   = 0; // by default, auto-detect the dimensions
    int height  = 0;
    int verbose = 0;

    mrcam_pixfmt_t pixfmt;

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|ssiip", keywords,
                                     &camera_name,
                                     &pixfmt_string,
                                     &width, &height,
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
             ")");
        goto done;
#undef SAY
    }
#undef PARSE


    if(!mrcam_init(&self->ctx,
                   camera_name,
                   pixfmt,
                   width, height))
    {
        BARF("Couldn't init mrcam camera");
        goto done;
    }

    if(verbose)
        mrcam_set_verbose();

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

#warning add timestamping
    const uint64_t timestamp_us = 0;


    char* keywords[] = {"timeout_us",
                        NULL};

    double timeout_sec = 0.0;

    if(currently_processing_image(self))
    {
        BARF("An image is already being processed...");
        goto done;
    }


    SET_SIGINT();

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|d", keywords,
                                     &timeout_sec))
        goto done;

    // generic type
    mrcal_image_uint8_t mrcal_image;

    switch(mrcam_output_type(self->ctx.pixfmt))
    {
    case MRCAM_uint8:
        {
            if(!mrcam_pull_uint8( (mrcal_image_uint8_t*)&mrcal_image,
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

    char* keywords[] = {"block",
                        NULL};

    int block = 0;

    SET_SIGINT();

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|p", keywords,
                                     &block))
        goto done;

    if(!block && !fd_has_data(self->fd_read))
    {
        BARF("Non-blocking mode requestd, but no data is available to be read");
        goto done;
    }

    PyObject* image  = NULL;

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


static PyMethodDef camera_methods[] =
    {
        PYMETHODDEF_ENTRY(, pull,           METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, request,        METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(, requested_image,METH_VARARGS | METH_KEYWORDS),
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
