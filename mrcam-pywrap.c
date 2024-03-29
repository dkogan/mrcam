#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <assert.h>

#include "mrcam.h"

#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}

#define BARF(fmt, ...) PyErr_Format(PyExc_RuntimeError, "%s:%d %s(): "fmt, __FILE__, __LINE__, __func__, ## __VA_ARGS__)


typedef struct {
    PyObject_HEAD

    mrcam_t camera;

} camera;


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
    int width  = 0; // by default, auto-detect the dimensions
    int height = 0;
    int verbose               = 0;

    mrcam_pixfmt_t pixfmt;

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|ssiip", keywords,
                                     &camera_name,
                                     &pixfmt_string,
                                     &width, &height,
                                     &verbose))
        goto done;

    if(0) ;

#define MRCAM_PIXFMT_PARSE(name, bytes_per_pixel)       \
    else if(0 == strcmp(pixfmt_string, #name))          \
        pixfmt = MRCAM_PIXFMT_ ## name;

    LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_PARSE)
    else
    {
#define MRCAM_PIXFMT_SAY(name, bytes_per_pixel) "'" #name "', "
        BARF("Unknown pixel format '%s'; I know about: ("
             LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_SAY)
             ")");
        goto done;
#undef MRCAM_PIXFMT_SAY
    }
#undef MRCAM_PIXFMT_PARSE

    if(!mrcam_init(&self->camera,
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
    mrcam_free(&self->camera);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
camera_get_frame(camera* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject* result = NULL;
    PyObject* image  = NULL;

    char* keywords[] = {"timeout_us",
                        NULL};

    double timeout_sec = 0.0;

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|d", keywords,
                                     &timeout_sec))
        goto done;


    if(self->camera.bytes_per_pixel == 1)
    {
        mrcal_image_uint8_t mrcal_image;
        if(!mrcam_get_frame_uint8( &mrcal_image,
                                   (uint64_t)(timeout_sec * 1e6),
                                   &self->camera))
            goto done;
        if(mrcal_image.stride !=
           mrcal_image.width * self->camera.bytes_per_pixel)
        {
            BARF("Image returned by mrcam_get_frame_...() is not contiguous");
            goto done;
        }
        image =
            PyArray_SimpleNewFromData(2,
                                      ((npy_intp[]){mrcal_image.height,
                                                    mrcal_image.width}),
                                      NPY_UINT8,
                                      mrcal_image.data);
    }
    else if(self->camera.bytes_per_pixel == 2)
    {
        mrcal_image_uint16_t mrcal_image;
        if(!mrcam_get_frame_uint16( &mrcal_image,
                                   (uint64_t)(timeout_sec * 1e6),
                                   &self->camera))
            goto done;
        if(mrcal_image.stride !=
           mrcal_image.width * self->camera.bytes_per_pixel)
        {
            BARF("Image returned by mrcam_get_frame_...() is not contiguous");
            goto done;
        }
        image =
            PyArray_SimpleNewFromData(2,
                                      ((npy_intp[]){mrcal_image.height,
                                                    mrcal_image.width}),
                                      NPY_UINT16,
                                      mrcal_image.data);
    }
    else
    {
        BARF("Supported values of bytes_per_pixel are (1,2), but got %d. Giving up", self->camera.bytes_per_pixel);
        goto done;
    }

    result = image;
    Py_INCREF(image);

    // result =
    //     Py_BuildValue("(KO)",
    //                   timestamp_us, image);

 done:
    Py_XDECREF(image);

    return result;
}
static const char camera_docstring[] =
#include "camera.docstring.h"
    ;
static const char camera_get_frame_docstring[] =
#include "camera_get_frame.docstring.h"
    ;

static PyMethodDef camera_methods[] =
    {
        PYMETHODDEF_ENTRY(camera_, get_frame, METH_VARARGS | METH_KEYWORDS),
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
