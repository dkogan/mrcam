#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>

#include "equalize.h"

#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        name ## _docstring}

#define BARF(fmt, ...) PyErr_Format(PyExc_RuntimeError, "%s:%d %s(): "fmt, __FILE__, __LINE__, __func__, ## __VA_ARGS__)

// the try...() macros in util.h will produce Python errors
#define ERR(fmt, ...) BARF(fmt, ##__VA_ARGS__)



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

static
PyObject* py_equalize(PyObject* NPY_UNUSED(self),
                      PyObject* args,
                      PyObject* kwargs)
{
    PyObject*      result       = NULL;
    PyArrayObject* py_image     = NULL;
    PyArrayObject* py_image_out = NULL;

    char* keywords[] = { "image",
                         NULL};
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O:equalize.equalize",
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

    if(!equalize(// out
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
        BARF("equalize() failed");
        goto done;
    }

    result = (PyObject*)py_image_out;

 done:
    if(result == NULL)
        Py_XDECREF(py_image_out);

    return result;
}

static const char equalize_docstring[] =
#include "equalize.docstring.h"
    ;

static PyMethodDef methods[] =
    {
        PYMETHODDEF_ENTRY(py_, equalize, METH_VARARGS | METH_KEYWORDS),
        {}
    };

#define MODULE_DOCSTRING \
    "equalize: fancy equalization for deep images\n"

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "equalize",
     MODULE_DOCSTRING,
     -1,
     methods,
    };

PyMODINIT_FUNC PyInit_equalize(void)
{
    PyObject* module = PyModule_Create(&module_def);

    import_array();

    return module;
}
