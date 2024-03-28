include /usr/include/mrbuild/Makefile.common.header

PROJECT_NAME := mrcam
ABI_VERSION  := 0
TAIL_VERSION := 0

VERSION := 0.0

LIB_SOURCES := \
  mrcam.c

BIN_SOURCES := \
  mrcam-test.c

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-parameter

mrcam-test.o: CPPFLAGS += -fopenmp
mrcam-test:   LDFLAGS  += -fopenmp
mrcam-test:   LDLIBS += -lmrcal

CFLAGS += $(shell pkg-config --cflags aravis-0.8)
LDLIBS += $(shell pkg-config --libs   aravis-0.8)



######### python stuff
mrcam-pywrap.o: $(addsuffix .h,$(wildcard *.docstring))
mrcam$(PY_EXT_SUFFIX): mrcam-pywrap.o libmrcam.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $(LDFLAGS) $(PY_MRBUILD_LDFLAGS) $< -lmrcam -o $@

PYTHON_OBJECTS := mrcam-pywrap.o

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
$(PYTHON_OBJECTS): CFLAGS += -Wno-cast-function-type
$(PYTHON_OBJECTS): CFLAGS += $(PY_MRBUILD_CFLAGS)

DIST_PY3_MODULES := mrcam

all: mrcam$(PY_EXT_SUFFIX)


include /usr/include/mrbuild/Makefile.common.footer
