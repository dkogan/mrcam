include choose_mrbuild.mk
include $(MRBUILD_MK)/Makefile.common.header

PROJECT_NAME := mrcam
ABI_VERSION  := 0
TAIL_VERSION := 0

VERSION := 0.0

LIB_SOURCES := \
  mrcam.c \
  clahe.cc \
  equalize-fieldscale.c

BIN_SOURCES := \
  mrcam-test.c

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-parameter

# to avoid benign warnings about casting the various mrcam_callback_image_TYPE_t
# to each other. These don't matter
CCXXFLAGS += -Wno-cast-function-type

# For the equalization
clahe.o: CCXXFLAGS += -I/usr/include/opencv4
LDLIBS             += -lopencv_imgproc -lopencv_core


# stolen docstring logic from Makefile.common.footer
%.usage.h: %.usage
	< $^ sed 's/\\/\\\\/g; s/"/\\"/g; s/^/"/; s/$$/\\n"/;' > $@
EXTRA_CLEAN += mrcam-test.usage.h

mrcam-test.o: mrcam-test.usage.h
mrcam-test.o: CPPFLAGS += -fopenmp
mrcam-test:   LDFLAGS  += -fopenmp
mrcam-test:   LDLIBS += -lmrcal

# use the arch-specific pkg-config if there is one
CFLAGS += $(shell $(or $(PKG_CONFIG),pkg-config) --cflags aravis-0.8)
LDLIBS += $(shell $(or $(PKG_CONFIG),pkg-config) --libs   aravis-0.8) -lavutil -lswscale




DIST_INCLUDE := \
  mrcam.h

DIST_BIN := \
	mrcam-equalize
DIST_MAN := $(addsuffix .1,$(DIST_BIN))
$(DIST_MAN): %.1: %.pod
	pod2man --center="mrcam: machine-vision camera interface" --name=MRCAM --release="mrcam $(VERSION)" --section=1 $< $@
%.pod: %
	$(MRBUILD_BIN)/make-pod-from-help $< > $@.tmp && cat footer.pod >> $@.tmp && mv $@.tmp $@
EXTRA_CLEAN += $(DIST_MAN) $(patsubst %.1,%.pod,$(DIST_MAN))


######### python stuff
mrcam-pywrap.o: $(addsuffix .h,$(wildcard *.docstring))
mrcam$(PY_EXT_SUFFIX): mrcam-pywrap.o libmrcam.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $(LDFLAGS) $(PY_MRBUILD_LDFLAGS) $< -lmrcam -lmrcal -o $@

PYTHON_OBJECTS := mrcam-pywrap.o

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
$(PYTHON_OBJECTS): CFLAGS += -Wno-cast-function-type
$(PYTHON_OBJECTS): CFLAGS += $(PY_MRBUILD_CFLAGS)

DIST_PY3_MODULES := mrcam$(PY_EXT_SUFFIX) mrcam_fltk.py

all: mrcam$(PY_EXT_SUFFIX)


include $(MRBUILD_MK)/Makefile.common.footer
