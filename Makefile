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


include /usr/include/mrbuild/Makefile.common.footer
