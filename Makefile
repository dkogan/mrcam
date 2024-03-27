include /usr/include/mrbuild/Makefile.common.header

PROJECT_NAME := mrcam
ABI_VERSION  := 0
TAIL_VERSION := 0

VERSION := 0.0


CCXXFLAGS += -fno-omit-frame-pointer
CCXXFLAGS += -Wno-unused-parameter -Wno-unused-function

LIB_SOURCES := \
  mrcam.c

BIN_SOURCES := \
  mrcam-test.c


CFLAGS := $(shell pkg-config --cflags aravis-0.8)
LDLIBS := $(shell pkg-config --libs   aravis-0.8)

mrcam-test: LDLIBS += -lmrcal

include /usr/include/mrbuild/Makefile.common.footer
