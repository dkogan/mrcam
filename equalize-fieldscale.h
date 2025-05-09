#pragma once

#include <mrcal/mrcal-image.h>
#include <stdbool.h>

bool equalize(// out
              mrcal_image_uint8_t*        image_out,
              // in
              const mrcal_image_uint16_t* image_in);
