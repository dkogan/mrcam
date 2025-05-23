#pragma once

/*
An equalization routine meant to boost the local contrast in 16-bit images.
Originally based on:

  Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared
  Image Hyeonjae Gil, Myeon-Hwan Jeon, and Ayoung Kim

  @article{gil2024fieldscale,
    title={Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared Image},
    author={Gil, Hyeonjae and Jeon, Myung-Hwan and Kim, Ayoung},
    journal={IEEE Robotics and Automation Letters},
    year={2024},
    publisher={IEEE}
  }

  Original author: Hyeonjae Gil
  Author email: h.gil@snu.ac.kr
*/

#include <mrcal/mrcal-image.h>
#include <stdbool.h>

bool equalize(// out
              mrcal_image_uint8_t*        image_out,
              // in
              const mrcal_image_uint16_t* image_in);
