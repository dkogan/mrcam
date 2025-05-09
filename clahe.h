#pragma once

#include <stdint.h>

void clahe( uint8_t* buffer, // input AND output
            int width, int height,
            double clipLimit);
