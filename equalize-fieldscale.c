/*
An equalization routine meant to boost the local contrast in 16-bit images.
Based on:

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


#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <mrcal/mrcal-image.h>

#include "clahe.h"
#include "util.h"


#define WRITE_INTERMEDIATE 0


/////// settings
static const int gridH = 8;
static const int gridW = 8;

// for local_extrema_suppression()
static const uint16_t max_diff       = 600;
static const uint16_t min_diff       = 600;
static const int      local_distance = 2;

// for message_passing()
static const int Niterations = 7;

static const double _gamma = 1.5;

static
bool gridwise_minmax(// out
                     // dense shape (gridH,gridW)
                     uint16_t* grid_min,
                     uint16_t* grid_max,
                     // in
                     const uint16_t* image_in,
                     const int W,
                     const int H)
{
    const int patchH = H / gridH;
    const int patchW = W / gridW;
    if(!(gridH*patchH == H &&
         gridW*patchW == W))
    {
        MSG("image must be subdivideable into and (%d,%d) grid of patches exactly",
            gridH, gridW);
        return false;
    }

    // Init min/max accumulators to the worst-possible values
    for(int i=0; i<gridH*gridW; i++) grid_min[i] = UINT16_MAX;
    for(int i=0; i<gridH*gridW; i++) grid_max[i] = 0;

    // I loop through the big image once to maximize cache locality
    int i=0;
    for(int grid_i=0; grid_i<gridH; grid_i++)
        for(int patch_i=0; patch_i<patchH; patch_i++, i++)
        {
            int j=0;
            for(int grid_j=0; grid_j<gridW; grid_j++)
            {
                uint16_t* grid_min_here = &grid_min[grid_i*gridW + grid_j];
                uint16_t* grid_max_here = &grid_max[grid_i*gridW + grid_j];

                for(int patch_j=0; patch_j<patchW; patch_j++, j++)
                {
                    uint16_t x = image_in[i*W + j];
                    if(*grid_min_here > x) *grid_min_here = x;
                    if(*grid_max_here < x) *grid_max_here = x;
                }
            }
        }

    return true;
}

static
void integral_image(// out
                    // dense shape (gridH,gridW)
                    uint32_t* integral,
                    // in
                    // dense shape (gridH,gridW)
                    const uint16_t* grid)
{
    // first row
    int i=0;

    int j=0;
    integral[i*gridW + j] = grid[i*gridW + j];
    for(int j=1; j<gridW; j++)
        integral[i*gridW + j] = integral[i*gridW + j-1] + grid[i*gridW + j];

    // subsequent rows
    for(int i=1; i<gridH; i++)
    {
        // first col
        int j=0;
        integral[i*gridW + j] =
            grid[i*gridW + j] +
            integral[(i-1)*gridW + j];

        // subsequent cols
        for(int j=1; j<gridW; j++)
            integral[i*gridW + j] =
                integral[i*gridW + j-1] +
                grid[i*gridW + j] +
                integral[(i-1)*gridW + j] -
                integral[(i-1)*gridW + j-1];
    }
}

static
void sum_from_integral(// out
                       int32_t*  sum,
                       int32_t*  N,
                       // in
                       const int i,
                       const int j,
                       const uint32_t* integral,
                       const int radius)
{
    // This function is trivial, except there's some non-obvious logic at the
    // edges
    //
    // The integral works like this:
    //   x        = [2 5  7  8  3  1  5]
    //   integral = [2 7 14 22 25 26 31]
    // At i==3 (x==8) I want sum(i+-2) to be 5+7+8+3+1 = 24
    // This is integral[i+2] - integral[i-2-1]
    //
    // Left edge. At i==1 (x==5) I want sum(i+-2) to be 2+5+7+8 = 22
    // This is integral[i+2] - 0
    //
    // Right edge. At i==5 (x==1) I want sum(i+-2) to be 8+3+1+5 = 17
    // This is integral[imax=6] - integral[i-2-1]

    int i0 = i-radius-1;
    if(i0 < 0) i0 = -1;
    int i1 = i+radius;
    if(i1 > gridH-1) i1 = gridH-1;

    int j0 = j-radius-1;
    if(j0 < 0) j0 = -1;
    int j1 = j+radius;
    if(j1 > gridW-1) j1 = gridW-1;

    const uint32_t integral11 = integral[i1*gridW + j1];
    const uint32_t integral01 =
        (i0 == -1) ? 0 : integral[i0*gridW + j1];
    const uint32_t integral10 =
        (j0 == -1) ? 0 : integral[i1*gridW + j0];
    const uint32_t integral00 =
        (i0 == -1 || j0 == -1) ? 0 : integral[i0*gridW + j0];

    *sum = integral11 - integral01 - integral10 + integral00;
    *N   = (i1-i0)*(j1-j0);
}

static
void local_extrema_suppression( // in,out
                                // dense shape (gridH,gridW)
                                uint16_t* grid)
{
    /* pseudo-code:

       for each pixel in grid
         Look at a neighborhood of pixels within diff_threshold of that pixel
           If we're over diff_threshold above the mean in the neighborhood,
           clamp us to diff_threshold within the mean
     */


    uint32_t integral[gridH*gridW];
    integral_image(integral, grid);

    for(int i=0; i<gridH; i++)
        for(int j=0; j<gridW; j++)
        {
            int32_t sum;
            int32_t N;
            sum_from_integral(&sum, &N,
                              i,j,
                              integral,
                              local_distance);

            const int ij = i*gridW + j;

            // subtract self
            sum -= grid[ij];
            N--;


            const int16_t Ndiff = N*(int32_t)grid[ij] - (int32_t)sum;
            if(      Ndiff > N*max_diff) grid[ij] = sum/N + max_diff;
            else if(-Ndiff > N*min_diff) grid[ij] = sum/N - min_diff;
        }
}

static
void message_passing( // in,out
                      // dense shape (gridH,gridW)
                      uint16_t*  grid,
                      const bool is_min)
{
    for(int iiter=0; iiter<Niterations; iiter++)
    {
        uint32_t integral[gridH*gridW];
        integral_image(integral, grid);


        for(int i=0; i<gridH; i++)
            for(int j=0; j<gridW; j++)
            {
                int32_t sum;
                int32_t N;
                sum_from_integral(&sum, &N,
                                  i,j,
                                  integral,
                                  1);

                const int ij = i*gridW + j;

                if(is_min)
                {
                    if(sum < N*(int32_t)grid[ij])
                        grid[ij] = sum/N;
                }
                else
                {
                    if(sum > N*(int32_t)grid[ij])
                        grid[ij] = sum/N;
                }
            }
    }
}

static
void interpolate(// out
                 uint16_t* grid_min_interpolated,
                 uint16_t* grid_max_interpolated,

                 // in
                 // dense shape (gridH,gridW)
                 const uint16_t* grid_min,
                 const uint16_t* grid_max,
                 const int i,
                 const int j,
                 const int patchW,
                 const int patchH)
{
    int16_t grid_i0 = (       i - patchH/2                 ) / patchH;
    float   qi      = (float)(i - patchH/2 - grid_i0*patchH) / (float)patchH;

    int16_t grid_j0 = (       j - patchW/2                 ) / patchW;
    float   qj      = (float)(j - patchW/2 - grid_j0*patchW) / (float)patchW;

    int16_t grid_i1 = grid_i0+1;
    int16_t grid_j1 = grid_j0+1;

    if(i - patchH/2 < 0)
    {
        grid_i0 = 0;
        grid_i1 = 0;
        qi = 0.f;
    }
    else if(grid_i1 >= gridH)
    {
        grid_i0 = gridH-1;
        grid_i1 = gridH-1;
        qi = 0.f;
    }
    if(j - patchW/2 < 0)
    {
        grid_j0 = 0;
        grid_j1 = 0;
        qj = 0.f;
    }
    else if(grid_j1 >= gridW)
    {
        grid_j0 = gridW-1;
        grid_j1 = gridW-1;
        qj = 0.f;
    }

    *grid_min_interpolated =
        (uint16_t)( 0.5f +

                    ( (float)grid_min[grid_i0*gridW + grid_j0] * (1.f-qj) +
                      (float)grid_min[grid_i0*gridW + grid_j1] * qj )
                    * (1.f-qi) +

                    ( (float)grid_min[grid_i1*gridW + grid_j0] * (1.f-qj) +
                      (float)grid_min[grid_i1*gridW + grid_j1] * qj )
                    * qi );
    *grid_max_interpolated =
        (uint16_t)( 0.5f +

                    ( (float)grid_max[grid_i0*gridW + grid_j0] * (1.f-qj) +
                      (float)grid_max[grid_i0*gridW + grid_j1] * qj )
                    * (1.f-qi) +

                    ( (float)grid_max[grid_i1*gridW + grid_j0] * (1.f-qj) +
                      (float)grid_max[grid_i1*gridW + grid_j1] * qj )
                    * qi );
}

static
void apply_minmax(// out
                  // dense shape (H,W)
                  uint8_t* out,
                  // in
                  // dense shape (H,W)
                  const uint16_t* in,
                  const int W, const int H,
                  // dense shape (gridH,gridW)
                  uint16_t*  grid_min,
                  uint16_t*  grid_max)
{
    const int patchH = H / gridH;
    const int patchW = W / gridW;


    for(int i=0; i<H; i++)
        for(int j=0; j<W; j++)
        {
            uint8_t*        out_here = &out[i*W + j];
            const uint16_t* in_here  = &in [i*W + j];

            uint16_t grid_min_interpolated;
            uint16_t grid_max_interpolated;
            interpolate(&grid_min_interpolated, &grid_max_interpolated,
                        grid_min, grid_max,
                        i,j,
                        patchW,patchH);


            if(*in_here <= grid_min_interpolated ||
               grid_min_interpolated >= grid_max_interpolated)
                *out_here = 0;
            else
            {
                uint16_t x =
                    (uint32_t)(*in_here - grid_min_interpolated) * (uint32_t)255 / (uint32_t)(grid_max_interpolated - grid_min_interpolated);
                if(x >= 255)
                    *out_here = 255;
                else
                    *out_here = (uint8_t)x;
            }
        }
}


bool mrcam_equalize_fieldscale(// out
                               mrcal_image_uint8_t*        image_out,
                               // in
                               const mrcal_image_uint16_t* image_in)
{
    const int W = image_in->width;
    const int H = image_in->height;
    if( !(W == image_out->width &&
          H == image_out->height) )
    {
        MSG("ERROR: image_out and image_in must have the same dimensions");
        return false;
    }

    if( image_in->stride != (int)sizeof(uint16_t)*W )
    {
        MSG("ERROR: image_in must be stored densely");
        return false;
    }
    if( image_out->stride != (int)sizeof(uint8_t)*W )
    {
        MSG("ERROR: image_out must be stored densely");
        return false;
    }


    uint16_t min_grid[gridH*gridW];
    uint16_t max_grid[gridH*gridW];

    if(!gridwise_minmax(min_grid, max_grid,
                        image_in->data, W,H))
        return false;

#if defined WRITE_INTERMEDIATE && WRITE_INTERMEDIATE
    FILE*fp;
    fp = fopen("/tmp/minmax0.dat","wb");
    fwrite(min_grid, sizeof(min_grid[0]), gridH*gridW, fp);
    fwrite(max_grid, sizeof(max_grid[0]), gridH*gridW, fp);
    fclose(fp);
#endif

    // apply max and min suppression ONLY to the max grid
    local_extrema_suppression(max_grid);

#if defined WRITE_INTERMEDIATE && WRITE_INTERMEDIATE
    fp = fopen("/tmp/minmax1.dat","wb");
    fwrite(min_grid, sizeof(min_grid[0]), gridH*gridW, fp);
    fwrite(max_grid, sizeof(max_grid[0]), gridH*gridW, fp);
    fclose(fp);
#endif

    message_passing(max_grid, false);
    message_passing(min_grid, true);

#if defined WRITE_INTERMEDIATE && WRITE_INTERMEDIATE
    fp = fopen("/tmp/minmax2.dat","wb");
    fwrite(min_grid, sizeof(min_grid[0]), gridH*gridW, fp);
    fwrite(max_grid, sizeof(max_grid[0]), gridH*gridW, fp);
    fclose(fp);
#endif

    apply_minmax(image_out->data,
                 image_in->data,
                 W,H,
                 min_grid, max_grid);



    for(int i=0; i<W*H; i++)
    {
        const double x =
            pow((double)(image_out->data[i]) / 255.,
                _gamma) * 255.;
        if(     x <= 0)   image_out->data[i] = 0;
        else if(x >= 255) image_out->data[i] = 255;
        else              image_out->data[i] = (uint8_t)(x + 0.5);
    }

    const double clipLimit = 2.0;
    clahe( image_out->data, W,H,
           clipLimit );

    return true;
}
