#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>
#include <string.h>
#include <omp.h>

#include "mrcam.h"
#include "util.h"


typedef struct { int width,height; } dimensions_t;
#define LIST_OPTIONS(_)                                                 \
    _(int,            Nframes, 1,                   required_argument, " N",           'N', "N:") \
    _(const char*,    outdir,  ".",                 required_argument, " DIR",         'o', "o:") \
    _(bool,           jpg,     false,               no_argument,       ,               'j', "j" ) \
    _(double,         period,  1.0,                 required_argument, " PERIOD_SEC",  'T', "T:") \
    _(mrcam_pixfmt_t, pixfmt,  MRCAM_PIXFMT_MONO_8, required_argument, " PIXELFORMAT", 'F', ""  ) \
    _(dimensions_t,   dims,    {},                  required_argument, " WIDTH,HEIGHT",'D', ""  ) \
    _(bool,           verbose, false,               no_argument,       ,               'v', "v")




#define OPTIONS_DECLARE(type, name, default, has_arg, placeholder, id, shortopt) \
    type name;
typedef struct
{
    int Ncameras;
    char** camera_names;
    LIST_OPTIONS(OPTIONS_DECLARE)
} options_t;






static int64_t gettimeofday_int64()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t) tv.tv_sec * 1000000LL + (int64_t) tv.tv_usec;
}


static bool parse_args(// out
                       options_t* options,
                       // in
                       int argc, char** argv)
{
    void sayusage(bool to_stderr)
    {

        // const char* usage =
        //     #include "mrcam-test.usage.h"
        //     ;

#define OPTIONS_HELP_MSG(type, name, default, has_arg, placeholder, id, shortopt) \
    "  [--" #name placeholder "]\n"

        FILE* fp = to_stderr ? stderr : stdout;
        fprintf(fp,
                "%s\n" LIST_OPTIONS(OPTIONS_HELP_MSG),
                argv[0]);
    }



#define OPTIONS_LONG_DEF(type, name, default, has_arg, placeholder, id, shortopt) \
    { #name, has_arg, NULL, id },
    struct option option_definition[] = {
        LIST_OPTIONS(OPTIONS_LONG_DEF)
        { "help",                       no_argument,       NULL, 'h' },
        {}
    };


#define OPTIONS_SHORT_DEF(type, name, default, has_arg, placeholder, id, shortopt) \
    shortopt
    const char* optstring = LIST_OPTIONS(OPTIONS_SHORT_DEF)
        // and --help
        "h";


#define OPTIONS_DECLARE_DEFAULT(type, name, default, has_arg, placeholder, id, shortopt) \
    , .name = default


    *options = (options_t){ .Ncameras = 0
                            LIST_OPTIONS(OPTIONS_DECLARE_DEFAULT) };

    int opt;
    do
    {
        opt = getopt_long(argc, argv, optstring, option_definition, NULL);
        switch(opt)
        {
        case -1:
            break;

        case 'h':
            sayusage(false);
            exit(0);

        case 'N':
            options->Nframes = atoi(optarg);
            if(options->Nframes <= 0)
            {
                MSG("I want Nframes > 0\n");
                sayusage(true);
                exit(1);
            }
            break;

        case 'o': options->outdir  = optarg;       break;
        case 'j': options->jpg     = true;         break;
        case 'T': options->period  = atof(optarg); break;
        case 'v': options->verbose = true;         break;

        case 'F':

            if(0) ;

#define MRCAM_PIXFMT_PARSE(name, ...)                           \
            else if(0 == strcmp(optarg, #name))                 \
                options->pixfmt = MRCAM_PIXFMT_ ## name;

            LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_PARSE)
            else
            {
                MSG("Unknown pixel format '%s'; I know about:", optarg);

#define MRCAM_PIXFMT_SAY(name, ...) MSG("  " #name);
                LIST_MRCAM_PIXFMT(MRCAM_PIXFMT_SAY);
#undef MRCAM_PIXFMT_SAY

                exit(1);
            }
#undef MRCAM_PIXFMT_PARSE
            break;

        case 'D':
            {
                int Nbytes_consumed;
                if( 2 != sscanf(optarg,
                                "%d,%d%n",
                                &options->dims.width,
                                &options->dims.height,
                                &Nbytes_consumed) ||
                    optarg[Nbytes_consumed] != '\0' )
                {
                    MSG("--dims MUST be followed by integer dimensions given by 'WIDTH,HEIGHT'. Couldn't parse '%s' that way",
                        optarg);
                    exit(1);
                }
                break;
            }

        case '?':
            MSG("Unknown option");
            sayusage(true);
            exit(1);
        }
    } while( opt != -1 );

    const int Noptions_remaining = argc - optind;
    if(Noptions_remaining > 0)
    {
        options->Ncameras     = Noptions_remaining;
        options->camera_names = &argv[optind];
    }
    else
    {
        // No cameras were requested; I use the first available one
        options->Ncameras = 1;
        static char* one_null[] = {NULL};
        options->camera_names = one_null;
    }

    return true;
}


int main(int argc, char **argv)
{
    int result = 1;

    options_t options;
    // function calls exit() if something went wrong; so no error checking
    parse_args(&options,
               argc, argv);

    if(options.verbose)
        mrcam_set_verbose();

    omp_set_num_threads(options.Ncameras);

    mrcam_t ctx[options.Ncameras];
    for(int icam=0; icam<options.Ncameras; icam++)
    {
        if(!mrcam_init(&ctx[icam],
                       options.camera_names[icam],
                       options.pixfmt,
                       options.dims.width,
                       options.dims.height))
            return 1;
    }


    union
    {
        mrcal_image_uint8_t  image_uint8 [options.Ncameras];
        mrcal_image_uint16_t image_uint16[options.Ncameras];
        mrcal_image_bgr_t    image_bgr   [options.Ncameras];
    } images;

    printf("# iframe icam cameraname t_system imagepath\n");

    for(int iframe=0; iframe<options.Nframes; iframe++)
    {
        int64_t t0 = gettimeofday_int64();

        for(int icam=0; icam<options.Ncameras; icam++)
        {
            if(ctx[icam].bytes_per_pixel == 1 && !ctx[icam].is_color)
            {
                if(!mrcam_get_frame_uint8 (&images.image_uint8 [icam], 0, &ctx[icam]))
                    goto done;
            }
            else if(ctx[icam].bytes_per_pixel == 2 && !ctx[icam].is_color)
            {
                if(!mrcam_get_frame_uint16(&images.image_uint16[icam], 0, &ctx[icam]))
                    goto done;
            }
            else if(ctx[icam].is_color)
            {
                if(!mrcam_get_frame_bgr(&images.image_bgr[icam], 0, &ctx[icam]))
                    goto done;
            }
            else
            {
                MSG("Unknown bytes_per_pixel=%d", ctx[icam].bytes_per_pixel);
                goto done;
            }
        }

        int64_t t1 = gettimeofday_int64();

        bool capturefailed = false;

        int icam;

#warning Disabled openmp to avoid gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=114509
// #pragma omp parallel for private(icam) num_threads(options.Ncameras)
        for(icam=0; icam<options.Ncameras; icam++)
        {
            const char* fmt = "%s/frame-%05d-cam%d.%s"; // in variable to not confuse MSG()
            char filename[1024];
            if( snprintf(filename, sizeof(filename),
                          fmt,
                          options.outdir,
                          iframe,
                          icam,
                          options.jpg ? "jpg" : "png") >= (int)sizeof(filename) )
            {
                MSG("Static buffer overflow. Increase sizeof(filename)");
                __atomic_store(&capturefailed, &(bool){true}, __ATOMIC_RELAXED);
                continue;

            }

            if(ctx[icam].bytes_per_pixel == 1 && !ctx[icam].is_color)
            {
                if(!mrcal_image_uint8_save( filename, &images.image_uint8 [icam]))
                {
                    MSG("Couldn't save to '%s'", filename);
                    __atomic_store(&capturefailed, &(bool){true}, __ATOMIC_RELAXED);
                    continue;
                }
            }
            else if(ctx[icam].bytes_per_pixel == 2 && !ctx[icam].is_color)
            {
                if(!mrcal_image_uint16_save(filename, &images.image_uint16[icam]))
                {
                    MSG("Couldn't save to '%s'", filename);
                    __atomic_store(&capturefailed, &(bool){true}, __ATOMIC_RELAXED);
                    continue;
                }
            }
            else if(ctx[icam].is_color)
            {
                if(!mrcal_image_bgr_save(filename, &images.image_bgr[icam]))
                {
                    MSG("Couldn't save to '%s'", filename);
                    __atomic_store(&capturefailed, &(bool){true}, __ATOMIC_RELAXED);
                    continue;
                }
            }
            else
            {
                MSG("Unknown bytes_per_pixel=%d", ctx[icam].bytes_per_pixel);
                __atomic_store(&capturefailed, &(bool){true}, __ATOMIC_RELAXED);
                continue;
            }

            printf("%d %d %s %ld.%06ld %s\n",
                   iframe, icam, options.camera_names[icam],
                   t1 / 1000000,
                   t1 % 1000000,
                   filename);
            fflush(stdout);
        }

        if(capturefailed)
            goto done;

        int64_t t2 = gettimeofday_int64();

        // I sleep the requested period, minus whatever time already elapsed
        int t_sleep = (int)(options.period*1e6 + 0.5) - (int)(t2-t0);
        if(t_sleep > 0)
            usleep(t_sleep);

    }

    result = 0;

 done:
    for(int icam=0; icam<options.Ncameras; icam++)
        mrcam_free(&ctx[icam]);

    return result;
}
