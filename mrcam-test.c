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
    _(int,            Nframes, Nframes, 1,                   required_argument, " N",           'N', "N:") \
    _(const char*,    outdir,  outdir,  ".",                 required_argument, " DIR",         'o', "o:") \
    _(bool,           jpg,     jpg,     false,               no_argument,       ,               'j', "j" ) \
    _(double,         period,  period,  1.0,                 required_argument, " PERIOD_SEC",  'T', "T:") \
    /* The default pixel format is MONO_*. Should match the one in camera_init() in mrcam-pywrap.c */ \
    _(mrcam_pixfmt_t, pixfmt,  pixfmt,  MRCAM_PIXFMT_MONO_8, required_argument, " PIXELFORMAT", 'F', ""  ) \
    _(dimensions_t,   dims,    dims,    {},                  required_argument, " WIDTH,HEIGHT",'D', ""  ) \
    _(mrcam_trigger_t,trigger, trigger, MRCAM_TRIGGER_SOFTWARE,required_argument, " TRIGGER",     't', ""  ) \
    _(bool,           recreate_stream_with_each_frame,recreate-stream-with-each-frame, false, no_argument, ,'R', "") \
    _(bool,           verbose, verbose, false,               no_argument,       ,               'v', "v")




#define OPTIONS_DECLARE(type, name_var, name_opt, default, has_arg, placeholder, id, shortopt) \
    type name_var;
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
#warning usage documentation
        // const char* usage =
        //     #include "mrcam-test.usage.h"
        //     ;

#define OPTIONS_HELP_MSG(type, name_var, name_opt, default, has_arg, placeholder, id, shortopt) \
    "  [--" #name_opt placeholder "]\n"

        FILE* fp = to_stderr ? stderr : stdout;
        fprintf(fp,
                "%s\n" LIST_OPTIONS(OPTIONS_HELP_MSG),
                argv[0]);
    }



#define OPTIONS_LONG_DEF(type, name_var, name_opt, default, has_arg, placeholder, id, shortopt) \
    { #name_opt, has_arg, NULL, id },
    struct option option_definition[] = {
        LIST_OPTIONS(OPTIONS_LONG_DEF)
        { "help",                       no_argument,       NULL, 'h' },
        {}
    };


#define OPTIONS_SHORT_DEF(type, name_var, name_opt, default, has_arg, placeholder, id, shortopt) \
    shortopt
    const char* optstring = LIST_OPTIONS(OPTIONS_SHORT_DEF)
        // and --help
        "h";


#define OPTIONS_DECLARE_DEFAULT(type, name_var, name_opt, default, has_arg, placeholder, id, shortopt) \
    , .name_var = default


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

        case 'o': options->outdir                          = optarg;       break;
        case 'j': options->jpg                             = true;         break;
        case 'T': options->period                          = atof(optarg); break;
        case 'v': options->verbose                         = true;         break;
        case 'R': options->recreate_stream_with_each_frame = true;         break;
        case 'F':

            if(0) ;

#define PARSE(name, ...)                                        \
            else if(0 == strcmp(optarg, #name))                 \
                options->pixfmt = MRCAM_PIXFMT_ ## name;

            LIST_MRCAM_PIXFMT(PARSE)
            else
            {
                MSG("Unknown pixel format '%s'; I know about:", optarg);

#define SAY(name, ...) MSG("  " #name);
                LIST_MRCAM_PIXFMT(SAY);
#undef SAY

                exit(1);
            }
#undef PARSE
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

        case 't':

            if(0) ;

#define PARSE(name, ...)                                        \
            else if(0 == strcmp(optarg, #name))                 \
                options->trigger = MRCAM_TRIGGER_ ## name;

            LIST_MRCAM_TRIGGER(PARSE)
            else
            {
                MSG("Unknown trigger mode '%s'; I know about:", optarg);

#define SAY(name, ...) MSG("  " #name);
                LIST_MRCAM_TRIGGER(SAY);
#undef SAY

                exit(1);
            }
#undef PARSE
            break;

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

    omp_set_num_threads(options.Ncameras);

    mrcam_t ctx[options.Ncameras];
    for(int icam=0; icam<options.Ncameras; icam++)
    {
        const mrcam_options_t mrcam_options =
            {
                .pixfmt                          = options.pixfmt,
                .width                           = options.dims.width,
                .height                          = options.dims.height,
                .trigger                         = options.trigger,
                .recreate_stream_with_each_frame = options.recreate_stream_with_each_frame,
                .verbose                         = options.verbose
            };
        if(!mrcam_init(&ctx[icam],
                       options.camera_names[icam],
                       &mrcam_options))
            return 1;
    }


    union
    {
        mrcal_image_uint8_t  image_uint8 [options.Ncameras];
        mrcal_image_uint16_t image_uint16[options.Ncameras];
        mrcal_image_bgr_t    image_bgr   [options.Ncameras];
    } images;

    uint64_t timestamps_us[options.Ncameras];

    printf("# iframe icam cameraname t_system imagepath\n");

    for(int iframe=0; iframe<options.Nframes; iframe++)
    {
        int64_t t0 = gettimeofday_int64();

        for(int icam=0; icam<options.Ncameras; icam++)
        {
            switch(mrcam_output_type(ctx[icam].pixfmt))
            {
            case MRCAM_uint8:
                if(!mrcam_pull_uint8( &images.image_uint8 [icam],
                                      &timestamps_us[icam],
                                      0, &ctx[icam]))
                    goto done;
                break;

            case MRCAM_uint16:
                if(!mrcam_pull_uint16(&images.image_uint16[icam],
                                      &timestamps_us[icam],
                                      0, &ctx[icam]))
                    goto done;
                break;

            case MRCAM_bgr:
                if(!mrcam_pull_bgr(   &images.image_bgr   [icam],
                                      &timestamps_us[icam],
                                      0, &ctx[icam]))
                    goto done;
                break;

            default:
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

            bool err = false;
            switch(mrcam_output_type(ctx[icam].pixfmt))
            {
            case MRCAM_uint8:
                if(!mrcal_image_uint8_save(  filename, &images.image_uint8 [icam]))
                {
                    MSG("Couldn't save to '%s'", filename);
                    __atomic_store(&capturefailed, &(bool){true}, __ATOMIC_RELAXED);
                    err = true;
                }
                break;

            case MRCAM_uint16:
                if(!mrcal_image_uint16_save( filename, &images.image_uint16[icam]))
                {
                    MSG("Couldn't save to '%s'", filename);
                    __atomic_store(&capturefailed, &(bool){true}, __ATOMIC_RELAXED);
                    err = true;
                }
                break;

            case MRCAM_bgr:
                if(!mrcal_image_bgr_save(   filename, &images.image_bgr   [icam]))
                {
                    MSG("Couldn't save to '%s'", filename);
                    __atomic_store(&capturefailed, &(bool){true}, __ATOMIC_RELAXED);
                    err = true;
                }
                break;

            default:
                err = true;
                break;
            }
            if(err) continue;


            printf("%d %d %s %ld.%06ld %s\n",
                   iframe, icam, options.camera_names[icam],
                   timestamps_us[icam] / 1000000,
                   timestamps_us[icam] % 1000000,
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
