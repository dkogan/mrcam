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
    _(const char*,    logdir,  logdir,  NULL,                required_argument, " DIR",         'l', "l:") \
    _(bool,           jpg,     jpg,     false,               no_argument,       ,               'j', "j" ) \
    _(double,         period,  period,  1.0,                 required_argument, " PERIOD_SEC",  'T', "T:") \
    /* The default pixel format is MONO_*. Should match the one in camera_init() in mrcam-pywrap.c */ \
    _(mrcam_pixfmt_t, pixfmt,  pixfmt,  MRCAM_PIXFMT_MONO_8, required_argument, " PIXELFORMAT", 'F', ""  ) \
    _(dimensions_t,   dims,    dims,    {},                  required_argument, " WIDTH,HEIGHT",'D', ""  ) \
    _(mrcam_trigger_t,trigger, trigger, MRCAM_TRIGGER_SOFTWARE,required_argument, " TRIGGER",     't', ""  ) \
    _(mrcam_acquisition_mode_t,acquisition_mode, acquisition-mode, MRCAM_ACQUISITION_MODE_SINGLE_FRAME,required_argument, " ACQUISITION-MODE",     'a', ""  ) \
    _(int,            time_decimation_factor,time-decimation-factor, 1, required_argument, " DECIMATION_FACTOR" ,'f', "") \
    _(bool,           acquisition_persistent,acquisition-persistent, false, no_argument, ,'p', "") \
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
        const char* usage =
            #include "mrcam-test.usage.h"
            ;

#define OPTIONS_HELP_MSG(type, name_var, name_opt, default, has_arg, placeholder, id, shortopt) \
    "  [--" #name_opt placeholder "]\n"

        FILE* fp = to_stderr ? stderr : stdout;
        fprintf(fp,
                "Usage: %s\n" LIST_OPTIONS(OPTIONS_HELP_MSG)
                "  [CAMERA0 [CAMERA1 ....]]\n\n"
                "%s",
                argv[0],
                usage);
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

        case 'l': options->logdir                          = optarg;       break;
        case 'j': options->jpg                             = true;         break;
        case 'T': options->period                          = atof(optarg); break;
        case 'v': options->verbose                         = true;         break;
        case 'p': options->acquisition_persistent          = true;         break;
        case 'F':

            if(0) ;

#define PARSE(name, name_genicam, ...)                          \
            else if(0 == strcasecmp(optarg, #name) ||               \
                    0 == strcasecmp(optarg, #name_genicam))         \
                options->pixfmt = MRCAM_PIXFMT_ ## name;

            LIST_MRCAM_PIXFMT(PARSE)
            else
            {
#define SAY(name, name_genicam, ...) "  " #name "\n  " #name_genicam "\n"
                MSG("Unknown pixel format '%s'; mrcam knows about:\n"
                    LIST_MRCAM_PIXFMT(SAY)
                    "These aren't all supported by each camera:\n"
                    "run 'arv-tool-0.8 features PixelFormat' to query the hardware",
                    optarg);
                exit(1);
#undef SAY
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
            else if(0 == strcasecmp(optarg, #name))                 \
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

        case 'a':
            if(0) ;
#define PARSE(name, ...)                                        \
            else if(0 == strcasecmp(optarg, #name))                 \
                options->acquisition_mode = MRCAM_ACQUISITION_MODE_ ## name;
            LIST_MRCAM_ACQUISITION_MODE(PARSE)
            else
            {
                MSG("Unknown acquisition_mode mode '%s'; I know about:", optarg);
#define SAY(name, ...) MSG("  " #name);
                LIST_MRCAM_ACQUISITION_MODE(SAY);
#undef SAY
                exit(1);
            }
#undef PARSE
            break;

        case 'f':
            options->time_decimation_factor = atoi(optarg);
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
                .acquisition_mode                = options.acquisition_mode,
                .acquisition_persistent          = options.acquisition_persistent,
                .time_decimation_factor          = options.time_decimation_factor,
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
            if(!mrcam_pull( &images.image_uint8 [icam],
                            &timestamps_us[icam],
                            0, &ctx[icam]))
                goto done;

        int64_t t1 = gettimeofday_int64();

        bool capturefailed = false;

        int icam;

#warning Disabled openmp to avoid gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=114509
// #pragma omp parallel for private(icam) num_threads(options.Ncameras)
        for(icam=0; icam<options.Ncameras; icam++)
        {
            char filename[1024] = "-";
            if(options.logdir != NULL)
            {
                const char* fmt = "%s/frame-%05d-cam%d.%s"; // in variable to not confuse MSG()
                if( snprintf(filename, sizeof(filename),
                             fmt,
                             options.logdir,
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
            }
            printf("%d %d %s %ld.%06ld %s\n",
                   iframe, icam,
                   options.camera_names[icam] == NULL ? "-" : options.camera_names[icam],
                   timestamps_us[icam] / 1000000,
                   timestamps_us[icam] % 1000000,
                   filename);
            fflush(stdout);
        }

        if(capturefailed)
            goto done;

        if(options.period > 0)
        {
            int64_t t2 = gettimeofday_int64();

            // I sleep the requested period, minus whatever time already elapsed
            int t_sleep = (int)(options.period*1e6 + 0.5) - (int)(t2-t0);
            if(t_sleep > 0)
                usleep(t_sleep);
        }
    }

    result = 0;

 done:
    for(int icam=0; icam<options.Ncameras; icam++)
        mrcam_free(&ctx[icam]);

    return result;
}
