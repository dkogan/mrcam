#!/usr/bin/python3

import sys
import re

def add_common_cmd_options(parser,
                           *,
                           Ncameras_expected = None):

    parser.add_argument('--verbose','-v',
                        action='store_true',
                        help='''If given, we turn on mrcam verbose reporting.
                        This is separate from the deeper diagnostics provided by
                        aravis using the ARV_DEBUG environment variable. See the
                        aravis documentation for those details''')
    parser.add_argument('--period','-T',
                        type=float,
                        default = 1.0,
                        help='''Image capture period, in seconds. Defaults to
                        1.0sec/frame. This is used to set the polling
                        frame-to-frame delay. If set to <= 0, we capture frames
                        as quickly as possible; with no frame-to-frame delay''')
    parser.add_argument('--features',
                        help='''A comma-separated list of features for which GUI
                        controls should be displayed. The available features can
                        be queried with "arv-tool-0.8". Each feature can be
                        specified as a regex to pick multiple features at once.
                        If the regex matches any category, everything in that
                        category will be selected''')
    parser.add_argument('--display-flip',
                        help='''Flip the image horizontally and/or vertically
                        for display. This changes the way the image is displayed
                        ONLY: the captured image data is unchanged. The argument
                        is a string 'x' or 'y' or 'xy' to be applied to ALL the
                        cameras OR a comma-separated list of these strings, one
                        per camera. If a list is given, its length MUST match
                        the number of cameras. Pass an empty string to flip
                        nothing. For instance, with 3 cameras, flipping the
                        second one, pass "--display-flip ,xy,"''')
    parser.add_argument('--dims',
                        action = 'append', # accept multiple instances of this option
                        help='''Imager dimensions given as WIDTH,HEIGHT.
                        Required for cameras where this cannot be auto-detected.
                        If given once, this is applied to ALL the cameras. If
                        given multiple times, the different dims are applied to
                        the cameras, in order. Pass '-' to mean "auto-detect
                        dimensions. If fewer --dims are given than cameras, the
                        remaining cameras will be auto-detected"''')
    parser.add_argument('--Nbuffers',
                        type    = int,
                        default = 10,
                        help='''How many buffers to allocate to store received images. As the camera sends
                        images, the driver writes them into a buffer in memory.
                        When the client program is ready to process the images,
                        it reads the buffer, and gives it back for the driver to
                        use for future images. If a camera is sending data very
                        quickly and the client cannot process them immediately,
                        we need more buffers to store the data until the client
                        can catch up.''')
    parser.add_argument('--pixfmt',
                        # should match the LIST_OPTIONS macro in mrcam-test.c
                        # and camera_init() in mrcam-pywrap.c
                        default = "MONO_8",
                        help='''The pixel format. If omitted, we use "MONO_8". Pass any invalid format (like
                        "") to get a list of valid values on stderr.''')
    parser.add_argument('--acquisition-mode',
                        # should match the LIST_OPTIONS macro in mrcam-test.c
                        # and camera_init() in mrcam-pywrap.c
                        default = "SINGLE_FRAME",
                        help='''The acquisition mode. If omitted, we use
                        "SINGLE_FRAME". Pass any invalid mode (like "") to get a
                        list of valid values on stderr.''')
    parser.add_argument('--trigger',
                        # should match the LIST_OPTIONS macro in mrcam-test.c
                        # and camera_init() in mrcam-pywrap.c
                        default = "SOFTWARE",
                        help='''The trigger mode. If omitted, we use "SOFTWARE". Pass any invalid mode (like
                        "") to get a list of valid values on stderr.''')
    parser.add_argument('--time-decimation-factor',
                        type    = int,
                        default = 1,
                        help='''The decimation factor. Report only every Nth
                        frame. By default this is 1: report every frame''')

    if Ncameras_expected != 1:
        parser.add_argument('--unlock-panzoom',
                            action='store_true',
                            help='''If given, the pan/zoom in the all the image
                            widgets are NOT locked together. By default they ARE
                            locked together''')

    camera_expect_description = ''
    if Ncameras_expected is not None:
        camera_expect_description = f' We expect exactly {Ncameras_expected} cameras'
    parser.add_argument('camera',
                        type = str,
                        nargs = '*',
                        default = (None,),
                        help=f'''Without --replay: the camera(s) to talk to.
                        One argument per camera. These are strings passed to
                        the arv_camera_new() function; see that function's
                        documentation for details. The strings can be IP
                        addresses or MAC addresses or
                        vendor-model-serialnumber strings. With --replay:
                        integers and/or A-B ranges of integers specifying
                        the camera indices in the log. If omitted, we use a
                        single device: the first camera.{camera_expect_description}''')

    # log/replay stuff
    parser.add_argument('--logdir',
                        help='''The directory to write the images and metadata.
                        If omitted, we do NOT log anything to disk. Exclusive
                        with --replay. When logging, simultaneous replaying is
                        ALWAYS enabled. If the requested directory does not
                        exist, we create it''')
    parser.add_argument('--jpg',
                        action='store_true',
                        help='''If given, we write the output images as .jpg
                        files, using lossy compression. If omitted, we write out
                        lossless .png files (bigger, much slower to compress,
                        decompress). Some pixel formats (deep ones, in
                        particular) do not work with.jpg''')

    parser.add_argument('--replay',
                        help='''If given, we replay the stored images
                        instead of talking to camera hardware. The log
                        directory must be given as an argument to --replay.
                        Exclusive with --logdir.''')
    parser.add_argument('--replay-from-frame',
                        type=int,
                        default=0,
                        help='''If given, we start the replay at the given
                        frame, instead of at the start of the log''')
    parser.add_argument('--image-path-prefix',
                        help='''Used with --replay. If given, we prepend the
                        given prefix to the image paths in the log. Exclusive
                        with --image-directory''')
    parser.add_argument('--image-directory',
                        help='''Used with --replay. If given, we extract the
                        filenames from the image paths in the log, and use the
                        given directory to find those filenames. Exclusive with
                        --image-path-prefix''')
    parser.add_argument('--utcoffset-hours',
                        type=float,
                        help='''Used to determine the mapping between the image
                        UNIX timestamps (in UTC) and local time. If omitted, we
                        default to local time''')


def parse_args_postprocess(args,
                           *,
                           Ncameras_expected = None):

    if args.features is not None:
        args.features = [ f for f in args.features.split(',') if len(f) ] # filter out empty features
    else:
        args.features = ()

    ### args.camera_params_noname_nodims is a subset of args.__dict__. That
    ### subset is camera parameters passed directly to mrcam.camera()
    # These must match keywords[] in camera_init() in mrcam-pywrap.c WITHOUT the
    # "name" field
    camera_param_noname_nodims_keys = \
        ("pixfmt",
         "acquisition_mode",
         "trigger",
         "time_decimation_factor",
         "Nbuffers",
         "verbose")
    args.camera_params_noname_nodims = dict()
    for k in camera_param_noname_nodims_keys:
        args.camera_params_noname_nodims[k] = getattr(args, k, None)

    if args.image_path_prefix is not None and \
       args.image_directory   is not None:
        print("--image-path-prefix and --image-directory are mutually exclusive",
              file=sys.stderr)
        sys.exit(1)


    if args.replay is not None and \
       args.logdir is not None:
        print("--replay and --logdir are mutually exclusive",
              file=sys.stderr)
        sys.exit(1)


    #### normalize args.camera into an iterable of camera IDs Ncameras_expected
    #### long. This will always be an interable of at least length 1. We handle
    #### replay differently because we accept ranges (like "0-2") where each
    #### token repreesnts multiple cameras
    if args.replay is not None:
        def camera_for_replay(s):
            if s is None:
                return [0]

            try:
                i = int(s)
                if i < 0:
                    print(f"--replay given, so the cameras must be a list of non-negative integers and/or A-B ranges. Invalid camera given: '{s}'",
                          file=sys.stderr)
                    sys.exit(1)
                return [i]
            except Exception as e:
                pass

            m = re.match("^([0-9]+)-([0-9]+)$", s)
            if m is None:
                print(f"--replay given, so the cameras must be a list of non-negative integers and/or A-B ranges. Invalid camera given: '{s}'",
                      file=sys.stderr)
                sys.exit(1)
            try:
                i0 = int(m.group(1))
                i1 = int(m.group(2))
            except Exception as e:
                print(f"--replay given, so the cameras must be a list of non-negative integers and/or A-B ranges. Invalid camera given: '{s}'",
                      file=sys.stderr)
                sys.exit(1)
            return list(range(i0,i1+1))


        args.camera = [c for cam in args.camera for c in camera_for_replay(cam)]

    if Ncameras_expected is not None and \
       len(args.camera) != Ncameras_expected:
        print(f"Expected {Ncameras_expected} cameras to be given, but got {len(args.camera)}",
              file=sys.stderr)
        sys.exit(1)

    if args.features and \
       args.replay   is not None:
        print("--replay and --features are mutually exclusive",
              file=sys.stderr)
        sys.exit(1)


    if args.dims is not None:

        def massage_dim(dim):
            if dim == '-':
                return None

            errmsg = f"--dims MUST be followed by '-' or integer dimensions given by 'WIDTH,HEIGHT'. Couldn't parse '{dim}' that way"
            m = re.match('([0-9]+),([0-9]+)$', dim)
            if m is None:
                print(errmsg, file = sys.stderr)
                sys.exit(1)

            try: w = int(m.group(1))
            except Exception:
                print(errmsg, file = sys.stderr)
                sys.exit(1)
            try: h = int(m.group(2))
            except Exception:
                print(errmsg, file = sys.stderr)
                sys.exit(1)
            if w <= 0 or h <= 0:
                print(errmsg, file = sys.stderr)
                sys.exit(1)

            return (w,h)

        Ncameras   = len(args.camera)
        Ndims_args = len(args.dims)

        # If given once, this is applied to ALL the cameras
        if Ndims_args == 1:
            args.dims *= Ncameras
            Ndims_args = Ncameras

        args.dims = [massage_dim(dim) for dim in args.dims]
        if Ndims_args <= Ncameras:
            args.dims += [None] * (Ncameras - Ndims_args)
        else:
            print("Received more --dims than cameras", file=sys.stderr)
            sys.exit(1)

    # The various machinery requires SOME value to exist, so I write one
    if Ncameras_expected == 1:
        args.unlock_panzoom = False

    Ncameras = len(args.camera)



    if args.display_flip is None:
        args.display_flip = ('') * Ncameras
    else:
        if ',' in args.display_flip:
            args.display_flip = args.display_flip.split(',')
            if len(args.display_flip) != Ncameras:
                print(f"--display-flip given a comma-separated list: MUST match {Ncameras=}", file=sys.stderr)
                sys.exit(1)
        else:
            args.display_flip = (args.display_flip,) * Ncameras
    def make_display_flip_one(display_flip):
        s = set(display_flip)
        return \
            ('x' in s,
             'y' in s)
    flip_xy = [make_display_flip_one(d) for d in args.display_flip]
    args.flip_x_allcams = [fxy[0] for fxy in flip_xy]
    args.flip_y_allcams = [fxy[1] for fxy in flip_xy]
    # flip_x_allcams and flip_y_allcams are boolean iterables of length args.camera
