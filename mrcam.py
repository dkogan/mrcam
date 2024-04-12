#!/usr/bin/python3

import sys
from _mrcam import *

def _add_common_cmd_options(parser,
                            *,
                            single_camera):

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
                        1.0sec/frame''')
    parser.add_argument('--single-buffered',
                        action='store_true',
                        help='''By default the image display is double-buffered
                        to avoid flickering. Some graphics hardare (in
                        particular my old i915-based system) are buggy, and
                        don't work right in this mode, so --single-buffered is
                        available to disable double-buffering to work around
                        those bugs''')
    parser.add_argument('--dims',
                        help='''Imager dimensions given as WIDTH,HEIGHT. Required for cameras where this
                        cannot be auto-detected''')
    parser.add_argument('--pixfmt',
                        # should match the LIST_OPTIONS macro in mrcam-test.c
                        # and camera_init() in mrcam-pywrap.c
                        default = "MONO_8",
                        help='''The pixel format. If omitted, we use "MONO_8". Pass any invalid format (like
                        "") to get a list of valid values on stderr.''')

    if single_camera:
        parser.add_argument('camera',
                            type = str,
                            nargs = '?',
                            default = None,
                            help='''The camera to talk to. This is a string passed
                            to the arv_camera_new() function; see that function's
                            documentation for details. The string can be IP
                            addresses or MAC addresses or vendor-model-serialnumber
                            strings. If omitted, we take the first available camera''')
    else:
        parser.add_argument('camera',
                            type = str,
                            nargs = '*',
                            default = (None,),
                            help='''The camera(s) to talk to. One argument per
                            camera. These are strings passed to the arv_camera_new()
                            function; see that function's documentation for details.
                            The strings can be IP addresses or MAC addresses or
                            vendor-model-serialnumber strings. If omitted, we
                            initialize a single device: the first available camera''')



def _parse_args_postprocess(args):
    if args.dims is not None:
        errmsg = f"--dims MUST be followed by integer dimensions given by 'WIDTH,HEIGHT'. Couldn't parse '{args.dims}' that way"
        m = re.match('([0-9]+),([0-9]+)$', args.dims)
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

        args.dims = (w,h)


def _set_up_image_capture(camera, window, image_widget, period,
                          *,
                          process_image_callback = None):

    from fltk import Fl


    def callback_image_ready(fd):
        frame = camera.requested_image()

        image        = frame['image']
        timestamp_us = frame['timestamp_us']

        if image is not None:
            # Update the image preview; deep images are shown as a heat map
            if image.itemsize > 1:
                if image.ndim > 2:
                    raise Exception("high-depth color images not supported yet")
                image_widget.update_image(image_data = mrcal.apply_color_map(image))
            else:
                image_widget.update_image(image_data = image)


            if process_image_callback is not None:
                process_image_callback(image)
        else:
            print("Error capturing the image. I will try again",
                  file=sys.stderr)

        Fl.add_timeout(period, update)

    def update(*args):

        # try block needed to avoid potential crashes:
        #   https://sourceforge.net/p/pyfltk/mailman/pyfltk-user/thread/875xx5ncgp.fsf%40secretsauce.net/#msg58754407
        try:
            camera.request()
        except Exception as e:
            window.hide()
            raise e




    Fl.add_fd( camera.fd_image_ready,
               callback_image_ready )

    update()
