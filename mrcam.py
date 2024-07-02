#!/usr/bin/python3

import sys
import re
from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget
from fltk import *
import mrcal
import time
import numpy as np
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
    parser.add_argument('--features',
                        help='''A comma-separated list of features for which GUI
                        controls should be displayed. The available features can
                        be queried with the "arv-tool-0.8" tool''')
    parser.add_argument('--single-buffered',
                        action='store_true',
                        help='''By default the image display is double-buffered
                        to avoid flickering. Some graphics hardare (in
                        particular my old i915-based system) are buggy, and
                        don't work right in this mode, so --single-buffered is
                        available to disable double-buffering to work around
                        those bugs''')
    parser.add_argument('--recreate-stream-with-each-frame',
                        action='store_true',
                        help='''If given, we create a new stream with each frame. This shouldn't be needed,
                        but the Emergent HR-20000 cameras don't work without
                        it''')


    parser.add_argument('--display-flip',
                        help='''Flip the image horizontally and/or vertically
                        for display. This changes the way the image is displayed
                        ONLY: the captured image data is unchanged. The argument
                        is a comma-separated string of "x" and/or "y"''')
    parser.add_argument('--dims',
                        help='''Imager dimensions given as WIDTH,HEIGHT. Required for cameras where this
                        cannot be auto-detected''')
    parser.add_argument('--pixfmt',
                        # should match the LIST_OPTIONS macro in mrcam-test.c
                        # and camera_init() in mrcam-pywrap.c
                        default = "MONO_8",
                        help='''The pixel format. If omitted, we use "MONO_8". Pass any invalid format (like
                        "") to get a list of valid values on stderr.''')
    parser.add_argument('--trigger',
                        # should match the LIST_OPTIONS macro in mrcam-test.c
                        # and camera_init() in mrcam-pywrap.c
                        default = "SOFTWARE",
                        help='''The trigger mode. If omitted, we use "SOFTWARE". Pass any invalid mode (like
                        "") to get a list of valid values on stderr.''')
    parser.add_argument('--logdir',
                        help='''The directory to write the images and metadata
                        (if no --replay) or to read them (if --replay). If
                        omitted, we do NOT log anything to disk. If --replay,
                        --logdir is required''')
    parser.add_argument('--jpg',
                        action='store_true',
                        help='''If given, we write the output images as .jpg
                        files, using lossy compression. If omitted, we write out
                        lossless .png files (bigger, much slower to compress,
                        decompress). Some pixel formats (deep ones, in
                        particular) do not work with.jpg''')

    parser.add_argument('--replay',
                        action='store_true',
                        help='''If given, we replay the stored images in
                        --logdir instead of talking to camera hardware''')
    parser.add_argument('--image-path-prefix',
                        help='''Used with --replay. If given, we prepend the
                        given prefix to the image paths in the log. Exclusive
                        with --image-directory''')
    parser.add_argument('--image-directory',
                        help='''Used with --replay. If given, we extract the
                        filenames from the image paths in the log, and use the
                        given directory to find those filenames. Exclusive with
                        --image-path-prefix''')
    parser.add_argument('--timezone-offset-hours',
                        default=0,
                        type=float,
                        help='''Used with --replay to determine the mapping
                        between the UNIX timestamps in the log file (in UTC) and
                        local time. Given in hours. For instance, Pacific
                        Standard Time is UTC-08:00, so pass
                        --timezone-offset-hours -8. If omitted, we default to
                        UTC''')

    if single_camera:
        parser.add_argument('camera',
                            type = str,
                            nargs = '?',
                            default = None,
                            help='''Without --replay: the camera to talk to.
                            This is a string passed to the arv_camera_new()
                            function; see that function's documentation for
                            details. The string can be IP addresses or MAC
                            addresses or vendor-model-serialnumber strings. With
                            --replay: an integer specifying the camera index in
                            the log. If omitted, we take the first camera''')
    else:
        parser.add_argument('camera',
                            type = str,
                            nargs = '*',
                            default = (None,),
                            help='''Without --replay: the camera(s) to talk to.
                            One argument per camera. These are strings passed to
                            the arv_camera_new() function; see that function's
                            documentation for details. The strings can be IP
                            addresses or MAC addresses or
                            vendor-model-serialnumber strings. With --replay:
                            integers and/or A-B ranges of integers specifying
                            the camera indices in the log. If omitted, we use a
                            single device: the first camera''')



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

    if args.features is not None:
        if args.replay:
            print("--replay and --feature are mutually exclusive",
                  file=sys.stderr)
            sys.exit(1)
        args.features = args.features.split(',')
    else:
        args.features = ()

    if args.display_flip is not None:
        args.display_flip = set(args.display_flip.split(','))
        set_remaining = args.display_flip - set( ('x','y'))
        if len(set_remaining):
            print(f"--display-flip takes a comma-separated list of ONLY 'x' and/or 'y': got unknown elements {set_remaining}", file = sys.stderr)
            sys.exit(1)
    else:
        args.display_flip = set()
    args.flip_x = 'x' in args.display_flip
    args.flip_y = 'y' in args.display_flip


    if args.image_path_prefix is not None and \
       args.image_directory   is not None:
        print("--image-path-prefix and --image-directory are mutually exclusive",
              file=sys.stderr)
        sys.exit(1)

    if args.replay and \
       args.logdir is None:
        print("--replay REQUIRES --logdir to specify the log being read",
              file=sys.stderr)
        sys.exit(1)



    if args.replay:
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


        if args.camera is None:
            # one camera; unspecified
            args.camera = 0
        elif isinstance(args.camera, str):
            # one camera; parse as int > 0
            try:
                args.camera = int(args.camera)
            except:
                print(f"--replay given, so the camera must be an integer >= 0",
                      file=sys.stderr)
                sys.exit(1)
            if args.camera < 0:
                print(f"--replay given, so the camera must be an integer >= 0",
                      file=sys.stderr)
                sys.exit(1)
        else:
            # multiple cameras
            args.camera = [c for cam in args.camera for c in camera_for_replay(cam)]



_time_last_request_image_set = None
def schedule_next_frame(f, period):
    # I want the image requests to fire at a constant rate, ignoring the
    # other processing
    global _time_last_request_image_set

    time_now = time.time()

    if _time_last_request_image_set is None:
        time_sleep = period
    else:
        time_sleep = _time_last_request_image_set + period - time_now

    if time_sleep <= 0:
        f()
        _time_last_request_image_set = time_now

    else:
        Fl.add_timeout(time_sleep, lambda *args: f())
        _time_last_request_image_set = time_now + time_sleep



class Fl_Gl_Image_With_Status_Widget(Fl_Gl_Image_Widget):
    def __init__(self,
                 *args,
                 group,
                 handle_extra = None,
                 **kwargs):

        self.group        = group;
        self.handle_extra = handle_extra;

        return super().__init__(*args, **kwargs)

    def handle(self, event):
        if event == FL_MOVE:
            try:
                q = self.map_pixel_image_from_viewport( (Fl.event_x(),Fl.event_y()), )
                self.group.status_widget.value(f"{q[0]:.1f},{q[1]:.1f}")
            except:
                self.group.status_widget.value("")
            # fall through to let parent handlers run

        if self.handle_extra is not None:
            self.handle_extra(self,event)
        return super().handle(event)



h_status         = 20
h_control        = 30
h_control_footer = 30 # for the label below the widget; needed for some widgets only

class Fl_Image_View_Group(Fl_Group):
    def __init__(self,
                 x,y,w,h,
                 *,
                 camera,
                 feature_names = (),
                 single_buffered = False,
                 status_widget   = None,
                 handle_extra    = None):

        super().__init__(x,y,w,h)


        self.camera = camera
        self.iframe = 0

        if feature_names: w_controls = 300
        else:             w_controls = 0

        if status_widget is None:
            # no global status bar; create one here
            h_status_here = h_status
        else:
            # use a global status bar
            h_status_here = 0

        self.image_widget = \
            Fl_Gl_Image_With_Status_Widget(x, y,
                                           w-w_controls, h-h_status_here,
                                           group           = self,
                                           double_buffered = not single_buffered,
                                           handle_extra    = handle_extra)
        if status_widget is None:
            self.status_widget = Fl_Output(x, y + h-h_status_here, w, h_status_here)
        else:
            self.status_widget = status_widget

        # Need group to control resizing: I want to fix the sizes of the widgets in
        # the group, so I group.resizable(None) later
        group = Fl_Group(x + w-w_controls, y,
                         w_controls, h-h_status_here)

        self.features                 = [dict() for i in feature_names]
        self.feature_dict_from_widget = dict()

        y = 0
        for i,name in enumerate(feature_names):

            feature_dict = self.features[i]
            desc = camera.feature_descriptor(name)

            t = desc['type']
            if t == 'integer' or t == 'float':
                if desc['unit']: label = f"{name} ({desc['unit']})"
                else:            label = name
                h_here = h_control + h_control_footer
                widget = Fl_Value_Slider(x + w-w_controls, y,
                                         w_controls, h_control,
                                         label)
                widget.align(FL_ALIGN_BOTTOM)
                widget.type(FL_HORIZONTAL)
                widget.bounds(*desc['bounds'])
                widget.callback(self.feature_callback_valuator)
            elif t == 'boolean':
                h_here = h_control
                widget = Fl_Check_Button(x + w-w_controls, y,
                                         w_controls, h_control,
                                         name)
                widget.callback(self.feature_callback_valuator)
            elif t == 'command':
                h_here = h_control
                widget = Fl_Button(x + w-w_controls, y,
                                   w_controls, h_control,
                                   name)
                widget.callback(self.feature_callback_button)
            elif t == 'enumeration':
                h_here = h_control + h_control_footer
                widget = Fl_Choice(x + w-w_controls, y,
                                   w_controls, h_control,
                                   name)
                for s in desc['entries']:
                    widget.add(s)
                widget.callback(self.feature_callback_enum)
                widget.align(FL_ALIGN_BOTTOM)
            else:
                raise Exception(f"Feature type '{t}' not supported")

            y += h_here

            feature_dict['widget']     = widget
            feature_dict['name']       = name
            feature_dict['descriptor'] = desc
            self.feature_dict_from_widget[id(widget)] = feature_dict

        self.sync_feature_widgets()

        group.resizable(None)
        group.end()

        self.resizable(self.image_widget)
        self.end()


    def feature_callback_valuator(self, widget):
        feature_dict = self.feature_dict_from_widget[id(widget)]
        self.camera.feature_value(feature_dict['descriptor'], widget.value())

        self.sync_feature_widgets()

    def feature_callback_button(self, widget):
        feature_dict = self.feature_dict_from_widget[id(widget)]
        self.camera.feature_value(feature_dict['descriptor'], 1)

        self.sync_feature_widgets()

    def feature_callback_enum(self, widget):
        feature_dict = self.feature_dict_from_widget[id(widget)]
        self.camera.feature_value(feature_dict['descriptor'], widget.text())

        self.sync_feature_widgets()

    def sync_feature_widgets(self):
        for feature_dict in self.features:
            value,metadata = self.camera.feature_value(feature_dict['descriptor'])
            widget = feature_dict['widget']
            if metadata['locked']: widget.deactivate()
            else:                  widget.activate()

            if isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                widget.value(value)
            elif isinstance(value, str):
                widget.value( widget.find_index(value) )

    def update_image_widget(self,
                            image,
                            *,
                            flip_x,
                            flip_y):

        if image is not None:
            # Update the image preview; deep images are shown as a heat map
            if image.itemsize > 1:
                if image.ndim > 2:
                    raise Exception("high-depth color images not supported yet")
                q = 5
                a_min = np.percentile(image, q = q)
                a_max = np.percentile(image, q = 100-q)
                heatmap = mrcal.apply_color_map(image,
                                                a_min = a_min,
                                                a_max = a_max)
                self.image_widget.update_image(image_data = heatmap,
                                               flip_x     = flip_x,
                                               flip_y     = flip_y)
            else:
                self.image_widget.update_image(image_data = image,
                                               flip_x     = flip_x,
                                               flip_y     = flip_y)

            self.sync_feature_widgets()
        else:
            print("Error capturing the image. I will try again",
                  file=sys.stderr)

    def set_up_image_capture(self,
                             *,
                             period         = None, # if given, we automatically recur
                             # guaranteed to be called with each frame; even on error
                             flip_x         = False,
                             flip_y         = False,
                             image_callback = None,
                             **image_callback_cookie):

        if self.camera is None:
            return

        def callback_image_ready(fd):
            frame = self.camera.requested_image()

            self.update_image_widget( image        = frame['image'],
                                      flip_x       = flip_x,
                                      flip_y       = flip_y,)
            if image_callback is not None:
                image_callback(image,
                               timestamp_us = timestamp_us,
                               iframe       = self.iframe,
                               **image_callback_cookie)

            self.iframe += 1

            if period is not None:
                # Ask for some data in the future
                schedule_next_frame(self.camera.request, period)


        # Tell FLTK to callback_image_ready() when data is available
        Fl.add_fd( self.camera.fd_image_ready,
                   callback_image_ready )

        if period is not None:
            # Ask for some data
            self.camera.request()

