#!/usr/bin/python3

r'''mrcam camera integration with FLTK

This is a higher-level Python interface to make full-featured mrcam-enabled FLTK
applications simple to build

'''

import sys
import re
from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget
from fltk import *
import mrcal
import time
import numpy as np
import numpysane as nps
import math
import datetime
import os
import mrcam


def add_common_cmd_options(parser,
                            *,
                            single_camera     = False,
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
                        is a comma-separated string of "x" and/or "y"''')
    parser.add_argument('--dims',
                        help='''Imager dimensions given as WIDTH,HEIGHT. Required for cameras where this
                        cannot be auto-detected''')
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
        parser.add_argument('--unlock-panzoom',
                            action='store_true',
                            help='''If given, the pan/zoom in the all the image
                            widgets are NOT locked together. By default they ARE
                            locked together''')
        parser.add_argument('camera',
                            type = str,
                            nargs = Ncameras_expected if Ncameras_expected is not None else '*',
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
                            single_camera = False):
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
        args.features = [ f for f in args.features.split(',') if len(f) ] # filter out empty features
    else:
        args.features = ()

    if args.display_flip is not None:
        args.display_flip = set(args.display_flip.split(','))

        set_remaining = args.display_flip - set( ('x','y','xy'))
        if len(set_remaining):
            print(f"--display-flip takes a comma-separated list of ONLY 'x' and/or 'y' and/or 'xy': got unknown elements {set_remaining}", file = sys.stderr)
            sys.exit(1)
    else:
        args.display_flip = set()
    args.flip_x = 'x' in args.display_flip or 'xy' in args.display_flip
    args.flip_y = 'y' in args.display_flip or 'xy' in args.display_flip

    ### args.camera_params_noname is a subset of args.__dict__. That subset is
    ### camera parameters passed directly to mrcam.camera()
    # These must match keywords[] in camera_init() in mrcam-pywrap.c WITHOUT the
    # "name" field
    camera_param_noname_keys = \
        ("pixfmt",
         "acquisition_mode",
         "trigger",
         "time_decimation_factor",
         "dims",
         "Nbuffers",
         "verbose")
    args.camera_params_noname = dict()
    for k in camera_param_noname_keys:
        args.camera_params_noname[k] = getattr(args, k, None)

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

    if args.features and \
       args.replay   is not None:
        print("--replay and --features are mutually exclusive",
              file=sys.stderr)
        sys.exit(1)

    if single_camera:
        # The various machinery requires SOME value to exist, so I write one
        args.unlock_panzoom = False
        # Lots of code assumes multiple cameras. I oblige
        args.camera = (args.camera,)



def schedule_next_frame(f, t0, period):
    # I want the image requests to fire at a constant rate, ignoring the other
    # processing. Analogous to mrcam_sleep_until_next_request(), but sets an
    # FLTK timer instead of sleeping.
    time_now = time.time()

    if t0 == 0:
        time_sleep = period
    else:
        time_sleep = t0 + period - time_now

    if time_sleep <= 0:
        f()
    else:
        Fl.add_timeout(time_sleep, lambda *args: f())


def displayed_image__default(image,
                             *,
                             do_equalize_fieldscale = False,
                             # extra stuff
                             **kwargs):
    if image is None:
        return None

    if image.itemsize == 1:
        # 8-bit image. Display as is
        return image

    # Deep image. Display as a heat map
    if image.ndim > 2:
        raise Exception("high-depth color images not supported yet")

    if not do_equalize_fieldscale:
        q = 5
        a_min = np.percentile(image, q = q)
        a_max = np.percentile(image, q = 100-q)
        return mrcal.apply_color_map(image,
                                     a_min = a_min,
                                     a_max = a_max)
    else:
        return mrcal.apply_color_map(mrcam.equalize_fieldscale(image),
                                     a_min = 0,
                                     a_max = 255)


def status_value__default(q, pixel_value_text):
    if q is not None:
        if pixel_value_text is not None:
            return f"{q[0]:.1f},{q[1]:.1f}{pixel_value_text}"
        else:
            return f"{q[0]:.1f},{q[1]:.1f}"
    else:
        return ""





class Fl_Gl_Image_with_handle(Fl_Gl_Image_Widget):
    def __init__(self,
                 *args,
                 handler = None,
                 locked_panzoom_groups = None,
                 **kwargs):

        self.handler               = handler
        self.locked_panzoom_groups = locked_panzoom_groups

        # I want keyboard commands to work the same regardless of which widget
        # is focused. Specifically, I want the arrow keys to always end up in
        # the time slider. So Fl_Gl_Image_Widget shouldn't handle these
        # keystrokes; but by default the navigation command logic WILL interpret
        # those. A discussion describes two ways to handle this:
        #   https://github.com/fltk/fltk/discussions/1048
        # One is to prevent the widget from being focused, which will block ALL
        # keyboard events. The other is to explicitly ignore the specific
        # keyboard events in a parent class. I do the former here because that's
        # simpler, and I don't need to process any keyboard events
        x = super().__init__(*args, **kwargs)
        self.visible_focus(0)
        return x

    def handle(self, event):
        res = self.handler(self,event)
        if res is not None:
            return res
        return super().handle(event)


    def set_panzoom(self,
                    x_centerpixel, y_centerpixel,
                    visible_width_pixels,
                    ratios           = False,
                    panzoom_siblings = True):
        r'''Pan/zoom the image

        This is an override of the function to do this: any request to
        pan/zoom the widget will come here first. panzoom_siblings
        dispatches any pan/zoom commands to all the widgets, so that they
        all work in unison.

        if ratios: the values are given not in pixels but in ratios of
        width/height. This is important if we're trying to lock the pan/zoom in
        cameras with unequal image sizes. ALL the calls to keep the panzoom
        locked use ratios=True

        '''

        if self.image is not None:
            (image_height,image_width) = self.image.shape[:2]
        else:
            (image_height,image_width) = (None,None)

        if not panzoom_siblings or \
           self.locked_panzoom_groups is None:

            if not ratios:
                return super().set_panzoom(x_centerpixel, y_centerpixel,
                                           visible_width_pixels)
            else:
                if image_width  is None or \
                   image_height is None:
                    return

                return super().set_panzoom(x_centerpixel        * image_width,
                                           y_centerpixel        * image_height,
                                           visible_width_pixels * image_width)

        # All the widgets should pan/zoom together
        if not ratios:
            if image_width  is None or \
               image_height is None:
                return
            return \
                all( g.image_widget. \
                     set_panzoom(x_centerpixel        / image_width,
                                 y_centerpixel        / image_height,
                                 visible_width_pixels / image_width,
                                 panzoom_siblings = False,
                                 ratios           = True) \
                     for g in self.locked_panzoom_groups )
        else:
            return \
                all( g.image_widget. \
                     set_panzoom(x_centerpixel,
                                 y_centerpixel,
                                 visible_width_pixels,
                                 panzoom_siblings = False,
                                 ratios           = True) \
                     for g in self.locked_panzoom_groups )


h_status         = 20
h_control        = 30
h_control_footer = 30 # for the label below the widget; needed for some widgets only

class Fl_Image_View_Group(Fl_Group):
    def __init__(self,
                 x,y,w,h,
                 *,
                 camera,
                 # iterable of strings. Might contain regex; might contain
                 # annotations such as [log] (applied to all regex matches). Any
                 # feature name that doesn't exist EXACTLY as given will be
                 # re-tried as a regex
                 features          = (),
                 status_widget,
                 unlock_panzoom,
                 # (function,cookie)
                 cb_handle_event_image_widget = None,
                 cb_displayed_image           = displayed_image__default,
                 cb_status_value              = status_value__default,
                 # usually will come from **fltk_application_context
                 image_view_groups,
                 do_equalize_fieldscale, # [value] and not value
                 # other stuff from the contexts that I don't need here
                 **kwargs):

        super().__init__(x,y,w,h)

        self.displayed_image = cb_displayed_image
        self.status_value    = cb_status_value

        self.camera = camera
        self.iframe = 0

        self.do_equalize_fieldscale = do_equalize_fieldscale

        if features: w_controls = 300
        else:        w_controls = 0

        self.status_widget = status_widget


        def handle_image_widget(self_image_widget, event):
            if event == FL_MOVE:
                try:
                    q = self_image_widget.map_pixel_image_from_viewport( (Fl.event_x(),Fl.event_y()), )

                    pixel_value_text = ''

                    if self.image_widget.image is not None:
                        qint_x = round(q[0])
                        qint_y = round(q[1])

                        (image_height,image_width) = self.image_widget.image.shape[:2]
                        if qint_x >= 0 and qint_x < image_width and \
                           qint_y >= 0 and qint_y < image_height:

                            pixel_value_text = f",{self.image_widget.image[qint_y,qint_x,...]}"

                    self.status_widget.value( self.status_value(q, pixel_value_text) )
                except:
                    self.status_widget.value( self.status_value(None, None) )

            if cb_handle_event_image_widget is not None:
                return cb_handle_event_image_widget[0](self,event,
                                                     **cb_handle_event_image_widget[1])

            return None # Use parent's return code

        self.image_widget = \
            Fl_Gl_Image_with_handle(x, y,
                                    w-w_controls, h,
                                    handler               = handle_image_widget,
                                    double_buffered       = True,
                                    locked_panzoom_groups = \
                                      None if unlock_panzoom else \
                                      image_view_groups)

        # Need group to control resizing: I want to fix the sizes of the widgets in
        # the group, so I group.resizable(None) later
        group = Fl_Group(x + w-w_controls, y,
                         w_controls, h)

        def expand_features(features_selected):
            feature_set = camera.features() if camera is not None else set()

            for f in features_selected:

                if not f.startswith('R['):
                    m = re.match(r"(.*?)" +
                                 r"(?:" +
                                   r"\[" +
                                     r"([^\[]*)" +
                                   r"\]" +
                                   r")?$", f) # should never fail

                    name = m.group(1).strip()
                    flags = m.group(2)
                    if flags and len(flags):
                        flags = set( [s.strip() for s in flags.split(',')] )
                    else:
                        flags = set()

                    if name in feature_set:
                        yield dict(name  = name,
                                   flags = flags)
                        continue

                    # name not found exactly; try regex
                    matched_any = False
                    for name_exists in feature_set:

                        # might get re.error
                        if re.search(name, name_exists):
                            matched_any = True
                            yield dict(name  = name_exists,
                                       flags = flags)
                    if not matched_any:
                        raise Exception(f"Feature '{name}' doesn't exist or isn't implemented; tried both exact searching and a regex")

                elif re.match(r'^R\[(0x[0-9a-fA-F]+)|[0-9]+\]$', f):
                    # is "R[....]". Treat it like a direct register access. Like arv-tool does
                    yield dict(name  = f,
                               flags = set())
                else:
                    raise Exception(f"Feature '{name}' doesn't exist or isn't implemented; it starts with R[ but doesn't have the expected R[ADDRESS] format")

        self.features = list(expand_features(features))
        self.feature_dict_from_widget = dict()

        y = 0
        for feature_dict in self.features:
            name  = feature_dict['name']
            flags = feature_dict['flags']

            try:
                desc = camera.feature_descriptor(name)
            except Exception as e:
                print(f"Warning: not adding widget for feature '{name}' because: {e}",
                      file = sys.stderr)
                continue

            if desc.get('representation','') == 'LOGARITHMIC':
                flags.add('log')

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

                if 'log' in flags:
                    if any(x <= 0 for x in desc['bounds']):
                        raise Exception(f"Requested log-scale feature '{name}' has non-positive bounds: {desc['bounds']}. Log-scale features must have strictly positive bounds")
                    widget.bounds(*[np.log(x) for x in desc['bounds']])
                else:
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
            feature_dict['descriptor'] = desc
            self.feature_dict_from_widget[id(widget)] = feature_dict

        # Keep only those features that were added successfully
        self.features = [f for f in self.features if 'widget' in f]

        self.sync_feature_widgets()

        group.resizable(None)
        group.end()

        self.resizable(self.image_widget)
        self.end()


    def feature_callback_valuator(self, widget):
        feature_dict = self.feature_dict_from_widget[id(widget)]
        value = np.exp(widget.value()) if 'log' in feature_dict['flags'] else widget.value()
        if feature_dict['descriptor']['type'] == 'integer':
            value = np.round(value)
        self.camera.feature_value(feature_dict['descriptor'],
                                  value)
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
            widget = feature_dict['widget']

            try:
                value,metadata = self.camera.feature_value(feature_dict['descriptor'])
            except Exception as e:
                if not feature_dict.get('warned-about-error-get-value'):
                    feature_dict['warned-about-error-get-value'] = True
                    name = feature_dict['name']
                    print(f"Warning: couldn't get the value of feature '{name}': {e}")
                    widget.deactivate()
                    widget.value(0)
                continue

            if metadata['locked']: widget.deactivate()
            else:                  widget.activate()

            if isinstance(value, bool):
                widget.value(value)
            elif isinstance(value, int) or isinstance(value, float):
                widget.value( np.log(value) if 'log' in feature_dict['flags'] else value )
            elif isinstance(value, str):
                widget.value( widget.find_index(value) )

    def update_image_widget(self,
                            image,
                            *,
                            iframe,
                            icam,
                            flip_x,
                            flip_y):

        self.image_widget.image = image

        # Will be None if the image was None (i.e. the capture failed)
        image_data = \
            self.displayed_image(image,
                                 iframe                 = iframe,
                                 icam                   = icam,
                                 do_equalize_fieldscale = self.do_equalize_fieldscale[0])
        self.image_widget.update_image(image_data = image_data,
                                       flip_x     = flip_x,
                                       flip_y     = flip_y)

        if image_data is not None:
            self.sync_feature_widgets()


    def set_up_image_capture(self,
                             *,
                             # if given, we automatically recur. Otherwise it is
                             # expected that the image_callback will request the
                             # next set of frames, if needed
                             period                   = None,
                             flip_x                   = False,
                             flip_y                   = False,
                             **image_callback__cookie):

        if self.camera is None:
            return

        def callback_mrcam(fd):

            frame = self.camera.requested_image()

            image_received_from_mrcam(iframe = self.iframe,
                                      frame  = frame,
                                      **image_callback__cookie)
            self.camera.push_buffer(frame['buffer']) # no-op if the buffer is None
            if not frame['off_decimation']:
                self.iframe += 1

            if period is not None:
                schedule_next_frame(self.camera.request, self.camera.timestamp_request_us/1e6, period)


        # Tell FLTK to callback_mrcam() when data is available
        Fl.add_fd( self.camera.fd_image_ready,
                   callback_mrcam )

        if period is not None:
            # Ask for some data
            self.camera.request()


def log_readwrite_init(camera_names,
                       *,
                       logdir_read       = None,
                       replay_from_frame = 0,
                       logdir_write      = None,
                       jpg               = False,
                       image_path_prefix = None,
                       image_directory   = None):

    ctx = dict(logdir_read       = logdir_read,
               replay_from_frame = replay_from_frame,
               logdir_write      = logdir_write,
               jpg               = jpg,
               image_path_prefix = image_path_prefix,
               image_directory   = image_directory)



    if logdir_read is not None and \
       logdir_write is not None:
        raise Exception("logdir_read and logdir_write are exclusive. parse_args_postprocess() should have checked this")

    Ncameras = len(camera_names)

    if logdir_read is not None:

        ### we're reading a log

        # I need at least vnlog 1.38 to support structured dtypes in vnlog.slurp().
        # See
        #   https://notes.secretsauce.net/notes/2024/07/02_vnlogslurp-with-non-numerical-data.html
        import vnlog

        path = f"{logdir_read}/images.vnl"
        max_len_imagepath = 128

        dtype = np.dtype([ ('time',      float),
                           ('iframe',    np.int32),
                           ('icam',      np.int8),
                           ('imagepath', f'U{max_len_imagepath}'),
                          ])
        log = vnlog.slurp(path, dtype=dtype)

        # I have the whole log. I cut it down to include ONLY the cameras that were
        # requested
        i = np.min( np.abs(nps.dummy(np.array(camera_names), -1) - \
                           log['icam']),
                    axis=-2) == 0
        log = log[i]

        if log.size == 0:
            print(f"The requested cameras {camera_names=} don't have any data in the log {path}",
                  file = sys.stderr)
            sys.exit(1)

        if max(len(s) for s in log['imagepath']) >= max_len_imagepath:
            print(f"Image paths in {path} are longer than the statically-defined value of {max_len_imagepath=}. Increase max_len_imagepath, or make the code more general", file=sys.stderr)
            sys.exit(1)

        # While the tool is running I want to be able to access the images in O(1),
        # so I pre-sort the log now to make that possible. For each constant iframe
        # I want a monotonically-increasing icam
        log = np.sort(log, order=('iframe','icam'))

        # I should have dense data over frames and cameras. I try to reshape it in
        # that way, and confirm that everything lines up
        Nframes  = len(log) // Ncameras
        if Nframes*Ncameras != len(log):
            print(f"The log {path} does not contain all dense combinations of {Ncameras=} and {Nframes=}. For the requested cameras it has {len(log)} entries",
                  file = sys.stderr)
            for icam in camera_names:
                print(f"  Requested camera {icam} has {np.count_nonzero(log['icam']==icam)} observed frames",
                      file = sys.stderr)
            print("These should all be idendical",
                  file=sys.stderr)
            sys.exit(1)

        if not (replay_from_frame >= 0 and \
                replay_from_frame < Nframes):
            print(f"We have {Nframes=} in the log '{path}', so --replay-from-frame must be in [0,{Nframes-1}], but we have --replay-from-frames {replay_from_frame}",
                  file = sys.stderr)
            sys.exit(1)

        log = log.reshape((Nframes,Ncameras))
        if np.any(log['icam'] - log['icam'][0,:]):
            print(f"The log {path} does not contain the same set of cameras in each frame",
                  file = sys.stderr)
            sys.exit(1)
        if np.any( np.sort(camera_names) - log['icam'][0,:] ):
            print(f"The log {path} does not contain exactly the cameras requested in {camera_names=}",
                  file = sys.stderr)
            sys.exit(1)
        if np.any(log['iframe'] - log['iframe'][:,(0,)]):
            print(f"The log {path} does not contain the same set of frames observing each camera",
                  file = sys.stderr)
            sys.exit(1)

        # Great. We have a dense set. We're done!
        ctx['logged_images'] = log
    elif logdir_write is not None:
        ### we're writing a log
        if not os.path.isdir(logdir_write):
            if os.path.exists(logdir_write):
                print(f"Error: requested logdir_write '{logdir_write}' is a FILE on disk. It should be a directory (that we will write to) or it shouldn't exist (we will create the directory)",
                      file=sys.stderr)
                sys.exit(1)
            try:
                os.mkdir(logdir_write)
            except Exception as e:
                print(f"Error: could not mkdir requested logdir_write '{logdir_write}': {e}",
                      file=sys.stderr)
                sys.exit(1)

        path = f"{logdir_write}/images.vnl"
        try:
            file_log = open(path, "w")
        except Exception as e:
            print(f"Error opening log file '{path}' for writing: {e}",
                  file=sys.stderr)
            sys.exit(1)

        for i,c in enumerate(camera_names):
            if c is not None:
                print(f"## Camera {i}: {c}", file=file_log)
        write_logline("# time iframe icam imagepath",
                      file_log = file_log);

        ctx['file_log'] = file_log
        # I can replay this log as I write it. 'logged_images' is set for both
        # reading and writing
        ctx['logged_images'] = []

    return ctx


def write_logline(l,
                  *,
                  # usually will come from **log_readwrite_context
                  file_log,
                  # other stuff from the contexts that I don't need here
                  **kwargs):
    if file_log is not None:
        print(l,file=file_log)
        file_log.flush()


def image_received_from_mrcam(*,
                              iframe,
                              frame, # dict from requested_image()

                              # All these are the cookie given to set_up_image_capture()
                              icam,
                              # usually will come from **log_readwrite_context
                              logged_images,
                              logdir_write,
                              file_log,
                              jpg,
                              # usually will come from **fltk_application_context
                              image_view_groups,
                              time_slider_widget,
                              flip_x,
                              flip_y,
                              utcoffset_sec,
                              tzname,
                              period,
                              Ncameras_seen_iframe,
                              logged_image_from_iframe,
                              # other stuff from the contexts that I don't need here
                              **kwargs):
    r'''Process the image

On return, we will push_buffer(frame['buffer']). If we do not want that (i.e. if
we will do that ourselves, set frame['buffer'] to None)

    '''

    Ncameras = len(image_view_groups)

    extension      = "jpg" if jpg else "png"

    image          = frame['image']
    timestamp      = frame['timestamp']
    buffer         = frame['buffer']
    off_decimation = frame['off_decimation']

    if not off_decimation:
        if image is None:
            print("Error capturing the image. I will try again",
                  file=sys.stderr)

        time_slider_at_max = False
        if logdir_write is not None:

            time_slider_now = int(time_slider_widget.value())
            time_slider_at_max = \
                time_slider_now == int(time_slider_widget.maximum())

            if not iframe in logged_image_from_iframe:
                # Started this set of cameras. Add empty record; fill it in as I get
                # frames.
                #
                # It is very strange to have a list of iframes here: each iframe
                # in the list will be the same. I do that because that's what I
                # get when I read logs, and I want to use the same code path
                # for the read- and write-log cases. This could be simplified if
                # I adjust the results in log_readwrite_init() to remove the
                # duplicated iframes
                logged_image_from_iframe[iframe] = dict(time      = [None] * Ncameras,
                                                        imagepath = [None] * Ncameras,
                                                        iframe    = [None] * Ncameras)
                logged_images.append( logged_image_from_iframe[iframe] )

            # write image to disk
            filename = f"frame{iframe:05d}-cam{icam}.{extension}"
            path = f"{logdir_write}/{filename}"

            if image is None:
                write_logline(f"{timestamp:.3f} {iframe} {icam} -",
                              file_log = file_log);
            else:
                write_logline(f"{timestamp:.3f} {iframe} {icam} {filename}",
                              file_log = file_log);

                image_view_groups[icam].camera. \
                    async_save_image_and_push_buffer(path,image,frame['buffer'])
                frame['buffer'] = None # indicate that the caller should NOT re-push the buffer

            logged_image_from_iframe[iframe]['time'     ][icam] = timestamp
            logged_image_from_iframe[iframe]['iframe'   ][icam] = iframe
            if image is not None:
                logged_image_from_iframe[iframe]['imagepath'][icam] = path
                # Otherwise, leave at None

        if logdir_write is None or time_slider_at_max:
            image_view_groups[icam].update_image_widget( image  = image,
                                                         iframe = iframe,
                                                         icam   = icam,
                                                         flip_x = flip_x,
                                                         flip_y = flip_y)

    # schedule the next set of images; do this even if off_decimation
    if not iframe in Ncameras_seen_iframe:
        Ncameras_seen_iframe[iframe] = 1
    else:
        Ncameras_seen_iframe[iframe] += 1

    # I need a Ncameras_seen_iframe to be a dict instead of a count for the
    # last frame, because in a free-runnning mode, I may get frames out of
    # the usual expected order
    if Ncameras_seen_iframe[iframe] >= Ncameras:
        # Every camera reported back. Finish up and ask for another frame
        del Ncameras_seen_iframe[iframe]

        if not off_decimation and time_slider_widget is not None:
            time_slider_widget.maximum(iframe)

            if time_slider_at_max:
                # We were at the end of the time slider. Update us so that we're
                # still at the end
                time_slider_widget.value(iframe)
                time_slider_update_label(iframe             = iframe,
                                         time               = timestamp,
                                         time_slider_widget = time_slider_widget,
                                         utcoffset_sec      = utcoffset_sec,
                                         tzname             = tzname)
            else:
                time_slider_update_label(iframe             = time_slider_now,
                                         time               = logged_images[time_slider_now]['time'][0],
                                         time_slider_widget = time_slider_widget,
                                         utcoffset_sec      = utcoffset_sec,
                                         tzname             = tzname)
                # The bounds changed, so the handle should be redrawn
                time_slider_widget.redraw()

        def request_image_set():
            for image_view_group in image_view_groups:
                image_view_group.camera.request()
        schedule_next_frame(request_image_set,
                            image_view_groups[0].camera.timestamp_request_us/1e6, period)



def complete_path(path,
                  *,
                  # usually will come from **log_readwrite_context
                  logdir_write      = None,
                  logdir_read       = None,
                  image_path_prefix = None,
                  image_directory   = None):

    if path is None or path == '-':
        return None

    if logdir_write is not None:
        # We're logging; we already have the full path
        return path

    if image_path_prefix is not None:
        return f"{image_path_prefix}/{path}"
    if image_directory is not None:
        return f"{image_directory}/{os.path.basename(path)}"
    if path[0] != '/':
        # The image filename has a relative path. I want it to be
        # relative to the log directory
        if logdir_read is not None:
            return f"{logdir_read}/{path}"
        raise Exception("We're replaying but both logdir and replay are None. This is a bug")

    return path


def time_slider_update_label(*,
                             iframe,
                             time,
                             time_slider_widget,
                             utcoffset_sec,
                             tzname):
    t = int(time + utcoffset_sec)
    t = datetime.datetime.fromtimestamp(t, datetime.UTC)
    time_slider_widget.label(f"iframe={iframe}/{int(time_slider_widget.maximum())} timestamp={time:.03f} {t.strftime('%Y-%m-%d %H:%M:%S')} {tzname}")


def update_all_images_from_replay(*,
                                  # usually will come from **log_readwrite_context
                                  logged_images,
                                  logdir_write      = None,
                                  logdir_read       = None,
                                  image_path_prefix = None,
                                  image_directory   = None,
                                  # usually will come from **fltk_application_context
                                  image_view_groups,
                                  flip_x,
                                  flip_y,
                                  time_slider_widget,
                                  # other stuff from the contexts that I don't need here
                                  **kwargs):
    i_iframe = round(time_slider_widget.value())

    try:
        record = logged_images[i_iframe]
    except IndexError:
        print(f"WARNING: {i_iframe=} is out-of-bounds in logged_images: {len(logged_images)=}. This is a bug")
        return

    Ncameras = len(image_view_groups)
    for icam in range(Ncameras):
        path = complete_path(record['imagepath'][icam],
                             logdir_write      = logdir_write,
                             logdir_read       = logdir_read,
                             image_path_prefix = image_path_prefix,
                             image_directory   = image_directory)
        if path is None:
            image = None # write an all-black image

        else:
            try:
                image = mrcal.load_image(path)
            except:
                print(f"Couldn't read image at '{path}'", file=sys.stderr)
                image = None

        image_view_groups[icam].update_image_widget( image,
                                                     iframe = record['iframe'][icam],
                                                     icam   = icam,
                                                     flip_x = flip_x,
                                                     flip_y = flip_y)



def time_slider_select(*,
                       # usually will come from **log_readwrite_context
                       logged_images,
                       logdir_write,
                       logdir_read,
                       image_path_prefix,
                       image_directory,
                       # usually will come from **fltk_application_context
                       image_view_groups,
                       time_slider_widget,
                       flip_x,
                       flip_y,
                       utcoffset_sec,
                       tzname,
                       # other stuff from the contexts that I don't need here
                       **kwargs):
    i_iframe = round(time_slider_widget.value())

    try:
        record = logged_images[i_iframe]
    except IndexError:
        print(f"WARNING: {i_iframe=} is out-of-bounds in logged_images: {len(logged_images)=}. This is a bug")
        return

    # shape (Ncameras,); all of these
    times   = record['time']
    iframes = record['iframe']

    time_slider_update_label(iframe             = iframes[0],
                             time               = times[0],
                             time_slider_widget = time_slider_widget,
                             utcoffset_sec      = utcoffset_sec,
                             tzname             = tzname)

    update_all_images_from_replay(logged_images     = logged_images,
                                  logdir_write      = logdir_write,
                                  logdir_read       = logdir_read,
                                  image_path_prefix = image_path_prefix,
                                  image_directory   = image_directory,
                                  image_view_groups = image_view_groups,
                                  flip_x            = flip_x,
                                  flip_y            = flip_y,
                                  time_slider_widget=time_slider_widget)

    # if live-updating we color the slider green
    if logdir_write is not None:
        if int(time_slider_widget.value()) == int(time_slider_widget.maximum()):
            time_slider_widget.color(FL_GREEN)
        else:
            time_slider_widget.color(FL_BACKGROUND_COLOR)


def handle_event_image_widget__e(image_view_group,
                                 event,
                                 *,
                                 # user-passed cookie
                                 # usually will come from **log_readwrite_context
                                 logged_images,
                                 logdir_write,
                                 logdir_read,
                                 image_path_prefix,
                                 image_directory,
                                 # usually will come from **fltk_application_context
                                 image_view_groups,
                                 flip_x,
                                 flip_y,
                                 time_slider_widget,
                                 # other stuff from the contexts that I don't need here
                                 **kwargs
                                 ):
    if event == FL_KEYUP:
        if Fl.event_key() == ord('e') or \
           Fl.event_key() == ord('E'):
            # Toggle the value shared between ALL the image_view_groups
            image_view_group.do_equalize_fieldscale[0] = \
                not image_view_group.do_equalize_fieldscale[0]
            # If I have a log (we're replaying or logging to disk) I update the
            # view. Otherwise I simply wait for the next frame to come in.
            # Hopefully that should be quick-enough
            if logged_images is not None:
                update_all_images_from_replay(logged_images     = logged_images,
                                              logdir_write      = logdir_write,
                                              logdir_read       = logdir_read,
                                              image_path_prefix = image_path_prefix,
                                              image_directory   = image_directory,
                                              image_view_groups = image_view_groups,
                                              flip_x            = flip_x,
                                              flip_y            = flip_y,
                                              time_slider_widget=time_slider_widget)
            return 1

    return None # Use parent's return code


def create_gui_elements__default(*,
                                 fltk_application_context,
                                 log_readwrite_context,
                                 W,
                                 H,
                                 H_footer,
                                 title,
                                 unlock_panzoom,
                                 features,
                                 cb_displayed_image,
                                 cb_status_value):

    H_footers = H_footer
    if log_readwrite_context.get('logged_images') is not None:
        H_footers += 2*H_footer

    kwargs = dict(fltk_application_context = fltk_application_context,
                  log_readwrite_context      = log_readwrite_context,
                  W                          = W,
                  H                          = H,
                  H_image_views              = H - H_footers,
                  W_image_views              = W,
                  H_footer                   = H_footer,
                  title                      = title,
                  unlock_panzoom             = unlock_panzoom,
                  features                   = features,
                  cb_displayed_image         = cb_displayed_image,
                  cb_status_value            = cb_status_value)

    create_gui_window     (**kwargs)
    create_gui_time_slider(**kwargs)
    create_gui_status     (**kwargs)
    create_gui_image_views(**kwargs)
    finish_gui_window     (**kwargs)


def create_gui_window(*,
                      fltk_application_context,
                      W,
                      H,
                      title,
                      # extra uneeded stuff
                      **kwargs):
    fltk_application_context['window'] = Fl_Window(W,H, title)


def create_gui_time_slider(*,
                           fltk_application_context,
                           log_readwrite_context,
                           W,
                           H_image_views,
                           H_footer,
                           # extra uneeded stuff
                           **kwargs):

    logged_images     = log_readwrite_context.get('logged_images')
    replay_from_frame = log_readwrite_context.get('replay_from_frame', 0)

    if logged_images is not None:
        time_slider_widget = \
            Fl_Slider(0, H_image_views,
                      W,H_footer)
        time_slider_widget.align(FL_ALIGN_BOTTOM)
        time_slider_widget.type(FL_HORIZONTAL)
        time_slider_widget.step(1)
        if len(logged_images) > 0:
            time_slider_widget.bounds(0, len(logged_images)-1)
            time_slider_widget.value(replay_from_frame)
        else:
            time_slider_widget.bounds(0, 0)
            time_slider_widget.value(0)
        time_slider_widget.callback( \
            lambda *args: \
                time_slider_select(**log_readwrite_context,
                                   **fltk_application_context))
    else:
        time_slider_widget = None
    fltk_application_context['time_slider_widget'] = time_slider_widget


def create_gui_status(*,
                      fltk_application_context,
                      W,
                      H,
                      H_footer,
                      # extra uneeded stuff
                      **kwargs):
    status_widget = Fl_Output(0, H-H_footer,
                              W, H_footer)
    status_widget.value('')

    # I want the status widget to be output-only and not user-focusable. This will
    # allow keyboard input to not be sent to THIS status widget, so that left/right
    # and 'u' go to the time slider and image windows respectively.
    status_widget.visible_focus(0)
    fltk_application_context['status_widget'] = status_widget


def create_gui_image_views(*,
                           fltk_application_context,
                           log_readwrite_context,
                           W_image_views,
                           H_image_views,
                           unlock_panzoom,
                           features,
                           cb_displayed_image,
                           cb_status_value,
                           # extra uneeded stuff
                           **kwargs):

    Ncameras = len(fltk_application_context['image_view_groups'])
    Ngrid = math.ceil(math.sqrt(Ncameras))
    Wgrid = Ngrid
    Hgrid = math.ceil(Ncameras/Wgrid)
    w_image = W_image_views // Wgrid
    h_image = H_image_views // Hgrid

    icam = 0
    y0   = 0
    image_views = Fl_Group(0, 0, W_image_views, H_image_views)
    fltk_application_context['image_views'] = image_views
    for i in range(Hgrid):
        x0 = 0

        for j in range(Wgrid):
            fltk_application_context['image_view_groups'][icam] = \
                Fl_Image_View_Group(x0,y0,
                                    w_image if j < Wgrid-1 else (W_image_views-x0),
                                    h_image if i < Hgrid-1 else (H_image_views-y0),
                                    camera          = fltk_application_context['cameras'][icam],
                                    features        = features,
                                    cb_handle_event_image_widget = (handle_event_image_widget__e,
                                                                    # the cookie
                                                                    dict(**log_readwrite_context,
                                                                         **fltk_application_context)),
                                    unlock_panzoom  = unlock_panzoom,
                                    cb_displayed_image = cb_displayed_image,
                                    cb_status_value    = cb_status_value,
                                    **fltk_application_context)
            x0   += w_image
            icam += 1

            if icam == Ncameras:
                break
        if icam == Ncameras:
            break

        y0 += h_image
    image_views.end()


def finish_gui_window(*,
                      fltk_application_context,
                      # extra uneeded stuff
                      **kwargs):

    window = fltk_application_context['window']

    window.resizable(fltk_application_context['image_views'])
    window.end()




def fltk_application_init(camera_params_noname,
                          camera_names,
                          *,
                          utcoffset_hours = None,
                          W               = 1280,
                          H               = 1024,
                          H_footer        = 30,
                          title = "mrcam stream",
                          flip_x            = False,
                          flip_y            = False,
                          unlock_panzoom    = False,
                          features          = (),
                          period            = 1.0,
                          # usually will come from **log_readwrite_context
                          logged_images     = None,
                          logdir_write      = None,
                          logdir_read       = None,
                          image_path_prefix = None,
                          image_directory   = None,
                          file_log          = None,
                          replay_from_frame = 0,
                          jpg               = False,

                          cb_create_gui_elements = create_gui_elements__default,
                          cb_displayed_image     = displayed_image__default,
                          cb_status_value        = status_value__default,
                          # other stuff from the contexts that I don't need here
                          **kwargs
                          ):

    Ncameras = len(camera_names)
    ctx = dict(cameras                = [None] * Ncameras,
               image_view_groups      = [None] * Ncameras,
               flip_x                 = flip_x,
               flip_y                 = flip_y,
               period                 = period,

               Ncameras_seen_iframe     = dict(),
               logged_image_from_iframe = dict(),

               # [False] and not False because I want this passed by reference
               # to functions. It should be modifiable.
               do_equalize_fieldscale = [False],
               )

    if logdir_read is None:
        # I init each camera. If we're sending the TTYS0 trigger signal, I want all
        # the cameras to be ready when the trigger comes in. Thus the LAST camera
        # will send the trigger; I set the rest to EXTERNAL triggering in that case
        camera_params_noname = dict(camera_params_noname) # make a copy
        for i,name in reversed(list(enumerate(camera_names))):
            ctx['cameras'][i] = \
                mrcam.camera(name = name,
                             **camera_params_noname)

            if camera_params_noname['trigger'] == 'HARDWARE_TTYS0':
                camera_params_noname['trigger'] = 'HARDWARE_EXTERNAL'



    if utcoffset_hours is not None:
        ctx['utcoffset_sec'] = utcoffset_hours*3600
        ctx['tzname']        = f"{'-' if utcoffset_hours<0 else ''}{int(abs(utcoffset_hours)):02d}:{round( (abs(utcoffset_hours) % 1)*60 ):02d}"
    else:
        import time
        t = time.localtime()
        ctx['utcoffset_sec'] = t.tm_gmtoff
        ctx['tzname']        = t.tm_zone


    cb_create_gui_elements(
      fltk_application_context   = ctx,
      W                          = W,
      H                          = H,
      H_footer                   = H_footer,
      title                      = title,
      unlock_panzoom             = unlock_panzoom,
      features                   = features,
      cb_displayed_image         = cb_displayed_image,
      cb_status_value            = cb_status_value,
      log_readwrite_context      = dict( logged_images     = logged_images,
                                         logdir_write      = logdir_write,
                                         logdir_read       = logdir_read,
                                         image_path_prefix = image_path_prefix,
                                         image_directory   = image_directory,
                                         file_log          = file_log,
                                         replay_from_frame = replay_from_frame,
                                         jpg               = jpg,
                                        ))


    for icam in range(Ncameras):
        if ctx['image_view_groups'][icam].camera is not None:
            ctx['image_view_groups'][icam].set_up_image_capture(# don't auto-recur. I do that myself,
                                                                # making sure ALL the cameras are processed
                                                                period         = None,
                                                                flip_x         = flip_x,
                                                                flip_y         = flip_y,
                                                                ### image_callback__cookie
                                                                icam = icam,
                                                                # pieces of the log_readwrite_context
                                                                logged_images = logged_images,
                                                                logdir_write  = logdir_write,
                                                                file_log      = file_log,
                                                                jpg           = jpg,
                                                                **ctx)
    ctx['window'].show()

    if logdir_read is None:
        # request the initial frame; will recur in image_callback
        for image_view_group in ctx['image_view_groups']:
            image_view_group.camera.request()
    else:
        time_slider_select(logged_images     = logged_images,
                           logdir_write      = logdir_write,
                           logdir_read       = logdir_read,
                           image_path_prefix = image_path_prefix,
                           image_directory   = image_directory,
                           **ctx)

    return ctx
