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
                        1.0sec/frame. If set to <= 0, we capture frames as
                        quickly as possible; with no frame-frame delay''')
    parser.add_argument('--features',
                        help='''A comma-separated list of features for which GUI
                        controls should be displayed. The available features can
                        be queried with "arv-tool-0.8". Each feature can be
                        postfixed with [log] to indicate that a log-scale widget
                        should be used. Each feature can be specified as a regex
                        to pick multiple features at once. If the regex matches
                        any category, everything in that category will be
                        selected''')
    parser.add_argument('--single-buffered',
                        action='store_true',
                        help='''By default the image display is double-buffered
                        to avoid flickering. Some graphics hardare (in
                        particular my old i915-based system) are buggy, and
                        don't work right in this mode, so --single-buffered is
                        available to disable double-buffering to work around
                        those bugs''')

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
    parser.add_argument('--acquisition-mode',
                        # should match the LIST_OPTIONS macro in mrcam-test.c
                        # and camera_init() in mrcam-pywrap.c
                        default = "SINGLE_FRAME",
                        help='''The acquisition mode. If omitted, we use
                        "SINGLE_FRAME". Pass any invalid mode (like "") to get a
                        list of valid values on stderr.''')
    parser.add_argument('--acquisition-persistent',
                        action='store_true',
                        help='''If given, we start the acquisition once, and
                        keep it active as the process runs. Usually you want
                        this in the CONTINUOUS acquisition mode, but different
                        cameras may demand different things''')
    parser.add_argument('--trigger',
                        # should match the LIST_OPTIONS macro in mrcam-test.c
                        # and camera_init() in mrcam-pywrap.c
                        default = "SOFTWARE",
                        help='''The trigger mode. If omitted, we use "SOFTWARE". Pass any invalid mode (like
                        "") to get a list of valid values on stderr.''')
    parser.add_argument('--time-decimation-factor',
                        # should match the LIST_OPTIONS macro in mrcam-test.c
                        # and camera_init() in mrcam-pywrap.c
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


def displayed_image(image):
    if image.itemsize == 1:
        # 8-bit image. Display as is
        return image

    # Deep image. Display as a heat map
    if image.ndim > 2:
        raise Exception("high-depth color images not supported yet")
    q = 5
    a_min = np.percentile(image, q = q)
    a_max = np.percentile(image, q = 100-q)
    return mrcal.apply_color_map(image,
                                 a_min = a_min,
                                 a_max = a_max)


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
                 single_buffered   = False,
                 status_widget     = None,
                 handle_extra      = None,
                 image_view_groups = None):

        super().__init__(x,y,w,h)


        self.camera = camera
        self.iframe = 0

        if features: w_controls = 300
        else:        w_controls = 0

        if status_widget is None:
            # no global status bar; create one here
            h_status_here = h_status
        else:
            # use a global status bar
            h_status_here = 0


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

                    self.status_widget.value(f"{q[0]:.1f},{q[1]:.1f}{pixel_value_text}")
                except:
                    self.status_widget.value("")

            if handle_extra is not None:
                return handle_extra(self_image_widget,event)

            return None # Use parent's return code

        self.image_widget = \
            Fl_Gl_Image_with_handle(x, y,
                                    w-w_controls, h-h_status_here,
                                    handler               = handle_image_widget,
                                    double_buffered       = not single_buffered,
                                    locked_panzoom_groups = image_view_groups)
        if status_widget is None:
            self.status_widget = Fl_Output(x, y + h-h_status_here, w, h_status_here)
            # We're making our own status widget. I'd like it to be output-only
            # and not user-focusable. This will allow keyboard input to not be
            # sent to THIS widget, so that left/right and 'u' go to the time
            # slider and image windows respectively. For status widgets that
            # were PASSED to us, we make the caller do this; because they might
            # actually want to do something different
            self.status_widget.visible_focus(0)
        else:
            self.status_widget = status_widget

        # Need group to control resizing: I want to fix the sizes of the widgets in
        # the group, so I group.resizable(None) later
        group = Fl_Group(x + w-w_controls, y,
                         w_controls, h-h_status_here)

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
                            flip_x,
                            flip_y):

        self.image_widget.image = image

        if image is None:
            # black image
            self.image_widget.update_image(image_data = None,
                                           flip_x     = flip_x,
                                           flip_y     = flip_y)
            return

        self.image_widget.update_image(image_data = displayed_image(image),
                                       flip_x     = flip_x,
                                       flip_y     = flip_y)

        self.sync_feature_widgets()


    def set_up_image_capture(self,
                             *,
                             # if given, we automatically recur. Otherwise it is
                             # expected that the image_callback will request the
                             # next set of frames, if needed
                             period                   = None,
                             flip_x                   = False,
                             flip_y                   = False,
                             auto_update_image_widget = True,

                             # guaranteed to be called with each frame; even on error
                             image_callback = None,
                             **image_callback_cookie):

        if self.camera is None:
            return

        def callback_mrcam(fd):

            frame = self.camera.requested_image()

            if auto_update_image_widget and not frame['off_decimation']:
                if frame['image'] is None:
                    print("Error capturing the image. I will try again",
                          file=sys.stderr)

                self.update_image_widget( image  = frame['image'],
                                          flip_x = flip_x,
                                          flip_y = flip_y )

            if image_callback is not None:
                image_callback(frame['image'],
                               timestamp      = frame['timestamp'],
                               iframe         = self.iframe,
                               off_decimation = frame['off_decimation'],
                               **image_callback_cookie)

            if not frame['off_decimation']:
                self.camera.callback_done_with_buffer(frame['buffer'])
                self.iframe += 1

            if period is not None:
                # Ask for some data in the future
                schedule_next_frame(self.camera.request, period)


        # Tell FLTK to callback_mrcam() when data is available
        Fl.add_fd( self.camera.fd_image_ready,
                   callback_mrcam )

        if period is not None:
            # Ask for some data
            self.camera.request()

