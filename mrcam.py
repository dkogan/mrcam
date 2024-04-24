#!/usr/bin/python3

import sys
import re
from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget
from fltk import *
import mrcal
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
    parser.add_argument('--power-cycle-at-startup',
                        action='store_true',
                        help='''If given, we cycle the camera power before we do anything else. Implemented
                        ONLY if --trigger TTYS0''')
    parser.add_argument('--power-down-when-finished',
                        action='store_true',
                        help='''If given, we power-down the camera when we're done. Implemented ONLY if
                        --trigger TTYS0''')
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

    if args.features is not None:
        args.features = args.features.split(',')
    else:
        args.features = ()


class Fl_Gl_Image_Widget_Derived(Fl_Gl_Image_Widget):
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
                 handle_extra    = None):

        super().__init__(x,y,w,h)


        self.camera = camera
        self.iframe = 0

        if feature_names: w_controls = 300
        else:             w_controls = 0

        self.image_widget = Fl_Gl_Image_Widget_Derived(x, y,
                                                       w-w_controls, h-h_status,
                                                       group           = self,
                                                       double_buffered = not single_buffered,
                                                       handle_extra    = handle_extra)
        self.status_widget = Fl_Output(x, y + h-h_status, w, h_status)

        # Need group to control resizing: I want to fix the sizes of the widgets in
        # the group, so I group.resizable(None) later
        group = Fl_Group(x + w-w_controls, y,
                         w_controls, h-h_status)

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




    def set_up_image_capture(self,
                             *,
                             period         = None, # if given, we automatically recur
                             # guaranteed to be called with each frame; even on error
                             image_callback = None,
                             **image_callback_cookie):

        def callback_image_ready(fd):
            frame = self.camera.requested_image()

            image        = frame['image']
            timestamp_us = frame['timestamp_us']

            if image is not None:
                # Update the image preview; deep images are shown as a heat map
                if image.itemsize > 1:
                    if image.ndim > 2:
                        raise Exception("high-depth color images not supported yet")
                    self.image_widget.update_image(image_data = mrcal.apply_color_map(image))
                else:
                    self.image_widget.update_image(image_data = image)


                self.sync_feature_widgets()

                if image_callback is not None:
                    image_callback(image,
                                   iframe = self.iframe,
                                   **image_callback_cookie)
            else:
                print("Error capturing the image. I will try again",
                      file=sys.stderr)

                if image_callback is not None:
                    image_callback(None, # no image; error
                                   iframe = self.iframe,
                                   **image_callback_cookie)

            self.iframe += 1

            if period is not None:
                Fl.add_timeout(period, lambda *args: self.camera.request())

        Fl.add_fd( self.camera.fd_image_ready,
                   callback_image_ready )

        if period is not None:
            self.camera.request()

