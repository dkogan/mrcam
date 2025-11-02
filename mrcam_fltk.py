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

# upstream importers can get everything with 'import mrcam_fltk'
from mrcam          import *
from mrcam_argparse import *

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



class Fl_mrcam_image(Fl_Gl_Image_Widget):
    r'''An image-display class that can tie pan/zoom for sibling widgets'''

    def __init__(self,
                 *args,
                 locked_panzoom_widgets= None,
                 # These may be None, to disable a few features
                 flip_x                = False,
                 flip_y                = False,
                 status_widget         = None,
                 icam                  = 0):

        self.do_equalize_fieldscale = False
        self.locked_panzoom_widgets = locked_panzoom_widgets
        self.status_widget          = status_widget
        self.icam                   = icam
        self.flip_x                 = flip_x
        self.flip_y                 = flip_y

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
        x = super().__init__(*args)
        self.visible_focus(0)
        return x

    def try_handle_move(self, event):
        if self.status_widget is not None and \
           event == FL_MOVE:
            try:
                q = self.map_pixel_image_from_viewport( (Fl.event_x(),Fl.event_y()), )

                pixel_value_text = ''

                if self.image is not None:
                    qint_x = round(q[0])
                    qint_y = round(q[1])

                    (image_height,image_width) = self.image.shape[:2]
                    if qint_x >= 0 and qint_x < image_width and \
                       qint_y >= 0 and qint_y < image_height:

                        pixel_value_text = f",{self.image[qint_y,qint_x,...]}"

                self.status_widget.value( self.status_value(q, pixel_value_text) )
            except:
                self.status_widget.value( self.status_value(None, None) )


    def handle(self, event,
               # a custom override might set this to false
               call_super = True):
        # default implementation; meant to be overridden and extended

        self.try_handle_move(event)

        if event == FL_KEYUP:
            if Fl.event_key() == ord('e') or \
               Fl.event_key() == ord('E'):

                self.do_equalize_fieldscale = not self.do_equalize_fieldscale
                self.update(self.image)
                return super().handle(event) # to keep handling the events, so
                                             # that the other widgets see this

        if call_super:
            return super().handle(event)
        else:
            return None


    def displayed_image(self,
                        image,
                        *,
                        same_image = False):

        # default implementation; meant to be overridden and extended
        if image is None:
            return None

        if image.itemsize == 1:
            # 8-bit image. Display as is
            return image

        # Deep image. Display as a heat map
        if image.ndim > 2:
            raise Exception("high-depth color images not supported yet")

        if not same_image:
            self.image_for_display_no_fieldscale  = None
            self.image_for_display_yes_fieldscale = None

        if not self.do_equalize_fieldscale:
            if same_image and self.image_for_display_no_fieldscale is not None:
                return self.image_for_display_no_fieldscale

            q = 5
            a_min = np.percentile(image, q = q)
            a_max = np.percentile(image, q = 100-q)
            self.image_for_display_no_fieldscale = \
                mrcal.apply_color_map(image,
                                      a_min = a_min,
                                      a_max = a_max)
            return self.image_for_display_no_fieldscale
        else:
            if same_image and self.image_for_display_yes_fieldscale is not None:
                return self.image_for_display_yes_fieldscale

            self.image_for_display_yes_fieldscale = \
                mrcal.apply_color_map(mrcam.equalize_fieldscale(image),
                                      a_min = 0,
                                      a_max = 255)
            return self.image_for_display_yes_fieldscale


    def update(self,
               image):

        # We might be updating the display settings, managed in
        # displayed_image()
        same_image = image is self.image

        self.image = image

        # Will be None if the image was None (i.e. the capture failed)
        image_data = self.displayed_image(image,
                                          same_image = same_image)
        self.update_image(image_data = image_data,
                          flip_x     = self.flip_x,
                          flip_y     = self.flip_y)


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

        if self.image is None:
            return

        (image_height,image_width) = self.image.shape[:2]

        if not panzoom_siblings or \
           self.locked_panzoom_widgets is None:

            if not ratios:
                return super().set_panzoom(x_centerpixel, y_centerpixel,
                                           visible_width_pixels)
            else:
                return super().set_panzoom(x_centerpixel        * image_width,
                                           y_centerpixel        * image_height,
                                           visible_width_pixels * image_width)

        # All the widgets should pan/zoom together
        if not ratios:
            return \
                all( w. \
                     set_panzoom(x_centerpixel        / image_width,
                                 y_centerpixel        / image_height,
                                 visible_width_pixels / image_width,
                                 panzoom_siblings = False,
                                 ratios           = True) \
                     for w in self.locked_panzoom_widgets )
        else:
            return \
                all( w. \
                     set_panzoom(x_centerpixel,
                                 y_centerpixel,
                                 visible_width_pixels,
                                 panzoom_siblings = False,
                                 ratios           = True) \
                     for w in self.locked_panzoom_widgets )

    def status_value(self, q, pixel_value_text):
        # default implementation; meant to be overridden and extended
        if q is not None:
            if pixel_value_text is not None:
                return f"{q[0]:.1f},{q[1]:.1f}{pixel_value_text}"
            else:
                return f"{q[0]:.1f},{q[1]:.1f}"
        else:
            return ""



h_control        = 30
h_control_footer = 30 # for the label below the widget; needed for some widgets only

class Fl_mrcam_image_group(Fl_Group):
    r'''The image widget itself and any genicam features that may be adjustable'''

    def __init__(self,
                 x,y,w,h,
                 *,
                 camera,
                 icam = 0,
                 # iterable of strings. Might contain regex; might contain
                 # annotations such as [log] (applied to all regex matches). Any
                 # feature name that doesn't exist EXACTLY as given will be
                 # re-tried as a regex
                 features          = (),
                 unlock_panzoom,

                 application,

                 # Custom classes
                 Fl_mrcam_image_custom = Fl_mrcam_image):

        super().__init__(x,y,w,h)

        self.camera      = camera
        self.iframe      = 0
        self.application = application

        if features: w_controls = 300
        else:        w_controls = 0

        self.image_widget = \
            Fl_mrcam_image_custom(x, y,
                                  w-w_controls, h,
                                  locked_panzoom_widgets = \
                                    None if unlock_panzoom else \
                                    application.image_view_groups, # may pass the group of widget here. The widgets don't exist yet
                                  flip_x        = application.flip_x_allcams[icam],
                                  flip_y        = application.flip_y_allcams[icam],
                                  icam          = icam,
                                  status_widget = application.status_widget)

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
                if 'log' in flags and \
                   any(x <= 0 for x in desc['bounds']):
                    print(f"Warning: Requested log-scale feature '{name}' has non-positive bounds: {desc['bounds']}. Log-scale features must have strictly positive bounds; not adding widget for this feature")
                    continue

                if desc['unit']: label = f"{name} ({desc['unit']})"
                else:            label = name
                h_here = h_control + h_control_footer
                widget = Fl_Value_Slider(x + w-w_controls, y,
                                         w_controls, h_control,
                                         label)
                widget.align(FL_ALIGN_BOTTOM)
                widget.type(FL_HORIZONTAL)

                if 'log' in flags:
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


    def set_up_image_capture(self,
                             *,
                             # if given, we automatically recur. Otherwise it is
                             # expected that the image_callback will request the
                             # next set of frames, if needed
                             period = None):

        if self.camera is None:
            return

        def callback_mrcam(fd):

            frame = self.camera.requested_image()

            self.application.image_received_from_mrcam(iframe = self.iframe,
                                                       frame  = frame,
                                                       icam   = self.image_widget.icam)
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

    # So that Fl_mrcam_image can take locked_panzoom_widgets or locked_panzoom_groups
    def set_panzoom(self, *args, **kwargs):
        return self.image_widget.set_panzoom(*args, **kwargs)

class Fl_mrcam_application:

    def __init__(self,
                 camera_params_noname_nodims,
                 camera_names,
                 *,
                 dims            = None,
                 utcoffset_hours = None,
                 W               = 1280,
                 H               = 1024,
                 H_footer        = 30,
                 title = "mrcam stream",
                 flip_x_allcams    = False,
                 flip_y_allcams    = False,
                 unlock_panzoom    = False,
                 features          = (),
                 period            = 1.0,

                 # logging stuff
                 logdir_write,
                 logdir_read,
                 replay_from_frame,
                 jpg,
                 image_path_prefix,
                 image_directory,

                 # Custom classes
                 Fl_mrcam_image_group_custom = Fl_mrcam_image_group,
                 Fl_mrcam_image_custom       = Fl_mrcam_image
                 ):

        Ncameras = len(camera_names)

        self.log_readwrite_init(camera_names,
                                logdir_write      = logdir_write,
                                logdir_read       = logdir_read,
                                replay_from_frame = replay_from_frame,
                                jpg               = jpg,
                                image_path_prefix = image_path_prefix,
                                image_directory   = image_directory)

        self.cameras                  = [None] * Ncameras
        self.image_view_groups        = [None] * Ncameras
        self.flip_x_allcams           = flip_x_allcams
        self.flip_y_allcams           = flip_y_allcams
        self.period                   = period
        self.Ncameras_seen_iframe     = dict()
        self.logged_image_from_iframe = dict()

        if self.logdir_read is None:
            # I init each camera. If we're sending the TTYS0 trigger signal, I want all
            # the cameras to be ready when the trigger comes in. Thus the LAST camera
            # will send the trigger; I set the rest to EXTERNAL triggering in that case
            camera_params_noname_nodims = dict(camera_params_noname_nodims) # make a copy
            for i,name in reversed(list(enumerate(camera_names))):
                self.cameras[i] = \
                    mrcam.camera(name = name,
                                 dims = dims[i] if dims is not None else None,
                                 **camera_params_noname_nodims)

                if camera_params_noname_nodims['trigger'] == 'HARDWARE_TTYS0':
                    camera_params_noname_nodims['trigger'] = 'HARDWARE_EXTERNAL'



        if utcoffset_hours is not None:
            self.utcoffset_sec = utcoffset_hours*3600
            self.tzname        = f"{'-' if utcoffset_hours<0 else ''}{int(abs(utcoffset_hours)):02d}:{round( (abs(utcoffset_hours) % 1)*60 ):02d}"
        else:
            import time
            t = time.localtime()
            self.utcoffset_sec = t.tm_gmtoff
            self.tzname        = t.tm_zone


        self.create_gui_elements(
            W                           = W,
            H                           = H,
            H_footer                    = H_footer,
            title                       = title,
            unlock_panzoom              = unlock_panzoom,
            features                    = features,
            Fl_mrcam_image_group_custom = Fl_mrcam_image_group_custom,
            Fl_mrcam_image_custom       = Fl_mrcam_image_custom)


        for icam in range(Ncameras):
            if self.image_view_groups[icam].camera is not None:
                self.image_view_groups[icam].set_up_image_capture(# don't auto-recur. I do that myself,
                                                                  # making sure ALL the cameras are processed
                                                                  period = None)
        self.window.show()

        if self.logdir_read is None:
            # request the initial frame; will recur in image_callback
            for image_view_group in self.image_view_groups:
                image_view_group.camera.request()
        else:
            self.time_slider_select()



    def log_readwrite_init(self,
                           camera_names,
                           *,
                           logdir_read,
                           replay_from_frame,
                           logdir_write,
                           jpg,
                           image_path_prefix,
                           image_directory):

        self.logdir_read       = logdir_read
        self.replay_from_frame = replay_from_frame
        self.logdir_write      = logdir_write
        self.jpg               = jpg
        self.image_path_prefix = image_path_prefix
        self.image_directory   = image_directory

        # default
        self.file_log      = None
        self.logged_images = None

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
                print("These should all be identical",
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
            self.logged_images = log

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
                self.file_log = open(path, "w")
            except Exception as e:
                print(f"Error opening log file '{path}' for writing: {e}",
                      file=sys.stderr)
                sys.exit(1)

            for i,c in enumerate(camera_names):
                if c is not None:
                    print(f"## Camera {i}: {c}", file=self.file_log)
            self.write_logline("# time iframe icam imagepath")

            # I can replay this log as I write it. 'logged_images' is set for both
            # reading and writing
            self.logged_images = []




    def create_gui_elements(self,
                            *,
                            W,
                            H,
                            H_footer,
                            title,
                            unlock_panzoom,
                            features,
                            Fl_mrcam_image_group_custom,
                            Fl_mrcam_image_custom):

        # default implementation; meant to be overridden and extended

        H_footers = H_footer
        if self.logged_images is not None:
            H_footers += 2*H_footer

        kwargs = dict(W                           = W,
                      H                           = H,
                      H_image_views               = H - H_footers,
                      W_image_views               = W,
                      H_footer                    = H_footer,
                      title                       = title,
                      unlock_panzoom              = unlock_panzoom,
                      features                    = features,
                      Fl_mrcam_image_group_custom = Fl_mrcam_image_group_custom,
                      Fl_mrcam_image_custom       = Fl_mrcam_image_custom)

        self.create_gui_window     (**kwargs)
        self.create_gui_time_slider(**kwargs)
        self.create_gui_status     (**kwargs)
        self.create_gui_body       (**kwargs)
        self.finish_gui_window     (**kwargs)


    def create_gui_window(self,
                          *,
                          W,
                          H,
                          title,
                          # extra uneeded stuff
                          **kwargs):
        # default implementation; meant to be overridden and extended
        self.window = Fl_Window(W,H, title)


    def create_gui_time_slider(self,
                               *,
                               W,
                               H_image_views,
                               H_footer,
                               # extra uneeded stuff
                               **kwargs):
        # default implementation; meant to be overridden and extended
        if self.logged_images is not None:
            self.time_slider_widget = \
                Fl_Slider(0, H_image_views,
                          W,H_footer)
            self.time_slider_widget.align(FL_ALIGN_BOTTOM)
            self.time_slider_widget.type(FL_HORIZONTAL)
            self.time_slider_widget.step(1)
            if len(self.logged_images) > 0:
                self.time_slider_widget.bounds(0, len(self.logged_images)-1)
                self.time_slider_widget.value(self.replay_from_frame)
            else:
                self.time_slider_widget.bounds(0, 0)
                self.time_slider_widget.value(0)
            self.time_slider_widget.callback( \
                lambda *args: \
                    self.time_slider_select())
        else:
            self.time_slider_widget = None


    def create_gui_status(self,
                          *,
                          W,
                          H,
                          H_footer,
                          # extra uneeded stuff
                          **kwargs):
        # default implementation; meant to be overridden and extended
        self.status_widget = Fl_Output(0, H-H_footer,
                                       W, H_footer)
        self.status_widget.value('')

        # I want the status widget to be output-only and not user-focusable. This will
        # allow keyboard input to not be sent to THIS status widget, so that left/right
        # and 'u' go to the time slider and image windows respectively.
        self.status_widget.visible_focus(0)


    def create_gui_body(self,
                        *,
                        W_image_views,
                        H_image_views,
                        unlock_panzoom,
                        features,
                        Fl_mrcam_image_group_custom,
                        Fl_mrcam_image_custom,
                        # extra uneeded stuff
                               **kwargs):
        # default implementation; meant to be overridden and extended

        Ncameras = len(self.image_view_groups)
        Ngrid = math.ceil(math.sqrt(Ncameras))
        Wgrid = Ngrid
        Hgrid = math.ceil(Ncameras/Wgrid)
        w_image = W_image_views // Wgrid
        h_image = H_image_views // Hgrid

        icam = 0
        y0   = 0
        self.image_views = Fl_Group(0, 0, W_image_views, H_image_views)
        for i in range(Hgrid):
            x0 = 0

            for j in range(Wgrid):
                self.image_view_groups[icam] = \
                    Fl_mrcam_image_group_custom(x0,y0,
                                                w_image if j < Wgrid-1 else (W_image_views-x0),
                                                h_image if i < Hgrid-1 else (H_image_views-y0),
                                                camera                       = self.cameras[icam],
                                                icam                         = icam,
                                                features                     = features,
                                                unlock_panzoom               = unlock_panzoom,
                                                application                  = self,
                                                Fl_mrcam_image_custom        = Fl_mrcam_image_custom)
                x0   += w_image
                icam += 1

                if icam == Ncameras:
                    break
            if icam == Ncameras:
                break

            y0 += h_image
        self.image_views.end()


    def finish_gui_window(self,
                          # extra uneeded stuff
                          **kwargs):
        # default implementation; meant to be overridden and extended
        self.window.resizable(self.image_views)
        self.window.end()



    def time_slider_select(self):
        i_iframe = round(self.time_slider_widget.value())

        try:
            record = self.logged_images[i_iframe]
        except IndexError:
            print(f"WARNING: {i_iframe=} is out-of-bounds in logged_images: {len(self.logged_images)=}. This is a bug")
            return

        # shape (Ncameras,); all of these
        times   = record['time']
        iframes = record['iframe']

        self.time_slider_update_label(iframe = iframes[0],
                                      time   = times[0])
        self.update_all_images_from_replay()

        # if live-updating we color the slider green
        if self.logdir_write is not None:
            if int(self.time_slider_widget.value()) == int(self.time_slider_widget.maximum()):
                self.time_slider_widget.color(FL_GREEN)
            else:
                self.time_slider_widget.color(FL_BACKGROUND_COLOR)

    def time_slider_update_label(self,
                                 *,
                                 iframe,
                                 time):
        t = int(time + self.utcoffset_sec)
        t = datetime.datetime.fromtimestamp(t, datetime.UTC)
        self.time_slider_widget.label(f"iframe={iframe}/{int(self.time_slider_widget.maximum())} timestamp={time:.03f} {t.strftime('%Y-%m-%d %H:%M:%S')} {self.tzname}")



    def update_all_images_from_replay(self):
        i_iframe = round(self.time_slider_widget.value())

        try:
            record = self.logged_images[i_iframe]
        except IndexError:
            print(f"WARNING: {i_iframe=} is out-of-bounds in logged_images: {len(self.logged_images)=}. This is a bug")
            return

        Ncameras = len(self.image_view_groups)
        for icam in range(Ncameras):
            path = self.complete_path(record['imagepath'][icam])
            if path is None:
                image = None # write an all-black image

            else:
                try:
                    image = mrcal.load_image(path)
                except:
                    image = None

            self.image_view_groups[icam].image_widget.update(image)


    def image_received_from_mrcam(self,
                                  *,
                                  iframe,
                                  frame, # dict from requested_image()

                                  # All these are the cookie given to set_up_image_capture()
                                  icam):
        r'''Process the image

On return, we will push_buffer(frame['buffer']). If we do not want that (i.e. if
we will do that ourselves, set frame['buffer'] to None)

        '''

        Ncameras = len(self.image_view_groups)

        extension      = "jpg" if self.jpg else "png"

        image          = frame['image']
        timestamp      = frame['timestamp']
        buffer         = frame['buffer']
        off_decimation = frame['off_decimation']

        if not off_decimation:
            if image is None:
                print("Error capturing the image. I will try again",
                      file=sys.stderr)

            time_slider_at_max = False
            if self.logdir_write is not None:

                time_slider_now = int(self.time_slider_widget.value())
                time_slider_at_max = \
                    time_slider_now == int(self.time_slider_widget.maximum())

                if not iframe in self.logged_image_from_iframe:
                    # Started this set of cameras. Add empty record; fill it in as I get
                    # frames.
                    #
                    # It is very strange to have a list of iframes here: each iframe
                    # in the list will be the same. I do that because that's what I
                    # get when I read logs, and I want to use the same code path
                    # for the read- and write-log cases. This could be simplified if
                    # I adjust the results in log_readwrite_init() to remove the
                    # duplicated iframes
                    self.logged_image_from_iframe[iframe] = dict(time      = [None] * Ncameras,
                                                                        imagepath = [None] * Ncameras,
                                                                        iframe    = [None] * Ncameras)
                    self.logged_images.append( self.logged_image_from_iframe[iframe] )

                # write image to disk
                filename = f"frame{iframe:05d}-cam{icam}.{extension}"
                path = f"{self.logdir_write}/{filename}"

                if image is None:
                    self.write_logline(f"{timestamp:.3f} {iframe} {icam} -")
                else:
                    self.write_logline(f"{timestamp:.3f} {iframe} {icam} {filename}")

                    self.image_view_groups[icam].camera. \
                        async_save_image_and_push_buffer(path,image,frame['buffer'])
                    frame['buffer'] = None # indicate that the caller should NOT re-push the buffer

                self.logged_image_from_iframe[iframe]['time'     ][icam] = timestamp
                self.logged_image_from_iframe[iframe]['iframe'   ][icam] = iframe
                if image is not None:
                    self.logged_image_from_iframe[iframe]['imagepath'][icam] = path
                    # Otherwise, leave at None

            if self.logdir_write is None or time_slider_at_max:
                self.image_view_groups[icam].image_widget.update(image = image)

                self.image_view_groups[icam].sync_feature_widgets()


        # schedule the next set of images; do this even if off_decimation
        if not iframe in self.Ncameras_seen_iframe:
            self.Ncameras_seen_iframe[iframe] = 1
        else:
            self.Ncameras_seen_iframe[iframe] += 1

        # I need a Ncameras_seen_iframe to be a dict instead of a count for the
        # last frame, because in a free-runnning mode, I may get frames out of
        # the usual expected order
        if self.Ncameras_seen_iframe[iframe] >= Ncameras:
            # Every camera reported back. Finish up and ask for another frame
            del self.Ncameras_seen_iframe[iframe]

            if not off_decimation and self.time_slider_widget is not None:
                self.time_slider_widget.maximum(iframe)

                if time_slider_at_max:
                    # We were at the end of the time slider. Update us so that we're
                    # still at the end
                    self.time_slider_widget.value(iframe)
                    self.time_slider_update_label(iframe = iframe,
                                                         time   = timestamp)
                else:
                    self.time_slider_update_label(iframe = time_slider_now,
                                                         time   = self.logged_images[time_slider_now]['time'][0])
                    # The bounds changed, so the handle should be redrawn
                    self.time_slider_widget.redraw()

            def request_image_set():
                for image_view_group in self.image_view_groups:
                    image_view_group.camera.request()
            schedule_next_frame(request_image_set,
                                self.image_view_groups[0].camera.timestamp_request_us/1e6, self.period)


    def write_logline(self,l):
        if self.file_log is not None:
            print(l,file=self.file_log)
            self.file_log.flush()


    def complete_path(self, path):
        if path is None or path == '-':
            return None

        if self.logdir_write is not None:
            # We're logging; we already have the full path
            return path

        if self.image_path_prefix is not None:
            return f"{self.image_path_prefix}/{path}"
        if self.image_directory is not None:
            return f"{self.image_directory}/{os.path.basename(path)}"
        if path[0] != '/':
            # The image filename has a relative path. I want it to be
            # relative to the log directory
            if self.logdir_read is not None:
                return f"{self.logdir_read}/{path}"
            raise Exception("We're replaying but both logdir and replay are None. This is a bug")

        return path
