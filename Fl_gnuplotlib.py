#!/usr/bin/python3

# A proof-of-concept gnuplot-based plotting widget for FLTK. Not "done", but
# quite functional. See Fl_gnuplotlib_demo.py and this blog post:
#
# http://notes.secretsauce.net/notes/2022/10/17_gnuplot-output-in-an-fltk-widget.html

import sys
import fltk
import gnuplotlib as gp


class Fl_Gnuplotlib_Window(fltk.Fl_Window):

    def __init__(self, x,y,w,h, **plot_options):
        super().__init__(x,y,w,h)
        self.end()

        self._plot                 = None
        self._delayed_plot_options = None

        self.init_plot(**plot_options)

    def init_plot(self, **plot_options):
        if 'terminal' in plot_options:
            raise Exception("Fl_Gnuplotlib_Window needs control of the terminal, but the user asked for a specific 'terminal'")

        if self._plot is not None:
            self._plot = None

        self._delayed_plot_options = None

        # Might fail if a buggy pyfltk is used:
        #   https://sourceforge.net/p/pyfltk/mailman/pyfltk-user/thread/87wmpllwt3.fsf%40secretsauce.net/#msg58754412
        xid = fltk.fl_xid(self)
        if xid == 0:
            # I don't have an xid (yet?), so I delay the init
            self._delayed_plot_options = plot_options
            return

        # will barf if we already have a terminal
        gp.add_plot_option(plot_options,
                           terminal = f'x11 window "0x{xid:x}"')

        self._plot = gp.gnuplotlib(**plot_options)

    def plot(self, *args, **kwargs):

        if self._plot is None:
            if self._delayed_plot_options is None:
                raise Exception("plot has not been initialized")

            self.init_plot(**self._delayed_plot_options)
            if self._plot is None:
                raise Exception("plot has not been initialized. Delayed initialization failed")

        self._plot.plot(*args, **kwargs)
