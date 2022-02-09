from vispy import gloo
from vispy import app

import numpy as np
from time import time

VERT_SHADER = """
#version 120

// y coordinate of the position.
attribute float a_time;
attribute float a_value;
attribute float a_yspan;

// row, col, and time index.
attribute vec2 a_index;
varying vec2 v_index;

// 2D scaling factor (zooming).
uniform float u_xspan;
uniform float u_now;

// Size of the table.
uniform vec2 u_size;

// Number of samples per signal.
uniform float u_n;

// Color.
attribute vec3 a_color;
varying vec4 v_color;

// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;

void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;

    // Compute the x coordinate from the time index.
    float x = 1 - 2 * (u_now - a_time) / u_xspan ;
    vec2 position = vec2(x, a_value / a_yspan);

    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows)*.9;
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
                  -1 + 2*(a_index.y+.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*position+b, 0.0, 1.0);

    v_color = vec4(a_color, 1.);
    v_index = a_index;

    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

FRAG_SHADER = """
#version 120

varying vec4 v_color;
varying vec2 v_index;

varying vec2 v_position;
varying vec4 v_ab;

void main() {
    gl_FragColor = v_color;

    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;

    // Clipping test.
    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    if ((test.x > 1) || (test.y > 1))
        discard;
}
"""

class VisualizationCanvas(app.Canvas):
    def __init__(self, data_source):
        app.Canvas.__init__(self, title='Use your wheel to zoom!',
                            keys='interactive')

        self.delay = None

        self.data_source = data_source

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        gloo.set_viewport(0, 0, *self.physical_size)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.program['u_xspan'] = 5.
        self.program['u_size'] = (2., 2.)

        self.program['a_yspan'] = np.zeros(1, dtype=np.float32)
        self.program['a_time'] = np.zeros(1, dtype=np.float32)
        self.program['a_value'] = np.zeros(1, dtype=np.float32)
        self.program['a_index'] = np.zeros((1, 2), dtype=np.float32)
        self.program['a_color'] = np.ones((1, 3), dtype=np.float32)

        self.set_data()

        self.show()

    def on_timer(self, event):
        self.set_data()

    def set_data(self):
        data = self.data_source.get_data()

        # print(data)

        if data:
            self.delay = data['tdelay']
            self.program['u_now'] = float(time() - (self.delay or 0.))

            self.program['u_xspan'] = 5.
            self.program['a_yspan'] = data['vspan']
            self.program['a_time'].set_data(data['time'].copy())
            self.program['a_value'].set_data(data['values'].copy())
            self.program['a_index'].set_data(data['index'].copy())
            self.program['a_color'].set_data(data['color'].copy())
            self.update()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        gloo.clear()
        self.program['u_now'] = float(time() - (self.delay or 0.))
        self.program.draw('line_strip')
