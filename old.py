import pandas as pd
from openbci import wifi as bci
from datetime import datetime
import csv

from time import time, sleep
from threading import Thread
import numpy as np

class OpenBciConsumer(Thread):
    def __init__(self, csv_prefix='data_'):
        Thread.__init__(self)

        self.channels = 4
        self.time = PlotData()
        self.data = [PlotData() for _ in range(4)]
        self.dupa = None
        self.time_delay = None

        filename = '{}{:%Y-%m-%d_%H-%M-%S}.csv'.format(csv_prefix, datetime.now())
        self.csv_writer = csv.writer(open(filename, 'w'))
        self.csv_writer.writerow(['Time', 'Number'] + ['Ch{}'.format(i + 1) for i in range(4)])

    def run(self, ip_address='192.168.1.34', sample_rate=6400):
        self.shield = bci.OpenBCIWiFi(ip_address=ip_address, log=True, high_speed=True, sample_rate=sample_rate)

        print("WiFi Shield Instantiated")
        self.shield.start_streaming(self.process_data)

        self.shield.loop()

    def process_data(self, sample):
        if sample.sample_number == 0:
            print(sample.channel_data)
            print("time", sample.timestamp, "valid:", sample.valid, "error:", sample.error)
        if sample.valid and len(sample.channel_data) == self.channels:
            if self.dupa is None:
                self.dupa = sample.timestamp
            sample.timestamp -= self.dupa
            self.time.append(sample.timestamp / 1000.0)
            td = time() - sample.timestamp / 1000.0
            self.time_delay = min(td, self.time_delay) if self.time_delay is not None else td
            for data, v in zip(self.data, sample.channel_data):
                data.append(v)
            self.csv_writer.writerow([sample.timestamp, sample.sample_number] + sample.channel_data)

    def get_data(self):
        n = self.time.size()

        if not n:
            return None

        single_times = self.time.get(n)
        values = np.concatenate([vdata.get(n) for vdata in self.data])

        print("aaaa", n)

        return {
            'time': np.concatenate([single_times for _ in range(4)]),
            'values': values,
            'index': np.vstack([np.vstack([np.full(n, i // 2, dtype=np.float32), np.full(n, i % 2, dtype=np.float32)]).transpose() for i in range(4)]),
            'color': np.full((4*n, 3), 0.9, dtype=np.float32),
            'vspan': np.abs(values).max(),
            'tdelay': self.time_delay
        }

from vispy import gloo
from vispy import app

class PlotData(object):
    def __init__(self, growth_factor = 0.1, min_growth = 100):
        self._n = 0
        self._capacity = min_growth
        self._data = np.empty(self._capacity, np.float32)
        self.growth_factor = growth_factor
        self.min_growth = min_growth

    def grow(self):
        new_capacity = self._capacity + min(int(self._capacity * self.growth_factor), self.min_growth)
        new_data = np.empty(new_capacity, np.float32)
        new_data[:self._n] = self._data[:self._n]

        self._capacity = new_capacity
        self._data = new_data

    def append(self, v):
        if self._n >= self._capacity:
            self.grow()
        self._data[self._n] = v
        self._n += 1

    def size(self):
        return self._n

    def get(self, limit = None):
        if limit is None:
            limit = self._n
        elif limit > self._n:
            limit = self._n

        return self._data[:limit]

VERT_SHADER = """
#version 120

// y coordinate of the position.
attribute float a_time;
attribute float a_value;

// row, col, and time index.
attribute vec2 a_index;
varying vec2 v_index;

// 2D scaling factor (zooming).
uniform vec2 u_span;
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
    float x = 1 - 2 * (u_now - a_time) / u_span.x ;
    vec2 position = vec2(x, a_value / u_span.y);

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

        self.program['u_span'] = (5., 1.)
        self.program['u_size'] = (2., 2.)

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

        if data:
            self.delay = data['tdelay']
            self.program['u_now'] = float(time() - (self.delay or 0.))

            self.program['u_span'] = (5., data['vspan'])
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

if __name__ == "__main__":
    consumer = OpenBciConsumer()
    consumer.start()

    while not consumer.get_data():
        print("Waiting for connection to be active")
        sleep(1)

    print(consumer.get_data())

    print("=====================")

    c = VisualizationCanvas(consumer)
    app.run()
