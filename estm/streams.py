import pandas as pd
from openbci import wifi as bci
from datetime import datetime
import csv

from time import time, sleep
from threading import Thread
import numpy as np

from rx.subject import Subject

class CsvSource(Thread):
    def __init__(self, path):
        Thread.__init__(self)
        self.csv = csv.reader(open(path, 'r'))
        next(self.csv)
        self.subject = Subject() 

    def start(self):
        self.start = time()
        self.i = 0
        Thread.start(self)

    def run(self):
        for row in self.csv:
            t = float(row[0])
            dt = max(0, t / 1000.0 - (time() - self.start))
            sleep(dt)

            data = [self.i, t, row[1]] + [float(x) for x in row[2:6]]
            self.subject.on_next(data)
            self.i += 1

        self.subject.dispose()

class OpenBciSource(Thread):
    def __init__(self, ip_address='10.14.22.121', sample_rate=6400, writer = None):
        Thread.__init__(self)

        self.ip_address = ip_address
        self.sample_rate = sample_rate
        self.channels = 4
        self.start_time = None
        self.time_delay = None
        self.writer = writer
        self.subject = Subject()
        self.i = 0

    def start(self):
        self.shield = bci.OpenBCIWiFi(ip_address=self.ip_address, log=True, high_speed=True, sample_rate=self.sample_rate)

        print("WiFi Shield Instantiated")
        self.shield.start_streaming(self.process_data)

        Thread.start(self)

    def run(self):
        self.shield.loop()

    def should_log(self, sample):
        return sample.sample_number == 0

    def process_data(self, sample):
        if self.should_log(sample):
            print(sample.channel_data)
            print("time", sample.timestamp, "valid:", sample.valid, "error:", sample.error)

        if sample.valid and len(sample.channel_data) == self.channels:
            if self.start_time is None:
                self.start_time = sample.timestamp
            sample.timestamp -= self.start_time

            t = sample.timestamp
            # td = time() - t
            # emit t, (sample.sample_number), *sample.channel_data
            self.subject.on_next([self.i, t, sample.sample_number] + sample.channel_data)
            if self.writer:
                self.writer.write([t], [sample.sample_number], [sample.channel_data], [[]])
            self.i += 1

def buffer_to_numpy(b):
    n = len(b)
    dim = 4
    dtype = np.float32
    data = np.zeros((n, dim), dtype)
    for i, row in enumerate(b):
        data[i] = row[2:]
    return [b[0][0], n, data]

class NumpyGrouper(object):
    def __init__(self, span = 301, skip = 301, dim = 4, dtype=np.float32):
        self.tdata = np.zeros(span, dtype)
        self.data = np.zeros((span, dim), dtype)
        self.pos = 0

    def on_data(self, d):
        if self.pos >= 0:
            p = self.pos
            self.tdata[p] = d[0]
            self.data[p][:dim] = d[1:(dim + 1)]

        self.pos += 1

        if self.pos >= span:
            #emit data
            old_tdata = self.tdata
            old_data = self.data
            self.tdata = np.zeros(span, dtype)
            self.data = np.zeros((span, dim), dtype)
            if self.skip < self.span:
                csize = self.span - self.skip
                self.tdata[:csize] = old_tdata[self.span:]
                self.data[:csize] = old_data[self.span:]
            self.pos -= self.skip

    def on_finished(self):
        pass


class DataCollector(object):
    def __init__(self, dim = 4, capacity = 3200, shift = 400, writer = None, dtype = np.float32):
        assert(dim == 4)
        assert(shift < capacity)
        self._dim = dim
        self._start = 0
        self._n = 0
        self._capacity = capacity
        self._shift = shift
        self._tdata = np.empty(self._capacity, np.float32)
        self._numdata = np.empty(self._capacity, np.int32)
        self._rdata = np.empty((self._capacity, dim), np.float32)
        self._data = np.empty((self._capacity, dim), np.float32)
        self._color = np.zeros((self._capacity, 3), np.float32)
        self._written = 0
        self.writer = writer

        self.time_delay = None

    def remodel(self):
        keys = ['_tdata', '_data', '_rdata', '_numdata', '_color']

        news = {}

        for key in keys:
            v = self.__getattribute__(key)
            shape = v.shape
            nv = np.empty(shape, v.dtype)

            nv[:(self._capacity - self._shift)] = v[self._shift:]
            news[key] = nv

        self._start += self._shift
        for key in keys:
            self.__setattr__(key, news[key])

    def should_remodel(self):
        return self._n - self._start >= self._capacity

    def on_raw_datum(self, index, value):
        assert(index == self._n)

        if self.should_remodel():
            self.remodel()

        td = time() - value[0] / 1000.0
        self.time_delay = min(td, self.time_delay) if self.time_delay is not None else td

        i = index - self._start

        self._tdata[i] = value[0]
        self._numdata[i] = value[1]
        self._data[i][:self._dim] = value[2:(2+self._dim)]
        self._rdata[i][:self._dim] = value[2:(2+self._dim)]
        self._color[i] = [0.6, 0, 0.6]

        self._n += 1

    def on_refined_data(self, index, length, values):
        print(index, length, self._n)
        assert(index >= self._start)
        assert(index + length <= self._n)

        i = index - self._start
        self._data[i:(i + length)] = values
        self._color[i:(i + length)] = [0.9, 0, 0.9]
        self.push_write(index + length)

    def push_write(self, write_end):
        assert (self._written >= self._start)
        assert (write_end <= self._n)

        def extract(arr):
            return arr[self._written : write_end]

        if self.writer:
            self.writer.write(extract(self._tdata), extract(self._numdata), extract(self._rdata), extract(self._data))

        self._written = write_end

    def get_data(self):
        n = self._n - self._start
        if n:
            d = self._data[:n]
            values = np.reshape(d, d.size, 'F')
            vspan = np.abs(d).max(0)
        else:
            values = np.empty(0, np.float32)
            vspan = np.ones((self._dim, 1), np.float32)

        return {
            'time': np.concatenate([self._tdata[:n] for _ in range(self._dim)]) / 1000.0,
            'values': values,
            'index': np.vstack([np.vstack([np.full(n, i // 2, dtype=np.float32), np.full(n, i % 2, dtype=np.float32)]).transpose() for i in range(4)]),
            'vspan': np.concatenate([np.full(n, v) for v in vspan]),
            'color': np.vstack([self._color[:n] for _ in range(self._dim)]),
            'tdelay': self.time_delay
        }

class CsvWriter(object):
    def __init__(self, filename = None, csv_prefix='data_'):
        if filename is None:
            filename = '{}{:%Y-%m-%d_%H-%M-%S}.csv'.format(csv_prefix, datetime.now())
        self._f = open(filename, 'w')
        self.csv_writer = csv.writer(self._f)
        self.csv_writer.writerow(['Time', 'Number'] + ['Ch{}'.format(i + 1) for i in range(4)] + ['Filt{}'.format(i + 1) for i in range(4)])

    def write(self, tdata, numdata, rdata, data):
        for i in range(len(tdata)):
            self.csv_writer.writerow([tdata[i]] + [numdata[i]] + list(rdata[i]) + list(data[i]))
        self._f.flush()
