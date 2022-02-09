import sys
import os
import argparse
from vispy import app
import scipy.signal as sig
from datetime import datetime

from estm import CsvSource, DataCollector, VisualizationCanvas, buffer_to_numpy, CsvWriter, OpenBciSource
from rx.scheduler import ThreadPoolScheduler
from rx import operators as ops

def err(e):
    print(repr(e), file = sys.stderr)

def wire(args, source, writer = None):
    sampling = int(args.sampling)

    #flt = sig.firwin(301, fs = sampling, window='blackman', cutoff = [48.0, 52.0], pass_zero = True)
    flt = sig.firwin(301, fs = sampling, window='blackman', cutoff = 48.0, pass_zero = True)

    def apply_filter(d):
        return None
        print("shape", d[-1].shape)
        res = d[:-1] + [sig.filtfilt(b=flt, a=1, x=d[-1], axis=0)]
        print("done")
        return res

    pool_scheduler = ThreadPoolScheduler(4)

    collector = DataCollector(capacity = 10*sampling, writer = writer)
    source.subject.subscribe(on_next = lambda row: collector.on_raw_datum(row[0], row[1:]), on_error = lambda err: print(repr(err)))

    def on_next_refined(row):
        try:
            pass#collector.on_refined_data(row[0], row[1], row[2])
        except:
            print(sys.exc_info(), file = sys.stderr)

    #skip = 320
    skip = 800
    source.subject.pipe(
            ops.observe_on(pool_scheduler),
            ops.map(lambda row: [row[0]] + row[2:]),
            ops.buffer_with_count(1600, skip=800),
            ops.map(buffer_to_numpy),
            ops.map(apply_filter)
    ).subscribe(on_next = on_next_refined, on_error = err)

    vis = VisualizationCanvas(collector)
    source.start()

    app.run()

def record(args):
    dirname = 'data/rec_{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())
    os.mkdir(dirname)

    backup_writer = CsvWriter(os.path.join(dirname, 'backup.csv'))
    source = OpenBciSource(writer = backup_writer)

    writer = CsvWriter(os.path.join(dirname, 'signal.csv'))
    wire(args, source, writer)

def replay(args):
    source = CsvSource(args.path)
    
    wire(args, source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = "nice.py")
    parser.add_argument('--sampling', type = int, help = 'sampling rate')
    subparsers = parser.add_subparsers(help='sub-command help')

    replay_parser = subparsers.add_parser('replay', help = 'replay help')
    replay_parser.add_argument('path', type = str, help = 'path help')
    replay_parser.set_defaults(func = replay)

    record_parser = subparsers.add_parser('record', help = 'record help')
    record_parser.set_defaults(func = record)

    args = parser.parse_args()
    print(repr(args))
    args.func(args)
