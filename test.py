import matplotlib.pyplot as plt
from openbci import wifi as bci
from time import time
import logging
import csv

from threading import Thread

datas = [[0.0] * 4000 for _ in range(4)]
lines = []

def processData(sample):
    #print(sample.sample_number)
    if sample.sample_number == 0:
        print(sample.channel_data)

    global data
    global csvf
    if len(sample.channel_data) == 4:
        csvf.writerow([int(1000*time()), sample.sample_number] + sample.channel_data)
        for sp in range(4):
            datas[sp] = datas[sp][1:] + [sample.channel_data[sp]]


class Dupa(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        logging.basicConfig(filename="test.log",format='%(asctime)s - %(levelname)s : %(message)s',level=logging.DEBUG)
        logging.info('---------LOG START-------------')
        # If you don't know your IP Address, you can use shield name option
        # If you know IP, such as with wifi direct 192.168.4.1, then use ip_address='192.168.4.1'
        shield_name = 'OpenBCI-E218'
        shield = bci.OpenBCIWiFi(ip_address='10.14.22.141', log=True, high_speed=True, sample_rate=6400)
        print("WiFi Shield Instantiated")
        shield.start_streaming(processData)

        shield.loop()


if __name__ == '__main__':
    plt.show()

    for sp in range(4):
        plt.subplot(2, 2, sp + 1)
        axes = plt.gca()
        axes.set_xlim(0, 4000)
        axes.set_ylim(-0.02, 0.02)
        xdata = list(range(0, 4000))
        line, = axes.plot(xdata, datas[sp], 'r-')
        lines.append(line)

    global csvf
    csvf = csv.writer(open('data.csv', 'w'))
    csvf.writerow(['Time', 'Number'] + ['Ch{}'.format(i + 1) for i in range(4)])

    Dupa().start()

    while plt:
        for sp in range(4):
            lines[sp].set_ydata(datas[sp])
        plt.draw()
        plt.pause(1e-17)

