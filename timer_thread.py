import time
from threading import Timer
import datetime


def printHello():
    # print('TimeNow:%s' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('TimeNow:%s' % time.time())
    t = Timer(1, printHello)
    t.start()


if __name__ == "__main__":
    printHello()