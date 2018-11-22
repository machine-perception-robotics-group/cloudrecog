#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import csv
from collections import OrderedDict
import argparse
from cloud_yolo.msg import *
import time
import sys
import threading
import signal
import random
import numpy as np
# signal.signal(signal.SIGINT, signal.SIG_DFL)

def mean(list):
    return sum(list)/len(list)

class TimeTable:
    def __init__(self, filename):
        # レイヤ情報の読み込み
        self.layers = OrderedDict()
        with open(filename, "r") as f:
            reader = csv.reader(f)
            head = next(reader)  # headerの読み飛ばし

            for row in reader:
                self.layers.update({row[0]: float(row[1])})
        self.time_table = OrderedDict()

        # print
        print("#### Div layer info ####")
        for layer,fmapsize in self.layers.iteritems():
            print("{} {}".format(layer, fmapsize))
        print("####  ####")

    def getLayers(self):
        return self.layers


class curr_sub:
        def __init__(self, c):
            self.c = c
            self.sub = rospy.Subscriber('processing_time', processing_time, self.callback)

        def callback(self, data):
            # print dir(data)
            self.c.setTimes(data.div_layer, data.Robottime, data.Cloudtime, data.Communicationtime)

class Curr_table:
    def __init__(self):
        self.currlayer = str()
        self.Rt = []
        self.Ct = []
        self.Com = []
        self.launched = False

    def setTimes(self, currlayer, rt, ct, com):
        '''
        set All times
        :param rt: Robot time
        :param ct: Cloud time
        :param com: Communication time
        :return: Nothing
        '''
        if self.currlayer != currlayer:
            self.Rt = []
            self.Ct = []
            self.Com = []
            self.currlayer = currlayer

        self.Rt.append(rt)
        self.Ct.append(ct)
        self.Com.append(com)

        # print len(self.Rt)

        while(len(self.Rt) >= 20):
            self.launched = True
            self.Rt.pop(0)
            self.Ct.pop(0)
            self.Com.pop(0)

    def getTimes(self):
        return self.currlayer, self.Rt, self.Ct, self.Com

    def getAvg(self):
        return self.currlayer, mean(self.Rt), mean(self.Ct), mean(self.Com)


class Controller():
    def __init__(self, event, args, cur):

        self.cur = cur

        self.time_table = OrderedDict()
        tt = TimeTable(args.layers)

        self.layers = tt.getLayers()
        self.latency = 200
        while (not self.cur.launched):
            time.sleep(1)

        print("#### Start init ####")
        # 初期計測の開始
        for layer in self.layers.keys():
            rospy.set_param('div_layer', layer)
            counter = 0
            sys.stdout.write("\r measuring... {}/{}".format(layer, len(self.layers)))
            sys.stdout.flush()
            while True:
                time.sleep(5)
                currlayer, rt, ct, com = self.cur.getAvg()
                self.time_table.update({currlayer: [rt, ct, com]})
                counter += 1
                if(currlayer==layer or counter > 10):
                    break
        print("\n")
        # 初期計測の終了 print
        for k, v in self.time_table.iteritems():
            print k, v
        print("####  ####")

        while not event.isSet():
            time.sleep(5)
            self.updateTable()
            self.selectLayer()


    def updateTable(self):
        # print("update")
        currlayer, rt, ct, com = self.cur.getAvg()
        # print com
        unit = com / self.layers[currlayer]

        for layer in self.layers.keys():
            if (layer == currlayer):
                # 測定したレイヤの結果はそのままupdate
                self.time_table.update({currlayer: [rt, ct, com]})
            else:
                # 測定したレイヤ以外は特徴マップのサイズ比から通信時間の見積もり後update
                rt_t, ct_t, _com_t = self.time_table[layer]
                com_t = unit*self.layers[layer]
                self.time_table.update({layer:[rt_t, ct_t, com_t]})

    def selectLayer(self):
        self.latency = rospy.get_param('latency', self.latency)
        # まずはレイテンシを満たすレイヤを絞り込み
        satisfied = []
        sum_of_times=[]
        for layer in self.layers.keys():
            sum_of_time = sum(self.time_table[layer])
            sum_of_times.append(sum_of_time)
            # print("{}: {} {} {} {}".format(layer, self.time_table[layer][0]*1000, self.time_table[layer][1]*1000, self.time_table[layer][2]*1000, sum_of_time*1000))
            if (sum_of_time*1000 <= self.latency):
                satisfied.append(layer)
        if (len(satisfied) > 0):
            # 条件を満たすレイヤが見つかったとき-> この中でクラウドの負荷減らす
            # print(satisfied)
            div_layer = satisfied[-1]
            rospy.set_param('div_layer', div_layer)
        else:
            # 条件を満たすレイヤが見つからなかったとき-> レイテンシが最小になるようにする

            div_layer = self.layers.items()[np.argmin(sum_of_times)][0]
            rospy.set_param('div_layer', div_layer)
        print("#### L= {:4.2f} ####".format(self.latency))
        for idx, layer in enumerate(self.layers.keys()):
            if (layer in satisfied) and (layer == div_layer):
                tp = "*"
            elif (layer in satisfied):
                tp = "o"
            elif (layer == div_layer):
                tp = "+"
            else:
                tp = ""
            # sys.stdout.write("{:5s}: {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:1s}\n".format(layer, self.time_table[layer][0]*1000, self.time_table[layer][1]*1000, self.time_table[layer][2]*1000, sum_of_times[idx]*1000, tp))
            print("{:5s}: {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:1s}".format(layer, self.time_table[layer][0]*1000, self.time_table[layer][1]*1000, self.time_table[layer][2]*1000, sum_of_times[idx]*1000, tp))
        print("")


def sighandler(event, signr, handler):
    event.set()

def main():
    rospy.init_node('controller_node_{}'.format(random.randint(1, 1000)), anonymous=True)
    parser = argparse.ArgumentParser(description='Controller Script')
    parser.add_argument('--layers', type=str, default='')
    args = parser.parse_args(rospy.myargv()[1:])

    e = threading.Event()
    signal.signal(signal.SIGINT, (lambda a, b: sighandler(e, a, b)))
    cur = Curr_table()

    th_callback = threading.Thread(target=curr_sub, name="cur_sub", args=(cur,))
    th_callback.start()

    th_con = threading.Thread(target=Controller, name="hoge", args=(e, args, cur, ))
    th_con.start()

    while th_con.isAlive():
        # print "alive"
        time.sleep(0.5)
    th_con.join()
    # while not e.isSet():
    #     # print("event")
    #     rospy.on_shutdown("hoge")
    # # Controller(args)

if __name__ == '__main__':
    main()