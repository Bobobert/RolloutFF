"""
    Miscelaneous functions
"""

import pickle
import re
import os
import time as Time

class Logger():
    """
    Class to save most of the print information in a file while the Experiment
    exists.

    Parameters
    ----------
    msg: str
        The string generated to log.
    prnt: bool
        If true a message will be displayed.
    time: bool
        If true a time stamp will be added to the message in the log.
    """

    last_time = 0
    def __init__(self):
        checkDir("Logs")
        self.logfile = open("./Logs/rollout_log_{}.txt".format(timeStr()),'wt')
        self("Logger initialized.",False)

    def __call__(self, msg, prnt=True, time=True, time_delta=False):
        if time_delta:
            if self.last_time == 0:
                self.last_time = Time.time()
            total_time = Time.time() - self.last_time
            delta = " -- delta-time {}h: {}m: {}s".format(\
                int(total_time//3600), int(total_time//60 - total_time//3600*60), int(total_time % 60))
            self.last_time = Time.time()
            msg += delta
        if prnt:
            print(msg,end="\n")
        if time:
            msg += " -- @ {}".format(timeStr())
        self.logfile.write(msg+"\n")
    
    def close(self):
        self("Clossing logger")
        self.logfile.close()

def timeStr():
    return Time.strftime("%d-%m-%Y-%H:%M:%S", Time.gmtime())

def checkDir(dir):
    assert isinstance(dir, str), "dir argument must be a string"
    ls = os.listdir(os.getcwd())
    if not dir in ls:
        print("Creating a folder {}".format(dir))
        os.mkdir(dir)
    Time.sleep(0.5)