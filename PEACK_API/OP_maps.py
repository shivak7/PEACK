import numpy as np
import csv

def PartsMap(fn):

    Map = {}
    with open(fn) as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            Map[row[0]] = int(row[1])


    return Map
