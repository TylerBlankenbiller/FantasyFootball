import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#game = pd.read_csv("testLast2.csv", low_memory = False)

filepath = 'qb.txt'  
with open(filepath) as fp:  
   line = fp.readline()
   cnt = 1
   while line:
       print("Line {}: {}".format(cnt, line.strip()))
       line = fp.readline()
       cnt += 1