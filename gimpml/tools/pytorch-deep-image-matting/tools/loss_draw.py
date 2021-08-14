import re
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

# args: log_name, match_rule, self_log_interval, smooth_log_interation
loss_file_name = "simple_loss"
title = "{}_Loss".format(loss_file_name)
f = open("../log/{}.log".format(loss_file_name))
pattern = re.compile(r"Loss:[ ]*\d+\.\d+")
self_inter = 10
smooth = 20

# read log file
lines = f.readlines()
print("Line: {}".format(len(lines)))
ys = []
k = 0
cnt = 0
sum_y = 0.0

# read one by one
for line in lines:
    obj = re.search(pattern, line)
    if obj:
        val = float(obj.group().split(":")[-1])
        sum_y += val
        k += 1
        if k >= smooth:
            ys.append(sum_y / k)
            sum_y = 0.0
            k = 0
            cnt += 1
            if cnt % 10 == 0:
                print("ys cnt: {}".format(cnt))
if k > 0:
    ys.append(sum_y / k)

ys = np.array(ys)
xs = np.arange(len(ys)) * self_inter * smooth

print(xs)
print(ys)

plt.plot(xs, ys)
plt.title(title)
plt.xlabel("Iter")
plt.ylabel("Loss")
plt.savefig("../log/{}.png".format(title))
plt.show()
