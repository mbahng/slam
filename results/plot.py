import matplotlib.pyplot as plt 
import numpy as np


def get_totalErr_vs_sumOfSubTrajErr(data:list): 

    total_err = data[-1] 
    sum_of_subtraj_err = sum(data[:-1])

    return total_err, sum_of_subtraj_err



with open("/home/iotlab/Desktop/ORB_SLAM3/results/nohup.out", "r") as f: 
    data = [eval(line.strip().split(" : ")[-1]) for line in f.readlines()]

rel_err_change = []
for elem in data: 
    changes = get_totalErr_vs_sumOfSubTrajErr(elem) 
    rel_err_change.append(changes)


bad_loop_close = [x[0] - x[1] for x in rel_err_change if x[0] > x[1]] 
good_loop_close = [x[1] - x[0] for x in rel_err_change if x[0] < x[1]] 

g_mean, g_std = np.mean(good_loop_close), np.std(good_loop_close)
b_mean, b_std = np.abs(np.mean(bad_loop_close)), np.std(bad_loop_close)

g_prop = round(len(good_loop_close)/(len(good_loop_close) + len(bad_loop_close)), 2), 
b_prop = round(len(bad_loop_close) / (len(good_loop_close) + len(bad_loop_close)), 2)

plt.plot()
plt.bar(
    [f"good ({g_prop})", f"bad ({b_prop})"], 
    [g_mean, b_mean], 
    yerr = [g_std, b_std] 
)
plt.savefig("bar") 
