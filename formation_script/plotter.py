import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def find_n_iterations(data):
    current_x = data[0]
    n_iterations = 1
    for x in range(1, len(data)):
        if data[x] == current_x:
            n_iterations += 1
        else:
            break
    return n_iterations

def averager(x_axis, y_axis):
    n_iterations = find_n_iterations(x_axis)
    n_data_points = int(len(x_axis)/n_iterations)
    clean_data = np.empty((n_data_points, 2))
    for i in range(n_data_points):
        x_value = x_axis[n_iterations*i]
        running_sum = 0
        for j in range(n_iterations):
            current_index = n_iterations*i+j
            running_sum += y_axis[current_index]
        clean_data[i, 0] = x_value
        clean_data[i, 1] = running_sum/n_iterations
    return clean_data

def coeff_of_var(y_axis):
    std = np.std(y_axis)
    avg = np.average(y_axis)
    return std/avg

def get_coeff_of_var_plot(n_files):
    clean_data = np.empty((n_files,2))
    for i in range(1, n_files+1):
        data = pd.read_csv(f"C:\\Users\\afons\\Desktop\\Simulations\\coefficient of variance\\fuel_saved\\B{i}")
        x_axis = data["manager_ratio"].to_numpy()
        y_axis = data["Fuel saving ratio"].to_numpy()
        coeff = coeff_of_var(y_axis)
        clean_data[i-1, 0] = find_n_iterations(x_axis)
        clean_data[i-1, 1] = coeff
    return clean_data

# data = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\data dumps\A2")
#
# x_axis = data["manager_ratio"].to_numpy()
# y_axis = data["Fuel saving ratio"].to_numpy()

# clean_data = averager(x_axis, y_axis)

clean_data = get_coeff_of_var_plot(3)
plt.scatter(clean_data[:,0], clean_data[:,1])
plt.xlabel("N iterations")
plt.ylabel("Coefficient of variation")
plt.show()
