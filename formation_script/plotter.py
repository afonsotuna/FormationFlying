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
        data = pd.read_csv(f"C:\\Users\\afons\\Desktop\\Simulations\\coefficient of variance (IN PROGRESS)\\all\\B{i}")
        x_axis = data["manager_ratio"].to_numpy()
        y_axis = data["Number of formations"].to_numpy()
        coeff = coeff_of_var(y_axis)
        clean_data[i-1, 0] = find_n_iterations(x_axis)
        clean_data[i-1, 1] = coeff
    #clean_data_2 = clean_data[clean_data[:,1] < 0.06]
    #clean_data_2 = np.vstack([[10, 0.43], clean_data_2])
    #clean_data = clean_data[clean_data[1:,1] < 5]
    return clean_data

# data = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\data dumps\A2")
#
# x_axis = data["manager_ratio"].to_numpy()
# y_axis = data["Fuel saving ratio"].to_numpy()

# clean_data = averager(x_axis, y_axis)

# clean_data = get_coeff_of_var_plot(25)
# plt.scatter(clean_data[:,0], clean_data[:,1])
# plt.xlabel("Number of iterations")
# plt.ylabel("Coefficient of variation")
# plt.title("Coefficient of variation for the number of formations")
# plt.show()

# data = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\CNP\change_alliance")
#
# x_axis = data["alliance_ratio"].to_numpy()
#
# # y_axis = data["Fuel saving ratio"].to_numpy()
# # plt.ylabel("Fuel saving ratio")
# # y_axis = data["Average delay"]
# # plt.ylabel("Average delay")
# y_axis = data["Alliance saving ratio"].to_numpy()
# plt.ylabel("Alliance saving ratio")
#
# clean_data = averager(x_axis, y_axis)
#
# plt.plot(clean_data[:, 0], clean_data[:, 1])
# plt.xlabel("Fraction of flights which belong to the alliance")
#
# plt.show()

# greedy = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\all negotiations\data\greedy")
# CNP = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\all negotiations\data\CNP")
# English = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\all negotiations\data\English")
# Vickrey = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\all negotiations\data\Vickrey")
# Japanese = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\all negotiations\data\Japanese")
#
# data_greedy = (greedy["Real saved fuel"]/greedy["n_flights"])*1.60 - (greedy["Average delay"])*0.9
# data_CNP = (CNP["Real saved fuel"]/CNP["n_flights"])*1.60 - (CNP["Average delay"])*0.9
# data_English = (English["Real saved fuel"]/English["n_flights"])*1.60 - (English["Average delay"])*0.9
# data_Vickrey = (Vickrey["Real saved fuel"]/Vickrey["n_flights"])*1.60 - (Vickrey["Average delay"])*0.9
# data_Japanese = (Japanese["Real saved fuel"]/Japanese["n_flights"])*1.60 - (Japanese["Average delay"])*0.9
#
# data = [data_greedy, data_CNP, data_English, data_Vickrey, data_Japanese]
# fig1, ax1 = plt.subplots()
# ax1.set_title('Average money flow per agent with varying negotiation methods')
# ax1.boxplot(data)
# plt.xticks([1, 2, 3, 4, 5], ["Greedy", "CNP", "English", "Vickrey", "Japanese"])
# plt.ylabel("Average money flow per agent [€/flight]")
#
# plt.show()

# data = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\CNP\data\change_range_extra")
#
# x_axis = data["communication_range"].to_numpy()
#
# #y_axis = data["Fuel saving ratio"].to_numpy()
# #plt.ylabel("Fuel saving ratio")
# #y_axis = data["Average delay"]
# #plt.ylabel("Average delay")
# y_axis = data["Alliance saving ratio"].to_numpy()
# plt.ylabel("Alliance saving ratio")
#
# clean_data = averager(x_axis, y_axis)
#
# plt.plot(clean_data[:, 0], clean_data[:, 1])
# plt.xlabel("Communication range")
#
# plt.show()


dense = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\destinations\dense")
scattered = pd.read_csv(r"C:\Users\afons\Desktop\Simulations\destinations\scattered")

data_dense = dense["Average delay"]
data_scattered = scattered["Average delay"]
data = [data_dense, data_scattered]
fig1, ax1 = plt.subplots()
ax1.set_title('Average delay for different destination airport distributions')
ax1.boxplot(data)
plt.xticks([1, 2], ["Dense", "Scattered"])
plt.ylabel("Steps of delay")

plt.show()