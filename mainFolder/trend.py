import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df0 = pd.read_csv('normalized_0.csv')
df1 = pd.read_csv('normalized_1.csv')
df2 = pd.read_csv('normalized_2.csv')
# print(attrition0)
# print("-----------------------------")
#
true0, false0 = df0[df0['Attrition'] == 1], df0[df0['Attrition'] == 0]
# print("---df0", df0.shape)
# print("true", true0.shape)
# print("false", false0.shape)
#
true1, false1 = df1[df1['Attrition'] == 1], df1[df1['Attrition'] == 0]
# print("---df1",df1.shape)
# print("true",true1.shape)
# print("false",false1.shape)
#
# df2.Attrition = df0.Attrition
#
true2, false2 = df2[df2['Attrition'] == 1], df2[df2['Attrition'] == 0]
# print("---df2",df2.shape)
# print("true",true2.shape)
# print("false",false2.shape)



# np.sum(df0['MonthlyIncome'].values)
# dfTrue = df0.columns.values
# print(dfTrue)

dataTrue = np.array([np.average(true0,axis=0), np.average(true1,axis=0), np.average(true2,axis=0)])
dfTrue = pd.DataFrame(dataTrue, columns = df0.columns.values)

dataFalse = np.array([np.average(false0,axis=0), np.average(false1,axis=0), np.average(false2,axis=0)])
dfFalse = pd.DataFrame(dataFalse, columns = df0.columns.values)

dataAll = np.array([np.average(df0,axis=0), np.average(df1,axis=0), np.average(df2,axis=0)])
dfAll = pd.DataFrame(dataAll, columns = df0.columns.values)

# data = np.array([dfTrue['Age']], dfFalse['Age'], dfAll['Age'])
# print(data)

attribute = np.array(["BusinessTravel", "JobLevel", "JobSatisfaction", "MaritalStatus", "PerformanceRating", "YearsAtCompany"])
plt.figure(1)
plt.suptitle('Categorical Plotting', fontsize=18)
for i in range(len(attribute)):
    x_axes = [2016, 2017, 2018]
    y_axes = np.array([dfTrue[attribute[i]], dfFalse[attribute[i]], dfAll[attribute[i]]]).T
    plt.subplot(2,3,i+1)
    plt.title(attribute[i] + " Plot", fontsize=10)
    plt.xlabel("Time Stamp", fontsize=7)
    plt.ylabel(attribute[i], fontsize=7)
    plt.plot(x_axes, y_axes, linewidth=1)
    plt.grid()
    plt.legend(("True", "False", "All"))
    plt.tick_params(axis='both', labelsize=9)
plt.show()

# attribute = np.array(["Age", "MonthlyIncome", "RelationshipSatisfaction"])
#
# fig, axes = plt.subplots(2, 3)
# plt.figure(1)
#
# fig.suptitle('Categorical Plotting')
# for i in range(len(attribute)):
#     x_axes = [2016, 2017, 2018]
#     y_axes = np.array([dfTrue[attribute[i]], dfFalse[attribute[i]], dfAll[attribute[i]]]).T
#     axes[0,i].plot(x_axes, y_axes, linewidth=1)
#     # axes[0,i].title(attribute[i], fontsize=19)
#     plt.xlabel("Time Stamp", fontsize=10)
#     plt.ylabel(attribute[i], fontsize=10)
#     # plt.legend(("True", "False", "All"))
#     # plt.tick_params(axis='both', labelsize=9)
#     fig.show()
#
# plt.show()

dataTrue = np.array([np.nanstd(true0,axis=0), np.nanstd(true1,axis=0), np.nanstd(true2,axis=0)])
dfTrue = pd.DataFrame(dataTrue, columns = df0.columns.values)

dataFalse = np.array([np.nanstd(false0,axis=0), np.nanstd(false1,axis=0), np.nanstd(false2,axis=0)])
dfFalse = pd.DataFrame(dataFalse, columns = df0.columns.values)

dataAll = np.array([np.nanstd(df0,axis=0), np.nanstd(df1,axis=0), np.nanstd(df2,axis=0)])
dfAll = pd.DataFrame(dataAll, columns = df0.columns.values)

attribute = np.array(["BusinessTravel", "JobLevel", "JobSatisfaction", "MaritalStatus", "PerformanceRating", "YearsAtCompany"])
plt.figure(1)
plt.suptitle('Categorical Standard Deviation Plotting', fontsize=18)
for i in range(len(attribute)):
    x_axes = [2016, 2017, 2018]
    y_axes = np.array([dfTrue[attribute[i]], dfFalse[attribute[i]], dfAll[attribute[i]]]).T
    plt.subplot(2,3,i+1)
    plt.title(attribute[i] + " Plot", fontsize=10)
    plt.xlabel("Time Stamp", fontsize=7)
    plt.ylabel(attribute[i], fontsize=7)
    plt.plot(x_axes, y_axes, linewidth=1)
    plt.grid()
    plt.legend(("True", "False", "All"))
    plt.tick_params(axis='both', labelsize=9)
plt.show()



