import matplotlib.pyplot as plt
import csv
import os
import re


ThisDir = '/home/villads/CLionProjects/pbct-simuleringer/CelleCount/'

def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

        
# Get experiment data.
DoseToVolume = {}

ManualBorDose = [0, 2, 4]
ManualBor = [1, 0.717398894773228, 0.235920399202413]

ManualNoBorDose = [0, 1, 2, 4]
ManualNoBor = [1, 0.8583, 0.7529, 0.3047]
# BaseCellCount = 125.
# Get cellcount data
label = []
for dir in os.listdir(ThisDir):
    if dir.endswith((".py", ".txt", ".jpg", ".png", ".csv")):
        continue

    with open(ThisDir + '/DoseToVolume.csv', 'r') as csvfileEx:
        experiment = csv.reader(filter(lambda row: row[0] != '#', csvfileEx), delimiter=',')
        for row in experiment:
            DoseToVolume.update({int(row[0]): float(row[1])})

    csvfileEx.close()
    name = []
    dose = []
    doseNoUnit = []
    counts = []
    SurvivalF = []
    with open(ThisDir + dir + '/Cellcount.csv', 'r') as csvfile:
        plots = csv.reader(filter(lambda row: row[0] != '#', csvfile), delimiter=',')
        for row in plots:
            for col in row:
                if any(c.isalpha() for c in col):
                    n = col[0:2]
                    d = col[3:5]
                    name.append(n)
                    dose.append(d)
                else:
                    counts.append(int(col))
    csvfile.close()
    # find same doses and avarege the cell counts
    doseint = []
    cellAve = []
    completedDose = []
    for i in dose:
        doseNo = re.findall('\d+', i)
        if int(doseNo[0]) in completedDose:
            continue

        completedDose.append(int(doseNo[0]))
        indxDose = indices(dose, i)
        cellAve_tmp = 0
        itr = 0
        for k in indxDose:
            cellAve_tmp += counts[k]
            itr += 1
        doseint.append(int(doseNo[0]))
        if int(doseNo[0]) == 0:
            BaseCellCount = cellAve_tmp/itr
        cellAve.append(cellAve_tmp/itr)
    for i in range(len(cellAve)):
        CellConc = cellAve[i]/DoseToVolume[doseint[i]]
        SurvivalF.append(CellConc / (BaseCellCount/DoseToVolume[0])) # (BaseCellCount/DoseToVolume[0])
        # print(doseint[i])
        # print(DoseToVolume[doseint[i]])
    # print(cellAve)

    # plt.scatter(doseint, SurvivalF)
    list_1_sorted, list_2_sorted = zip(*sorted(zip(doseint, SurvivalF), reverse=False))
    plt.plot(list_1_sorted[0:4], list_2_sorted[0:4])
    plt.scatter(list_1_sorted[0:4], list_2_sorted[0:4])
    # if len(list_1_sorted) > 5:
    #     plt.errorbar(list_1_sorted[0:4], list_2_sorted[0:4], [0, 0.0573, 0.0991, 0.0787])
    # else:
    #     plt.plot(list_1_sorted[0:4], list_2_sorted[0:4])

    label.append(dir)
    print("The Survival fractions for {} at {} are {} and with Counts {}".format(dir, list_1_sorted, list_2_sorted, cellAve) )

plt.plot(ManualBorDose, ManualBor)
plt.scatter(ManualBorDose, ManualBor)
label.append('ManualBorDose')
plt.plot(ManualNoBorDose, ManualNoBor)
plt.scatter(ManualNoBorDose, ManualNoBor)
label.append('ManualNoBorDose')
label2 = ['Boron (python)', 'No Boron (python)', 'Boron (Manual)', 'No Boron (Manual)']
plt.legend(label2)
plt.grid(True)
plt.xlabel('Dose [Gy]')
plt.ylabel('SF [%]')
plt.yscale('log')
plt.subplots_adjust(left=0.18)
plt.savefig('SF.png')
# plt.show()
plt.close()
# print(len(SurvivalF))
# plt.table(rowLabels=[doseint], colLabels=['ManualNoBorDose'], cellText=[SurvivalF])
# plt.show()
# plt.savefig('plots/Protonhist_' + dir + '_CellFlask' + str(i))
# plt.clf()
# csvfile.close()
# y.clear()
