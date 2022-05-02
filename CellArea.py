import matplotlib.pyplot as plt
import csv
import os
import re


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)


ThisDir = '/home/villads/CLionProjects/pbct-simuleringer/CelleCount/'

#TODO can ikke samle filer for samme dosis 
for dir in os.listdir(ThisDir):
    
    if dir.endswith((".py", ".txt", ".jpg", ".png", ".csv")):
        continue
    pattern = "_blob.csv"
    print(dir)
    matching_files = [f for f in os.listdir(ThisDir + dir) if pattern in f]
    doses = []
    for files in matching_files:
        print(files)
        if not files[3:5] in doses:
            doses.append(files[3:5])
    print(doses)
    for dose in doses:
        patterndose = dose
        matching_dose = [f for f in os.listdir(ThisDir + dir) if patterndose in f]
        Radius = []
        filesdone = []
        # print("For dose {} we found {}".format(dose, matching_dose))
        for dosefile in matching_dose:
            if dosefile in filesdone or dosefile.endswith(".png"):
                continue
            filesdone.append(dosefile)
            # print(ThisDir + dir + '/' + dosefile)
            with open(ThisDir + dir + '/' + dosefile, 'r') as csvfile:
                data = csv.reader(filter(lambda row: row[0] != '#', csvfile), delimiter=',')
                for row in data:
                    Radius.append(float(row[2]))
                csvfile.close()
        plt.hist(Radius, bins=10)
        plt.savefig(ThisDir + dir + '/' + dose + '_hist.png')
        plt.close()

    # for files in matching_files:
    #     dose = files[3:5]
    #     doseint = re.findall('\d+', dose)[0]
    #     Radius = []

    #     if doseint in CompletedDose:
    #         continue

    #     CompletedDose.append(doseint)
    #     indxDose = indices(dose, i)
    #     cellAve_tmp = 0
    #     itr = 0
    #     for k in indxDose:
    #         cellAve_tmp += counts[k]
    #         itr += 1
    #     doseint.append(int(doseNo[0]))

    #     with open(ThisDir + dir + '/' + files, 'r') as csvfile:
    #         data = csv.reader(filter(lambda row: row[0] != '#', csvfile), delimiter=',')
    #         for row in data:
    #             Radius.append(float(row[2]))
    #     csvfile.close()
    #     plt.hist(Radius, bins=10)
    #     plt.savefig(ThisDir + dir + '/' + files[0:5] + '_hist.png')
    #     plt.close()
