"""
This program parses the breastcancer.csv, mosquito.csv files, and colon.csv files
"""




import numpy as np

def main():

  #data_file = open('breastcancer.csv', "r")  #Open dataset file
  #data_list = parse_restData(data_file)

  breast_file = open('formattedData.csv', 'r')
  breast_data, breast_labels = parse_breastData(breast_file)

  #print breast_data
  print breast_labels

  print len(breast_data)
  print len(breast_labels)


def parse_mosquitoData(file):
    returnlist = []
    for line in file:
        line = line.strip("\n").split("\t")
        line = np.array(line).astype(float)
        returnlist.append(line)

    returnlist = np.array(returnlist)
    file.close()
    return returnlist

def parse_mosquitoLabels(file):
    returnlist = []
    for line in file:
        line = line.strip("\n")
        returnlist.append(line)

    returnlist = np.array(returnlist)
    file.close()
    return returnlist


def parse_breastData(file):
    returnlist = []
    labels = []
    file.readline()
    for line in file:
        line = line.strip("\n").split(",")
        if 'null' in line:
            line = np.array(line)
            indices = np.where(line == 'null')
            for i in indices[0]:
                line[i] = 0
            line = line.astype(float)
        else:
            line = np.array(line).astype(float)
        returnlist.append(line[:-1])
        labels.append(line[-1])


    returnlist = np.array(returnlist)
    labels = np.array(labels)
    file.close()
    return returnlist, labels

def create_colonData(file):
    returnlist = []
    for line in file:
        line = line.strip("\n").split(" ")
        if len(line) != 1:
            returnlist.append(line)

    returnlist = np.array(returnlist)
    returnlist = returnlist.transpose()
    file.close()
    return returnlist


def parse_colonLabels(file):
    returnlist = []
    for line in file:
        line = line.strip("\n")
        returnlist.append(line)

    returnlist = np.array(returnlist)
    returnlist = returnlist.transpose()
    file.close()
    return returnlist



#main()
