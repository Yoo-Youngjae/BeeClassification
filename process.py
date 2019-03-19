from glob import glob
import random
import numpy as np

def getTrainList(list, start, batch_size):
    return list[start * batch_size: start * batch_size + batch_size]

def getTestList(test_size):
    test_full_list = glob('test/*.jpg')
    ran_num = random.randrange(0, len(test_full_list)-test_size)
    test_list = test_full_list[ran_num: ran_num+test_size]
    return test_list

def getTestLabel(name):
    f = open("test_label", 'r')
    res_arr = np.array([1,0])
    while True:
        line = f.readline()
        try:
            if line.index(name) >= 0:
                res = line[len(name) + 1:len(name) + 2]
                print(res)
                if res == '0':
                    # print('this is cerana')
                    res_arr = np.array([1, 0])
                elif res == '1':
                    # print('this is mellifera')
                    res_arr = np.array([0, 1])
                break
        except:
            pass

    f.close()
    return res_arr