from glob import glob
import random
import numpy as np

def getTrainList(start, batch_size):
    f = open("train_data", 'r')
    lines = f.readlines()
    f.close()

    return lines[start * batch_size: start * batch_size + batch_size]
def getTrainLabel(item):
    end = item.index('jpg')
    train_Xs = item[:end+3]
    trainlabel = item[end+4:end+5]
    if trainlabel == '0':
        train_Ys = np.array([1, 0])
    else:
        train_Ys = np.array([0, 1])

    return train_Xs, train_Ys

def getTestList(test_size):
    test_full_list = glob('test/*.jpg')
    test_list = test_full_list
    # ran_num = random.randrange(0, len(test_full_list)-test_size)
    # test_list = test_full_list[ran_num: ran_num+test_size]
    return test_list

def getTestLabel(name):
    start = name.index('/')
    end = name.index('.')
    name = name[start+1:end] #/이후부터 . 이전까지의 이름 찾아냄
    f = open("test_label", 'r')
    res_arr = np.array([1,0])
    while True:
        line = f.readline()
        try:
            if line.index(name) >= 0:
                # test_label 에서 정확히 정답만 골라냄.
                res = line[len(name)+1:len(name)+2]
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