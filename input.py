def getTrainList(list, start, batch_size):
    return list[start * batch_size: start * batch_size + batch_size]

def getTestList(list, start, batch_size, test_size):
    _test_start = start * batch_size + batch_size
    return list[_test_start: _test_start + test_size]