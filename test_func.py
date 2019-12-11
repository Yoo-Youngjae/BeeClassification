## 본 파일은 test_label 을 만들어 주는 기능을 함.
import random
from glob import glob



f = open("test_label", 'w')
#mellifera 의 사진 번호를 보여줌.
#mellifera 면 1
folder_name_mel = 'test/*.jpg'
data_list = glob(folder_name_mel)
for i in data_list:
    dot_index = i.index('.')
    print(i[10:])
    f.write(i[10: dot_index]+" 1\n")

#cerana 의 사진 번호를 보여줌.
#cerana 가 0
# folder_name_cer = 'cerana/*.jpg'
# data_list_2 = glob(folder_name_cer)
# for i in data_list_2:
#     dot_index = i.index('.')
#     print(i[7:])
#     f.write(i[7: dot_index] + " 0\n")
#
# f.close()
