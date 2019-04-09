## 본 파일은 두 폴더에서 train set 을 골고루 만들어 주는 기능을함.
import random
from glob import glob


f = open("train_data", 'w')
# mellifera 의 사진 번호를 보여줌.


mel_list = glob('mellifera/*.jpg')
cer_list = glob('cerana/*.jpg')
print("mel_size: ",len(mel_list))
print("cer_list: ",len(cer_list))
smaller = len(mel_list) <len(cer_list)  and len(mel_list) or len(cer_list)
print("smaller: ",smaller)
ran_list = []

for i in range(smaller):
    # mellifera 면 1
    ran_list.append(mel_list[i]+" 1\n")
    # cerana 가 0
    ran_list.append(cer_list[i] + " 0\n")

    # f.write(mel_list[i]+" 1\n")
    #
    # f.write(cer_list[i] + " 0\n")

random.shuffle(ran_list)

for i in ran_list:
    f.write(i)

f.close()