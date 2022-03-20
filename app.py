import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import math

y1, sr1 = librosa.load('tp4.wav')
y2, sr2 = librosa.load('t2.wav')

def dot(A,B): 
    return (sum(a*b for a,b in zip(A,B)))

def cosine_similarity(a,b):
    return dot(a,b) / ( (dot(a,a) **.5) * (dot(b,b) ** .5) )
  
def cosine_similarity1(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
  


#Showing multiple plots using subplot
plt.subplot(1, 2, 1) 
mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values
display.specshow(mfcc1)

plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
display.specshow(mfcc2)

from dtw import dtw
from numpy.linalg import norm

dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print('Normalized distance between the two sounds: '+ dist.__str__())

# import numpy as np
# array1 = np.array(mfcc1)
# array2 = np.array(mfcc2)

# number_of_equal_element = np.sum(array1 == array2)
# total_elements = np.multiply(*array1.shape)
# percentage = number_of_equal_element/total_elements
# print('Number of equal elements: '+ format(number_of_equal_element))
# print('number of identical elements: \t\t{}'.format(number_of_equal_element))
# print('number of different elements: \t\t{}'.format(total_elements-number_of_equal_element))
# print('percentage of identical elements: \t{:.2f}%'.format(percentage*100))

array1 = []
for nums in mfcc1:
    for val in nums:
        array1.append(val)
        
array2 = []
for nums in mfcc2:
    for val in nums:
        array2.append(val)
        
print(cosine_similarity(array1, array2))
        
# print(array1)
# print(array2)



# set1 = set(array1)
# set2 = set(array2)
# total = sorted(set1|set2)

# new_list1 = [x if x in set1 else "MISSING" for x in total]
# new_list2 = [x if x in set2 else "MISSING" for x in total]

# # print(new_list1)
# # print(new_list2)
# from numpy import dot
# from numpy.linalg import norm

# cos_sim = dot(new_list1, new_list2) / (norm(new_list1) * norm(new_list2))
# print('Cosine similarity: '+ cos_sim.__str__())


# def cosine_similarity1(v1,v2):
# #     "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
#     sumxx, sumxy, sumyy = 0, 0, 0
#     for i in range(len(v1)):
#         x = v1[i]; y = v2[i]
#         sumxx += x*x
#         sumyy += y*y
#         sumxy += x*y
#     return sumxy/math.sqrt(sumxx*sumyy)


# -------------------------------------------------------#
# hop_length = 1024
# y_ref, sr = librosa.load("t1.wav")
# y_comp, sr = librosa.load("coba.wav")
# chroma_ref = librosa.feature.chroma_cqt(y=y_ref, 
#   sr=sr,hop_length=hop_length)
# chroma_comp = librosa.feature.chroma_cqt(y=y_comp, 
#   sr=sr, hop_length=hop_length)

# x_ref = librosa.feature.stack_memory(
#   chroma_ref, n_steps=10, delay=3)
# x_comp = librosa.feature.stack_memory(
#   chroma_comp, n_steps=10, delay=3)
# xsim = librosa.segment.cross_similarity(x_comp, x_ref)

# # print(xsim)

# fig, ax = plt.subplots()
# display.specshow(xsim, x_axis='s', y_axis='time', hop_length=hop_length, ax=ax)
# plt.show()


# ------------------------------------------------ #
# plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
# plt.plot(path[0], path[1], 'w')   #creating plot for DTW

# plt.show()