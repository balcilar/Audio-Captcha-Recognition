from CaptchaAnalysis import CaptchaAnalysis
import os
import sys
import math
from collections import defaultdict
import matplotlib.pyplot as plt    
import numpy as np

# distance calculation, if we reach the end of the signal2 (not reference the other)
# then the distance is always 0. unless the distance is absolute difference
def dist(A,B,s):
    if A==B:
        return 0
    return 1 #np.abs(A-B)

def dynamicTimeWarp(seqA, seqB, nap):
    # take lenght of signal1 and signal 2 and set infinity to cost matrix
    numRows, numCols = len(seqA), len(seqB)
    cost = [[np.inf for _ in range(numCols)] for _ in range(numRows)]
 
    # set first element of cost matrix.
    cost[0][0] = dist(seqA[0], seqB[0],False)
    # set the initial cost of matching between first element of reference and all element of signal2
    for i in range(1, numRows):
        cost[i][0] = cost[i-1][0] + dist(seqA[i], seqB[0],False)
    # set initial cost of between first element of signal 2 to all element of reference as 0. because the signal might have 
    # some initial shift
    for j in range(1, numCols):
        cost[0][j] = 0 #cost[0][j-1] + dist(seqA[0], seqB[j],j==numCols-1)
 
    # do all element in reference signal
    for i in range(1, numRows):
        # take the search range the seach range is [bas son]
        jj=i*1.0*numCols/numRows
        bas=max(1,int(jj)-nap)
        son=min(numCols,int(jj)+nap)
        
        for j in range(bas, son):
            # take 3 different choice
            choices = [cost[i-1][j], cost[i][j-1], cost[i-1][j-1]]
            # calculate cost of concerned move
            cost[i][j] = min(choices) + dist(seqA[i], seqB[j],j==numCols-1) 
    ## backtracing part to calculate path
    #path = []
    ## take last element of cost matrix as a first element of path
    #i, j = numRows-1, numCols-1
    #while not (i == j == 0):
    #    # add i,j into the path
    #    path.append((i, j))
    #    # select minimum costed neighbour
    #    if cost[i-1][j-1]<= min( [cost[i][j-1],cost[i-1][j]]):
    #        i=i-1
    #        j=j-1
    #    elif cost[i-1][j]<= min( [cost[i][j-1],cost[i-1][j-1]]):
    #        i=i-1
    #    elif cost[i][j-1]<= min( [cost[i-1][j],cost[i-1][j-1]]):
    #        j=j-1
    ## reverse path
    #path.reverse()
    # return cost and path
    return cost[-1][-1]







print 'Using instruction......'
print 'python captcha_test traineddatafile [filename1 | --all] <filename2> <filename3>, .....'
print '---------------------'

 
traineddatafile= "dataset/train/train_features_50.0_0.9" 
param="--all"

if len(sys.argv)>1:
    traineddatafile=str(sys.argv[1])    

if len(sys.argv)>2:
    param=str(sys.argv[2])



ca=CaptchaAnalysis() 
ca.loadtrainparam(traineddatafile)

print 'Using trained data file: ', traineddatafile

d=0
t=0
dd=0
tt=0

if param=="--all":
    for file in os.listdir(os.getcwd()+'/dataset/test/'):
        if file.endswith(".wav"):
            result=ca.test('dataset/test/'+str(file))
            #result=ca.testNB('dataset/test/'+str(file))
            cost=dynamicTimeWarp(result, file[0:-4], 4)
            tt+=len(file[0:-4])
            dd+=len(file[0:-4])-cost

            if result==file[0:-4]:
                d+=1
            t+=1
            print str(file), '\t------->', result, '\t precision: ', d*1.0/t, '\t digit precision: ', dd*1.0/tt
else:
    for i in range(2,len(sys.argv)):
        result=ca.test(str(sys.argv[i]))
        fn=sys.argv[i]
        if result==fn[0:-4]:
            d+=1
        t+=1
    
        print str(sys.argv[i]), '\t------->', result, '\t precision: ', d*1.0/t


    
print ' General precision :', d*1.0/t, '\t digit precision: ', dd*1.0/tt
