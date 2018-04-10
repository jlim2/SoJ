import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

path = "/Users/JJ/SoJ/HW3_CV/HW3_task2_match_results"
matchResultFiles = listdir(path)

for matchResultFile in matchResultFiles:
    currPath = path + "/" + matchResultFile
    imgAndOriNames = matchResultFile[10:-18]    #<image name>_<orientation>


    matchResult = pd.read_csv(currPath)
    match1ValCount = matchResult.match1.value_counts(normalize=True)
    match2ValCount = matchResult.match2.value_counts(normalize=True)
    match3ValCount = matchResult.match3.value_counts(normalize=True)
    print("------------------"+imgAndOriNames+"------------------")

    print('-------Match 1-------')
    # df1 = pd.DataFrame(match1ValCount, index=match1ValCount.index)
    # print(df1)
    # df1['percent'] = (match1ValCount['match1'])/len(matchResult)*100
    print(match1ValCount)
    print('-------Match 2-------')
    print(match2ValCount)
    print('-------Match 3-------')
    print(match3ValCount)






