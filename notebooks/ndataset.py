import pandas as pd
dataset1 = pd.DataFrame(data=[
           [10,2,8],[20,13,7]],
            index=pd.Series(['Female','Male']),
            columns=['Total','player','not-player'])
dataset2 = pd.DataFrame(data=[
           [14,6,8],[16,9,7]],
            index=pd.Series(['ClassA','ClassB']),
            columns=['Total','player','not-player'])
print(dataset1.shape,dataset2.shape)   
print(dataset1)
print('-----------------------------------')
print(dataset2)
