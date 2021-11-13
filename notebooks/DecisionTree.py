from math import log

def Entropy_value(total,pickup,grp,gp):
    p = gp/grp
    q = (grp-gp)/grp
    return -p*int(log(p,2))-q*int(log(q,2))

def Gini_value(total,pickup,grp,gp):
    p = gp/grp
    q = (grp-gp)/grp
    return p**2+q**2

total=30
pick=15

print('By Gender:')
Female = [10,2,8]
Male = [20,13,7]

e1 = Entropy_value(total,pick,Female[0],Female[1])
e2 = Entropy_value(total,pick,Male[0],Male[1])

Result = Female[0]/total * e1 + Male[0]/total * e2
print('Entropy: (%4.2f %4.2f), 加權後:%4.2f'%(e1,e2,Result))

g1 = Gini_value(total,pick,Female[0],Female[1])
g2 = Gini_value(total,pick,Male[0],Male[1])

Result1 = Female[0]/total * g1 + Male[0]/total * g2
print('Gini value: %4.2f %4.2f), 加權後: %4.2f'%(g1,g2,Result1))
print('-------------------------------------------------')
print('By Class:')
classA = [14,6,8]
classB = [16,9,7]

e3 = Entropy_value(total,pick,classA[0],classA[1])
e4 = Entropy_value(total,pick,classB[0],classB[1])

Result3 = classA[0]/total * e3 + classB[0]/total * e4
print('Entropy:(%4.2f %4.2f), 加權後:%4.2f'%(e3,e4,Result3))

g3 = Gini_value(total,pick,classA[0],classA[1])
g4 = Gini_value(total,pick,classB[0],classB[1])

Result4 = classA[0]/total * g3 + classB[0]/total * g4
print('Gini value: (%4.2f %4.2f), 加權後:%4.2f'%(g3,g4,Result4))

print('================================================')
if(Result<Result3):
    print('Entropy: 使用性別當作節點')
else:
    print('Entropy: 使用班級當作節點')
print('\t(選擇 Entropy 小的)')
if(Result1>Result4):
    print('Gini: 使用性別當作節點')
else:
    print('Gini: 使用班級當作節點')
print('\t(選擇 Gini值 大的)')