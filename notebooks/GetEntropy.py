from math import log

def Entropy_value(total,pickup,grp,gp):
    p = gp/grp
    q = (grp-gp)/grp
    return -p*int(log(p,2))-q*int(log(q,2))

total=30
pick=15
Female = [10,2,8]
Male = [20,13,7]

e1 = Entropy_value(total,pick,Female[0],Female[1])
e2 = Entropy_value(total,pick,Male[0],Male[1])

Result = Female[0]/total * e1 + Male[0]/total * e2
print('%4.2f %4.2f %4.2f'%(e1,e2,Result))

classA = [14,6,8]
classB = [16,9,7]

e3 = Entropy_value(total,pick,classA[0],classA[1])
e4 = Entropy_value(total,pick,classB[0],classB[1])

Result1 = classA[0]/total * e3 + classB[0]/total * e4
print('%4.2f %4.2f %4.2f'%(e3,e4,Result1))
