from math import log
n=30
s=15
p=s/n
q=(n-s)/n
print(p,q)
Entropy=-p*int(log(p,2))-q*int(log(q,2))
print('Entropy:',Entropy)
Gains=p**2+q**2
print('Gains:',Gains)