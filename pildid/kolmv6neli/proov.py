#! /usr/bin/env python3
import numpy as np 

a=np.array([[1,2,3],[1,2,3],[1,2,3]])
b=np.array([[4,5,6],[4,5,6],[4,5,6]])
c=np.array([[7,8,9],[7,8,9],[7,8,9]])
e=np.array([[7,8,9],[7,8,9],[7,8,9]])

vaja=a 
print(a)
print(b)
print(c)

vaja=np.dstack([b,vaja])
vaja=np.dstack([c,vaja])
vaja=np.dstack([e,vaja])


print(vaja)
vaja.shape=(4,3,3)
print(vaja)
