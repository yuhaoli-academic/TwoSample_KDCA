# %%
import sys
from pathlib import Path

# Get the directory of the current script
script_dir = Path(__file__).parent.absolute()

# Add the script's directory to Python's module search path
sys.path.insert(0, str(script_dir))

from functions import *


# %%
mm =100
nn = 100
j = int(1)
Nb =  500
Nrep = 1000
dd_candidates = [5,10,25,50,75,100,150]
# %%
print("p=0, Gaussian")
p = 0

for d in dd_candidates:
    print("multi kernel:")
    print(f"d={d}:",multi(mm,nn,d,p,Nrep = Nrep,Nb=Nb,j=j))
    

    print("single kernel:")
    print(f"d={d}:",single(mm,nn,d,p,Nrep = Nrep,Nb=Nb,j=j))
   

    print("MMD with permutation:")
    print(f"d={d}:",mmd(mm,nn,d,p,Nrep = Nrep,Nb=Nb))
    
# %%
print("p=1, t-distribution")
p = 1

for d in dd_candidates:
    X = X_gen(mm, d, p)
    Y = Y_gen(nn, d, p)
    print("multi kernel:")
    print(f"d={d}:",multi(mm,nn,d,p,Nrep = Nrep,Nb=Nb,j=j))
    

    print("single kernel:")
    print(f"d={d}:",single(mm,nn,d,p,Nrep = Nrep,Nb=Nb,j=j))
   

    print("MMD with permutation:")
    print(f"d={d}:",mmd(mm,nn,d,p,Nrep = Nrep,Nb=Nb))

# %%
print("p=0.5, mixture distribution with equal probability")
p = 0.5

for d in dd_candidates:
    X = X_gen(mm, d, p)
    Y = Y_gen(nn, d, p)
    print("multi kernel:")
    print(f"d={d}:",multi(mm,nn,d,p,Nrep = Nrep,Nb=Nb,j=j))
    

    print("single kernel:")
    print(f"d={d}:",single(mm,nn,d,p,Nrep = Nrep,Nb=Nb,j=j))
   

    print("MMD with permutation:")
    print(f"d={d}:",mmd(mm,nn,d,p,Nrep = Nrep,Nb=Nb))
# %%
