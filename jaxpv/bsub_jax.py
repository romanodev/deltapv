import numpy as np
import time
import sys
import jax.numpy as jnp
from jax import random
import jax
from jax import lax
from jax import grad, jit, vmap

key = random.PRNGKey(1)

n = 10


#sys.setrecursionlimit(2*n)

#Jax---------------------------------------
A_jax = random.uniform(key,(n,n))*jnp.tri(n).T
b_jax = random.uniform(key,(n,1))
x_jax = jnp.linalg.solve(A_jax,b_jax)
#Initialize xcomp
xcomp_jax = jax.ops.index_update(np.zeros(n), jax.ops.index[-1],b_jax[-1,0]/A_jax[-1,-1])
#------------------------------------------

#Numpy-------------------------------------
A = np.random.randn(n, n) * np.tri(n).T
b = np.random.randn(n)
x = np.linalg.solve(A,b)
#Initialize xcomp
xcomp_numpy = np.zeros(n)
xcomp_numpy[-1] = b[-1]/A[-1,-1]
#------------------------------------------

def one_shot_numpy(xcomp):

 for i in range(n - 2, -1, -1):

    xcomp[i] = (b[i] - np.dot(A[i, i + 1:], xcomp[i + 1:])) / A[i, i]

 return xcomp



#Attempt at scan. https://gist.github.com/shoyer/dc33a5850337b6a87d48ed97b4727d29 
#It needs to be updated with dynamic slicing
def scan(xcomp_jax):

 def f(xcomp_jax,i):

    jnp.dot(A_jax[i, i + 1:], xcomp_jax[i + 1:])

    tmp = (b_jax[i] - jnp.dot(A_jax[i, i + 1:], xcomp_jax[i + 1:])) / A_jax[i, i]

    xcomp_jax = jax.ops.index_update(xcomp_jax, jax.ops.index[i],tmp[0])


 #xcomp_jax = lax.scan(f,xcomp_jax, jnp.arange(n-2,-1,-1))
 xcomp_jax = lax.scan(f,xcomp_jax, jnp.arange(n))

 return xcomp_jax



def one_shot_jax(xcomp_jax):

 for i in range(n - 2, -1, -1):

    tmp = (b_jax[i] - jnp.dot(A_jax[i, i + 1:], xcomp_jax[i + 1:])) / A_jax[i, i]

    xcomp_jax = jax.ops.index_update(xcomp_jax, jax.ops.index[i],tmp[0])

 return xcomp_jax

def recursive_numpy(xcomp,i):

    xcomp[i] = (b[i] - np.dot(A[i, i + 1:], xcomp[i + 1:])) / A[i, i]

    i -=1
    if i  == -1 : return xcomp

    return recursive_numpy(xcomp,i)

def recursive_jax(xcomp_jax,i):

    tmp = (b_jax[i] - jnp.dot(A_jax[i, i + 1:], xcomp_jax[i + 1:])) / A_jax[i, i]

    xcomp_jax = jax.ops.index_update(xcomp_jax, jax.ops.index[i],tmp[0])

    i -=1
    if i  == -1 : return xcomp_jax

    return recursive_jax(xcomp_jax,i)


a = time.time()
xcomp_numpy = one_shot_numpy(xcomp_numpy)
print(np.allclose(x,xcomp_numpy))
print('Numpy (One shot) : ', time.time()-a,np.allclose(x,xcomp_numpy))
xcomp_numpy = recursive_numpy(xcomp_numpy,n-2)
print('Numpy (Recursive): ', time.time()-a,np.allclose(x,xcomp_numpy))
xcomp_jax = one_shot_jax(xcomp_jax)
print('JAX   (One shot) : ', time.time()-a,np.allclose(x_jax.T[0],xcomp_jax))
xcomp_jax = recursive_jax(xcomp_jax,n-2)
print('JAX   (Recursive) : ', time.time()-a,np.allclose(x_jax.T[0],xcomp_jax))



#print(xcomp)
#print(x.T[0])

