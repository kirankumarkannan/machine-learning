#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import scipy.io



def normalize(X, mean, std):
    X = (X - mean) / std

    return X

    
def pca(X): 
    m, n = X.shape 
    mean = np.mean(X ,axis=0)
    
    # Compute covariance matrix 
    C = np.dot(X.T, X) / (m-1)
    
    # Eigen decomposition 
    eigen_values, eigen_vectors = np.linalg.eig(C) 

    orig_PCA = np.dot(X, eigen_vectors)

    print('PCA Matrix', orig_PCA)

    #Reduce dimensions
    
    X_pca = np.dot(X, eigen_vectors[:,:2])
    
    #Top 2 eigen vectors
    pc = eigen_vectors.T[0:2]

    pc = np.dot(pc, pc.T)
    print('Feature vectors of PCA ', pc)
    plt.scatter(X_pca[:,0], X_pca[:,1], color='red', label='PCA')
    plt.plot(pc[0], label='PCA_eigen_vector1', color = 'yellow')
    plt.plot(pc[1], label='PCA_eigen_vector2', color = 'purple')
    plt.legend(loc=(1.04,0))
    return X_pca, orig_PCA

def svd(X):
    m, n = X.shape
 
    U, sigma, V = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    
    # Reduce Dimensions
    X_svd = np.dot(U, np.diag(sigma)[:,:2])
    
    orig_svd = np.dot(U, np.diag(sigma))
    
    print('SVD Data instances',orig_svd)

    A = np.dot(np.dot(V,np.diag(sigma**2)),V.T) / (m-1)
    eigen_values, eigen_vectors = np.linalg.eig(A)
    pc= eigen_vectors.T[0:2]
    pc = np.dot(pc, pc.T)
    print("Feature vectors of SVD", pc)
    plt.scatter(X_svd[:,0], X_svd[:,1], color='blue', label='SVD')
    plt.plot(pc[0], label='SVD_PC1', color = 'yellow')
    plt.plot(pc[1], label='SVD_PC2', color = 'purple')
    plt.legend(loc=(1.04,0))
    return X_svd, orig_svd


file = 'cars.mat'
data = scipy.io.loadmat(file)

X = data['X']
X = pd.DataFrame(X)
name_col = points['names']
names = pd.DataFrame(name_col)
names = np.array(names)
print(names.item(1))

#Ignoring unnecessary columns from the dataset
X = X.iloc[:,7:18]
m, n = X.shape

meanX = np.mean(X, axis=0)
stdX = np.std(X,axis=0)
sumX = np.sum(X,axis=0)

#Apply Normalization here
X = normalize(X, meanX, stdX)

#Apply PCA here
X_pca, orig_pca = pca(X)
X_svd, orig_svd = svd(X)

print('PCA projection: ', X_pca)
print('SVD projection: ', X_svd)

#Check PCA and SVD vectors similarity from the folowing statements
if (orig_pca==orig_svd).all():
    print('PCA and SVD vectors are same')
else:
     print('PCA and SVD vectors are not same')   
        
#Check if the projections are same in the statements
if (X_pca == X_svd).all():
    print('PCA and SVD projections are same')
else:
    print('PCA and SVD projections are not same')
    
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA and SVD projections')
plt.figure(figsize=(60, 20), dpi=80, facecolor='w', edgecolor='k')
plt.show()


# In[ ]:




