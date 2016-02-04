# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:09:56 2016

@author: ed203246
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from matplotlib.patches import Ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

n_samples, n_features = 100, 2
mean1, mean2 = np.array([0, 0]), np.array([0, 2])
Cov = np.array([[1, .8],[.8, 1]])

X1 = np.random.multivariate_normal(mean1, Cov, n_samples)
X2 = np.random.multivariate_normal(mean2, Cov, n_samples)

x = np.array([2, 2])

plt.scatter(X1[:, 0], X1[:, 1], color='b')
plt.scatter(X2[:, 0], X2[:, 1], color='r')
plt.scatter(mean1[0], mean1[1], color='b', s=200, label="m1")
plt.scatter(mean2[0], mean2[1], color='r', s=200, label="m2")
plt.scatter(x[0], x[1], color='k', s=200, label="x")
plot_cov_ellipse(Cov, pos=mean1, facecolor='none', linewidth=2, edgecolor='b')
plot_cov_ellipse(Cov, pos=mean2, facecolor='none', linewidth=2, edgecolor='r')
plt.legend(loc='upper left')

#
d2_m1x = scipy.spatial.distance.euclidean(mean1, x)
d2_m1m2 = scipy.spatial.distance.euclidean(mean1, mean2)

Covi = scipy.linalg.inv(Cov)
dm_m1x = scipy.spatial.distance.mahalanobis(mean1, x, Covi)
dm_m1m2 = scipy.spatial.distance.mahalanobis(mean1, mean2, Covi)

print('Euclidian dist(m1, x)=%.2f > dist(m1, m2)=%.2f' % (d2_m1x, d2_m1m2))
print('Mahalanobis dist(m1, x)=%.2f < dist(m1, m2)=%.2f' % (dm_m1x, dm_m1m2))


'''
- Write a function `euclidian(a, b)` that compute the euclidian distance
- Write a function `mahalanobis(a, b, Covi)` that compute the euclidian 
  distance, with the inverse of the covariance matrix. Use `scipy.linalg.inv(Cov)`
  to invert your matrix.
'''
def euclidian(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def mahalanobis(a, b, cov_inv):
    return np.sqrt(np.dot(np.dot((a - b), cov_inv),  (a - b).T))

assert mahalanobis(mean1, mean2, Covi) == dm_m1m2
assert euclidian(mean1, mean2)  == d2_m1m2
