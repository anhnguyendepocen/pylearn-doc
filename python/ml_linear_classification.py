# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:23:23 2016

@author: ed203246
"""

'''
Fisher's linear discriminant
============================
'''

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

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
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, label="toto", **kwargs)

    ax.add_artist(ellip)
    return ellip

n_samples, n_features = 100, 2
mean0, mean1 = np.array([0, 0]), np.array([0, 2])
Cov = np.array([[1, .8],[.8, 1]])
np.random.seed(42)
X0 = np.random.multivariate_normal(mean0, Cov, n_samples)
X1 = np.random.multivariate_normal(mean1, Cov, n_samples)

def fisher_lda(X, y):
    mean0_hat, mean1_hat = X[y_true == 0].mean(axis=0),  X[y_true == 1].mean(axis=0)
    Xcentered = np.vstack([(X[y_true == 0] - mean0_hat), (X[y_true == 1] - mean1_hat)])
    Cov_hat = np.cov(Xcentered.T)
    beta = np.dot(np.linalg.inv(Cov_hat), (mean1 - mean0))
    beta /= np.linalg.norm(beta)
    thres = 1 / 2 * np.dot(beta, (mean1 - mean0))
    return beta, thres, mean0_hat, mean1_hat, Cov_hat

X = np.vstack([X0, X1])
y_true = np.array([0] * X0.shape[0] + [1] * X1.shape[0])
beta, thres, mean0_hat, mean1_hat, Cov_hat = fisher_lda(X, y_true)
#beta_nrom = beta / np.linalg.norm(beta)

y_proj = np.dot(X, beta)
y_pred = np.asarray(y_proj > thres, dtype=int)
errors = y_pred != y_true 
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred)))

#%matplotlib inline
%matplotlib qt

from matplotlib import rc
plt.rc('text', usetex=True)
font = {'family' : 'serif'}
plt.rc('font', **font)

fig = plt.figure(figsize=(7, 7))
# Threshold coordinate. xy of the point equi-distant to m0, m1
thres_xy = thres * beta
# vector supporting the seprating hyperplane 
sep_vec = np.array([-beta[1], beta[0]])
# 2 points lying on the seprating hyperplane
sep_p1_xy = thres_xy - 5 * sep_vec
sep_p2_xy = thres_xy + 3 * sep_vec

err = plt.scatter(X[errors, 0], X[errors, 1], color='k', marker="x", s=100, lw=2)
plt.scatter(X1[:, 0], X1[:, 1], color='r')
plt.scatter(X0[:, 0], X0[:, 1], color='b')
m1 = plt.scatter(mean0_hat[0], mean0_hat[1], color='b', s=200, label="m0")
m2 = plt.scatter(mean1_hat[0], mean1_hat[1], color='r', s=200, label="m2")
plot_cov_ellipse(Cov_hat, pos=mean0_hat, facecolor='none', linewidth=2, edgecolor='m', ls='-')
Sw = plot_cov_ellipse(Cov_hat, pos=mean1_hat, facecolor='none', linewidth=2, edgecolor='m', ls='-')
#ax.legend([ell], ['label1'])
# projection vector
proj = plt.arrow(equi_xy[0], equi_xy[1], beta[0], beta[1], fc="k", ec="k", head_width=0.2, head_length=0.2, linewidth=2)
# Points along the separating hyperplance
hyper = plt.plot([sep_p1_xy[0], sep_p2_xy[0]], [sep_p1_xy[1], sep_p2_xy[1]], color='k', linewidth=4, ls='--')
plt.plot()

plt.legend([m1, m2, Sw, proj, err], ['$\mu_0$', '$\mu_1$', '$S_W$', "$w$", 'Errors'], loc='lower right', fontsize=18)


'''
Linear discriminant analysis (LDA)
==================================
'''
import numpy as np
from sklearn.lda import LDA

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dataset
n_samples, n_features = 100, 2
mean0, mean1 = np.array([0, 0]), np.array([0, 2])
Cov = np.array([[1, .8],[.8, 1]])
np.random.seed(42)
X0 = np.random.multivariate_normal(mean0, Cov, n_samples)
X1 = np.random.multivariate_normal(mean1, Cov, n_samples)
X = np.vstack([X0, X1])
y = np.array([0] * X0.shape[0] + [1] * X1.shape[0])

# LDA with scikit-learn
lda = LDA()
proj = lda.fit(X, y).transform(X)
y_pred = lda.predict(X)

errors =  y_pred != y
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred)))

# Use pandas & seaborn for convinience
data = pd.DataFrame(dict(x0=X[:, 0], x1=X[:, 1], y=["c"+str(v) for v in y]))
plt.figure()
g = sns.PairGrid(data, hue="y")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()



plt.figure()
for lab in np.unique(y):
    sns.distplot(proj.ravel()[y == lab], label=str(lab))

plt.legend()
plt.title("Distribution of projected data using LDA")


