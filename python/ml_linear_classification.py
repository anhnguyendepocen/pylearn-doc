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
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

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
    thres = 1 / 2 * np.dot(beta, (mean1 - mean0))
    return beta, thres, mean0_hat, mean1_hat, Cov_hat

X = np.vstack([X0, X1])
y_true = np.array([0] * X0.shape[0] + [1] * X1.shape[0])
beta, thres, mean0_hat, mean1_hat, Cov_hat = fisher_lda(X, y_true)
beta_nrom = beta / np.linalg.norm(beta)

y_proj = np.dot(X, beta)
y_pred = np.asarray(y_proj > thres, dtype=int)
errors = y_pred !=y_true 
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred)))

plt.scatter(X[errors, 0], X[errors, 1], color='k', marker="x", s=200, lw=2)
plt.scatter(X1[:, 0], X1[:, 1], color='r')
plt.scatter(X0[:, 0], X0[:, 1], color='b')
plt.scatter(mean0_hat[0], mean0_hat[1], color='b', s=200, label="m0")
plt.scatter(mean1_hat[0], mean1_hat[1], color='r', s=200, label="m2")
plot_cov_ellipse(Cov_hat, pos=mean0_hat, facecolor='none', linewidth=2, edgecolor='k', ls='--')
plot_cov_ellipse(Cov_hat, pos=mean1_hat, facecolor='none', linewidth=2, edgecolor='k', ls='--')
plt.arrow( mean0_hat[0], mean0_hat[1], beta_nrom[0], beta_nrom[1], linewidth=3, fc="k", ec="k", head_width=0.3, head_length=0.3)
plt.legend(loc='upper left')

sns.kdeplot(y_proj[y_true==0], shade=True, color='b')
sns.kdeplot(y_proj[y_true==1], shade=True, color='r')
plt.axvline(thres, color='k', ls='--')

beta, thres, mean0_hat, mean1_hat, Cov_hat = fisher_lda(X, y_true)



mean0_hat = mean0_hat[:, None]
mean1_hat = mean1_hat[:, None]
diff = mean1_hat - mean0_hat

B = np.dot(diff, diff.T)
#np.outer(diff, diff)
W = Cov_hat

u = np.random.randn(2, 1)
Wi = scipy.linalg.inv(W)
Wu = np.dot(W, u)
Bu = np.dot(B, u)


# u' B u /  u' W u
np.dot(u.T, Bu) / np.dot(u.T, Wu)

# u' (B . W^-1) u
np.dot(np.dot(u.T, np.dot(B, Wi)), u)





