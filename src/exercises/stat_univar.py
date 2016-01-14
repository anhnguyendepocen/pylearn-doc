# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:50:14 2016

@author: ed203246
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

url = 'ftp://ftp.cea.fr/pub/unati/people/educhesnay/pylearn_doc/data/salary_table.csv'
salary = pd.read_csv(url)
salary.E = salary.E.map({1:'Bachelor', 2:'Master', 3:'Ph.D'})
salary.M = salary.M.map({0:'N', 1:'Y'})

## Outcome
## S: salaries for IT staff in a corporation.

## Predictors:
## X: experience (years)
## E: education (1=Bachelor's, 2=Master's, 3=Ph.D)
## M: management (1=management, 0=not management)


from scipy import stats
import numpy as np
y, x = salary.S, salary.X
beta, beta0, r_value, p_value, std_err = stats.linregress(x,y)
print "y=%f x + %f  r:%f, r-squared:%f, p-value:%f, std_err:%f" % (beta, beta0, r_value, r_value**2, p_value, std_err)

# plotting the line
yhat = beta * x  +  beta0 # regression line
plt.plot(x, yhat, 'r-', x, y,'o')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.show()

## Exercise partition of variance formula.

## Compute:
## $\bar{y}$ `y_mu`

y_mu = np.mean(y)

## $SS_\text{tot}$: `ss_tot`

ss_tot = np.sum((y - y_mu) ** 2)

## $SS_\text{reg}$: `ss_reg`
ss_reg = np.sum((yhat - y_mu) ** 2)

## $SS_\text{res}$: `ss_res`
ss_res = np.sum((y - yhat) ** 2)

## Check partition of variance formula based on SS using `assert np.allclose(val1, val2, atol=1e-05)`
assert np.allclose(ss_tot - (ss_reg + ss_res), 0, atol=1e-05)

## What np.allclose does ?

## What assert does

## What is it worth for ?

## Compute $R^2$ and compare with `r_value` above
r2 = ss_reg / ss_tot

assert np.sqrt(r2) == r_value

## Compute F score
n = y.size
fval = ss_reg / (ss_res / (n - 2))

## Plot the F(1, n) distribution for 100 f values within [10, 25] 
## Depict P(F(1, n) > F) ie. folor the surface defined by x values larger than F beloww the F(1, n)
fvalues = np.linspace(10, 25, 100)

plt.plot(fvalues, f.pdf(fvalues, 1, 30), 'b-', label="F(1, 30)")

upper_fval_fvalues = fvalues[fvalues > fval]
plt.fill_between(upper_fval_fvalues, 0, f.pdf(upper_fval_fvalues, 1, 30), alpha=.8)

# pdf(x, df1, df2): Probability density function at x of the given RV.
plt.legend()

## P(F(1, n) > F) is the p-value, compute it

from scipy.stats import f
# Survival function (1 - `cdf`)
pval = f.sf(fval, 1, n - 2)



## With statmodels
from statsmodels.formula.api import ols
model = ols('S ~ X', salary)
results = model.fit()
print results.summary()
smry.tables[0].data

## sklearn
import sklearn.feature_selection
#sklearn.feature_selection.f_regression??
sklearn.feature_selection.f_regression(x.reshape((n, 1)), y)

"""
## center = True
## degrees_of_freedom = y.size - (2 if center else 1)
## F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
## pv = stats.f.sf(F, 1, degrees_of_freedom)

Comapre the residual sum of squares of two models, 1 and 2, where model 1 is 'nested' within model 2. Model 1 is the Restricted model, and Model 2 is the Unrestricted one. That is, model 1 has $p1$ parameters, and model 2 has $p2$ parameters, where $p2 > p1$. The model with more parameters will always be able to fit the data at least as well as the model with fewer parameters. Thus typically model 2 will give a better (i.e. lower error) fit to the data than model 1. But one often wants to determine whether model 2 gives a significantly better fit to the data.

If there are $n$ data points to estimate parameters of both models from, then one can calculate the F statistic, given by

$$
    F = \frac{\left(\frac{\text{RSS}_1 - \text{RSS}_2 }{p_2 - p_1}\right)}{\left(\frac{\text{RSS}_2}{n - p_2 + 1}\right)} ,
$$

where RSSi is the residual sum of squares of model i. Under the null hypothesis that model 2 does not provide a significantly better fit than model 1, F will have an F distribution, with ($p_2-p_1, n-p_2+1$) degrees of freedom. The null hypothesis is rejected if the F calculated from the data is greater than the critical value of the F-distribution for some desired false-rejection probability (e.g. 0.05).

In our case: $\text{RSS}_1=SS_\text{tot}$, $\text{RSS}_2 = SS_\text{res}$, thus $\text{RSS}_1 - \text{RSS}_2 = SS_\text{reg}$ $p_1=1$, $p_2=2$.
"""
