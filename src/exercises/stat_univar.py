import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(seed=42)  # make example reproducible


'''
## Estimator of main statistical measures

- Generate 2 ramdom samples $x \sim(1.78, 0.1)$, $y \sim(1.66, 0.1)$ both of size 10.

- Compute xbar $\bar{x}, \sigma_x, \sigma_{xy}$ using only `np.sum()` operation. 
Explore `np.` module to find out the numpy functions that does the same 
computations and compare them (using `assert`) with your previous results.
'''
n = 10
x  = np.random.normal(loc=1.78, scale=.1, size=n)
y  = np.random.normal(loc=1.78, scale=.1, size=n)

xbar = np.sum(x) / n
assert np.mean(x) == xbar

xvar = np.sum((x - xbar) ** 2) / (n - 1)
assert np.var(x, ddof=1) == xvar

ybar = np.sum(y) / n
xycov = np.sum((x - xbar) * (y - ybar)) / (n - 1)

xy = np.vstack((x, y))
Cov = np.cov(xy, ddof=1)  # or bias = True is the default behavior 
assert Cov[0, 0] == xvar
assert Cov[0, 1] == xycov
assert np.all(np.cov(xy, ddof=1) == np.cov(xy))

'''
###  One sample t-test (no IV)

- 
Given the following samples, test whether its true mean is 1.75.
Warning, when computing the std or the variance set ddof=1. The default
value 0, leads to the biased estimator of the variance.

'''
import scipy.stats as stats
n = 100
x = np.random.normal(loc=1.78, scale=.1, size=n)

'''
- Compute the t-value (tval)

- Plot the T(n-1) distribution for 100 tvalues values within [0, 10]. Draw P(T(n-1)>tval) 
  ie. color the surface defined by x values larger than tval below the T(n-1).
  Using the code.

- Compute the p-value: P(T(n-1)>tval).

- The p-value is one-sided: a two-sided test would test P(T(n-1) > tval)
  and P(T(n-1) < -tval). What would be the two sided p-value ?
  
- Compare the two-sided p-value with the one obtained by stats.ttest_1samp
using `assert np.allclose(arr1, arr2)`
'''


xbar, s, xmu, = np.mean(x), np.std(x, ddof=1), 1.75

tval = (xbar - xmu) / (s / np.sqrt(n))

tvalues = np.linspace(-10, 10, 100)
plt.plot(tvalues, stats.t.pdf(tvalues, n-1), 'b-', label="T(n-1)")
upper_tval_tvalues = tvalues[tvalues > tval]
plt.fill_between(upper_tval_tvalues, 0, stats.t.pdf(upper_tval_tvalues, n-1), alpha=.8)
plt.legend()

# Survival function (1 - `cdf`)
pval = stats.t.sf(tval, n - 1)

pval2sided = pval * 2
# do it with sicpy
assert np.allclose((tval, pval2sided), stats.ttest_1samp(x, xmu))


'''
###  Two sample t-test (no IV)

Given the following two sample, test whether their means are equals.

'''
import scipy.stats as stats
nx, ny = 50, 25
x = np.random.normal(loc=1.76, scale=.1, size=nx)
y = np.random.normal(loc=1.70, scale=.12, size=ny)

'''
- Compute the t-value.
'''

xbar, ybar = np.mean(x), np.mean(y)
xvar, yvar = np.var(x, ddof=1), np.var(y, ddof=1)

se = np.sqrt(xvar / nx + yvar / ny)

tval = (xbar - ybar) / se
stats.ttest_ind(x, y, equal_var=False)

tval

'''
Use the following function to approximate the df neede for the p-value
'''

def unequal_var_ttest_df(v1, n1, v2, n2):
    vn1 = v1 / n1
    vn2 = v2 / n2
    df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))
    return df

df = unequal_var_ttest_df(xvar, nx, yvar, ny)

'''
- Compute the p-value.

- The p-value is one-sided: a two-sided test would test P(T > tval)
  and P(T < -tval). What would be the two sided p-value ?
'''

pval = stats.t.sf(tval, df)
pval2sided = pval * 2


'''
- Compare the two-sided p-value with the one obtained by stats.ttest_ind
using `assert np.allclose(arr1, arr2)`
'''
# do it with sicpy
assert np.allclose((tval, pval2sided), stats.ttest_ind(x, y, equal_var=False))

'''
Plot of the two sample t-test
'''
xjitter = np.random.normal(loc=-1, size=len(x), scale=.01)
yjitter = np.random.normal(loc=+1, size=len(y), scale=.01)
plt.plot(xjitter, x, "ob", alpha=.5)
plt.plot(yjitter, y, "ob", alpha=.5)
plt.plot([-1, +1], [xbar, ybar], "or", markersize=15)

#left, left + width, bottom, bottom + height
#plt.bar(left=0, height=se, width=0.1, bottom=ybar-se/2)
## effect size error bar
plt.errorbar(-.1, ybar + (xbar - ybar) / 2, yerr=(xbar - ybar) / 2, 
             elinewidth=3, capsize=5, markeredgewidth=3,
             color='r')

plt.errorbar([-.8, .8], [xbar, ybar], yerr=np.sqrt([xvar, yvar]) / 2, 
             elinewidth=3, capsize=5, markeredgewidth=3,
             color='b')

plt.errorbar(.1, ybar, yerr=se / 2, 
             elinewidth=3, capsize=5, markeredgewidth=3,
             color='b')

plt.savefig("/tmp/two_samples_ttest.svg")
#plt.savefig("/tmp/two_samples_ttest.png")

plt.clf()

'''
## Simple linear regression (one continuous independant variable (IV))
'''


url = 'https://raw.github.com/duchesnay/pylearn-doc/master/data/salary_table.csv'
salary = pd.read_csv(url)
salary.E = salary.E.map({1:'Bachelor', 2:'Master', 3:'Ph.D'})
salary.M = salary.M.map({0:'N', 1:'Y'})

## Outcome
## S: salaries for IT staff in a corporation.

## Predictors:
## X: experience (years)
## E: education (1=Bachelor's, 2=Master's, 3=Ph.D)
## M: management (1=management, 0=not management)


from scipy.stats as stats
import numpy as np
y, x = salary.S, salary.X
beta, beta0, r_value, p_value, std_err = stats.linregress(x,y)

print("y=%f x + %f  r:%f, r-squared:%f, p-value:%f, std_err:%f" % (beta, beta0, r_value, r_value**2, p_value, std_err))

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

'''
- Compute the p-value:
  * Plot the F(1,n) distribution for 100 f values within [10, 25]. Draw P(F(1,n)>F) ie. color the surface defined by x values larger than F below the F(1,n).
  * P(F(1,n)>F) is the p-value, compute it.
'''

from scipy.stats as stats
fvalues = np.linspace(10, 25, 100)

plt.plot(fvalues, f.pdf(fvalues, 1, 30), 'b-', label="F(1, 30)")

upper_fval_fvalues = fvalues[fvalues > fval]
plt.fill_between(upper_fval_fvalues, 0, f.pdf(upper_fval_fvalues, 1, 30), alpha=.8)

# pdf(x, df1, df2): Probability density function at x of the given RV.
plt.legend()


# Survival function (1 - `cdf`)
pval = f.sf(fval, 1, n - 2)


## With statmodels
from statsmodels.formula.api import ols
model = ols('S ~ X', salary)
results = model.fit()
print(results.summary())

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
