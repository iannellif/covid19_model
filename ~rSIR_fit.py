import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
from numpy import exp
from matplotlib.ticker import 
from lmfit import Model
from scipy.optimize import curve_fit, least_squares
from scipy.special import gammainc, gamma
#
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
    \usepackage[utf8]{inputenc}
    \usepackage{mathtools}'''
plt.rcParams['hist.bins'] = 'auto'
#   model parameters
mu = 1/4
R_0F = 2.4
betaF = mu * R_0F
#   r(t) simple SIR model: A * exp(B*t) + C
def rSIR(t, A, B, C):
    return A * exp(B*t) + C

#   directories 
dir_IT = 'data/COVID-19-master_IT_2703/dati-andamento-nazionale/'
dir_IT_reg = 'data/COVID-19-master_IT_2703/dati-regioni/'
#   import
ITdat = pd.read_csv(dir_IT+'dpc-covid19-ita-andamento-nazionale.csv')
regdat = pd.read_csv(dir_IT_reg+'dpc-covid19-ita-regioni.csv')
#   fit
arr_casi = ITdat['totale_casi'][:9]
t = np.arange(len(arr_casi))
x0 = 1, 1, 1
popt, pcov = curve_fit(rSIR, t, arr_casi, x0)
#   values of r(0), rho(0), R_0
A, B, C = popt
R_0 = B/mu + 1
rho0 = A * B/mu
r0 = C + A
k = R_0/2.4
print(k, rho0, r0, C)
#   plot
fig, ax = plt.subplots(figsize=(5, 3))
lin, = ax.plot(t, arr_casi, lw=.75, c='red')
mark, = ax.plot(t, arr_casi, marker='^', ms=6, markerfacecolor=(1, 0, 0, .4),
    markeredgecolor='red', markeredgewidth=.75, lw=.75, c='red')
fit, = ax.plot(t, rSIR(t, *popt), lw=.75, c='blue')
ax.legend([(lin, mark), fit], ['dati', 'fit'])
ax.set_ylabel('Casi totali (Morti, contagiati, guariti)')
ax.set_xlabel('Giorni dal 24 Febbraio')
fig.tight_layout()
fig.savefig('plot/rSIR_ITdat_tot-20Febb.pdf')
plt.close(fig)
