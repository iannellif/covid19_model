import numpy as np
import matplotlib.pyplot as plt
import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')
from matplotlib.ticker import AutoMinorLocator
from numpy import exp
from scipy.optimize import curve_fit
from scipy.special import gammainc, gamma
mu =0.25


import warnings
warnings.filterwarnings("ignore",
	message="invalid value encountered in multiply")
warnings.filterwarnings("ignore",
	message="invalid value encountered in double_scalars")

def rSIR(t, A, B, C):
    return A * exp(B*t) + C

def phi(t, A, R_0):
    return gamma(A*R_0 + 1) * (gammainc(A*R_0 + 1, A + mu*t) -
        gammainc(A*R_0 + 1, A))

def rMF1(t, A, R_0, R0, RHO0):
    return R0 + RHO0 * exp(A)/(A**(A*R_0)) * phi(t, A, R_0)

def rMF2(t, eta, R_0, R0, RHO0):
    return R0 + RHO0 * exp(mu/eta)/((mu/eta)**(mu*R_0/eta)) * phi(t, mu/eta, R_0)

def rho(t, eta, g, R_0):
    return (1+eta*t)**g*exp(-mu*t)

def rMF(R0, RHO0, R_0):
    def make_rMFrMF(t, A):
        return R0 + RHO0 * exp(A)/(A**(A*R_0)) * phi(t, A, R_0)
    return make_rMFrMF

def rMFRHO0R_0(R0):
    def make_rMF(t, A, R_0, RHO0):
        return R0 + RHO0 * exp(A)/(A**(A*R_0)) * phi(t, A, R_0)
    return make_rMF

def rMFRHO0R_0eta(R0):
    def make_rMF(t, eta, R_0, RHO0):
        return R0 + RHO0 * exp(mu/eta)/((mu/eta)**(R_0*mu/eta)) * phi(t, mu/eta, R_0)
    return make_rMF

class fit_plot():
    def __init__(self, dat, func, name, popt, title, figs=(7.5, 4.5),
            str_fit_model='fit', yscale='log', ylabel='y', xlabel='x', t1=0):
        self.figs = figs
        self.dat = dat
        self.func = func
        self.str_fit_model = str_fit_model
        self.yscale = yscale
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.popt = popt
        self.name = name
        self.t1 = t1
        self.title = title
    def make(self):
        lw = .75
        lc, lc2, lc3 = (0.960, 0.721, 0.4), (0.776, 0.4, 0.960), (0.4, 0.960, 0.776)
        lca = (0.960, 0.721, 0.4, .4)
        t = np.arange(len(self.dat))
#         print('debug')
        if self.t1 > 0:
            t1r = np.arange(len(self.dat)+self.t1)
        else:
            t1r = t
        #
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=self.figs)
        #
#         print('debug')
#         eta, gam, R_0 = mu/popt[0], popt[0]*popt[1], popt[1]
#         print(eta, gam)
        lin, = ax.plot(t, self.dat, lw=lw, c=lc)
        mark, = ax.plot(t, self.dat, marker='^', ms=6, lw=lw, c=lc,
            markerfacecolor=(lca), markeredgecolor=lc, markeredgewidth=.75)
        fit, = ax.plot(t1r, self.func(t1r, *self.popt), '--', lw=.75, c=lc2)
#         rhot, = ax.plot(t1r, rho(t1r, eta, gam, R_0), '--', lw=.75, c=lc3)
        #
        ax.legend([(lin, mark), fit], ['dati', self.str_fit_model])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major', ls='--', lw=.25)
        ax.grid(True, which='minor', ls='--', lw=.08)
#         ax.grid(True, which='minor', ls='--', lw=.15)
        ax.set(yscale=self.yscale,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            title=self.title)
        #
        fig.tight_layout()
        fig.savefig(self.name+'.pdf')
#         fig.savefig(self.name+'.png', dpi=600)
        plt.close('all')
        print('OK - '+self.name+' plotted')