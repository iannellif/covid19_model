import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')
"""useful functions of pandas
    .info()
    .sort_values(by=['data'])
    .drop(columns=['stato', 'note_it', 'note_en'])"""
#   special import
from numpy import exp
from calendar import month_name
from lmfit import Model
from scipy.optimize import curve_fit, least_squares
from scipy.special import gammainc, gamma
#   rcParams set
# plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"""
    \usepackage[utf8]{inputenc}
    \usepackage{mathtools}
    % more packages here
"""
#   fixed parameters
mu = 0.25
#   directories 
day = 3003
dir_dat = 'data/'
dir_datPC = 'data/COVID-19_IT_'
dir_IT = dir_datPC+str(day)+'/dati-andamento-nazionale/'
dir_IT_reg = dir_datPC+str(day)+'/dati-regioni/'
dir_plot = 'plot/'
#   import
ITdat = pd.read_csv(dir_IT+'dpc-covid19-ita-andamento-nazionale.csv')
regdat = pd.read_csv(dir_IT_reg+'dpc-covid19-ita-regioni.csv')
ITdat2= pd.read_csv(dir_dat+
    'COVID-19-geographic-disbtribution-worldwide-2020-03-26.csv',sep=';') 
#   cleaning
ITdat2 = ITdat2[ITdat2.Countries =='Italy']
ITdat.data = ITdat.data.str[5:10]
ITdat.data = ITdat.data.str.replace('-', '')
ITdat['ospedalizzati_attivi'] = (ITdat['totale_ospedalizzati'] - 
    ITdat['deceduti'] - ITdat['dimessi_guariti'])#  ospedalizzati_attivi
regdat.data = regdat.data.str[5:10]
regdat.data = regdat.data.str.replace('-', '')
regdat['ospedalizzati_attivi'] = (regdat['totale_ospedalizzati'] - 
    regdat['deceduti'] - regdat['dimessi_guariti'])#    ospedalizzati_attivi
#   def fit
def rSIR(t, A, B, C):
    return A * exp(B*t) + C

def phi(t, A, R_0):
    return gamma(A*R_0 + 1) * (gammainc(A*R_0 + 1, A + mu*t) -
        gammainc(A*R_0 + 1, A))

def rho(t, eta, g, R_0):
    return (1+eta*t)**g*exp(-mu*R_0/eta)

def rMF(R0, RHO0, R_0):
    def make_rMFrMF(t, A):
        return R0 + RHO0 * exp(A)/A * phi(t, A, R_0)
    return make_rMFrMF

def rMF1(t, A, R_0, R0, RHO0):
    return R0 + RHO0 * exp(A)/(A**(A*R_0)) * phi(t, A, R_0)

def rMFR_0(R0, RHO0):
    def make_rMFrMF(t, A, R_0):
        return R0 + RHO0 * exp(A)/A * phi(t, A, R_0)
    return make_rMFrMF

def rMFRHO0(R0, R_0):
    def make_rMFrMF(t, A, RHO0):
        return R0 + RHO0 * exp(A)/A * phi(t, A, R_0)
    return make_rMFrMF

def rMFRHO0R0(R_0):
    def make_rMFrMF(t, A, RHO0, R0):
        return R0 + RHO0 * exp(A)/A * phi(t, A, R_0)
    return make_rMFrMF

def rMFRHO0R_0(R0):
    def make_rMFrMF(t, A, RHO0, R_0):
        return R0 + RHO0 * exp(A)/A * phi(t, A, R_0)
    return make_rMFrMF

class fit_plot():
    def __init__(self, dat, func, name, popt, title, figs=(5, 3),
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
        lc, lc2 = (0.960, 0.721, 0.4), (0.776, 0.4, 0.960)
        lca = (0.960, 0.721, 0.4, .4)
        t = np.arange(len(self.dat))
        if self.t1 > 0:
            t1r = np.arange(len(self.dat)+self.t1)
        else:
            t1r = t
        #
        fig, ax = plt.subplots(figsize=self.figs)
        #
        lin, = ax.plot(t, self.dat, lw=lw, c=lc)
        mark, = ax.plot(t, self.dat, marker='^', ms=6, lw=lw, c=lc,
            markerfacecolor=(lca), markeredgecolor=lc, markeredgewidth=.75)
        fit, = ax.plot(t1r, self.func(t1r, *popt), '--', lw=.75, c=lc2)
        #
        ax.legend([(lin, mark), fit], ['dati', self.str_fit_model])
        ax.set(yscale=self.yscale,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            title=self.title)
        #
        fig.tight_layout()
        fig.savefig(dir_plot+self.name+'.pdf')
        plt.close('all')
        print('OK - '+self.name+' plotted')

"""PIANO DI LAVORO |===========================================================|
- il parametro r(0) si prende dai dati, il 24 Febbraio in italia il totale dei
    casi da Covid misurati e' 229 OPPURE si stima dal fit del modello SIR
        r(t) = A * exp(B*t) + C;                                             (1)
- il parametro rho(0) e' stimabile da questo fit, essendo legato ai parametri
    del fit A, B, C;
- analogamente vale per R_0, stimabile dalla conoscenza di A, B, C;
- il fit del modello con intervento viene preso dalla data dell'8 Marzo, quindi
    occorrono le quantita' r(8 Marzo) che si calcola inserendo t = 8/03 in (1),
    e analogamente rho(8 Marzo) si stima dall'espressione di rho(t) 
    er il modello SIR
        rho(t) = rho(0)*exp(-mu*t). 
    NOTA: 08/03 => ELEMENTO 13
|===========================================================================|"""
day0 = 0
day1 = 13
giorno = lambda a : ITdat['data'][a][2:]
mese = lambda a : ITdat['data'][a][:2]
mese_n = lambda a : str(month_name[int(a)])
"""     1. fit con modello SIR (rSIR)"""
casiSIR = ITdat['totale_casi'][:day1]
t = np.arange(len(casiSIR))
x0 = 10000, 1, 1
popt, pcov = curve_fit(rSIR, t, casiSIR, x0)
A, B, C = popt
R_0, rho0, r0 = B/mu + 1, A * B/mu, C + A
print(R_0, rho0, r0)
"""     1a. plot del fit SIR"""
plotSIR = fit_plot(dat=casiSIR,
    func=rSIR, popt=popt,
    name='ITdattot_rSIR_'+mese(day0)+giorno(day0)+'-'+mese(day1)+giorno(day1),
    str_fit_model='fit $r(t) = Ae^{Bt} + C$',
    yscale='log', title='$r(t)$',
    ylabel='totale\_casi',
    xlabel='giorni dal '+giorno(day0)+' '+mese(day0))
plotSIR.make()
#
"""     2. calcolo di r(day1), rho(day1)"""
#
r1 = rSIR(day1, A, B, C)
rho1 = rSIR(day1, rho0, mu*(R_0-1), 0)
#
"""     3. fit con modello Mean Field (rMF)"""
#
casiMF = ITdat['totale_casi'][day1:]
t = np.arange(len(casiMF))
t1 = 30
x0 = 1
func = rMF(r1, rho1, R_0)
popt, pcov = curve_fit(func, t, casiMF, x0)# maxfev = 5000
"""     3a. plot del fit MF di parametro A"""
plotSIR = fit_plot(dat=casiMF,
    func=func, popt=popt,
    name='ITdattot_rMF-a_'+mese(day1)+giorno(day1),
    str_fit_model='fit $r(t) = r(0) + ' +
        '\\rho(0)\\frac{e^{\\alpha}}{\\alpha}\\phi(\\alpha, t)$',
    yscale='log', t1=t1,
    ylabel='totale\_casi',
    xlabel='giorni dal '+giorno(day1)+' '+mese_n(mese(day1)))
plotSIR.make()
"""     4. fit con modello Mean Field con fit su R_0 (rMF)"""
casiMF = ITdat['totale_casi'][day1:]
t = np.arange(len(casiMF))
t1 = np.arange(len(casiMF)+30)
x0 = 1, 1
func = rMFR_0(r1, rho1)
popt, pcov = curve_fit(func, t, casiMF, x0)# maxfev = 5000
"""     4a. plot del fit MF di parametri A, R_0"""
fig, ax = plt.subplots(figsize=(5, 3))
lin, = ax.plot(t, casiMF, lw=.75, c=(0.960, 0.721, 0.4))
mark, = ax.plot(t, casiMF, marker='^', ms=6,
    markerfacecolor=(0.827, 0.572, 0.780, .4),
    markeredgecolor=(0.670, 0.450, 0.631, 1),
    markeredgewidth=.75, lw=.75, c=(0.960, 0.721, 0.4))
fit, = ax.plot(t1, func(t1, *popt), lw=.75, c=(0.682, 0.431, 0.949))
ax.set_yscale('log')
ax.legend([(lin, mark), fit], ['dati', 'fit $r(t) = r(0) + ' + 
    '\\rho(0)\\frac{e^{\\alpha}}{\\alpha}\\phi(\\alpha, t)$'])
ax.set_ylabel('Casi totali (Morti, contagiati, guariti)')
ax.set_xlabel('Giorni dal ' + giorno + ' ' + mese)
fig.tight_layout()
fig.savefig(dir_plot+'ITdat_tot_MFaR_0_'+str(day)+'.pdf')
plt.close(fig)
"""     5. fit con modello Mean Field con fit su RHO0 (rMF)"""
casiMF = ITdat['totale_casi'][day1:]
t = np.arange(len(casiMF))
t1 = np.arange(len(casiMF)+30)
x0 = 1, 1
func = rMFRHO0(r1, R_0)
popt, pcov = curve_fit(func, t, casiMF, x0)# maxfev = 5000
"""     5a. plot del fit MF di parametri A, RHO0"""
fig, ax = plt.subplots(figsize=(5, 3))
lin, = ax.plot(t, casiMF, lw=.75, c=(0.960, 0.721, 0.4))
mark, = ax.plot(t, casiMF, marker='^', ms=6,
    markerfacecolor=(0.827, 0.572, 0.780, .4),
    markeredgecolor=(0.670, 0.450, 0.631, 1),
    markeredgewidth=.75, lw=.75, c=(0.960, 0.721, 0.4))
fit, = ax.plot(t1, func(t1, *popt), lw=.75, c=(0.682, 0.431, 0.949))
ax.set_yscale('log')
ax.legend([(lin, mark), fit], ['dati', 'fit $r(t) = r(0) + ' + 
    '\\rho(0)\\frac{e^{\\alpha}}{\\alpha}\\phi(\\alpha, t)$'])
ax.set_ylabel('Casi totali (Morti, contagiati, guariti)')
ax.set_xlabel('Giorni dal ' + giorno + ' ' + mese)
fig.tight_layout()
fig.savefig(dir_plot+'ITdat_tot_MFaRHO0_'+str(day)+'.pdf')
plt.close(fig)
"""     5. fit con modello Mean Field con fit su RHO0, R0 (rMF)"""
casiMF = ITdat['totale_casi'][day1:]
t = np.arange(len(casiMF))
t1 = np.arange(len(casiMF)+30)
x0 = 1, 1, 1
func = rMFRHO0R0(R_0)
popt, pcov = curve_fit(func, t, casiMF, x0)# maxfev = 5000
"""     5a. plot del fit MF di parametri A, RHO0"""
fig, ax = plt.subplots(figsize=(5, 3))
lin, = ax.plot(t, casiMF, lw=.75, c=(0.960, 0.721, 0.4))
mark, = ax.plot(t, casiMF, marker='^', ms=6,
    markerfacecolor=(0.827, 0.572, 0.780, .4),
    markeredgecolor=(0.670, 0.450, 0.631, 1),
    markeredgewidth=.75, lw=.75, c=(0.960, 0.721, 0.4))
fit, = ax.plot(t1, func(t1, *popt), lw=.75, c=(0.682, 0.431, 0.949))
ax.set_yscale('log')
ax.legend([(lin, mark), fit], ['dati', 'fit $r(t) = r(0) + ' + 
    '\\rho(0)\\frac{e^{\\alpha}}{\\alpha}\\phi(\\alpha, t)$'])
ax.set_ylabel('Casi totali (Morti, contagiati, guariti)')
ax.set_xlabel('Giorni dal ' + giorno + ' ' + mese)
fig.tight_layout()
fig.savefig(dir_plot+'ITdat_tot_MFaRHO0R0_'+str(day)+'.pdf')
plt.close(fig)
"""REGIONI
"""
"""     1. """
#
regioni = []
for i in range(20):
    regioni.append(regdat[regdat['codice_regione'] == i+1])
#     casiMF = regioni[i]['totale_casi'][day1+10:]
#     t = np.arange(len(casiMF))
#     t1 = 30
#     x0 = 1, 100, 2, 10
#     R0 = regioni[i].iloc[day1]['totale_casi']
#     func = rMF1
#     popt, pcov = curve_fit(func, t, casiMF, x0, maxfev = 15000)
#     reg_name = regioni[i].iloc[day1]['denominazione_regione']
#     print(reg_name, popt)
#     plotSIR = fit_plot(dat=casiMF,
#         func=func, popt=popt,
#         name=reg_name+'tot_rMF-a_'+mese(day1)+giorno(day1),
#         str_fit_model='fit $r(t) = r(0) + ' +
#             '\\rho(0)\\frac{e^{\\alpha}}{\\alpha}\\phi(\\alpha, t)$',
#         yscale='log', t1=t1,
#         title='$\\alpha$={:3f}, $R_0$={:3f}, $\\rho(0)$={:3f}, $r(0)$={:3f}'.format(*popt),
#         ylabel='totale\_casi '+reg_name,
#         xlabel='giorni dal '+giorno(day1)+' '+mese_n(mese(day1)))
#     plotSIR.make()

regioni = []
for i in range(20):
    regioni.append(regdat[regdat['codice_regione'] == i+1])

MIN_CASE_FIT = 100
i = 1
k = 0
while regioni[i].iloc[day1+k]['totale_casi'] < MIN_CASE_FIT:
    k = k + 1

casiMF = regioni[i]['totale_casi'][day1+k:]
t = np.arange(len(casiMF))
t1 = 30
x0 = 2, 10, 10, 0.1#(A, R_0, R0, RHO0):
# R0 = regioni[i].iloc[day1]['totale_casi']
func = rMF1
popt, pcov = curve_fit(func, t, casiMF, x0)
reg_name = regioni[i].iloc[day1]['denominazione_regione']
print(reg_name, popt)










plotSIR = fit_plot(dat=casiMF,
    func=func, popt=popt,
    name=reg_name+'tot_rMF-a_'+mese(day1)+giorno(day1),
    str_fit_model='fit $r(t) = r(0) + ' +
        '\\rho(0)\\frac{e^{\\alpha}}{\\alpha}\\phi(\\alpha, t)$',
    yscale='log', t1=t1,
    title='$\\alpha$={:3f}, $R_0$={:3f}, $\\rho(0)$={:3f}, $r(0)$={:3f}'.format(*popt),
    ylabel='totale\_casi '+reg_name,
    xlabel='giorni dal '+giorno(day1)+' '+mese_n(mese(day1)))
plotSIR.make()













































''''Trash |---------------------------------------------------------------->>>>>


def rho(t, eta, R_0):
    return (1+eta*t)**(mu*R_0/eta)*exp(-mu*t)

alpha = popt[1]
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(t1, rho0*rho(t1, alpha/mu, R_0), lw=.75, c='red')
ax.set_xlabel('Giorni dal ' + giorno + ' ' + mese)
fig.tight_layout()
fig.savefig(dir_plot+'ITdat_tot_rho_'+str(day)+'.pdf')
plt.close(fig)

#___________________________________________________________________________
#   rMF plot
fig, ax = plt.subplots(figsize=(5, 3))
lin, = ax.plot(t, casiMF, lw=.75, c='red')
mark, = ax.plot(t, casiMF, marker='^', ms=6, markerfacecolor=(1, 0, 0, .4),
    markeredgecolor='red', markeredgewidth=.75, lw=.75, c='red')
fit, = ax.plot(t, rMF(t, *popt), lw=.75, c='blue')
ax.set_yscale('log')
ax.legend([(lin, mark), fit], ['dati', 'fit $r(t) = r(0) + \\rho(0)\\frac{e^{\\alpha}}{\\alpha}\\phi(\\alpha, t)$'])
ax.set_ylabel('Casi totali (Morti, contagiati, guariti)')
ax.set_xlabel('Giorni dal ' + ITdat['data'][day1][3:] + ' di ' + 
    str(month_name[int(ITdat['data'][day1][:2])]))
fig.tight_layout()
fig.savefig('plot/ITdat_total_AAA.pdf')
plt.clf()
#   rMF with ITdat2
casiMF = ITdat2['Cases']
casiMF = np.cumsum(ITdat2['Cases'][::-1]).to_numpy()[50:]
t = np.arange(len(casiMF))
x0 = 5000, 0.1, 1
popt, pcov = curve_fit(rMF, t, casiMF, x0, maxfev = 5000)
#   rMF plot
fig, ax = plt.subplots(figsize=(5, 3))
lin, = ax.plot(t, casiMF, lw=.75, c='red')
mark, = ax.plot(t, casiMF, marker='^', ms=6, markerfacecolor=(1, 0, 0, .4),
    markeredgecolor='red', markeredgewidth=.75, lw=.75, c='red')
fit, = ax.plot(t, rMF(t, *popt), lw=.75, c='blue')
ax.set_yscale('log')
ax.legend([(lin, mark), fit], ['dati', 'fit $r(t) = \\rho(0)\\frac{e^{\\alpha}}{\\alpha}\\phi(\\alpha, t)$'])
ax.set_ylabel('Casi totali (Morti, contagiati, guariti)')
ax.set_xlabel('Giorni')
fig.tight_layout()
fig.savefig(dir_plot+'ITdat2_total_AAA.pdf')
plt.clf()





#TAMPONI PER BALDI-------------------------------------------------------------

casiMF = ITdat['tamponi']
casiMFT = ITdat['totale_casi']
t = np.arange(len(casiMF))
x0 = 100, 1, 1

popt, pcov = curve_fit(rSIR, t, casiMF, x0)
popt1 = [671.3866389926839, 0.1868547024373594, -500.25302711774515]
fig, ax = plt.subplots(figsize=(6, 4))
lin, = ax.plot(t, casiMF, lw=.75, c='orange')
mark, = ax.plot(t, casiMF, marker='^', ms=6, markerfacecolor=(1, 0.584, 0.180, .4),
    markeredgecolor='orange', markeredgewidth=.75, lw=.75, c='orange')
lin1, = ax.plot(t, casiMFT, lw=.75, c='red')
mark1, = ax.plot(t, casiMFT, marker='^', ms=6, markerfacecolor=(1, 0, 0, .4),
    markeredgecolor='red', markeredgewidth=.75, lw=.75, c='red')
fit, = ax.plot(t1, rSIR(t1, *popt), '--',lw=.75, c=(0.682, 0.431, 0.949))
fit2, = ax.plot(t1, rSIR(t1, *popt1), '--', lw=.75, c='b')
ax.set_yscale('log')
ax.legend([(lin, mark), fit, fit2, (lin1, mark1)], ['dati tamponi', 'fit $\mathrm{tamponi}(t) = A e^{Bt} + C$ con'+' $B = {:2f}$'.format(popt[1]), 'contagiati SIR', 'dati casi'])
ax.set_ylabel('\# Tamponi cumulativi')
ax.set_xlabel('Giorni dal 24 Febbraio')
ax.set_title('Alla gentile attenzione del gruppo !!BALDICrazyTG24 COVID\\_19!?')
fig.tight_layout()
fig.savefig(dir_plot+'tamponi.pdf')
plt.clf()

#
#___________________________________________________________________________
"""
    pre-8th March exponential fitting of national growth (ITdat)
"""
t = np.array([i for i in range(len(ITdat['data'][:13]))])
para, pcov = curve_fit(rho_exp, t, ITdat['totale_casi'][:13])
s = rho_exp(t, para[0], para[1])
#
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(t, s)
ax.plot(t, ITdat['totale_casi'][:13])
ax.set_ylabel('Casi totali (Morti, contagiati, guariti)')
ax.set_xlabel('Giorni dal 24 Febbraio')
fig.tight_layout()
fig.savefig('plot/ITdat<8thMarch.pdf')
plt.clf()
"""
    post-8th March Gumbel & linearised Gumbel
"""
#   totale_casi con gumbel
arr_casi = ITdat['totale_casi'][13:]
t = np.array([i for i in range(len(arr_casi))])
t1 = np.array([i for i in range(len(arr_casi+50))])
x0 = 1000, 0.01, 1
func = gumbel
popt, pcov = curve_fit(func, t, arr_casi, x0, maxfev=5000)
#
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(t1, gumbel(t, *popt))
plt.plot(t, ITdat['totale_casi'][13:])
ax.set_ylabel('Casi totali (Morti, contagiati, guariti)')
ax.set_xlabel('Giorni dal 8 Marzo')
fig.tight_layout()
fig.savefig('plot/ITdat>8thMarch.pdf')
plt.clf()
""""


t = np.array([i for i in range(len(ITdat['data'][13:]))])
popt, pcov = curve_fit(rho_approx, t, ITdat['totale_casi'][13:], maxfev=5000)
s = rho_approx(t, popt[0], popt[1], popt[2])
plt.plot(t, s)
plt.plot(t, ITdat['totale_casi'][13:])
plt.savefig('plot/after8thMarchIT.pdf')
plt.clf()

x = np.array([i for i in range(len(ITdat['data'][13:]))])
y = ITdat['totale_casi'][13:]

model = Model(rho)
params = model.make_params()
result = model.fit(y, x=x, rho0=1, eta=1, ts=1)

print(result.fit_report())

plt.plot(x, y, 'bo')
plt.plot(x, result.init_fit, 'k--', label='initial fit')
plt.plot(x, result.best_fit, 'r-', label='best fit')
plt.legend(loc='best')
plt.show()




# plt.yscale('log')
# for i in range(reg_n):
#     regione = a[a['codice_regione'] == i+1].sort_values(by=['data'])
#     plt.plot(regione['data'], regione['totale_casi'])
# R_0 = 2.3 condizione iniziale


# def gumbel(t, a, b):
#     return a * b * np.exp(-b * np.exp(-a * t) - a * t)
# 
# def rho(x, rho0, eta):
#     K = np.exp(R_0)/R_0
#     return rho0 * (K/eta)**(mu/eta) * gumbel(x, eta, R_0)**(mu/eta)
