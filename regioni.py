import fmod
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import locale
locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')
#   special import
from calendar import month_name
from scipy.optimize import curve_fit
#   rcParams set
plt.rc('font', family='serif', size=10)
plt.rc('text', usetex=True)
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
dir_plot = 'plot/'
dir_datPC = 'data/COVID-19_IT_'
dir_IT_reg = dir_datPC+str(day)+'/dati-regioni/'
#   import
regdat = pd.read_csv(dir_IT_reg+'dpc-covid19-ita-regioni.csv')
#   cleaning
regdat.data = regdat.data.str[5:10]
regdat.data = regdat.data.str.replace('-', '')
regdat['ospedalizzati_attivi'] = (regdat['totale_ospedalizzati'] - 
    regdat['deceduti'] - regdat['dimessi_guariti'])#    ospedalizzati_attivi
#   list regioni
d = {'ricoverati_con_sintomi': 'sum',
    'terapia_intensiva': 'sum',
    'totale_ospedalizzati': 'sum',
    'isolamento_domiciliare': 'sum',
    'totale_attualmente_positivi': 'sum',
    'nuovi_attualmente_positivi': 'sum',
    'dimessi_guariti': 'sum',
    'deceduti': 'sum',
    'totale_casi': 'sum',
    'tamponi': 'sum', 'ospedalizzati_attivi': 'sum',
    'denominazione_regione':'first',
    'codice_regione':'first', 'lat':'first', 'long':'first'}
regioni = []
for i in range(20):
    regioni.append(regdat[regdat['codice_regione'] == i+1].drop(
        columns=['stato', 'note_it', 'note_en']))

regioni[3] = regioni[3].groupby('data', as_index=False).aggregate(d).reindex(
    columns=regioni[3].columns)

"""|========================================================================|"""
#   global
N = 60317116
t1 = 20 # tempo in piu' dopo i dati
day1 = 13   # 8 Marzo (24 Febbraio + 13 d), giorno di inizio restrizioni
MIN_CASE_FIT = 300  # numero minimo di casi al day1 per iniziare il fit
giorno = lambda a, i : regioni[i]['data'].iloc[a][2:]
mese = lambda a, i : regioni[i]['data'].iloc[a][:2]
mese_n = lambda a: str(month_name[int(a)])
#
#
for i in range(20):
    reg_name = regioni[i].iloc[day1]['denominazione_regione']
    len_spacin = 80 - len(reg_name) - 2
    spacin =len_spacin*'-'
    print(reg_name+spacin+'>>')
    regioni[i].iloc[day1]['denominazione_regione']
    if regioni[i]['totale_casi'].max() < 5*MIN_CASE_FIT:
        print('Troppi pochi casi in '+reg_name)
    else:
        k = 0
        while regioni[i].iloc[day1+k]['totale_casi'] < MIN_CASE_FIT:
            k = k + 1
        day_tmp = day1+k
        casiMF = regioni[i]['totale_casi'][day_tmp:]
        t = np.arange(len(casiMF))
        
        R_00 = 3
        alpha0 = 4
        R0 = regioni[i].iloc[day_tmp]['totale_casi']
        x0 = alpha0, R_00, R0, (R0+5*R0)#(A, R_0, R0, RHO0), (A, R_0, RHO0)
        func = fmod.rMF1
        popt, pcov = curve_fit(func, t, casiMF, x0, maxfev = 50000)
        print(k, popt)
        title = '$\\eta = \\mu/\\alpha$ = {:3f}, $R_0$ = {:3f}'
        #
        plot = fmod.fit_plot(dat=casiMF,
            func=func, popt=popt,
            name=dir_plot+str(regioni[i]['codice_regione'].iloc[day_tmp])+'-' +\
                reg_name+'_tot_rMF-a_'+mese(day_tmp, i)+giorno(day_tmp, i),
            str_fit_model='fit $r(t) = r(0) + ' +
                '\\rho(0)\\frac{e^{\\alpha}}{\\alpha}\\phi(\\alpha, t)$',
            yscale='log', t1=t1+k,
            title=title.format(mu/popt[0], popt[1]),
            ylabel='totale\_casi '+reg_name,
            xlabel='giorni dal '+giorno(day_tmp, i)+' '+mese_n(mese(day_tmp, i)))
        plot.make()
        #
        print('STOP')








"""




plt.yscale('log')
for i in range(20):
    reg = regioni[i]['totale_casi']
    t = np.arange(len(reg))
    max(t)
    plt.plot(t, reg)"""









