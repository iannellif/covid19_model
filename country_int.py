import fmod
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#   rcParams set
plt.rc('font', family='serif', size=15)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"""
    \usepackage[utf8]{inputenc}
    \usepackage{mathtools}
    % more packages here
"""
#
day = 0402
dir_dat = 'data/'
dir_plot = 'plot/'
dir_datPC = dir_dat+'COVID-19_WORLD_'+str(day)+'/csse_covid_19_data/'\
    'csse_covid_19_time_series/'
dat_raw = pd.read_csv(dir_datPC+'time_series_covid19_confirmed_global.csv')
dat = dat_raw.drop(columns=['Lat', 'Long'])
#
mu = 0.25
MIN_CASE_FIT = 100
MIN_CASE_FIT1 = 1000
c_lab = 'Country/Region'
p_lab = 'Province/State'
c_list = ['Italy', 'US', 'Korea, South', 'Germany', 'China', 'France', 'Spain']
c_europe = ['Italy', 'France', 'Germany', 'Spain']

#
agg_p = {'Province/State': 'first', '1/22/20': 'sum', '1/23/20': 'sum', '1/24/20': 'sum', '1/25/20': 'sum',
    '1/26/20': 'sum', '1/27/20': 'sum', '1/28/20': 'sum', '1/29/20': 'sum',
    '1/30/20': 'sum', '1/31/20': 'sum', '2/1/20': 'sum', '2/2/20': 'sum',
    '2/3/20': 'sum', '2/4/20': 'sum', '2/5/20': 'sum',  '2/6/20': 'sum',
    '2/7/20': 'sum', '2/8/20': 'sum', '2/9/20': 'sum', '2/10/20': 'sum',
    '2/11/20': 'sum', '2/12/20': 'sum', '2/13/20': 'sum', '2/14/20': 'sum',
    '2/15/20': 'sum', '2/16/20': 'sum', '2/17/20': 'sum', '2/18/20': 'sum',
    '2/19/20': 'sum', '2/20/20': 'sum', '2/21/20': 'sum', '2/22/20': 'sum',
    '2/23/20': 'sum', '2/24/20': 'sum', '2/25/20': 'sum', '2/26/20': 'sum',
    '2/27/20': 'sum', '2/28/20': 'sum', '2/29/20': 'sum', '3/1/20': 'sum',
    '3/2/20': 'sum',  '3/3/20': 'sum',  '3/4/20': 'sum',  '3/5/20': 'sum',
    '3/6/20': 'sum',  '3/7/20': 'sum',  '3/8/20': 'sum',  '3/9/20': 'sum',
    '3/10/20': 'sum', '3/11/20': 'sum', '3/12/20': 'sum', '3/13/20': 'sum',
    '3/14/20': 'sum', '3/15/20': 'sum', '3/16/20': 'sum', '3/17/20': 'sum',
    '3/18/20': 'sum', '3/19/20': 'sum', '3/20/20': 'sum', '3/21/20': 'sum',
    '3/22/20': 'sum', '3/23/20': 'sum', '3/24/20': 'sum', '3/25/20': 'sum',
    '3/26/20': 'sum', '3/27/20': 'sum', '3/28/20': 'sum', '3/29/20': 'sum',
    '3/30/20': 'sum', '3/31/20': 'sum'}
#   data cleaning
def cclean(dat, c):
    d_tmp = dat[dat[c_lab] == c].groupby(c_lab, as_index=False).agg(agg_p)
    dat.drop(dat[dat[c_lab] == c].index , inplace=True)
    dat = dat.append(d_tmp, sort=False).sort_values(by=[c_lab]
        ).reset_index(drop=True)
    dat.loc[dat[c_lab] == c, p_lab] = np.nan
    return dat

dat = cclean(dat, 'China')
dat = cclean(dat, 'Canada')
dat = cclean(dat, 'Australia')
cond = ~dat['Province/State'].isin([np.nan])
dat.drop(dat[cond].index, inplace = True)
#
def find_start(c_arr, MIN):
    k=0
    while c_arr[k] < MIN:
        k = k + 1
    return k

def make_arr(c):
    return dat[dat[c_lab]==c].drop(columns=[c_lab, p_lab]).to_numpy().flatten()

def fit_one(c, func, x0, t_star, maxfev=90000):
    c_arr = make_arr(c)
#     k = find_start(c_arr, MIN_CASE_FIT)
    k = t_star
    tot_cases = c_arr[k:]
    t = np.arange(len(tot_cases))
    popt, pcov = curve_fit(func, t, tot_cases, x0, maxfev=maxfev)
    print(c, k, '\n', popt, '\n', np.sqrt(pcov.diagonal()))
    return popt, pcov

def fit_countries(c_arr, k, func, x0):
    tot_cases = c_arr[k:]
    t = np.arange(len(tot_cases))
    popt, pcov = curve_fit(func, t, tot_cases, x0, maxfev=70000)
    print(c, k, '\n', popt, '\n', np.sqrt(pcov.diagonal()))

def error_Dtc(a, R_0, Da, DR_0):
    R_0err = a/mu * DR_0
    aerr = (1-R_0)/mu * Da
    return np.sqrt(R_0err**2 + aerr**2)
#   plot countries total cases
# marker = itertools.cycle(3*['.', '+', '*', '.', '1', 'p', 's', 'H', 'D'])
# fig, ax = plt.subplots(figsize=(10, 6))
# plt.yscale('log')
# for c in c_list1:
#     ydat_t = make_arr(c)
#     k = find_start(ydat_t, MIN_CASE_FIT1)
#     ydat = ydat_t[k:]
#     if len(ydat) > 40:
#         ydat = ydat_t[k:k+50]
#     xdat = np.arange(len(ydat))
#     ax.plot(xdat, ydat, ls='-', marker=next(marker), label=c)
# 
# plt.legend()
# fig.tight_layout()
# fig.savefig(dir_plot+'TOT_count_cases'+str(MIN_CASE_FIT1)+'.pdf')












dict_dateCount = {'Italy': '3/4/20', 'US': '3/16/20', 'Spain': '3/13/20',
    'France': '3/13/20', 'United Kingdom': '3/20/20', 'Switzerland': '3/13/20',
    'Germany': '3/13/20'}

##
out_file = open("nations.txt","w")
rho_val = []
#   fit c_list1
for c in dict_dateCount:
    c_arr = make_arr(c)
    t_star = dict_dateCount[c][:4]
    k = dat.columns.get_loc(dict_dateCount[c])
#     np.savetxt(dir_dat+'dat_gnuplot'+c+'.dat', np.c_[np.arange(len(c_arr[t_star:])), c_arr[t_star:]])
#     k = find_start(c_arr, MIN_CASE_FIT)
#     k = t_star
    x0 = 4, 3, 100
    func = fmod.rMFRHO0R_0eta(c_arr[k])
    popt, pcov = fit_one(c, func, x0, k)
    #
    eta, R_0, rho0 = popt
    Deta, DR_0, Drho0 = np.sqrt(np.diag(pcov))
    kav = popt[1]/2.4
    tM, gam = (R_0-1)/eta, mu*R_0/eta
    #
    Dtc = error_Dtc(mu/eta, R_0, Deta, DR_0)
    Dkav = DR_0/2.4
    Dgam = mu*R_0*Deta/(eta**2)
    
    out_file.write(c+r"& \num{{{:.2f}}} $\pm {:.2f}$ & \num{{{:.2f}}} {{\tiny $\pm {:.2f}$}} & \num{{{:.2f}}}"\
        r" {{\tiny $\pm {:.2f}$}} & \num{{{:d}}} {{\tiny $\pm {:d}$}} &"\
        r"{}\\".format(R_0, DR_0, kav, Dkav, eta, Deta, int(tM), int(Dtc), 
        t_star)+'\n')
    with plt.style.context('seaborn-colorblind'):
        popt_IT = popt
        pcov_IT = pcov
        fig, ax1 = plt.subplots()
        ax1.set_yscale('log')
        tot_cases = c_arr[k:]
        coltri = ["darkred", "skyblue", "orange"]
        x = np.arange(len(tot_cases))
        ax1.plot(x, tot_cases, 'o-', color=coltri[2], mfc='none', lw=1,
            label=r'data')
        x = np.arange(80)
        ax1.plot(x, func(x, *popt_IT), '--', color='black', lw=1, label=r'fit')
        left, bottom, width, height = [0.57, 0.25, 0.3, 0.3]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(x, fmod.rho(x, eta, gam, R_0), 'o', alpha=0.7, lw=1,
            ms = 3,mfc='none', color='purple', label=r'$\rho(t)$')# 
#     ax1.axvline(tM, ls='--', lw=1, label='$t_M$')
#     ax2.axvline(tM, ls='--', lw=1, label='$t_M$')
        ax1.legend(loc=1,prop={'size':15},frameon=True)
        ax1.set_xlabel(r"days since 100 cases", fontsize=15)
#     ax2.set_xlabel(r"days since 100 cases", fontsize=15)
        ax1.set_ylabel(r"cases", fontsize=15)
#         ax2.set_ylabel(r"$\rho(t)$", fontsize=10)
        ax1.tick_params(which = 'major', axis='both', width=1.5, length = 10, 
            labelsize=15,direction='in')
        ax1.tick_params(which = 'minor', axis='both', width=1.5, length = 5, 
            labelsize=15,direction='in')
        ax1.set_title(c)
    #plt.xticks(fontsize = 20)
    #plt.yticks(fontsize = 20)
#     ax1.set_yscale('log')
#     plt.grid(True)
    # plt.xscale('log')
#     plt.title('ITALY',fontsize = 20)
        fig.savefig(dir_plot+'fla'+c+'.pdf')
    
    
    
#     rho_val.append([rho0, eta, gam, R_0])
#     #
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plt.yscale('log')
#     t = np.arange(50)
#     t1 = np.arange(len(c_arr[k:]))
#     ax.plot(t1, c_arr[k:], marker='o', label=c)
#     ax.plot(t, func(t, *popt))
#     plt.legend()
#     fig.savefig(dir_plot+'fit'+str(MIN_CASE_FIT)+c+'.pdf')

out_file.close()
# fig, ax = plt.subplots(figsize=(10, 6))
# plt.yscale('log')
# t = np.arange(100)
# #     fig.savefig(dir_plot+'rho'+c+'.pdf')
# for i,c in enumerate(c_good):
#     ax.plot(t, rho_val[i][0]*fmod.rho(t, rho_val[i][1], rho_val[i][2], rho_val[i][3]))
# 
# fig.savefig(dir_plot+'rho.pdf')
#   fit lista di paesi
# for c in c_list:
#     c_arr = make_arr(c)
#     k = find_start(c_arr, MIN_CASE_FIT)
#     x0 = 1, 3, 10#
#     print(c_arr[k])
#     func = fmod.rMFRHO0R_0(c_arr[k])
#     fit_countries(c_arr, k, func, x0)



'''


#   fit singolo paese

#   fit lista di paesi
for c in c_list:
    c_arr = dat[dat[c_lab]==c].drop(columns=[c_lab, p_lab]).to_numpy().flatten()
    k = find_start(c_arr, MIN_CASE_FIT)
    x0 = 1, 3, 10#
    print(c_arr[k])
    func = fmod.rMFRHO0R_0(c_arr[k])
    fit_countries(c_arr, k, func, x0)
# 
# for c in c_list:
#     c_arr = dat[dat[c_lab]==c].drop(columns=[c_lab, p_lab]).to_numpy().flatten()
#     x0 = 3, 5, 1000, 11#
#     func = fmod.rMF1
#     fit_countries(c_arr, k, func, x0)
# 




'''




