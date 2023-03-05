import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
iso_st = '3166-1'
f = open('iso'+iso_st+'.json', 'r')
country_dict = json.loads(f.read())
import warnings
warnings.filterwarnings("ignore",
    message="invalid value encountered in double_scalars")
#
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from numpy import exp, log, loadtxt, pi, sqrt
from scipy.special import gammainc, gamma
from scipy.optimize import curve_fit
#
plt.rc('font', family='sans-serif', size=15)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"""
    \usepackage[utf8]{inputenc}
    \usepackage{mathtools}
    % more packages here
"""
#
dir_d = 'data/'
dir_dc = dir_d+'clean/'
dir_plot = 'plot/'
cases_inspect = 'deaths'
name = 'cleaned_covid19_'+cases_inspect+'_global.csv'
c_lab = 'Country/Region'
#
mu=1/14
#
def rho(t, eta, R_0):
    return (1+eta*t)**(mu*R_0/eta)*exp(-mu*t)
def phi(t, a, R_0):
    return gamma(a*R_0 + 1) * (gammainc(a*R_0 + 1, a + mu*t) - 
        gammainc(a*R_0 + 1, a))
def rMF(r0):
    def make_rMF(t, eta, R_0, rho0):
        a = mu/eta
        return r0 + rho0 * exp(a)/(a**(a*R_0))*phi(t, a, R_0)
    return make_rMF
def error_Dtc(a, R_0, Da, DR_0):
    R_0err = a/mu * DR_0
    aerr = (1-R_0)/mu * Da
    return np.sqrt(R_0err**2 + aerr**2)
def error_DrhotM(tM, DtM, rho0, Drho0, eta, Deta, R_0, DR_0):
    rho_f = rho(tM, eta, R_0)
    rho_t = rho0*rho_f
    D_rho0 = Drho0*rho_f
    D_tM = DtM*mu*rho_t*(R_0/(1+eta*tM)-1)
    D_eta = Deta*rho_t*(mu*R_0/(eta*(1+eta*tM))-mu*R_0/eta**2*log(1+eta*tM))
    D_R_0 = DR_0*rho_t*mu/eta*log(1+eta*tM)
    return np.sqrt(D_rho0**2 + D_tM**2 + D_eta**2 + D_R_0**2)
#
dict_dateCount = {#'Korea, South':'2/25/20',
#     'Japan': '3/2/20',
    'Italy': '3/4/20',
    'Brazil': '3/12/20',
    'Norway': '3/12/20',
    'Belgium': '3/13/20',
    'Switzerland': '3/13/20',
    'Germany': '3/13/20',
    'Spain': '3/13/20',
    'France': '3/13/20',
#     'Israel': '3/13/20',
    'Austria': '3/16/20',
#     'Canada': '3/16/20',
    'Netherlands': '3/16/20',
    'US': '3/16/20',
#     'Turkey':'3/16/20',
    'Sweden': '3/18/20',
    'United Kingdom': '3/20/20' 
    }
df = pd.read_csv(dir_dc+name)
#
ofname = 'latex_table_countries_'+cases_inspect+'.tex'
out_file = open(ofname,'w')
for c in dict_dateCount:
    ts = df.columns.get_loc(dict_dateCount[c])
    cdf = df[df[c_lab]==c].to_numpy().flatten()
    cn = cdf[0]
    cdf_t = np.array(list(np.delete(cdf, 0)))
    cdf = cdf_t[ts:]
    t = np.arange(len(cdf))
    #
    x0 = 0.07, 3, 100
    func = rMF(cdf[0])
    popt, pcov = curve_fit(func, t, cdf, x0, maxfev = 50000)#maxfev=70000
    #
    eta, R_0, rho0 = popt
    tc = (R_0-1)/eta
    err_arr = Deta, DR_0, Drho0 = np.sqrt(np.diag(pcov))
    kav = R_0/2.4
    tc = (R_0-1)/eta
    #
    Dtc = error_Dtc(mu/eta, R_0, Deta, DR_0)
    Dkav = DR_0/2.4
    rhotc = rho0 * rho(tc, eta, R_0)
    Drhotc = error_DrhotM(tc, Dtc, rho0, Drho0, eta, Deta, R_0, DR_0)
    #
    idx = [i for i, d in enumerate(country_dict[iso_st])
    if c in d.values()][0]
    abb_c = country_dict[iso_st][idx]['alpha_2']
    date = datetime.strptime(dict_dateCount[c], '%m/%d/%y')
    date2 = datetime.strftime(date,'%m-%d')
    date3 = datetime.strftime(date,'%d/%m')
    date = datetime.strftime(date,'%d %b')
    out_file.write(abb_c+" & "+date3+r"& \num{{{:.2f}}} $\scriptstyle\pm "\
        r"\num{{{:.2f}}}$ & \num{{{:.2f}}} $\scriptstyle\pm \num{{{:.2f}}}$ "\
        r"& \num{{{:.3f}}} $\scriptstyle\pm \num{{{:.3f}}}$ & \num{{{:d}}} "\
        r"\newline $\scriptstyle\pm \num{{{:d}}}$\\".format(R_0, DR_0, kav, 
            Dkav, eta, Deta, int(tc), int(Dtc))+'\n')
    #
    x_minus_shift = 10
    tf = np.arange(0, 80)
    with plt.style.context('seaborn-colorblind'):
        coltri = ["darkred", "skyblue", "orange"]
        left, bottom, width, height = [0.57, 0.2, 0.3, 0.3]
        low, upp = popt - err_arr, popt + err_arr
        if any(x<0 for x in low) == True:
            low = popt*0.5
        # print(low, upp)
        # print(popt)
        lower_bound = func(tf, *low)
        upper_bound = func(tf, *upp)
        
        #
        fig, ax1 = plt.subplots()
        #
        t_minus = np.arange(-x_minus_shift, 1)
        ax1.fill_between(tf, lower_bound, upper_bound,
            facecolor=coltri[2], alpha=0.2)
        ax1.set_yscale('log')
        ax1.plot(t_minus, cdf_t[ts-x_minus_shift:ts+1], 'o-', 
            color=coltri[2], mfc='none', lw=1)
        ax1.plot(t, cdf, 'o-', color=coltri[2], mfc='none', lw=1, label=r'data')
        ax1.plot(tf, func(tf, *popt), '-', color=coltri[0], lw=1, label=r'fit')
        #
        ax1.set_yscale('log')
        ax1.set_ylim(100, cdf[-1]*10)
        ax1.axvline(tc, ls='--', lw=1, label='$t_M$')
        #
        ax1.set_xlabel(r"days since "+date, fontsize=15)
        ax1.set_ylabel(r"cases", fontsize=15)
        ax1.tick_params(which = 'major', axis='both', width=1.5, length = 10, 
                labelsize=15,direction='in')
        ax1.tick_params(which = 'minor', axis='both', width=1, length = 5, 
                labelsize=15,direction='in')
        #
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(tf, rho0*rho(tf, eta, R_0), 'o', alpha=0.7, lw=1, ms = 3,
            mfc='none', color='purple', label=r'$\rho(t)$')#
        ax2.axvline(tc, ls='--', lw=1, label='$t_M$')
        ax2.set_yscale('log')
#         ax2.set_ylim(, max(func(tf, *popt))*1.5)
        ax2.tick_params(which = 'major', axis='both', width=1.5, length = 10, 
                labelsize=15,direction='in')
        ax2.tick_params(which = 'minor', axis='both', width=1.5, length = 5, 
                labelsize=15,direction='in')
        ax2.set_title(r"$N\rho(t)$".format(abb_c), fontsize=15)
        # ax2.yaxis.set_major_formatter(ScalarFormatter())
        # ax2.yaxis.get_major_formatter().set_powerlimits((-1, 1))
        # fig.tight_layout()
        ax1.legend(loc=2,prop={'size':15},frameon=True)
        fig.savefig(dir_plot+abb_c+'_'+date2+'tot_'+cases_inspect+'.pdf')
    print(c, dict_dateCount[c], ' |---> done √')

out_file.close()
print(ofname, ' |---> done √')











#     ax2.set_xlabel(r"days since 100 cases", fontsize=15)
    
#         ax2.set_ylabel(r"$\rho(t)$", fontsize=10)
        
    #plt.xticks(fontsize = 20)
    #plt.yticks(fontsize = 20)
#     ax1.set_yscale('log')
#     plt.grid(True)
    # plt.xscale('log')
#     plt.title('ITALY',fontsize = 20)
        







