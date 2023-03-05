#data_cleaner.py - v.1.0-------------------------------------------------------#
import numpy as np
import pandas as pd
#
day = '0415'
dir_d = 'data/'
dir_dc = dir_d+'clean/'
dir_plot = 'plot/'
#
list_name = ['confirmed', 'deaths']
for ln in list_name:
    dir_dat = dir_d+'COVID-19_WORLD_'+day+'/csse_covid_19_data/'\
        'csse_covid_19_time_series/'
    dat = pd.read_csv(dir_dat+'time_series_covid19_'+ln+'_global.csv')
    name = 'cleaned_covid19_'+ln+'_global.csv'
    #
    c_lab = 'Country/Region'
    p_lab = 'Province/State'
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
    dat = dat.drop(columns=['Lat', 'Long', 'Province/State'])
    dat.to_csv(dir_dc+name, index=False)
    print('File ' + name + ' correctly cleaned √')
#______________________________________________________________________________#