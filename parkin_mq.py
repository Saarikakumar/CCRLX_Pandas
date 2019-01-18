import os
import pandas as pd
import pylab as plt
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import seaborn as sns
sns.set_style("ticks")
plt.rcParams["axes.linewidth"] = 2.5

d1 = "/Users/SaarikaKumar/Downloads/"
mq='parkin_mq.tab'
sq='parkin_sq.tab'
vdlist='vdlist'
set_reorder = True


def fit(x, p, c):
    """Fitting function from Sun, Tugarinov and Kay."""
    a = np.sqrt((p[1]**2) + p[0]**2)
    num = c * p[1] * np.tanh(a*sp.array(x))
    dem = a - (p[0] * np.tanh(a*sp.array(x)))
    quo = num / dem
    return quo


def sep(dir, file):
    os.chdir(dir)
    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        if(line.startswith('VARS')):
            index = (line.split())
    df_orig = pd.read_csv(file, sep='\s+', names = index[1:], index_col = 0, header=None, skiprows=12)
    filter_col = [col for col in df_orig if col.startswith('Z_A')]
    filter_col.append('ASS')
    filter_col.append('HEIGHT')
    df_clean = pd.DataFrame(df_orig, columns=filter_col)
    return df_clean

def none (df):
    df.ASS.dropna()
    return df

def remove_dup(df):
     df_nodup = df.drop_duplicates(subset= 'ASS', keep=False)
     return df_nodup

def remove_incomplete(data_mq, data_sq):
    data_mq_2 = data_mq[data_mq["ASS"].isin(data_sq["ASS"])]
    data_sq_2 = data_sq[data_sq["ASS"].isin(data_mq["ASS"])]
    return data_mq_2, data_sq_2

def get_heights(df):
    df = df.reset_index(drop=True)
    filter_col = [col for col in df if col.startswith('Z_A')]
    for item in filter_col:
        for index, row in df.iterrows():
            df.at[index, item] = df.loc[index, item] * df.loc[index, "HEIGHT"]
    return(df)

def mult_df(df, const):
    filter_col = [col for col in df if col.startswith('Z_A')]
    for item in filter_col:
        for index, row in df.iterrows():
            df.at[index, item] = df.loc[index, item] * const
    return (df)

def div_all (df_1, df_2):
    df_1 = df_1.reset_index(drop=True)
    df_2 = df_2.reset_index(drop=True)
    filter_col_ = [col for col in df_1 if col.startswith('Z_A')]
    for item in filter_col_:
        for index, row in df_1.iterrows():
            df_1.at[index, item] = df_1.loc[index, item] / df_2.loc[index, item]
    return(df_1)

def div_data (mq_nc, sq_nc):
    nc_proc = (2.0**(-1.0*mq_nc))/(2.0**(-1.0*sq_nc))
    return nc_proc

def div_nc_proc(df, nc_proc):
    df_1 = df.reset_index(drop=True)
    filter_col_ = [col for col in df_1 if col.startswith('Z_A')]
    for item in filter_col_:
        for index, row in df.iterrows():
            df_1.at[index, item] = df_1.loc[index, item] / nc_proc
    return(df_1)

def getdup(dir, f_1):
    os.chdir(dir)
    f = open(f_1, 'r')
    lines = f.readlines()
    x_val = []
    for line in lines:
        x_val.append(float(line.strip('s\n')))
        set_x = set(x_val)
        set_x_l = list(set_x)
        set_x_l.sort()
        dup = x_val[len(set_x_l):]
    return set_x_l, dup


def reorder(df_mq, df_mq_last4, df_sq, uniq_val, dup_val):
    col = [col for col in df_mq]
    col.reverse()
    df_new = (df_mq.loc[:, col[(len(uniq_val)-len(dup_val) - 2):]])
    parkin_mq_reorder = df_new.join(df_mq_last4)
    parkin_mq_reorder.columns = list(df_sq)
    return parkin_mq_reorder

def fit(x, p, c):
    """Fitting function from Sun, Tugarinov and Kay."""
    a = np.sqrt((p[1]**2) + p[0]**2)
    num = c * p[1] * np.tanh(a*sp.array(x))
    dem = a - (p[0] * np.tanh(a*sp.array(x)))
    quo = num / dem
    return quo


def residuals(p, y, x, c):
    """Error function for minimization.

    -----
    Chi-squared.
    """
    err = ((y - fit(x, p, c)))
    return err

def l_square(df_final, x, p, c, num_dup):
    for index, row in final_df.iterrows():
        y = row.tolist()
        res = leastsq(residuals, p, args=(y[:-2], x, c), full_output=1)
        #print (res[0])
        expected_values = fit(x, res[0], c)
        x_smth = np.linspace(0,0.03,1000)
        chi_sq = ((res[2]['fvec'])**2).sum()
        dof = len(unique_vd) - len(p)
        red_chi_sq = np.round(chi_sq/dof, 4)
        red_chi_sq = str(red_chi_sq)
        plt.plot(x[:-2], y[:(-2 - len(num_dup))], 'ok', label='original data', markersize=10)
        plt.plot(x_smth, fit(x_smth, res[0], c), '-b', linewidth = 2, label='fitted line')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("I3Q/ISQ", fontsize=12)
        plt.tick_params(axis="both", labelsize=14, width=2.5)
        plt.figtext(0.68, 0.3, "chi square value = " + red_chi_sq, fontsize=12)
        #plt.figtext(0.83, 0.3, red_chi_sq, fontsize=14)
        observed_values = (y[:-2])
        sns.despine()
        #plt.show()

def dup_err (df, uniq_val, dup_val):
    dataindex = []
    dup_subtract = []
    all_dup_subtract = []
    for dup in dup_val:
        dataindex.append(uniq_val.index(dup)) # get the index of the element in list of uniques
    for index, item in enumerate(dataindex):
        for rowindex, row in df.iterrows():
            dup_subtract.append(df.iloc[rowindex, item] - df.iloc[rowindex, len(uniq_val)+index])
        all_dup_subtract.append(dup_subtract)
        dup_subtract = []
        #print(all_dup_subtract)

    #np_diff_subt = (np.array(all_dup_subtract).reshape(len(all_dup_subtract[0]),len(dup_val)))
    np_diff_subt = np.transpose(np.array(all_dup_subtract))
    df_diff = pd.DataFrame(np_diff_subt, columns=["Dup_diff_1", "Dup_diff_2"])
    #print(df_diff)
    #print(df_diff["Dup_diff_2"])
    return(df.join(df_diff))
    #df_diff_2 = pd.DataFrame(dup_subtract[90:181], columns=["Dup_diff_2"])
    #dup_diff_final=(df_diff_1.join(df_diff_2.join(df)))
    #return dup_diff_final

def std_dev(df, unique_val, dup_val):
    col = [col for col in df]
    sd = []
    #print(dup_val)
    for index, col in enumerate(dup_val):
        sd.append(np.std(df.iloc[:, len(unique_val) + len(dup_val) + 2 + index].tolist(),ddof=1)/np.sqrt(2.0))
    return sd


parkin_mq = sep(d1, mq)
parkin_sq = sep(d1, sq)


unique_vd, dup_vd = getdup(d1, vdlist)
x_val = unique_vd + dup_vd

p = [-20, 30]

c1 = 0.75
c2 = 0.5

c = c1

parkin_mq_last4 = parkin_mq.iloc[:, len(unique_vd):len(unique_vd) + len(dup_vd) + 2 ]

if(set_reorder == False):
    parkin_mq = reorder(parkin_mq, parkin_mq_last4, parkin_sq, unique_vd, dup_vd)

parkin_mq = reorder(parkin_mq, parkin_mq_last4, parkin_sq, unique_vd, dup_vd)


parkin_mq = remove_dup(parkin_mq)
parkin_sq = remove_dup(parkin_sq)

none(parkin_mq)
none(parkin_sq)

parkin_mq, parkin_sq = remove_incomplete(parkin_mq, parkin_sq)


parkin_sq_h = get_heights(parkin_sq)
parkin_mq_h = get_heights(parkin_mq)


parkin_mq_sorted = parkin_mq_h.sort_values(by=["ASS"])
parkin_sq_sorted = parkin_sq_h.sort_values(by=["ASS"])

parkin_mq_sorted = parkin_mq_sorted.reset_index(drop=True)
parkin_sq_sorted = parkin_sq_sorted.reset_index(drop=True)


parkins_mq_mult = mult_df(parkin_mq_sorted, (32/120))

df_dup_subtr_mq = (dup_err(parkins_mq_mult, unique_vd, dup_vd))
df_dup_subtr_sq = (dup_err(parkin_sq_sorted, unique_vd, dup_vd))



std_dev_mq = std_dev(df_dup_subtr_mq, unique_vd, dup_vd)

std_dev_sq = std_dev(df_dup_subtr_sq, unique_vd, dup_vd)


div_val = div_all(parkins_mq_mult, parkin_sq_sorted)


nc_proc = div_data(-5.0, -4.0)
final_df = div_nc_proc(div_val, nc_proc)



#for index, row in final_df.iterrows():
            #test = row.tolist()
            #plt.plot(unique_vd + dup_vd, test[:-2], "ok")
            #plt.show()


if(c == 0.75):
    fit_func = fit(x_val, p, c1)
else:
    fit_func = fit(x_val, p, c2)

for index, row in final_df.iterrows():
    y = row.tolist()


#print(l_square(final_df, x_val, p, c, dup_vd ))
