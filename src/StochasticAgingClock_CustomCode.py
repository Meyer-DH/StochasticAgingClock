import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import ttest_ind, pearsonr, spearmanr
from fitter import Fitter, get_common_distributions, get_distributions
import pickle
import matplotlib.ticker as plticker
from sklearn import linear_model
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet, ElasticNetCV
import pingouin as pg
import statsmodels
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

input_path = ''
plot_path = ''
if not os.path.exists(f'{plot_path}'):
    os.mkdir(f'{plot_path}')
filetype = '.pdf'
fontsize=7
height=2.17 
scatter_size=1
sns.set(font='Times New Roman', style='white')
dot_color = 'grey' 
line_color = 'black'
xlab = 'Simulated age'
ylab = 'Predicted age'
n_jobs=3


def correct_Bio_Age(corr_arr):
    corr_arr_BioAgeCorr = corr_arr.copy()
    bio = corr_arr_BioAgeCorr.Bio_Age.values
    bio2 = [(((y - (scipy.special.erfinv(
        0.5 - (((scipy.special.erf(((372 - y) / (192 / 3)) / np.sqrt(2)) / 2) + 0.5) / 2)) * np.sqrt(2) * 2 * (
                            192 / 3))))) for y in bio]
    corr_arr_BioAgeCorr.Bio_Age = bio2

    return corr_arr_BioAgeCorr


def make_binary(df, filter_genes='WBG', log=False, q=0.5):
    df_div = df.copy()
    df_div = df_div.filter(regex=filter_genes)
    df_div[df_div == 0] = np.nan
    df_div['Median'] = df_div.quantile(q=q, axis=1)  # q=0.5 is the Median
    df_div = df_div.filter(regex=filter_genes).div(df_div.Median, axis=0)
    df_div[df_div.isna()] = 0
    df_div[df_div <= 1] = 0
    df_div[df_div > 1] = 1

    df_div['Bio_Age'] = df.Bio_Age

    return df_div



# no prediction possible as long as the values are not kept between 0 and 1
def random_epi_nolimit(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.05, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = ground + noise_ground * np.random.randn(epi_sites)

            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = x + noise
            ages.append(age)
            samples.append(x)
    return samples, ages


# same as above, but limit the data between 0 and 1
def random_epi(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.05, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = ground + noise_ground * np.random.randn(epi_sites)
            x[x > 1] = 1
            x[x < 0] = 0

            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = x + noise
                x[x > 1] = 1
                x[x < 0] = 0
            ages.append(age)
            samples.append(x)
    return samples, ages


# with logit
from scipy.special import logit, expit
def random_epi_logit(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.2, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = logit(ground)
            x = x + noise_ground * np.random.randn(epi_sites)
            
            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = x + noise
            x = expit(x)
            ages.append(age)
            samples.append(x)
    return samples, ages

def random_epi_logit_only(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.2, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = logit(ground)
            x = x + noise_ground * np.random.randn(epi_sites)
            
            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = x + noise

            ages.append(age)
            samples.append(x)
    return samples, ages


def random_epi_logit_everystep(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.05, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = logit(ground)
            x = x + noise_ground * np.random.randn(epi_sites)
            x = expit(x)
            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = logit(x)
                x = x + noise
                x = expit(x)
            ages.append(age)
            samples.append(x)
    return samples, ages

@ignore_warnings(category=ConvergenceWarning)
def pred_and_plot(samples, ages, samples2, ages2, outname, xlab, ylab, savepic=True, tick_step=25, fontsize=12,
                         height=3.2, regr=None, filetype='.pdf', scatter_size=1, color='grey', n_jobs=1, line_color='black'):
    stats = []
    if regr:
        pred_y = regr.predict(samples2)
    else:
        regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], n_jobs=n_jobs)
        regr.fit(samples, ages)
        pred_y = regr.predict(samples2)

    if savepic:
        sns.set(font='Times New Roman', style='white')
        g = sns.jointplot(x=ages2, y=pred_y, kind='reg', height=height,
                          scatter_kws={'s': scatter_size}, color=color, joint_kws={'line_kws':{'color':line_color}}) 
        g.ax_joint.set_ylim([0, 99])
        lims = [0, 99]  
        g.ax_joint.plot(lims, lims, ':k')
        g.set_axis_labels(xlab, ylab, fontsize=fontsize)
        loc = plticker.MultipleLocator(base=tick_step)
        g.ax_joint.xaxis.set_major_locator(loc)
        g.ax_joint.yaxis.set_major_locator(loc)
        if isinstance(tick_step, int):
            g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
            g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
        else:
            g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
            g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(f'{outname}_fontsize{fontsize}_height{height}{filetype}')
        plt.close()
    stats.append(pearsonr(ages2, pred_y))
    stats.append(spearmanr(ages2, pred_y))
    stats.append(r2_score(ages2, pred_y))
    stats.append(np.median(abs(ages2 - pred_y)))
    return regr, stats



def get_noise_func_parm(t, start_ind, end_ind, step=10, normalize=None):
    '''
    compute the difference between start_ind - end_ind and subsequent the noise function
    :param t: Dataframe with datasets in the columns
    :param start_ind: left side of subtraction
    :param end_ind: right side
    :param step: how many quantiles to compute for the noise function
    :return: 
    '''
    d = {}
    d['Q1'] = []
    d['Q2'] = []
    d['Param'] = []

    for i in np.array(range(step)) / step:
        c1 = t.iloc[:, start_ind]
        c2 = t.iloc[:, end_ind]
        q1 = c1.quantile(q=i)
        q2 = c1.quantile(q=i + (1 / step))
        # get the quantile from the young dataset
        if i == 0:
            q1 = 0  
            d['Q1'].append(q1)
            d['Q2'].append(q2)
            c1_q = c1[(c1 >= q1) & (c1 <= q2)]
        elif i == (step - 1) / step:
            q2 = 1
            d['Q1'].append(q1)
            d['Q2'].append(q2)
            c1_q = c1[(c1 > q1) & (c1 <= q2)]
        else:
            d['Q1'].append(q1)
            d['Q2'].append(q2)
            c1_q = c1[(c1 > q1) & (c1 <= q2)]
        # get the same sites for the older dataset
        c2_q = c2[c2.index.isin(c1_q.index)]
        rs = len(c1_q)
        # old - young
        dif1 = (c2_q - c1_q).values
        if normalize == 'Mean':
            dif1 = dif1 - np.mean(dif1)
        elif normalize == 'Median':
            dif1 = dif1 - np.median(dif1)

        listdif = list(dif1)
        f = Fitter(listdif, distributions=['lognorm'], timeout=120)
        f.fit()
        d['Param'].append(f.get_best())
    df = pd.DataFrame(d)
    return df


# take the ground array, and the noise df to generate random noise fitting to the ground truth
def apply_biol_noise_ground(ground, noise_ground_df):
    c3 = pd.DataFrame(ground)
    c3.columns = ['Ground']
    c3['Rand'] = np.nan
    for i in range(len(noise_ground_df)):
        q1 = noise_ground_df['Q1'][i]
        q2 = noise_ground_df['Q2'][i]
        if q1 == 0:
            r1 = scipy.stats.lognorm.rvs(size=len(c3[(c3['Ground'] >= q1) & (c3['Ground'] <= q2)]),
                                         scale=noise_ground_df['Param'][i]['lognorm']['scale'],
                                         loc=noise_ground_df['Param'][i]['lognorm']['loc'],
                                         s=noise_ground_df['Param'][i]['lognorm']['s'])
            c3.loc[(c3['Ground'] >= q1) & (c3['Ground'] <= q2), 'Rand'] = r1
        else:
            r1 = scipy.stats.lognorm.rvs(size=len(c3[(c3['Ground'] > q1) & (c3['Ground'] <= q2)]),
                                         scale=noise_ground_df['Param'][i]['lognorm']['scale'],
                                         loc=noise_ground_df['Param'][i]['lognorm']['loc'],
                                         s=noise_ground_df['Param'][i]['lognorm']['s'])
            c3.loc[(c3['Ground'] > q1) & (c3['Ground'] <= q2), 'Rand'] = r1
    return c3



def random_epi_biol_age(ground, noise_ground_df, noise_age_df, samples_per_age=3, epi_sites=20000, age_steps=1,
                        age_start=0, age_end=100, noise_norm=1):
    samples = []
    ages = []
    for age in range(age_start, age_end, age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            n = apply_biol_noise_ground(ground, noise_ground_df)
            x = (n.Ground + n.Rand).values
            x[x > 1] = 1
            x[x < 0] = 0

            for _ in range(age):
                n = apply_biol_noise_ground(ground, noise_age_df) # measure noise always form ground
                x = (x + (n.Rand / noise_norm)).values
                x[x > 1] = 1
                x[x < 0] = 0
            ages.append(age)
            samples.append(x)
    return samples, ages



####### single cell
# the maintenance is fixed to the same value
# use binary NOT to do all states at once
def update_cells_fast_fixed(cells, Em, cell_num=100):
    lens = len(cells)
    flip = np.random.uniform(size=[lens,cell_num])>=Em
    cells[flip] = ~cells[flip]
    return cells

def simulate_cells_for_age_fixed(ground, Em, Ed=0, samples_per_age=3, age_steps=30, cell_num=100, deviate_ground=True):
    samples = []
    ages = []
    # generate cell_num cells for each site
    for _ in range(samples_per_age):
        for age in range(1, age_steps):
            x = ground
            if deviate_ground:
                x = x + 0.01 * np.random.randn(len(ground)) # slightly deviate the ground state
                x[x > 1] = 1
                x[x < 0] = 0
            cells = np.array([int(cell_num * g) * [True] + (cell_num - int(cell_num * g)) * [False] for g in x])
            for _ in range(age):
                cells = update_cells_fast_fixed(cells, Em, cell_num=cell_num)
            samples.append(cells.mean(axis=1))  # compute bulk average again and append
            ages.append(age)
    return samples, ages


def get_noise_Em_all_new(t, sample_name, Em_lim=0.95, Ed_lim=0.23):
    '''
    Compute the average Em value for sample_name based on 
    1+Ed(z-1)/z, where z is the equilibrium value
    :param t: Dataframe with datasets in the columns
    :param sample_name: sample of interest, usually the oldest to get the equilibrium value

    :param step: how many quantiles to compute for the noise function
    :return: 
    '''
    d = {}
    d['Site'] = []
    d['Value'] = []
    d['Em'] = []
    d['Ed'] = []

    c1 = t[sample_name]

    for row in c1.index:
        d['Site'].append(row)
        eq = c1.loc[row]  # median equilibrium value that we define
        d['Value'].append(eq)
        Ed = ((Em_lim - 1) * eq) / (eq - 1)
        if Ed > Ed_lim:
            Ed = np.random.randint(1, int(Ed_lim * 1000)) / 1000 
        Em = 1 + Ed * (eq - 1) / eq

        d['Em'].append(Em)
        d['Ed'].append(Ed)
    df = pd.DataFrame(d)
    return df




def update_cells_fast_empirical_noquantile(cells, eml, edl, cell_num=100):
    
    lenc = len(cells[0])
    for i, c in enumerate(cells):  # this loops over the 2000 sites --> every site has a specific Em
        Em = eml[i]
        Ed = edl[i]
        flip = ((c == False) & (np.random.uniform(size=lenc) <= Ed)) | ((c == True) & (np.random.uniform(size=lenc) >= Em))
        c[flip] = ~c[flip]
    return cells


def simulate_for_age_empirical_noquantile(ground, Em_df, samples_per_age=3, age_steps=30, cell_num=100, deviate_ground=True):
    samples = []
    ages = []
    Em_df = Em_df.sort_values(by='Site')
    eml = Em_df.Em.values
    edl = Em_df.Ed.values
    for _ in range(samples_per_age):
        for age in range(1, age_steps):
            x = ground
            if deviate_ground:
                x = x + 0.01 * np.random.randn(len(ground)) # slightly deviate the ground state
                x[x > 1] = 1
                x[x < 0] = 0
            cells = np.array([int(cell_num * g) * [True] + (cell_num - int(cell_num * g)) * [False] for g in x])

            for _ in range(age):
                cells = update_cells_fast_empirical_noquantile(cells, eml, edl, cell_num=cell_num)
            samples.append(cells.mean(axis=1))
            ages.append(age)
    return samples, ages

#### Code for published clocks
def anti_transform_age(exps):
    adult_age = 20
    ages = []
    for exp in exps:
        if exp < 0:
            age = (1 + adult_age) * (np.exp(exp)) - 1
            ages.append(age)
        else:
            age = (1 + adult_age) * exp + adult_age
            ages.append(age)
    ages = np.array(ages)
    return ages


def get_clock(clock_csv_file, sep=','):
    coef_data = pd.read_csv(clock_csv_file, sep=sep)
    if ('Intercept' in coef_data.CpGmarker.values) or ('(Intercept)' in coef_data.CpGmarker.values):
        intercept = coef_data[0:1].CoefficientTraining[0]
        coef_data = coef_data.drop(0)
    else:
        intercept = 0
    coef_data = coef_data[coef_data.CpGmarker.str.startswith('cg')]
    coef_data = coef_data.sort_values(by='CpGmarker')
    horvath_cpgs = np.array(coef_data.CpGmarker)
    coefs = np.array(coef_data.CoefficientTraining)
    horvath_model = linear_model.LinearRegression()
    horvath_model.coef_ = coefs
    horvath_model.intercept_ = intercept
    return horvath_cpgs, horvath_model


def get_clock_data(dat, cpg_sites, young16y='GSM1007467', old88y='GSM1007832'):
    h = dat.loc[cpg_sites]
    data = h[[young16y, old88y]]  
    data = data.sort_index()
    return data


def get_samples(dat, cpg_sites, age_steps=30, cell_num=100, Em_lim=0.95, Ed_lim=0.23, young16y='GSM1007467',
                old88y='GSM1007832'):
    data = get_clock_data(dat, cpg_sites=cpg_sites, young16y=young16y, old88y=old88y)
    Em_df_new = get_noise_Em_all_new(data, old88y, Em_lim=Em_lim, Ed_lim=Ed_lim)
    ground = np.array(data[young16y])
    samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new,
                                                                                        samples_per_age=1,
                                                                                        age_steps=age_steps,
                                                                                        cell_num=cell_num)
    return samples_emp_noquantile, ages_emp_noquantile, data[[young16y]]




def get_samples_fixed(dat, cpg_sites, Em, Ed, samples_per_age=1, age_steps=30, cell_num=100, young16y='GSM1007467', old88y='GSM1007832'):
    data = get_clock_data(dat, cpg_sites=cpg_sites, young16y=young16y, old88y=old88y)
    ground = np.array(data[young16y])
    samples_emp_noquantile, ages_emp_noquantile = simulate_cells_for_age_fixed(ground, Em, Ed, samples_per_age=samples_per_age,
                                                                               age_steps=age_steps, cell_num=cell_num)
    return samples_emp_noquantile, ages_emp_noquantile, data[[young16y]]



# complete random Em and Ed
def get_noise_Em_all_random(t, Em_lim=0.95, Ed_lim=0.23):

    d = {}
    d['Site'] = []
    d['Em'] = []
    d['Ed'] = []

    for row in t.index:
        d['Site'].append(row)
        # completely random within the limits
        Ed = np.random.randint(1, int(Ed_lim * 10000)) / 10000
        Em =np.random.randint(int(Em_lim * 10000), 9999) / 10000

        d['Em'].append(Em)
        d['Ed'].append(Ed)
    df = pd.DataFrame(d)
    return df


def get_samples_random(dat, cpg_sites, age_steps=30, cell_num=100, Em_lim=0.95, Ed_lim=0.23, young16y='GSM1007467',
                old88y='GSM1007832'):
    data = get_clock_data(dat, cpg_sites=cpg_sites, young16y=young16y, old88y=old88y)
    Em_df_new = get_noise_Em_all_random(data, Em_lim=Em_lim, Ed_lim=Ed_lim)
    ground = np.array(data[young16y])
    samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new,
                                                                                        samples_per_age=1,
                                                                                        age_steps=age_steps,
                                                                                        cell_num=cell_num)
    return samples_emp_noquantile, ages_emp_noquantile, data[[young16y]]




@ignore_warnings(category=ConvergenceWarning)
def get_prediction(samples_emp_noquantile, ages_emp_noquantile, data, clock_model, outname, scatter_size=1,
                   tick_step=25, fontsize=12, height=3.2, filetype='.pdf', kind='scatter', lim_ax = False, loc_tick=False, color='grey', line_color='black', dot_color = 'grey', tight=True,xlab='Simulated age', ylab='Predicted epigenetic age'):

    for i, s in enumerate(samples_emp_noquantile):
        data[f'Sample{i}'] = s
    data = data.T
    if len(clock_model.coef_) == 353:
        pred = anti_transform_age(clock_model.predict(data))
    else:
        pred = clock_model.predict(data)
    pear = pearsonr(ages_emp_noquantile, pred[1:])
    spear = spearmanr(ages_emp_noquantile, pred[1:])
    r2 = r2_score(ages_emp_noquantile, pred[1:])


    sns.set(font='Times New Roman', style='white')
    if kind=='scatter':
        g = sns.jointplot(x=ages_emp_noquantile, y=pred[1:], kind=kind, height=height, s=4, color=color)

    else:
        g = sns.jointplot(x=ages_emp_noquantile, y=pred[1:], kind=kind, height=height, scatter_kws={'s': scatter_size}, color=color, joint_kws={'line_kws':{'color':line_color}})
    if lim_ax:
        g.ax_joint.set_ylim([0, 80])
        lims = [0, 80] 
    g.set_axis_labels(xlab, ylab, fontsize=fontsize)
    if loc_tick:
        loc = plticker.MultipleLocator(base=tick_step)
        g.ax_joint.xaxis.set_major_locator(loc)
        g.ax_joint.yaxis.set_major_locator(loc)
    if isinstance(tick_step, int):
        g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
        g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
    else:
        g.ax_joint.set_xticklabels([round(tt,3) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
        g.ax_joint.set_yticklabels([round(tt,3) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
    if tight:
        plt.tight_layout()

    plt.savefig(f'{plot_path}{outname}_Prediction.pdf')
    plt.close()

    xlab = 'Ground state values'
    ylab = 'Ground state values + 100x Single cell noise'
    tick_step = 0.25
    sns.set(font='Times New Roman', style='white')

    g = sns.jointplot(x=data.iloc[0], y=data.iloc[-1], kind='scatter', height=height, s=4, color=dot_color)

    g.set_axis_labels(xlab, ylab, fontsize=fontsize)
    loc = plticker.MultipleLocator(base=tick_step)
    g.ax_joint.xaxis.set_major_locator(loc)
    g.ax_joint.yaxis.set_major_locator(loc)
    g.ax_joint.plot([0,1], [0,1], ':k')
    g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
    g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(
        f'{plot_path}{outname}_100xNoise_fontsize{fontsize}_height{height}{filetype}')
    plt.close()
    return pear, spear, r2, pred



def get_clock_correlations(Tissues, prediction_output_file, numsam=5):
    tmp = pd.read_csv(prediction_output_file, index_col=0)
    tmp = tmp.dropna(subset=['Age'])
    tmp.Age = tmp.Age.astype(float)
    tmp = tmp[~tmp.MaxAge.isin(['Cervus canadensis', 'Cervus elaphus'])]  # no proper meta data
    tmp.MaxAge = tmp.MaxAge.astype(float)
    tmp['RelAge'] = tmp.Age / tmp.MaxAge
    tmp = tmp[(tmp.Experiment.isna())]

    df_median_all = pd.DataFrame()
    cl = prediction_output_file.split('/')[-1][:-4]

    pears = {}
    pears['Corr'] = []
    pears['p'] = []
    pears['Tissue'] = []
    pears['Organism'] = []
    pears['NumSamples'] = []

    for s in tmp.Organism.unique():
        for t in Tissues:
            try:
                pears['Corr'].append(pearsonr(tmp[(tmp.Organism == s) & (tmp.Tissue == t)].RelAge, tmp[
                    (tmp.Organism == s) & (tmp.Tissue == t)].clock)[0])
                pears['p'].append(pearsonr(tmp[(tmp.Organism == s) & (tmp.Tissue == t)].RelAge, tmp[
                    (tmp.Organism == s) & (tmp.Tissue == t)].clock)[1])
                pears['NumSamples'].append(len(tmp[(tmp.Organism == s) & (tmp.Tissue == t)]))
                pears['Organism'].append(s)

                pears['Tissue'].append(t)
            except:
                continue

    df = pd.DataFrame(pears)
    df = df.dropna()
    df = df[df.NumSamples >= numsam]
    df['FDR_bh'] = statsmodels.stats.multitest.multipletests(df.p, method='fdr_bh')[1]
    df['Bonferroni'] = statsmodels.stats.multitest.multipletests(df.p, method='Bonferroni')[1]

    df_median = df.groupby('Organism').Corr.median().sort_values()
    df_median = df_median.to_frame()

    maxage = tmp[['Organism', 'Species_Name', 'MaxAge']]
    maxage = maxage.drop_duplicates()
    maxage = maxage.set_index('Organism')

    df_median = df_median.join(maxage)
    df_median = df_median.sort_values(by='MaxAge')

    df_median['Clock'] = cl
    if len(df_median_all) == 0:

        df_median = df_median[['Corr']]
        df_median.columns = [cl]
        df_median_all = df_median
    else:
        df_median = df_median[['Corr']]
        df_median.columns = [cl]
        df_median_all = df_median_all.join(df_median[[cl]], how='outer')


    df_median_all = df_median_all.join(maxage)
    
    df_median_all = df_median_all.sort_values(by='MaxAge')
    df_median_all = df_median_all.dropna()
    
    return df_median_all



def plot_circles(Tissues, prediction_output_file,numsam, outname):
    df_median_all = get_clock_correlations(Tissues=Tissues,prediction_output_file=prediction_output_file, numsam=numsam) 
    # add taxonomy data
    tax = pd.read_csv(f'{input_path}taxonomy_anage_meta.csv', index_col=0)
    df_median_all =df_median_all.join(tax[['Order']])

    # add missing manually
    df_median_all.loc['Notamacropus rufogriseus', 'Order'] = 'Diprotodontia'
    df_median_all.loc['Osphranter rufus', 'Order'] = 'Diprotodontia'
    df_median_all.loc['Panthera leo', 'Order'] = 'Carnivora'
    df_median_all.loc['Equus asinus somalicus', 'Order'] = 'Perissodactyla'
    cl = prediction_output_file.split('/')[-1][:-4]
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    # plot on a circle
    ydata = df_median_all[cl].values
    data_len = len(ydata)
    theta = np.linspace(0, (2 - 2 / len(ydata)) * np.pi, len(ydata))
    r = ydata
    ax.plot(theta, r, 'g')

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    for angle, radius, label in zip(theta, r, df_median_all.Species_Name.values):
        x = angle
        y = 1.3
        ax.text(x, y, label, ha='center',
                fontsize=8, rotation=np.degrees(-angle), va='center')

    ax.set_xticklabels([])
    ax.set_yticklabels([round(x,1) for x in ax.get_yticks()], fontsize=8)
    plt.tight_layout(pad=0.1)
    plt.legend([f'Clock1 Median Correlation: {df_median_all[cl].median():.2f}'], #noise_clock_Empirical_maxage67_l10.001_coefs_
loc='upper right')
    # add the colors
    taxo = {s:i for i,s in enumerate(df_median_all.Order.unique())}
    colors = plt.cm.tab20.colors[:len(taxo)]
    for i in range(len(theta)-1):
        ax.fill_between([theta[i]- np.pi / len(ydata), theta[i + 1]- np.pi / len(ydata)], 0, 0.85, 
                    color=colors[taxo[df_median_all.iloc[i].Order]], alpha=0.5)

    ax.fill_between([ theta[i+1]- np.pi / len(ydata), np.pi*2- np.pi / len(ydata)], 0, 0.85,
                    color=colors[taxo[df_median_all.iloc[i +1].Order]], alpha=0.5)


    plt.savefig(f'{plot_path}CirclePlot_{outname}.pdf')
    plt.close()

    fig, ax = plt.subplots()
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i,0), 1, 1, color=color))
    ax.set_xlim(0,len(colors))
    ax.set_xticks(range(len(colors)))
    ax.set_xticklabels(taxo.keys())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{plot_path}CirclePlot_{outname}_COLORMAP.pdf')
    plt.close()


#################################################################################################################33
#################################################################################################################33
#################################################################################################################33
#################################################################################################################33
#################################################################################################################33

'''
Figure 1
'''
##1C
# Data without limit
size = 2000
ground = np.random.randint(0,1000, size=size)/1000
samples, ages = random_epi_nolimit(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.05, 
                                   age_steps = 100)


samples_2, ages_2 = random_epi_nolimit(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.05, 
                                   age_steps = 100)
o=f'1c_random_clock_nolimit'
regr, stats = pred_and_plot(samples=samples, 
                                   ages=ages, 
                                   samples2=samples_2, 
                                   ages2=ages_2, 
                                   outname=f'{plot_path}{o}',
                                   xlab=xlab,
                                   ylab=ylab, 
                                   fontsize=fontsize, 
                                   height=height, 
                                   tick_step=25, color=dot_color, line_color=line_color, n_jobs=n_jobs)



##1D
# Limit between 0 and 1 with gaussian noise
size = 2000
ground = np.random.randint(0,1000, size=size)/1000
samples, ages = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.05, 
                                   age_steps = 100)


samples_2, ages_2 = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.05, 
                                   age_steps = 100)
o=f'1c_random_clock_withlimit'
regr, stats = pred_and_plot(samples=samples, 
                                   ages=ages, 
                                   samples2=samples_2, 
                                   ages2=ages_2, 
                                   outname=f'{plot_path}{o}',
                                   xlab=xlab,
                                   ylab=ylab, 
                                   fontsize=fontsize, 
                                   height=height, 
                                   tick_step=25, color=dot_color, line_color=line_color, n_jobs=n_jobs)




##1E
stats_dict = {}
stats_dict['Noise_Age'] = []
stats_dict['Pearson'] = []
stats_dict['Spearman'] = []
stats_dict['R2'] = []
stats_dict['MAD'] = []
size = 2000
for noise_age in [0.005,0.0075, 0.01,0.025,0.05, 0.075,0.1]:
    for _ in range(3):
        
        ground = np.random.randint(0,1000, size=size)/1000
        samples, ages = random_epi(ground, 
                                           samples_per_age = 3, 
                                           epi_sites = size, 
                                           noise_ground = 0.01,
                                           noise_age = noise_age, 
                                           age_steps = 100)
        
        
        samples_2, ages_2 = random_epi(ground, 
                                           samples_per_age = 3, 
                                           epi_sites = size, 
                                           noise_ground = 0.01,
                                           noise_age = noise_age, 
                                           age_steps = 100)

        o=f'1e_random_clock_withlimit_{noise_age}'
        regr, stats = pred_and_plot(samples=samples, 
                                           ages=ages, 
                                           samples2=samples_2, 
                                           ages2=ages_2, 
                                           outname=f'{plot_path}{o}',
                                           xlab=xlab,
                                           ylab=ylab, 
                                           fontsize=fontsize, 
                                           height=height, 
                                           tick_step=25, color=dot_color, line_color=line_color, n_jobs=n_jobs)
        stats_dict['Noise_Age'].append(noise_age)
        stats_dict['Pearson'].append(stats[0])
        stats_dict['Spearman'].append(stats[1])
        stats_dict['R2'].append(stats[2])
        stats_dict['MAD'].append(stats[3])
        

stats_df = pd.DataFrame(stats_dict)
fig = plt.figure(figsize=(height,height))
g=sns.swarmplot(x='Noise_Age', y='R2', data=stats_df, color='black', s=2)
g=sns.boxplot(x='Noise_Age', y='R2', data=stats_df, color='white')
g.set_xlabel('Noise',fontsize=fontsize)
g.set_ylabel('R2', fontsize=fontsize)
g.set_ylim(0,1)
loc = plticker.MultipleLocator(base=0.2)
g.yaxis.set_major_locator(loc)
g.set_yticklabels([round(tt,3) for tt in g.get_yticks()], fontsize=fontsize)
g.set_xticklabels([0.005,0.0075, 0.01,0.025,0.05, 0.075,0.1], fontsize=fontsize)
g.set_xlabel('Noise standard deviation')
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}1e_random_clock_withlimit_noiseage_comp.pdf')
plt.close()



##1F
samples, ages = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.05, 
                                   age_steps = 100)


samples_2, ages_2 = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.05, 
                                   age_steps = 100)

regr_2, stats = pred_and_plot(samples=samples, 
                                   ages=ages, 
                                   samples2=samples_2, 
                                   ages2=ages_2, 
                                   outname=f'{plot_path}{o}',
                                   xlab=xlab,
                                   ylab=ylab, 
                                   fontsize=fontsize, 
                                   height=height, 
                                   tick_step=25,savepic=False, color=dot_color, line_color=line_color, n_jobs=n_jobs)

tick_step=0.5
xlab='Elastic net regression coefficients run 1'
ylab='Elastic net regression coefficients run 2'
sns.set(font='Times New Roman', style='white')
g = sns.jointplot(x=regr.coef_, y=regr_2.coef_, kind='reg', height=height, scatter_kws={'s':1}, color=dot_color, joint_kws={'line_kws':{'color':line_color}}) 
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc = plticker.MultipleLocator(base=tick_step)
g.ax_joint.xaxis.set_major_locator(loc)
g.ax_joint.yaxis.set_major_locator(loc)
if isinstance(tick_step, int):
    g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
    g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
else:
    g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
    g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}1f_random_clock_withlimit_reproducible.pdf')
plt.close()


##1G
# Regression to the mean
tick_step=0.5
xlab='Ground state values'
ylab='Elastic net regression coefficients'
sns.set(font='Times New Roman', style='white')

g = sns.jointplot(x=ground, y=regr.coef_, kind='reg', height=height, scatter_kws={'s':1}, color=dot_color, joint_kws={'line_kws':{'color':line_color}}) 
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc = plticker.MultipleLocator(base=tick_step)
g.ax_joint.xaxis.set_major_locator(loc)
g.ax_joint.yaxis.set_major_locator(loc)
if isinstance(tick_step, int):
    g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
    g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
else:
    g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
    g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}1g_random_clock_withlimit_RegressionToTheMean.pdf')
plt.close()

##1H
stats_dict = {}
stats_dict['Size'] = []
stats_dict['Pearson'] = []
stats_dict['Spearman'] = []
stats_dict['R2'] = []
stats_dict['MAD'] = []
stats_dict['alpha'] = []
stats_dict['l1'] = []
s = 0
e = 1000
for size in [pow(2, i) for i in range(12)]:
    r = 3
    for _ in range(r):
        ground = np.random.randint(s, e, size=size) / 1000

        samples, ages = random_epi(ground, epi_sites=size)
        samples_2, ages_2 = random_epi(ground, epi_sites=size)

        o = f'ground_{s}_{e}_{size}'
        regr, stats = pred_and_plot(samples=samples, ages=ages,
                                           samples2=samples_2, ages2=ages_2,
                                           outname=f'{plot_path}{o}.png', savepic=False,
                                           xlab=None, ylab=None)
        stats_dict['Size'].append(size)
        stats_dict['Pearson'].append(stats[0])
        stats_dict['Spearman'].append(stats[1])
        stats_dict['R2'].append(stats[2])
        stats_dict['MAD'].append(stats[3])
        stats_dict['alpha'].append(regr.alpha_)
        stats_dict['l1'].append(regr.l1_ratio_)

    
stats_df = pd.DataFrame(stats_dict)
fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Size', y='R2', data=stats_df, color='black', s=2)
ax=sns.boxplot(x='Size', y='R2', data=stats_df, color='white')
ax.set_xlabel('Feature Size',fontsize=fontsize)
ax.set_ylabel('R2', fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout(pad=1)
plt.savefig(f'{plot_path}1h_random_clock_withlimit_FeatureSizeComp.pdf')
plt.close()


##1I
size = 2000
ground = np.random.randint(0,1000, size=size)/1000
samples, ages = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.05, 
                                   age_steps = 100)


samples_2, ages_2 = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.05, 
                                   age_steps = 100)

samples_3, ages_2 = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.1, 
                                   age_steps = 100)

samples_4, ages_2 = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.2, 
                                   age_steps = 100)
samples_5, ages_2 = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.025, 
                                   age_steps = 100)

xlab = 'Simulated age'
ylab = 'Predicted age'
regr, stats = pred_and_plot(samples=samples, 
                                   ages=ages, 
                                   samples2=samples_2, 
                                   ages2=ages_2, 
                                   outname=f'',
                                   xlab=xlab,
                                   ylab=ylab, 
                                   fontsize=fontsize, 
                                   height=height, 
                                   tick_step=25, savepic=False)

d = {}
d['Learned Noise'] = regr.predict(samples_2)
d['2x Noise'] = regr.predict(samples_3)
d['4x Noise'] = regr.predict(samples_4)
d['0.5x Noise'] = regr.predict(samples_5)
d['True # of noise cycles'] = ages
df = pd.DataFrame(d)
df = df.melt(id_vars=['True # of noise cycles'])

sns.set(font='Times New Roman', style='white')
fig = plt.figure(figsize=(height, height))
g=sns.swarmplot(x='True # of noise cycles', y='value', hue='variable', data=df, s=1, palette='colorblind')
loc = plticker.MultipleLocator(base=10)
plt.ylim(0,160)
g.xaxis.set_major_locator(loc)
g.set_xlabel(xlab, fontsize=fontsize)
g.set_ylabel(ylab, fontsize=fontsize)
plt.text(60, 30, 'N(0,0.025)' ,fontsize=fontsize)
plt.text(60, 85, 'N(0,0.05)' ,fontsize=fontsize)
plt.text(60, 105, 'N(0,0.1)' ,fontsize=fontsize)
plt.text(60, 145, 'N(0,0.2)' ,fontsize=fontsize)
g.get_legend().remove()
g.set_xticklabels([int(tt) for tt in g.get_xticks()], fontsize=fontsize)
g.set_yticklabels([int(tt) for tt in g.get_yticks()], fontsize=fontsize)
plt.tight_layout(pad=1)
plt.savefig(f'{plot_path}1i_random_clock_withlimit_NoiseComp.pdf')
plt.close()




'''
Figure 2
'''
### BitAge 
#2A
meta = pd.read_csv(f'{input_path}SupplementaryTable1.csv', index_col=0)
meta['Bio_Age'] = meta.Biological_Age_in_Hours
meta.Bio_Age = meta.Bio_Age.astype(float)
meta = correct_Bio_Age(meta)
meta['Bio_Age'] = meta.Bio_Age / 24
meta['Chronological_Age_Days'] = meta['Chronological_Age_in_Hours'] / 24
dat = pd.read_csv(f'{input_path}Celegans_RNAseq_counts.csv', index_col=0)
dat.columns = [c[:-3] for c in dat.columns]
dat = dat.T
dat = dat[dat.sum(axis=1) != 0]
dat = dat.join(meta[['Bio_Age']])
dat = dat.dropna()
meta = meta.dropna(subset=['Bio_Age'])
meta = meta[meta.index.isin(dat.index)]
y = dat.Bio_Age.values
train = dat.drop('Bio_Age', axis=1)
train += 1
train = np.log10(train)
train = train.T
train = train.div(train.max())
train = train[train.sum(axis=1) != 0]
train = train.T
train = train.join(meta[['Bio_Age']])
train_bin = make_binary(train)
train = train.drop('Bio_Age', axis=1)
train = train.T
train_bin = train_bin.drop('Bio_Age', axis=1)
train_bin = train_bin.T
step = 5
youngest_list = meta[meta.Bio_Age == meta.Bio_Age.min()].index
youngest = youngest_list[0]
forquant = train[[youngest]]
forquant['Old'] = train[
                             meta[meta.Bio_Age == meta.Bio_Age.max()].index].iloc[:, 0].values
forquant.columns = ['Young', 'Old']
ground_sites = forquant.index
ground_sites = np.sort(ground_sites)
ground = forquant[forquant.index.isin(ground_sites)].loc[:, 'Young'].values
coefs = pd.read_csv(f'{input_path}BitAgeClock.csv', index_col=0, sep='\t')
intercept = 103.55

noise_age=0.01
x='Bio_Age'
samples, ages = random_epi(ground,
                                       samples_per_age=10,
                                       epi_sites=len(ground),
                                       noise_ground=0.01,
                                       noise_age=noise_age,
                                       age_steps=16)
samples_df = pd.DataFrame(samples)
samples_df.columns = ground_sites
samples_df['Bio_Age'] = ages
samples_bin = make_binary(samples_df)
x_train = samples_bin.filter(regex='WB').values
y = samples_bin.Bio_Age.values
samples_bin = samples_bin.drop('Bio_Age', axis=1)
samples_bin = samples_bin.T
coefs_samples = coefs.join(samples_bin)
coefs_samples = coefs_samples.dropna()
df = {'Name': [], 'Prediction': []}
for i in range(1, len(coefs_samples.columns)):
    df['Name'].append(coefs_samples.columns[i])
    df['Prediction'].append(sum(coefs_samples.iloc[:, 0] * coefs_samples.iloc[:, i]) + intercept)
df = pd.DataFrame(df)
df['Bio_Age'] = y
df['Prediction_Days'] = df.Prediction/24
xlab='Simulated age'
ylab='Predicted age (BitAge)'
sns.set(font='Times New Roman', style='white')
g=sns.jointplot(x='Bio_Age', y='Prediction_Days', kind='reg', 
                height=height, scatter_kws={'s':1}, color=dot_color, joint_kws={'line_kws':{'color':line_color}},
                data=df)
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
g.ax_joint.set_xticks(range(0,16,2))
g.ax_joint.set_yticks(range(1,4))
g.ax_joint.set_xticklabels(range(0,16,2), fontsize=fontsize)
g.ax_joint.set_yticklabels(range(1,4), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}2a_predict_worm_correlation_comp_noiseage_example_BitAge.pdf')
plt.close()

#2B
noise_age=0.01
x='Bio_Age'
samples, ages = random_epi(ground,
                                       samples_per_age=10,
                                       epi_sites=len(ground),
                                       noise_ground=0.01,
                                       noise_age=noise_age,
                                       age_steps=16)
samples_df = pd.DataFrame(samples)
samples_df.columns = ground_sites
samples_df['Bio_Age'] = ages
samples_bin = make_binary(samples_df)
x_train = samples_bin.filter(regex='WB').values
y = samples_bin.Bio_Age.values
y=y*2
regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    alphas=[1])  
regr.fit(x_train, y) 

test = train_bin.T
test = test[ground_sites]
test = test[~test.index.isin([youngest])] 

y_pred = regr.predict(test.values)
test['Predicted'] = y_pred
meta = meta.join(test[['Predicted']])
meta = meta[~meta.index.isin([youngest])] 
xlab='Biological age in days'
ylab='Predicted age (Stochastic data-based)'
sns.set(font='Times New Roman', style='white')
g=sns.jointplot(x='Bio_Age', y='Predicted', kind='reg', 
                height=height, scatter_kws={'s':1}, color=dot_color, joint_kws={'line_kws':{'color':line_color}},
                data=meta)

g.set_axis_labels(xlab, ylab, fontsize=fontsize)

g.ax_joint.set_xticks(range(0,18,2))
g.ax_joint.set_yticks(range(0,18,2))
g.ax_joint.set_xticklabels(range(0,18,2), fontsize=fontsize)
g.ax_joint.set_yticklabels(range(0,18,2), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}2b_predict_worm_correlation_comp_noiseage_example_excludeyoungest_rescaled.pdf')
plt.close()



#2c
fig = plt.figure(figsize=(height,height))
g=sns.boxplot(x='Treatment', y='Predicted', data=meta[(meta.GEO=='GSE63528')&(meta.Chronological_Age_Days==5)], 
              order=['FUDR', '2microM Mianserin+FUDR', '10microM Mianserin+FUDR', '50microM Mianserin+FUDR'], color='white')
g=sns.swarmplot(x='Treatment', y='Predicted', data=meta[(meta.GEO=='GSE63528')&(meta.Chronological_Age_Days==5)], 
              order=['FUDR', '2microM Mianserin+FUDR', '10microM Mianserin+FUDR', '50microM Mianserin+FUDR'], color='black', s=2)
plt.xticks(rotation=30, ha='right')
g.set_xticklabels(g.get_xticklabels(), fontsize=fontsize)
plt.xlabel(None)
plt.ylabel('Predicted Noise Age', fontsize=fontsize)
loc = plticker.MultipleLocator(base=1)
g.yaxis.set_major_locator(loc)
g.set_yticklabels(g.get_yticklabels(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}2c_Mianserin_Noise_clock.pdf')
plt.close()
#STATS
aov = pg.anova(dv='Predicted', between=['Treatment'], data=meta[(meta.GEO=='GSE63528')&(meta.Chronological_Age_Days==5)], detailed=True, ss_type=2)
aov.to_csv(f'{plot_path}2c_Mianserin_day5_ANOVA.csv')
results = pairwise_tukeyhsd(list(meta[(meta.GEO=='GSE63528')&(meta.Chronological_Age_Days==5)].Predicted.values),
                                list(meta[(meta.GEO=='GSE63528')&(meta.Chronological_Age_Days==5)].Treatment.values))
df = pd.DataFrame(data=results._results_table.data[1:], columns=results._results_table.data[0])
df.to_csv(f'{plot_path}2c_Mianserin_day5_Tukey.csv')


#2d
ax=sns.lmplot(x='Chronological_Age_Days', y='Predicted', hue='Treatment', 
           data=meta[(meta.GEO=='GSE63528')&(meta.Treatment.isin(['FUDR', '50microM Mianserin+FUDR']))], 
              height=height, legend=False,scatter_kws={'s':5})
legend=plt.legend(loc='upper left', title='Treatment', frameon=False, fontsize=fontsize)
plt.setp(legend.get_title(), fontsize=fontsize)
plt.xlabel('Chronological Age', fontsize=fontsize)
plt.ylabel('Predicted Noise Age', fontsize=fontsize)
loc = plticker.MultipleLocator(base=2)
ax.ax.yaxis.set_major_locator(loc)
ax.set_yticklabels(ax.ax.get_yticklabels(), fontsize=fontsize)
loc = plticker.MultipleLocator(base=5)
ax.ax.xaxis.set_major_locator(loc)
ax.set_xticklabels(ax.ax.get_xticklabels(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}2d_Mianserin_timecourse_Noise_clock.pdf')
plt.close()
aov = pg.anova(dv='Predicted', between=['Treatment', 'Chronological_Age_Days'], 
               data=meta[(meta.GEO=='GSE63528')&(meta.Treatment.isin(['FUDR', '50microM Mianserin+FUDR']))], detailed=True, ss_type=2)
aov.to_csv(f'{plot_path}2d_Mianserin_50microM_timecourse_ANOVA.csv')

#2e
meta['Lifespan'] = 'Normal'
meta.loc[meta.Correction_Factor<0.8, 'Lifespan'] = 'Long-lived'
meta.loc[meta.Correction_Factor>1.2, 'Lifespan'] = 'Short-lived'
ax=sns.lmplot(x='Chronological_Age_Days', y='Predicted', data=meta, hue='Lifespan', legend=False, height=height,scatter_kws={'s':5})
plt.xlabel('Chronological Age', fontsize=fontsize)
plt.ylabel('Predicted Noise Age', fontsize=fontsize)
legend=plt.legend(loc='upper left', title='Lifespan', frameon=False, fontsize=fontsize)
plt.setp(legend.get_title(), fontsize=fontsize)
loc = plticker.MultipleLocator(base=5)
ax.ax.yaxis.set_major_locator(loc)
ax.set_yticklabels(ax.ax.get_yticklabels(), fontsize=fontsize)
loc = plticker.MultipleLocator(base=10)
ax.ax.xaxis.set_major_locator(loc)
ax.set_xticklabels(ax.ax.get_xticklabels(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}2e_ChronoAge_vs_NoiseAge_3classes.pdf')
plt.close()


'''
Figure 3
'''
meta = pd.read_csv(f'{input_path}GSE41037_meta.csv', index_col=0, sep='\t')
meta = meta.T
meta = meta[meta['!Sample_source_name_ch1'].str.contains('control')]
meta.columns = ['Name', 'Gender', 'Age', 'Disease']
meta.Age = meta.Age.astype(int)
meta = meta[['Gender', 'Age']]
dat = pd.read_csv(f'{input_path}GSE41037.csv', sep='\t', index_col=0)
dat = dat.iloc[:-1]
dat = dat.dropna()
dat = dat[meta.index]
dat = dat.T
dat = dat.join(meta)
dat = dat.sort_values(by='Age')
blood = dat.iloc[:,:-2].T
young16y = 'GSM1007467' 
ground_sites = np.random.choice(blood[young16y].index, replace=False, size=2000)
ground_sites = np.sort(ground_sites)
ground = blood[blood.index.isin(ground_sites)].loc[:,young16y].values
cell_num = 1000

# 3B
ground_sites = np.random.choice(blood[young16y].index, replace=False, size=500) 
ground_sites = np.sort(ground_sites)
ground = blood[blood.index.isin(ground_sites)].loc[:,young16y].values
d = {}
d['Em'] = []
d['Ed'] = []
d['Eu'] = []
d['R2'] = []
d['Pearson r'] = []
d['Pearson p'] = []
d['Spearman r'] = []
d['Spearman p'] = []
for Em in [0.9, 0.95, 0.99, 0.999, 0.9999, 0.99995]:
    for _ in range(3):
        Ed = 1 - Em
        samples, ages = simulate_cells_for_age_fixed(ground, Em, Ed, samples_per_age=3, age_steps=100,
                                                     cell_num=cell_num)
        samples_2, ages_2 = simulate_cells_for_age_fixed(ground, Em, Ed, samples_per_age=3, age_steps=100,
                                                         cell_num=cell_num)

        regr, stats = pred_and_plot(samples=samples,
                                           ages=ages,
                                           samples2=samples_2,
                                           ages2=ages_2, savepic=False,
                                           outname=f'',
                                           xlab=xlab,
                                           ylab=ylab,
                                           fontsize=fontsize,
                                           height=height,
                                           tick_step=25)

        d['Em'].append(Em)
        d['Ed'].append(Ed)
        d['Eu'].append(Em)
        d['R2'].append(stats[2])
        d['Pearson r'].append(stats[0][0])
        d['Pearson p'].append(stats[0][1])
        d['Spearman r'].append(stats[1][0])
        d['Spearman p'].append(stats[1][1])
df = pd.DataFrame(d)

fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Em', y='R2', data=df, color='black', s=2)
ax=sns.boxplot(x='Em', y='R2', data=df, color='white')
ax.set_xlabel('Methylation maintenance efficiency (%)',fontsize=fontsize)
ax.set_ylabel('R2', fontsize=fontsize)
ax.set_xticklabels([90, 95, 99, 99.9, 99.99, 99.995], fontsize=fontsize)
plt.ylim((0,1.1))
loc = plticker.MultipleLocator(base=0.5)
ax.yaxis.set_major_locator(loc)
ax.set_yticklabels([-0.5,0.0,0.5,1.0], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}3b_Em_vs_R2.pdf')
plt.close()


#Figure 3C
Em = 0.999
cell_num = 1000
samples, ages = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num)
samples_2, ages_2 = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num)

xlab = 'Simulated age'
ylab = 'Predicted age'
regr, stats = pred_and_plot(samples=samples, 
                                   ages=ages, 
                                   samples2=samples_2, 
                                   ages2=ages_2, 
                                   outname=f'{plot_path}3c_single_cell_simulation_Em{Em}',
                                   xlab=xlab,
                                   ylab=ylab, 
                                   fontsize=fontsize, 
                                   height=height, 
                                   tick_step=25, color=dot_color, line_color=line_color, n_jobs=n_jobs)


# Figure 3D
Em = 0.999
cell_num = 1000
d={}
d['Number of features'] = []
d['Pearson r'] = []
d['Pearson p'] = []
d['Spearman r'] = []
d['Spearman p'] = []
d['R2'] = []
for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    if size<1000:
        r = 10
    else:
        r = 3
    for _ in range(r):
        ground_sites = np.random.choice(blood[young16y].index, replace=False, size=size)
        ground_sites = np.sort(ground_sites)
        ground = blood[blood.index.isin(ground_sites)].loc[:, young16y].values
        samples, ages = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100,
                                                     cell_num=cell_num)
        samples_2, ages_2 = simulate_cells_for_age_fixed(ground, Em,samples_per_age=3, age_steps=100,
                                                         cell_num=cell_num)

        regr, stats = pred_and_plot(samples=samples,
                                           ages=ages,
                                           samples2=samples_2,
                                           ages2=ages_2,
                                           outname=f'',
                                           xlab=xlab,
                                           ylab=ylab,
                                           fontsize=fontsize,
                                           height=height,
                                           tick_step=25, savepic=False)

        d['Number of features'].append(size)
        d['Pearson r'].append(stats[0][0])
        d['Pearson p'].append(stats[0][1])
        d['Spearman r'].append(stats[1][0])
        d['Spearman p'].append(stats[1][1])
        d['R2'].append(stats[2])

df =pd.DataFrame(d)
fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Number of features', y='R2', data=df, color='black', s=2)
ax=sns.boxplot(x='Number of features', y='R2', data=df, color='white')
ax.set_xlabel('Feature Size',fontsize=fontsize)
ax.set_ylabel('R2', fontsize=fontsize)
ax.set_ylim(0,1.1)
ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], fontsize=fontsize)
ax.set_yticklabels([0.0,0.5,1.0], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout(pad=1)
plt.savefig(f'{plot_path}3d_FixedEm_Size_vs_R2.pdf')
plt.close()

# 3E
size=2000
Em = 0.999
cell_num = 1000
ground_sites = np.random.choice(blood[young16y].index, replace=False, size=size)
ground_sites = np.sort(ground_sites)
ground = blood[blood.index.isin(ground_sites)].loc[:,young16y].values
samples, ages = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num)
regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], n_jobs=n_jobs)
regr.fit(samples, ages)

Em=0.95
samples_2, ages_2 = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num)    

Em=0.99
samples_3, ages_3 = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num)    

Em=0.999
samples_4, ages_4 = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num)    

Em=0.9999
samples_5, ages_5 = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num)       
d = {}
d['Em=95%'] = regr.predict(samples_2)
d['Em=99%'] = regr.predict(samples_3)
d['Em=99.9%'] = regr.predict(samples_4)
d['Em=99.99%'] = regr.predict(samples_5)
d['True # of noise cycles'] = ages
df = pd.DataFrame(d)
df = df.melt(id_vars =['True # of noise cycles'])
sns.set(font='Times New Roman', style='white')
fig = plt.figure(figsize=(height, height))
g=sns.swarmplot(x='True # of noise cycles', y='value', hue='variable', data=df, s=1, palette='colorblind')
loc = plticker.MultipleLocator(base=10)
g.xaxis.set_major_locator(loc)
g.set_xlabel('Simulated age', fontsize=fontsize)
g.set_ylabel('Predicted age', fontsize=fontsize)
plt.text(60, 500, 'Em=95%' ,fontsize=fontsize)
plt.text(60, 370, 'Em=99%' ,fontsize=fontsize)
plt.text(60, 100, 'Em=99.9%' ,fontsize=fontsize)
plt.text(60, 25, 'Em=99.99%' ,fontsize=fontsize)
g.get_legend().remove()
loc = plticker.MultipleLocator(base=100)
g.yaxis.set_major_locator(loc)
g.set_xticklabels([int(tt) for tt in g.get_xticks()], fontsize=fontsize)
g.set_yticklabels([int(tt) for tt in g.get_yticks()], fontsize=fontsize)
plt.tight_layout(pad=1)
plt.savefig(f'{plot_path}3e_TrainedOn999_PredictedVariableEm_fontsize{fontsize}_height{height}{filetype}')
plt.close()


# Figure 3F
o=f'3f_SingleCell_EmpiricalEm_Ed'
old88 = 'GSM1007832'
Em_lim=0.95
Ed_lim=0.23
Em_df_new = get_noise_Em_all_new(blood, old88, Em_lim=Em_lim, Ed_lim=Ed_lim)
Em_df_new = Em_df_new[Em_df_new.Site.isin(ground_sites)]
samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new, samples_per_age=3, age_steps=100)
samples_emp_noquantile_2, ages_emp_noquantile_2 = simulate_for_age_empirical_noquantile(ground, Em_df_new, samples_per_age=3, age_steps=100)
regr, stats = pred_and_plot(samples=samples_emp_noquantile, 
                                   ages=ages_emp_noquantile, 
                                   samples2=samples_emp_noquantile_2, 
                                   ages2=ages_emp_noquantile_2, 
                                   outname=f'{plot_path}{o}',
                                   xlab='Simulated age',
                                   ylab='Predicted age', 
                                   fontsize=fontsize, 
                                   height=height, 
                                   tick_step=25)



'''
Figure 4
'''

meta = pd.read_csv(f'{input_path}GSE41037_meta.csv', index_col=0, sep='\t')
meta = meta.T
meta = meta[meta['!Sample_source_name_ch1'].str.contains('control')]
meta.columns = ['Name', 'Gender', 'Age', 'Disease']
meta.Age = meta.Age.astype(int)
meta = meta[['Gender', 'Age']]
dat = pd.read_csv(f'{input_path}GSE41037.csv', sep='\t', index_col=0)
dat = dat.iloc[:-1]
cell_num=1000
young16y = 'GSM1007467'
old88y = 'GSM1007832'

Em=0.99
Ed = 1-Em
age_steps=74
kind='scatter'

#4A
d = {}
d['Em_lim'] = []
d['Ed_lim'] = []
d['Pearson r'] = []
d['Pearson p'] = []
d['R2'] = []
horvath =f'{input_path}horvath_clock_coef.csv'
horvath_cpgs, horvath_model = get_clock(clock_csv_file = horvath)
for _ in range(3):
    for Em_lim in [0.95,0.97,0.99]:
        for Ed_lim in [0.01,0.05,0.23]:  
            o = f'4a_Horvath_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells_Emlim{Em_lim}_Edlim{Ed_lim}'
            
            samples_horvath, ages_horvath, data =  get_samples(dat, 
                    cpg_sites=horvath_cpgs, 
                    age_steps=74, 
                    cell_num=cell_num,
                    Em_lim=Em_lim, 
                    Ed_lim=Ed_lim, 
                    young16y = young16y, 
                    old88y = old88y)
            samples_horvath = samples_horvath
            ages_horvath = [i + 15 for i in ages_horvath]
            pear, spear, r2, pred=get_prediction(samples_horvath, 
                       ages_horvath, 
                       data, 
                       clock_model=horvath_model, 
                       outname=o, 
                       scatter_size=scatter_size, 
                       tick_step=25, 
                       fontsize=fontsize, 
                               height=height, kind=kind)
        
            d['Em_lim'].append(Em_lim)
            d['Ed_lim'].append(Ed_lim)
            d['Pearson r'].append(pear[0])
            d['Pearson p'].append(pear[1])
            d['R2'].append(r2)
df = pd.DataFrame(d)
fig = plt.figure(figsize=(height,height))
ax=sns.boxplot(x='Em_lim', y='R2', hue='Ed_lim', data=df, color='white')
ax=sns.swarmplot(x='Em_lim', y='R2', hue='Ed_lim', data=df, dodge=True, s=4)
leg=plt.legend(ax.get_legend_handles_labels()[0][3:], ax.get_legend_handles_labels()[1][3:], 
               fontsize=fontsize, handletextpad=0, labelspacing=0.05, frameon=False)
leg.set_title('Ed limit', prop={'size':fontsize})
leg.legendHandles[0]._sizes = [4]
leg.legendHandles[1]._sizes = [4]
leg.legendHandles[2]._sizes = [4]
plt.xlabel('Em limit', fontsize=fontsize)
plt.ylabel('R2', fontsize=fontsize)
plt.ylim(-0.1,1)
ax.set_yticks([0,0.25,0.5,0.75,1])
ax.set_yticklabels([0,0.25,0.5,0.75,1], fontsize=fontsize)
ax.set_xticklabels([0.95,0.97,0.99], fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}4a_horvath_limits_comp.pdf')
plt.close()

# 4B
age_steps=74
for Em_lim, Ed_lim in [(0.95,0.23),(0.97,0.05)]: 
    o = f'4b_Horvath_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells_Emlim{Em_lim}_Edlim{Ed_lim}'
    horvath = f'{input_path}horvath_clock_coef.csv'
    horvath_cpgs, horvath_model = get_clock(clock_csv_file = horvath)
    
    samples_horvath, ages_horvath, data =  get_samples(dat, 
                cpg_sites=horvath_cpgs, 
                age_steps=age_steps, 
                cell_num=cell_num,
                Em_lim=Em_lim, 
                Ed_lim=Ed_lim, 
                young16y = young16y, 
                old88y = old88y)
    
    ages_horvath = [i + 15 for i in ages_horvath]

    pear, spear, r2, pred=get_prediction(samples_horvath, 
                   ages_horvath, 
                   data, 
                   clock_model=horvath_model, 
                   outname=o,
                   scatter_size=scatter_size, 
                   tick_step=25, 
                   fontsize=fontsize, 
                   height=height, kind=kind, xlab='Simulated age\n(estimated maintenance efficiencies)', ylab='Predicted age (Horvath)')
    
    


#4C
o = f'4c_Horvath_FixedEm{Em}_FixedEd{Ed}_{age_steps}NoiseCycles_{cell_num}SimulatedCells'
samples_horvath, ages_horvath, data = get_samples_fixed(dat, 
                                                        cpg_sites=horvath_cpgs, 
                                                        Em=Em, 
                                                        Ed=Ed, 
                                                        age_steps=age_steps, 
                                                        cell_num=cell_num, 
                                                        young16y = young16y, 
                                                        old88y = old88y)
ages_horvath = [i + 15 for i in ages_horvath]
pear, spear, r2, pred=get_prediction(samples_horvath, 
               ages_horvath, 
               data, 
               clock_model=horvath_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, xlab='Simulated age\n(universal maintenance efficiency)', ylab='Predicted age (Horvath)')


#4D
# PHENOAGE
d = {}
d['Em_lim'] = []
d['Ed_lim'] = []
d['Pearson r'] = []
d['Pearson p'] = []
d['R2'] = []
phenoage = f'{input_path}phenoage_clock_coef.csv'
phenoage_cpgs, phenoage_model = get_clock(clock_csv_file = phenoage)
for _ in range(3):
    for Em_lim in [0.95,0.97,0.99]:
        for Ed_lim in [0.01,0.05,0.23]:    
            print(Em_lim, Ed_lim) 
            o = f'4d_Phenoage_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells_Emlim{Em_lim}_Edlim{Ed_lim}'
            
            samples_phenoage, ages_phenoage, data = get_samples(dat,
                                                                cpg_sites=phenoage_cpgs,
                                                                age_steps=74,
                                                                cell_num=cell_num,
                                                                Em_lim=Em_lim,
                                                                Ed_lim=Ed_lim,
                                                                young16y=young16y,
                                                                old88y=old88y)

            ages_phenoage = [i + 15 for i in ages_phenoage]
            pear, spear, r2, pred=get_prediction(samples_phenoage, 
               ages_phenoage, 
               data, 
               clock_model=phenoage_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind)
        
            d['Em_lim'].append(Em_lim)
            d['Ed_lim'].append(Ed_lim)
            d['Pearson r'].append(pear[0])
            d['Pearson p'].append(pear[1])
            d['R2'].append(r2)
df = pd.DataFrame(d)
fig = plt.figure(figsize=(height,height))
ax=sns.boxplot(x='Em_lim', y='R2', hue='Ed_lim', data=df, color='white')
ax=sns.swarmplot(x='Em_lim', y='R2', hue='Ed_lim', data=df, dodge=True, s=4)
leg=plt.legend(ax.get_legend_handles_labels()[0][3:], ax.get_legend_handles_labels()[1][3:], 
               fontsize=fontsize, handletextpad=0, labelspacing=0.05, frameon=False)
leg.set_title('Ed limit', prop={'size':fontsize})
leg.legendHandles[0]._sizes = [4]
leg.legendHandles[1]._sizes = [4]
leg.legendHandles[2]._sizes = [4]

plt.xlabel('Em limit', fontsize=fontsize)
plt.ylabel('R2', fontsize=fontsize)
plt.ylim(-0.1,1)
ax.set_yticks([0,0.25,0.5,0.75,1])
ax.set_yticklabels([0,0.25,0.5,0.75,1], fontsize=fontsize)
ax.set_xticklabels([0.95,0.97,0.99], fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}4d_phenoage_limits_comp.pdf')
plt.close()


# 4E
age_steps = 74
for Em_lim, Ed_lim in [(0.95,0.23),(0.97,0.05)]: 
    o = f'4e_PhenoAge_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells_Emlim{Em_lim}_Edlim{Ed_lim}'
    phenoage = f'{input_path}phenoage_clock_coef.csv'
    phenoage_cpgs, phenoage_model = get_clock(clock_csv_file = phenoage)
    
    samples_phenoage, ages_phenoage, data =  get_samples(dat, 
                cpg_sites=phenoage_cpgs, 
                age_steps=age_steps, 
                cell_num=cell_num,
                Em_lim=Em_lim, 
                Ed_lim=Ed_lim, 
                young16y = young16y, 
                old88y = old88y)
    ages_phenoage = [i + 15 for i in ages_phenoage]
    pear, spear, r2, pred=get_prediction(samples_phenoage, 
                   ages_phenoage, 
                   data, 
                   clock_model=phenoage_model, 
                   outname=o, 
                   scatter_size=scatter_size, 
                   tick_step=25, 
                   fontsize=fontsize, 
                   height=height, kind=kind, xlab='Simulated age\n(estimated maintenance efficiencies)', ylab='Predicted age (PhenoAge)')
    
    


#4F
# Phenoage on Fixed
samples_phenoage, ages_phenoage, data = get_samples_fixed(dat, 
                                                        cpg_sites=phenoage_cpgs, 
                                                        Em=Em, 
                                                        Ed=Ed, 
                                                        age_steps=age_steps, 
                                                        cell_num=cell_num, 
                                                        young16y = young16y, 
                                                        old88y = old88y)
ages_phenoage = [i + 15 for i in ages_phenoage]
pear, spear, r2, pred=get_prediction(samples_phenoage, 
               ages_phenoage, 
               data, 
               clock_model=phenoage_model, 
               outname=f'4f_PhenoAge_FixedEm{Em}_FixedEd{Ed}_{age_steps}NoiseCycles_{cell_num}SimulatedCells', 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, xlab='Simulated age\n(universal maintenance efficiency)', ylab='Predicted age (PhenoAge)')



'''
Figure 5
'''

# Estimate and correct for cell-type heterogeneity
'''
The cell-type proportion estimation was done with epiDISH in R:

library(EpiDISH)


beta = read.csv(filename, row.names=1, sep='\t')
beta = na.omit(beta) 
beta = as.matrix(beta)
beta=beta[intersect(row.names(beta), row.names(tmp)),,drop=FALSE]
out.l = epidish(beta.m = beta, ref.m = centDHSbloodDMC.m, method='RPC')
write.csv(out.l$estF, 'GSE41037_estCellTypes.csv')

out.l = read.csv('GSE41037_estCellTypes.csv', row.names=1, sep=',')

beta.lm = apply(beta, 1, function(x){
   out.l[colnames(beta),] -> blood
   lm(x ~ B+ NK + CD4T + CD8T + Mono + Neutro + Eosino, data=blood)
})

residuals = t(sapply(beta.lm, function(x) residuals(summary(x))))
colnames(residuals) = colnames(beta)
adj.betas = residuals + matrix(apply(beta,1,mean), nrow=nrow(residuals), ncol=ncol(residuals))
write.csv(adj.betas, 'GSE41037_adjustedbetas.csv')
'''

cell_num=1000
young16y = 'GSM1007467'
old88y = 'GSM1007832'

Em=0.99
Ed = 1-Em
age_steps=74
kind='scatter'
meta = pd.read_csv(f'{input_path}GSE41037_meta.csv', index_col=0, sep='\t')
meta = meta.T
meta = meta[meta['!Sample_source_name_ch1'].str.contains('control')]
meta.columns = ['Name', 'Gender', 'Age', 'Disease']
meta.Age = meta.Age.astype(int)
meta = meta[['Gender', 'Age']]
dat = pd.read_csv(f'{input_path}GSE41037_adjustedbetas.csv', sep='\t', index_col=0) # calculated in R
dat = dat.iloc[:-1]
dat = dat.dropna() 
dat = dat[meta.index]
dat = dat.T
dat = dat.join(meta)
dat = dat.sort_values(by='Age') 
blood = dat.iloc[:,:-2].T
horvath = f'{input_path}horvath_clock_coef.csv'
horvath_cpgs, horvath_model = get_clock(clock_csv_file = horvath)
size=len(horvath_cpgs)
ground_sites = np.sort(horvath_cpgs)
ground = blood[blood.index.isin(ground_sites)].loc[:, young16y].values
old88 = 'GSM1007832'
Em_lim = 0.97
Ed_lim = 0.05  
cell_num = 1000
Em_df_new = get_noise_Em_all_new(blood, old88, Em_lim=Em_lim, Ed_lim=Ed_lim)
Em_df_new = Em_df_new[Em_df_new.Site.isin(ground_sites)]
samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new,
                                                                                    samples_per_age=1, age_steps=74,
                                                                                    cell_num=cell_num)


tmp_ages = ages_emp_noquantile
rescaled_ages = ((tmp_ages - np.min(tmp_ages))/(np.max(tmp_ages) - np.min(tmp_ages))*400)-120

regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    alphas=[1])  
regr.fit(samples_emp_noquantile, rescaled_ages)
pred = regr.predict(blood[blood.index.isin(ground_sites)].T)
pred = pd.DataFrame(pred)
pred.columns = ['Predicted']
pred['Sample'] = blood.columns
pred = pred.set_index('Sample')
pred = pred.join(meta)
pred['Disease'] = 'Healthy'
pred = pred[~pred.index.isin([young16y, old88])]  # exclude the starting point and end point

pear = pearsonr(pred.Predicted, pred.Age)
spear = spearmanr(pred.Predicted, pred.Age)

g = sns.jointplot(x=pred.Age, y=pred.Predicted, kind='reg', height=height, scatter_kws={'s': scatter_size},
                       color=dot_color, joint_kws={'line_kws': {'color': line_color}})
g.set_axis_labels('Chronological age', 'Predicted age (Stochastic data-based)', fontsize=fontsize)
g.ax_joint.set_xticks([20,40,60,80])
g.ax_joint.set_yticks([0,20,40,60,80,100])
g.ax_joint.set_xticklabels([20,40,60,80], fontsize=fontsize)
g.ax_joint.set_yticklabels([0,20,40,60,80,100], fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}4a_RandomClock_Horvath_adjusted.pdf')
plt.close()




# 5C pan-mammalian clock
'''
The mydata_GitHub data is the example data from the UniversalPanMammalianClock Github Page:
https://github.com/shorvath/MammalianMethylationConsortium/blob/main/UniversalPanMammalianClock/ClockParameters/mydata_GitHub.Rds

First we simulate samples starting form the young biological ground state (here the youngest sample from the UniversalPanMammalianClock example data, in Supplement Figure 10 the ground state is correspondingly changed to a different species).
The samples are either generated with an empirically estimated maintenance rate, or with a fixed universal rate as described above.
The calculated coefficients are saved in "clockcoefficients" and subsequently used to predict the independent samples, the same way the UniversalPanMammalianClock calculates its predictions in R:

pred=as.numeric(as.matrix(subset(independentbetas,select=as.character(clockcoefficients$)))%*%clockcoefficients$Coef)
DNAmRelativeAge=exp(-exp(-1*pred))

where independentbetas contains the beta values of the independent samples and clockcoefficients the saved coefficients as calculated below.

'''


young = '204529320059_R04C02' #204529320059_R04C02 0.57y
old = '203877620075_R03C02' #203877620075_R03C02 57.87
meta= pd.read_csv(f'{plot_path}mydata_GitHub_SampleAnnotation.csv', index_col=2)
tmp = pd.read_csv(f'{plot_path}mydata_GitHub_meth_betas.csv', index_col=0)
# cpgs in rows, samples in columns
ground_sites = np.sort(tmp[young].index)
tmp = tmp[tmp.index.isin(ground_sites)]
age_steps = 67 # maxage of the species
und_sites = np.sort(tmp[young].index)
Em_lim = 0.97
Ed_lim = 0.05
samples_horvath, ages_horvath, data =  get_samples(tmp, 
                cpg_sites=ground_sites, 
                age_steps=maxage, 
                cell_num=1000,
                Em_lim=Em_lim, 
                Ed_lim=Ed_lim, 
                young16y = young, 
                old88y = old)

regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.01, 0.001, 0.1],
                    alphas=[1])
clock2_age = [-np.log(-np.log((age) / (maxage))) for age in ages_horvath]
regr.fit(samples_horvath, clock2_age)
coefs = pd.DataFrame(regr.coef_, index = ground_sites)
coefs.columns = ['Coef']
coefs=coefs[coefs.Coef!=0]
coefs.loc['Intercept'] = regr.intercept_

## example plot, note that running stochastic variation accumulation simulations with subsequent Elastic net regression
# to build a clock will due to the random nature produce slightly different results and coefficients everytime
plot_circles(Tissues=['Blood'], prediction_output_file=f'{input_path}panmammalian_prediction_noise_clock_Empirical_maxage67_l10.001_coefs_.csv',numsam=5, outname='Empirical_maxage67_l10.001')


# Figure 5D
'''
The ArrayConverterAlgorithm for the human datasets was used as described in:
https://github.com/shorvath/MammalianMethylationConsortium/tree/main/UniversalPanMammalianClock/R_code/ArrayConverterAlgorithm
'''

Clock1 = 'noise_clock_Empirical_maxage67_l10.001_coefs_'
Clock2 = 'noise_clock_maxage67_0.99_l10.001_coefs_'
Clock3 = 'noise_clock_Empirical_maxage67_l10.01_ALLcoefs_'
Clock4 = 'noise_clock_maxage67_0.99_l10.001_ALLcoefs_'

df_all = pd.DataFrame()
for prediction_output_file in [Clock1, Clock2, Clock3, Clock4]:
    smoking_df = pd.read_csv(f'{plot_path}{prediction_output_file}_Smoking_Predictions.csv')

    smoking_df['RelAge'] = smoking_df.Age / 122.5

    smoking_df['Ex_smoker'] = np.nan
    smoking_df.loc[smoking_df.Smoking != 1, 'Ex_smoker'] = 0
    smoking_df.loc[smoking_df.Smoking == 1, 'Ex_smoker'] = 1
    smoking_df['Smoker'] = np.nan
    smoking_df.loc[smoking_df.Smoking != 2, 'Smoker'] = 0
    smoking_df.loc[smoking_df.Smoking == 2, 'Smoker'] = 1

    smoking_df['Interaction_Ex'] = smoking_df['Age'] * smoking_df['Ex_smoker']
    smoking_df['Interaction_Smoker'] = smoking_df['Age'] * smoking_df['Smoker']
    Interaction = ols('clock ~ Age + Ex_smoker + Smoker + Interaction_Ex + Interaction_Smoker', data=smoking_df).fit()

    d = {}
    d['Tissue'] = []
    d['Intervention'] = []
    d['t'] = []
    d['p'] = []

    d['Tissue'].append('Blood')
    d['Intervention'].append('Smoking')
    d['t'].append(-Interaction.summary2().tables[1].loc['Interaction_Smoker'][
        't'])  # negative since the pan-mammalian clock paper used negative values to indicate a positive age acceleration
    d['p'].append(Interaction.summary2().tables[1].loc['Interaction_Smoker']['P>|t|'])

    # MOUSE
    tmp = pd.read_csv(f'{input_path}panmammalian_prediction_{prediction_output_file}.csv', index_col=0)
    tmp = tmp.dropna(subset=['Age'])
    tmp.Age = tmp.Age.astype(float)
    tmp = tmp[~tmp.MaxAge.isin(['Cervus canadensis', 'Cervus elaphus'])]  # discard, since no proper meta data
    tmp.MaxAge = tmp.MaxAge.astype(float)
    tmp['RelAge'] = tmp.Age / tmp.MaxAge
    tmp = tmp[~(tmp.Experiment.isna())]

    for t in ['Liver', 'CerebralCortex', 'Kidney']:
        d['Tissue'].append(t)
        d['Intervention'].append('GHRKO.Het.KO')
        # GHRKO are not 0.5y
        tt = ttest_ind(tmp[(tmp.Experiment == 'Genetic perturbation') & (tmp.ClockTraining == 'no') & (
                tmp.Tissue == t) & (tmp.Intervention == 'WT') & (tmp.Age != 0.5)].clock,
                       tmp[(tmp.Experiment == 'Genetic perturbation') & (tmp.ClockTraining == 'no') & (
                               tmp.Tissue == t) & (tmp.Intervention == 'GHRKO.Het.KO') & (tmp.Age != 0.5)].clock)
        d['t'].append(tt[0])
        d['p'].append(tt[1])

    for t in ['CerebralCortex', 'Striatum']:
        d['Tissue'].append(t)
        d['Intervention'].append('Tet3KO.Het')
        # Tet3 are all 0.5y
        tt = ttest_ind(tmp[(tmp.Experiment == 'Genetic perturbation') & (tmp.ClockTraining == 'no') & (
                tmp.Tissue == t) & (tmp.Intervention == 'WT') & (tmp.Age == 0.5)].clock,
                       tmp[(tmp.Experiment == 'Genetic perturbation') & (tmp.ClockTraining == 'no') & (
                               tmp.Tissue == t) & (tmp.Intervention == 'Tet3KO.Het') & (tmp.Age == 0.5)].clock)
        d['t'].append(tt[0])
        d['p'].append(tt[1])

        # CR
    tt = ttest_ind(tmp[(tmp.Experiment == 'CR') & (tmp.Tissue == 'Liver') & (tmp.Intervention == 'Control')].clock,
                   tmp[(tmp.Experiment == 'CR') & (tmp.Tissue == 'Liver') & (tmp.Intervention == 'CR')].clock)
    d['Tissue'].append('Liver')
    d['Intervention'].append('CR')
    d['t'].append(tt[0])
    d['p'].append(tt[1])

    df = pd.DataFrame(d)
    df['FDR_bh'] = statsmodels.stats.multitest.multipletests(df.p, method='fdr_bh')[1]
    df['Clock'] = prediction_output_file
    if len(df_all) == 0:
        df_all = df
    else:
        df_all = df_all.append(df)


## plot heatmap
sns.set(font='Times New Roman', style='white')
df_all['Tissue_Clock'] = df_all['Tissue']+df_all['Clock']
df_all['ylabel'] = ''
df_all.Tissue_Clock = df_all.Tissue_Clock.str.replace(Clock1, '_Clock1')
df_all.Tissue_Clock = df_all.Tissue_Clock.str.replace(Clock2, '_Clock2')
df_all.Tissue_Clock = df_all.Tissue_Clock.str.replace(Clock3, '_Clock3')
df_all.Tissue_Clock = df_all.Tissue_Clock.str.replace(Clock4, '_Clock4')
tdata = df_all[['t', 'Tissue_Clock', 'Intervention']].pivot(index='Tissue_Clock', columns='Intervention')
tdata.columns = [c[1] for c in tdata.columns]
fdrdata = df_all[['FDR_bh', 'Tissue_Clock', 'Intervention']].pivot(index='Tissue_Clock', columns='Intervention') 
fdrdata.columns = [c[1] for c in fdrdata.columns]

fdrdata = fdrdata[['GHRKO.Het.KO', 'Tet3KO.Het', 'CR', 'Smoking']]
tdata = tdata[['GHRKO.Het.KO', 'Tet3KO.Het', 'CR', 'Smoking']]
fig = plt.figure(figsize=(2*height,2*height))
ax = sns.heatmap(tdata, cmap='vlag', vmin=-df_all.t.max(), vmax=df_all.t.max())

for i in range((len(fdrdata))):
    for j in range(len(fdrdata.columns)):
        try:
            if np.isnan(fdrdata.iloc[i, j]):
                val = ''
        except:
            print('')


        val = "{:.2e}".format(fdrdata.iloc[i, j])
        plt.text(j + 0.5, i + 0.5, str(val), ha='center', va='center', color='black', fontsize=fontsize)

ax.set_xticklabels(['GHRKO Exp.1', 'Tet3','CR', 'Smoking'], fontsize=fontsize)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
plt.tight_layout()
plt.ylabel(None)
plt.savefig(f'{plot_path}3d_Validation_Heatmap.pdf')
plt.close()


'''
Supplement Figure 1
'''

#######
# LOGIT Figure 1

##Sup 1B and C
size = 2000
ground = np.random.randint(0,1000, size=size)/1000
stats_dict = {}
stats_dict['Noise_Age'] = []
stats_dict['Pearson'] = []
stats_dict['Spearman'] = []
stats_dict['R2'] = []
stats_dict['MAD'] = []
size = 2000
noise_age_list = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,0.3,0.5]
for noise_age in noise_age_list:
    for _ in range(3):
        ground = np.random.randint(0, 1000, size=size) / 1000
        samples, ages = random_epi_logit(ground,
                                   samples_per_age=3,
                                   epi_sites=size,
                                   noise_ground=0.01,
                                   noise_age=noise_age,
                                   age_steps=100)

        samples_2, ages_2 = random_epi_logit(ground,
                                       samples_per_age=3,
                                       epi_sites=size,
                                       noise_ground=0.01,
                                       noise_age=noise_age,
                                       age_steps=100)

        o = f'1e_random_clock_withlimit_{noise_age}_LOGIT'
        regr, stats = pred_and_plot(samples=samples,
                                    ages=ages,
                                    samples2=samples_2,
                                    ages2=ages_2,
                                    outname=f'{plot_path}{o}',
                                    xlab=xlab,
                                    ylab=ylab,
                                    fontsize=fontsize,
                                    height=height,
                                    tick_step=25, color=dot_color, line_color=line_color, n_jobs=n_jobs)
        stats_dict['Noise_Age'].append(noise_age)
        stats_dict['Pearson'].append(stats[0])
        stats_dict['Spearman'].append(stats[1])
        stats_dict['R2'].append(stats[2])
        stats_dict['MAD'].append(stats[3])

stats_df = pd.DataFrame(stats_dict)
fig = plt.figure(figsize=(height, height))
g = sns.swarmplot(x='Noise_Age', y='R2', data=stats_df, color='black', s=2)
g = sns.boxplot(x='Noise_Age', y='R2', data=stats_df, color='white')
g.set_xlabel('Noise', fontsize=fontsize)
g.set_ylabel('R2', fontsize=fontsize)
g.set_ylim(0, 1)
loc = plticker.MultipleLocator(base=0.2)
g.yaxis.set_major_locator(loc)
g.set_yticklabels([round(tt, 3) for tt in g.get_yticks()], fontsize=fontsize)
g.set_xticklabels(noise_age_list, fontsize=fontsize)
g.set_xlabel('Noise standard deviation')
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}1e_random_clock_withlimit_noiseage_comp_LOGIT.pdf')
plt.close()



##Sup1D
samples, ages = random_epi_logit(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.2, 
                                   age_steps = 100)


samples_2, ages_2 = random_epi_logit(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.2, 
                                   age_steps = 100)

regr_2, stats = pred_and_plot(samples=samples, 
                                   ages=ages, 
                                   samples2=samples_2, 
                                   ages2=ages_2, 
                                   outname=f'{plot_path}{o}',
                                   xlab=xlab,
                                   ylab=ylab, 
                                   fontsize=fontsize, 
                                   height=height, 
                                   tick_step=25,savepic=False, color=dot_color, line_color=line_color, n_jobs=n_jobs)

tick_step=0.5
xlab='Elastic net regression coefficients run 1'
ylab='Elastic net regression coefficients run 2'
sns.set(font='Times New Roman', style='white')
g = sns.jointplot(x=regr.coef_, y=regr_2.coef_, kind='reg', height=height, scatter_kws={'s':1}, color=dot_color, joint_kws={'line_kws':{'color':line_color}}) 
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc = plticker.MultipleLocator(base=tick_step)
g.ax_joint.xaxis.set_major_locator(loc)
g.ax_joint.yaxis.set_major_locator(loc)
if isinstance(tick_step, int):
    g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
    g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
else:
    g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
    g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}1f_random_clock_withlimit_reproducible_LOGIT.pdf')
plt.close()


## Sup 1E
# Regression to the mean
tick_step=0.5
xlab='Ground state values'
ylab='Elastic net regression coefficients'
sns.set(font='Times New Roman', style='white')

g = sns.jointplot(x=ground, y=regr.coef_, kind='reg', height=height, scatter_kws={'s':1}, color=dot_color, joint_kws={'line_kws':{'color':line_color}}) 
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc = plticker.MultipleLocator(base=tick_step)
g.ax_joint.xaxis.set_major_locator(loc)
g.ax_joint.yaxis.set_major_locator(loc)
if isinstance(tick_step, int):
    g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
    g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
else:
    g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
    g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}1g_random_clock_withlimit_RegressionToTheMean_LOGIT.pdf')
plt.close()

##1F
stats_dict = {}
stats_dict['Size'] = []
stats_dict['Pearson'] = []
stats_dict['Spearman'] = []
stats_dict['R2'] = []
stats_dict['MAD'] = []
stats_dict['alpha'] = []
stats_dict['l1'] = []
s = 0
e = 1000
for size in [pow(2, i) for i in range(14)]:
    r = 3
    for _ in range(r):
        ground = np.random.randint(s, e, size=size) / 1000

        samples, ages = random_epi_logit(ground, epi_sites=size)
        samples_2, ages_2 = random_epi_logit(ground, epi_sites=size)

        o = f'ground_{s}_{e}_{size}'
        regr, stats = pred_and_plot(samples=samples, ages=ages,
                                           samples2=samples_2, ages2=ages_2,
                                           outname=f'{plot_path}{o}.png', savepic=False,
                                           xlab=None, ylab=None)
        stats_dict['Size'].append(size)
        stats_dict['Pearson'].append(stats[0])
        stats_dict['Spearman'].append(stats[1])
        stats_dict['R2'].append(stats[2])
        stats_dict['MAD'].append(stats[3])
        stats_dict['alpha'].append(regr.alpha_)
        stats_dict['l1'].append(regr.l1_ratio_)

    

stats_df = pd.DataFrame(stats_dict)
fig = plt.figure(figsize=(height, height))
ax = sns.swarmplot(x='Size', y='R2', data=stats_df, color='black', s=2)
ax = sns.boxplot(x='Size', y='R2', data=stats_df, color='white')
ax.set_xlabel('Feature Size', fontsize=fontsize)
ax.set_ylabel('R2', fontsize=fontsize)
plt.xticks(rotation=60)
plt.ylim(0,1.1)
ax.set_yticklabels([0,0.5,1], fontsize=fontsize)
ax.set_xticklabels([pow(2, i) for i in range(14)], fontsize=fontsize)
plt.tight_layout(pad=1)
plt.savefig(f'{plot_path}1h_random_clock_withlimit_FeatureSizeComp_LOGIT.pdf')
plt.close()



##1G
size = 2000
ground = np.random.randint(0,1000, size=size)/1000
samples, ages = random_epi_logit(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.2, 
                                   age_steps = 100)


samples_2, ages_2 = random_epi_logit(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.2, 
                                   age_steps = 100)

samples_3, ages_2 = random_epi_logit(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.4, 
                                   age_steps = 100)

samples_4, ages_2 = random_epi_logit(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.8, 
                                   age_steps = 100)
samples_5, ages_2 = random_epi_logit(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.1, 
                                   age_steps = 100)

xlab = 'Simulated age'
ylab = 'Predicted age'
regr, stats = pred_and_plot(samples=samples, 
                                   ages=ages, 
                                   samples2=samples_2, 
                                   ages2=ages_2, 
                                   outname=f'',
                                   xlab=xlab,
                                   ylab=ylab, 
                                   fontsize=fontsize, 
                                   height=height, 
                                   tick_step=25, savepic=False)

d = {}
d['Learned Noise'] = regr.predict(samples_2)
d['2x Noise'] = regr.predict(samples_3)
d['4x Noise'] = regr.predict(samples_4)
d['0.5x Noise'] = regr.predict(samples_5)
d['True # of noise cycles'] = ages
df = pd.DataFrame(d)
df = df.melt(id_vars=['True # of noise cycles'])

sns.set(font='Times New Roman', style='white')
fig = plt.figure(figsize=(height, height))
g=sns.swarmplot(x='True # of noise cycles', y='value', hue='variable', data=df, s=1, palette='colorblind')
plt.ylim(0,250)
loc = plticker.MultipleLocator(base=10)
locy = plticker.MultipleLocator(base=50)
g.xaxis.set_major_locator(loc)
g.yaxis.set_major_locator(locy)
g.set_xlabel(xlab, fontsize=fontsize)
g.set_ylabel(ylab, fontsize=fontsize)
plt.text(60, 30, 'N(0,0.1)' ,fontsize=fontsize)
plt.text(60, 85, 'N(0,0.2)' ,fontsize=fontsize)
plt.text(60, 150, 'N(0,0.4)' ,fontsize=fontsize)
plt.text(60, 230, 'N(0,0.8)' ,fontsize=fontsize)
g.get_legend().remove()
g.set_xticklabels([int(tt) for tt in g.get_xticks()], fontsize=fontsize)
g.set_yticklabels([int(tt) for tt in g.get_yticks()], fontsize=fontsize)
plt.tight_layout(pad=1)
plt.savefig(f'{plot_path}1i_random_clock_withlimit_NoiseComp_LOGIT.pdf')
plt.close()


'''
Supplement Figure 2
'''

meta = pd.read_csv(f'{input_path}SupplementaryTable1.csv', index_col=0)
meta['Bio_Age'] = meta.Biological_Age_in_Hours
meta.Bio_Age = meta.Bio_Age.astype(float)
meta = correct_Bio_Age(meta)
meta['Bio_Age'] = meta.Bio_Age / 24
meta['Chronological_Age_Days'] = meta['Chronological_Age_in_Hours'] / 24
dat = pd.read_csv(f'{input_path}Celegans_RNAseq_counts.csv', index_col=0)
dat.columns = [c[:-3] for c in dat.columns]

dat = dat.T
dat = dat[dat.sum(axis=1) != 0]
dat = dat.join(meta[['Bio_Age']])
dat = dat.dropna()
meta = meta.dropna(subset=['Bio_Age'])
meta = meta[meta.index.isin(dat.index)]


y = dat.Bio_Age.values
train = dat.drop('Bio_Age', axis=1)
train += 1
train = np.log10(train)
train = train.T
train = train.div(train.max())
train = train[train.sum(axis=1) != 0]
train = train.T
train = train.join(meta[['Bio_Age']])
train_bin = make_binary(train)
train = train.drop('Bio_Age', axis=1)
train = train.T
train_bin = train_bin.drop('Bio_Age', axis=1)
train_bin = train_bin.T
youngest_list = meta[meta.Bio_Age == meta.Bio_Age.min()].index
youngest = youngest_list[0]
forquant = train[[youngest]]
forquant['Old'] = train[
                             meta[meta.Bio_Age == meta.Bio_Age.max()].index].iloc[:, 0].values

forquant.columns = ['Young', 'Old']

ground_sites = forquant.index
ground_sites = np.sort(ground_sites)
ground = forquant[forquant.index.isin(ground_sites)].loc[:, 'Young'].values

coefs = pd.read_csv(f'{input_path}BitAgeClock.csv', index_col=0, sep='\t')
intercept = 103.55

x='Bio_Age'
d = {}
d['Noise_Age'] = []
d['Pearson r'] = []
d['Pearson r - BitAge'] = []
d['Pearson p'] = []
d['Pearson p - BitAge'] = []
d['R2'] = []
d['R2 - BitAge'] = []
for _ in range(5):
    for noise_age in [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125]:
        print(noise_age)
        samples, ages = random_epi(ground,
                                               samples_per_age=10,
                                               epi_sites=len(ground),
                                               noise_ground=0.01,
                                               noise_age=noise_age,
                                               age_steps=16)
        
        samples_df = pd.DataFrame(samples)
        samples_df.columns = ground_sites
        samples_df['Bio_Age'] = ages
        samples_bin = make_binary(samples_df)
        x_train = samples_bin.filter(regex='WB').values
        y = samples_bin.Bio_Age.values
        regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            alphas=[1])  # 
        regr.fit(x_train, y)  

        test = train_bin.T
        test = test[ground_sites]
        test = test[~test.index.isin([youngest])] 

        y_pred = regr.predict(test.values)
        test['Predicted'] = y_pred
        meta = meta.join(test[['Predicted']])
        meta = meta[~meta.index.isin([youngest])] 
        d['Noise_Age'].append(noise_age)
        d['Pearson r'].append(pearsonr(meta[x], meta.Predicted)[0])
        d['Pearson p'].append(pearsonr(meta[x], meta.Predicted)[1])
        d['R2'].append(r2_score(meta[x], meta.Predicted))
        meta = meta.drop('Predicted', axis=1)

        y = samples_bin.Bio_Age
        samples_bin = samples_bin.drop('Bio_Age', axis=1)
        samples_bin = samples_bin.T
        coefs_samples = coefs.join(samples_bin)
        coefs_samples = coefs_samples.dropna()
        df = {'Name': [], 'Prediction': []}
        for i in range(1, len(coefs_samples.columns)):
            df['Name'].append(coefs_samples.columns[i])
            df['Prediction'].append(sum(coefs_samples.iloc[:, 0] * coefs_samples.iloc[:, i]) + intercept)
        df = pd.DataFrame(df)
        df['Bio_Age'] = y
        d['Pearson r - BitAge'].append(pearsonr(df.Bio_Age, df.Prediction)[0])
        d['Pearson p - BitAge'].append(pearsonr(df.Bio_Age, df.Prediction)[1])
        d['R2 - BitAge'].append(r2_score(df.Bio_Age, df.Prediction))
df = pd.DataFrame(d)
fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Noise_Age', y='Pearson r - BitAge', data=df, color='black', s=2)
ax=sns.boxplot(x='Noise_Age', y='Pearson r - BitAge', data=df, color='white')
ax.set_xlabel('Noise standard variation',fontsize=fontsize)
ax.set_ylabel('Pearson correlation', fontsize=fontsize)
ax.set_ylim(0,1)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
ax.set_yticklabels([0,0.25,0.5,0.75,1], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}Sup2a_predict_worm_correlation_comp_bitage_newdata_excludeyoungest.pdf')
plt.close()



# sup 2b,c
noise_age = 0.01
meta['Bio_Age_Permuted'] = np.random.permutation(meta.Bio_Age)
d = {}
d['Size'] = []
d['Pearson r'] = []
d['Pearson p'] = []
d['R2'] = []
d['Pearson r - Permuted'] = []
d['Pearson p - Permuted'] = []
d['R2 - Permuted'] = []
for _ in range(20):
    for size in [100,500, 1000,2000,3000,4000,5000,10000, 30000]:
        ground_sites = np.random.choice(forquant.index, replace=False, size=size)
        ground_sites = np.sort(ground_sites)
        ground = forquant[forquant.index.isin(ground_sites)].loc[:, 'Young'].values
        print(noise_age)
        samples, ages = random_epi(ground,
                                               samples_per_age=10,
                                               epi_sites=len(ground),
                                               noise_ground=0.01,
                                               noise_age=noise_age,
                                               age_steps=16)
        samples_df = pd.DataFrame(samples)
        samples_df.columns = ground_sites
        samples_df['Bio_Age'] = ages
        samples_bin = make_binary(samples_df)
        x_train = samples_bin.filter(regex='WB').values
        y = samples_bin.Bio_Age.values
        regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            alphas=[1])  # 
        regr.fit(x_train, y)  

        test = train_bin.T
        test = test[ground_sites]

        y_pred = regr.predict(test.values)
        test['Predicted'] = y_pred
        meta = meta.join(test[['Predicted']])
        d['Size'].append(size)
        d['Pearson r'].append(pearsonr(meta[x], meta.Predicted)[0])
        d['Pearson p'].append(pearsonr(meta[x], meta.Predicted)[1])
        d['R2'].append(r2_score(meta['Bio_Age_Permuted'], meta.Predicted))
        d['Pearson r - Permuted'].append(pearsonr(meta['Bio_Age_Permuted'], meta.Predicted)[0])
        d['Pearson p - Permuted'].append(pearsonr(meta['Bio_Age_Permuted'], meta.Predicted)[1])
        d['R2 - Permuted'].append(r2_score(meta['Bio_Age_Permuted'], meta.Predicted))
        meta = meta.drop('Predicted', axis=1)
df = pd.DataFrame(d)

# Sup 2b
fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Size', y='Pearson r', data=df, color='black', s=2)
ax=sns.boxplot(x='Size', y='Pearson r', data=df, color='white')
ax.set_xlabel('Feature size',fontsize=fontsize)
ax.set_ylabel('Pearson correlation', fontsize=fontsize)
ax.set_ylim(0,1)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}Sup2b_predict_worm_correlation_comp_featuresize.pdf')
plt.close()
# Sup 2c
fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Size', y='Pearson r - Permuted', data=df, color='black', s=2)
ax=sns.boxplot(x='Size', y='Pearson r - Permuted', data=df, color='white')
ax.set_xlabel('Feature size',fontsize=fontsize)
ax.set_ylabel('Pearson correlation', fontsize=fontsize)
ax.set_ylim(0,1)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}Sup2c_predict_worm_correlation_comp_featuresize.pdf')
plt.close()





'''
Supplement Figure 3
'''
size = 2000
ground = np.random.randint(0,1000, size=size)/1000
samples, ages = random_epi(ground, 
                                   samples_per_age = 3, 
                                   epi_sites = size, 
                                   noise_ground = 0.01,
                                   noise_age = 0.05, 
                                   age_steps = 100)
##sup3A
scatter_size=1
xlab='Ground state values'
ylab='Ground state values + 1x Gaussian noise'
tick_step=0.25
sns.set(font='Times New Roman', style='white')
g=sns.jointplot(x=ground, y=samples[0], height=height, kind='scatter',s=4, color=dot_color)

g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc = plticker.MultipleLocator(base=tick_step)
g.ax_joint.xaxis.set_major_locator(loc)
g.ax_joint.yaxis.set_major_locator(loc)

g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
plt.savefig(f'{plot_path}Sup3a_random_clock_withlimit_Sample1Noise.pdf')
plt.close()

##Sup3b
scatter_size=1
xlab='Ground state values'
ylab='Ground state values + 100x Gaussian noise'
tick_step=0.25
sns.set(font='Times New Roman', style='white')
g=sns.jointplot(x=ground, y=samples[-1], kind='scatter',height=height, s=4, color=dot_color)

g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc = plticker.MultipleLocator(base=tick_step)
g.ax_joint.xaxis.set_major_locator(loc)
g.ax_joint.yaxis.set_major_locator(loc)

g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
plt.savefig(f'{plot_path}Sup3b_random_clock_withlimit_Sample100Noise.pdf')
plt.close()

# Supplement 3C
meta = pd.read_csv(f'{input_path}GSE41037_meta.csv', index_col=0, sep='\t')
meta = meta.T
meta = meta[meta['!Sample_source_name_ch1'].str.contains('control')]
meta.columns = ['Name', 'Gender', 'Age', 'Disease']
meta.Age = meta.Age.astype(int)
meta = meta[['Gender', 'Age']]
dat = pd.read_csv(f'{input_path}GSE41037.csv', sep='\t', index_col=0)
dat = dat.iloc[:-1]
dat = dat.dropna()
dat = dat[meta.index]
dat = dat.T
dat = dat.join(meta)
dat = dat.sort_values(by='Age') 
blood = dat.iloc[:,:-2].T

o=f'Sup3c_Bio_Young_Old'
xlab='Young DNAm sample'
ylab='Old DNAm sample'
young16y = 'GSM1007467' 
old88y = 'GSM1007832'
tick_step=0.25
sns.set(font='Times New Roman', style='white')
g=sns.jointplot(x=young16y, y=old88y, data=blood, kind='reg', height=height, scatter_kws={'s':scatter_size}, color=dot_color)
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc = plticker.MultipleLocator(base=tick_step)
g.ax_joint.xaxis.set_major_locator(loc)
g.ax_joint.yaxis.set_major_locator(loc)
g.ax_joint.plot([0,1], [0,1], ':k')
g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
plt.tight_layout(pad=0.5)
plt.savefig(f'{plot_path}{o}_fontsize{fontsize}_height{height}{filetype}')
plt.close()

#Sup3d
young16y = 'GSM1007467' 
young17y = 'GSM1007336'
blood = blood[[young16y, young17y]] 
blood['Diff'] = blood.loc[:,young17y] - blood.loc[:,young16y]
ax=blood[[young16y]].plot.hist(bins=100, figsize=(height,height))
plt.vlines(x = 0, ymin=0, ymax=3000, color='black')
plt.vlines(x = blood[[young16y]].quantile(q=0.2).values[0], ymin=0, ymax=3000, color='black')
plt.vlines(x = blood[[young16y]].quantile(q=0.4).values[0], ymin=0, ymax=3000, color='black')
plt.vlines(x = blood[[young16y]].quantile(q=0.6).values[0], ymin=0, ymax=3000, color='black')
plt.vlines(x = blood[[young16y]].quantile(q=0.8).values[0], ymin=0, ymax=3000, color='black')
plt.vlines(x =1, ymin=0, ymax=3000, color='black')
plt.xlabel('Ground state', fontsize=fontsize)
plt.ylabel('Frequency', fontsize=fontsize)
ax.set_xticklabels([round(x,2) for x in ax.get_xticks()], fontsize=fontsize)
ax.set_yticklabels([int(x) for x in ax.get_yticks()], fontsize=fontsize)
plt.legend('')
plt.tight_layout()
plt.savefig(f'{plot_path}Sup3d_Quantiles_dist.pdf')
plt.close()

#Sup3e
ax = sns.jointplot(x=young16y, y='Diff', data=blood, height=height,s=4, color='grey')
ax.ax_joint.set_xlabel('Ground state: Young 16y', fontsize=fontsize)
ax.ax_joint.set_ylabel('Young 17y - Young 16y', fontsize=fontsize)
loc = plticker.MultipleLocator(base=0.5)
ax.ax_joint.xaxis.set_major_locator(loc)
ax.ax_joint.yaxis.set_major_locator(loc)
ax.ax_joint.set_xticklabels(ax.ax_joint.get_xticks(), fontsize=fontsize)
ax.ax_joint.set_yticklabels(ax.ax_joint.get_yticks(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}Sup3e_Young_diff.pdf')
plt.close()

#Sup3f
young16y = 'GSM1007467' 
young17y = 'GSM1007336'
young_blood = blood[[young16y, young17y]]
noise_age_df = get_noise_func_parm(young_blood, start_ind=0, end_ind=1, step=5,
                                   normalize='None')  

df = pd.DataFrame()
for i in range(5):
    q1 = noise_age_df['Q1'][i]
    q2 = noise_age_df['Q2'][i]
    print(i, q1,q2)
    if q1 == 0:
        # generate noise based on the parameters estimated by get_noise_func_parm()
        r1 = scipy.stats.lognorm.rvs(size=2000,
                                     scale=noise_age_df['Param'][i]['lognorm']['scale'],
                                     loc=noise_age_df['Param'][i]['lognorm']['loc'],
                                     s=noise_age_df['Param'][i]['lognorm']['s'])
    else:
        r1 = scipy.stats.lognorm.rvs(size=2000,
                                     scale=noise_age_df['Param'][i]['lognorm']['scale'],
                                     loc=noise_age_df['Param'][i]['lognorm']['loc'],
                                     s=noise_age_df['Param'][i]['lognorm']['s'])
    df[f'Quantile {i+1}'] = r1
fig, ax = plt.subplots(1, 1, figsize=(height,height))
for s in df.columns:
    df[s].plot(kind='density')
fig.show()
plt.legend(fontsize=6)
plt.xlim(-0.15, 0.15)

ax.set_xlabel('Noise values', fontsize=fontsize)
ax.set_ylabel('Density', fontsize=fontsize)
loc = plticker.MultipleLocator(base=0.1)
ax.xaxis.set_major_locator(loc)
ax.set_xticklabels([round(tt,2) for tt in ax.get_xticks()], fontsize=fontsize)
ax.set_yticklabels([int(tt) for tt in ax.get_yticks()], fontsize=fontsize)
plt.tight_layout(pad=0.5)
plt.savefig(f'{plot_path}Sup3f_Quantile_Noise_Dists.pdf')
plt.close()


# sup 3g-i
young16y = 'GSM1007467' 
young17y = 'GSM1007336'
young_blood = blood[[young16y, young17y]]
# compute the amount of noise for one age step, i.e from 16 to 17
stats_dict = {}
stats_dict['Quant'] = []
stats_dict['Pearson r'] = []
stats_dict['Pearson p'] = []
stats_dict['Spearman r'] = []
stats_dict['Spearman p'] = []
stats_dict['R2'] = []
for step in [5,10,15,20]:
    noise_age_df = get_noise_func_parm(young_blood, start_ind=0, end_ind=1, step=step, normalize='None')
    for _ in range(3):
        ground = np.random.choice(young_blood.iloc[:,0].values, replace=False, size=2000) 
        
        samples, ages = random_epi_biol_age(ground = ground, 
                                            noise_ground_df = noise_age_df, 
                                            noise_age_df = noise_age_df, 
                                            samples_per_age = 3, 
                                            epi_sites = len(ground), 
                                            age_end = 100, 
                                            age_start=0, noise_norm=1)
        
        samples_2, ages_2 = random_epi_biol_age(ground = ground, 
                                            noise_ground_df = noise_age_df, 
                                            noise_age_df = noise_age_df, 
                                            samples_per_age = 3, 
                                            epi_sites = len(ground), 
                                            age_end = 100, 
                                            age_start=0, noise_norm=1)
        
        o=f'Sup3i_BioData_Quant{step}_None_cv'
        scatter_size=1
        xlab='Ground state values'
        ylab='Ground state values + 100x empirical noise'
        tick_step=0.25
        sns.set(font='Times New Roman', style='white')
        g=sns.jointplot(x=ground, y=samples[-1], height=height, kind='scatter',s=4, color=dot_color)
        
        g.set_axis_labels(xlab, ylab, fontsize=fontsize)
        loc = plticker.MultipleLocator(base=tick_step)
        g.ax_joint.xaxis.set_major_locator(loc)
        g.ax_joint.yaxis.set_major_locator(loc)
        g.ax_joint.plot([0,1], [0,1], ':k')
        g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
        g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
        plt.tight_layout(pad=0.5)
        plt.savefig(f'{plot_path}{o}_Sample100EmpiricalNoise_fontsize{fontsize}_height{height}{filetype}')
        plt.close()
        
        scatter_size=1
        xlab='Ground state values'
        ylab='Ground state values + 1x empirical noise'
        tick_step=0.25
        sns.set(font='Times New Roman', style='white')
        g=sns.jointplot(x=ground, y=samples[0], height=height, kind='scatter',s=4, color=dot_color)
        
        g.set_axis_labels(xlab, ylab, fontsize=fontsize)
        loc = plticker.MultipleLocator(base=tick_step)
        g.ax_joint.xaxis.set_major_locator(loc)
        g.ax_joint.yaxis.set_major_locator(loc)
        g.ax_joint.plot([0,1], [0,1], ':k')
        g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
        g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
        plt.tight_layout(pad=0.5)
        plt.savefig(f'{plot_path}{o}_Sample1EmpiricalNoise_fontsize{fontsize}_height{height}{filetype}')
        plt.close()
        
     
        xlab = 'Simulated age'
        ylab = 'Predicted age'
        o=f'Sup3h_BioData_Quant{step}_None_cv'
        
        
        regr, stats = pred_and_plot(samples=samples, 
                                           ages=ages, 
                                           samples2=samples_2, 
                                           ages2=ages_2, 
                                           outname=f'{plot_path}{o}',
                                           xlab=xlab,
                                           ylab=ylab, 
                                           fontsize=fontsize, 
                                           height=height, 
                                           tick_step=25, color=dot_color, line_color=line_color, n_jobs=n_jobs)
        
        stats_dict['Quant'].append(step)
        stats_dict['Pearson r'].append(stats[0][0])
        stats_dict['Pearson p'].append(stats[0][1])
        stats_dict['Spearman r'].append(stats[1][0])
        stats_dict['Spearman p'].append(stats[1][1])
        stats_dict['R2'].append(stats[2])



stats_df = pd.DataFrame(stats_dict)

fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Quant', y='R2', data=stats_df, color='black', s=2)
ax=sns.boxplot(x='Quant', y='R2', data=stats_df, color='white')
ax.set_xlabel('Number of quantiles',fontsize=fontsize)
ax.set_ylabel('R2', fontsize=fontsize)
ax.set_xticklabels([5,10,15,20], fontsize=fontsize)
ax.set_ylim(0,1)
ax.set_yticklabels([0.0,0.25,0.5,0.75,1.0], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout(pad=1)
plt.savefig(f'{plot_path}Sup3g_Quant_vs_R2.pdf')
plt.close()


'''
Supplement Figure 5
'''
#sup5a
Em = 0.999
cell_num = 1000
samples, ages = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num)

o=f'Sup5a_SingleCell_Fixed_Em{Em}'
xlab='Ground state values'
ylab = 'Ground state values +\n100x single-cell noise (fixed)'
tick_step=0.25
sns.set(font='Times New Roman', style='white')
g=sns.jointplot(x=ground, y=samples[-1], kind='scatter', height=height, s=4, color=dot_color)
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc = plticker.MultipleLocator(base=tick_step)
g.ax_joint.xaxis.set_major_locator(loc)
g.ax_joint.yaxis.set_major_locator(loc)
g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
plt.savefig(f'{plot_path}{o}_Em{Em}_100xNoise_fontsize{fontsize}_height{height}{filetype}')
plt.close()




#Sup5B
xlab = 'Simulated age'
ylab = 'Predicted age'
ground_size = 2000
cell_num = 100
Em=0.99
o=f'Sup3b_All_0.5_groundsize{ground_size}'
ground = [0.5]*ground_size
Em = 0.999
samples, ages = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num, deviate_ground=False)
samples_2, ages_2 = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num, deviate_ground=False)

regr, stats = pred_and_plot(samples=samples,
                                   ages=ages,
                                   samples2=samples_2,
                                   ages2=ages_2,
                                   outname=f'{plot_path}{o}',
                                   xlab=xlab,
                                   ylab=ylab,
                                   fontsize=fontsize,
                                   height=height,
                                   tick_step=25, savepic=True)

#Sup5C
### Just a slight deviation to 0.51 allows for a prediction
o=f'Sup5c_All_0.51_groundsize{ground_size}'
ground = [0.51]*ground_size
samples, ages = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num, deviate_ground=False)
samples_2, ages_2 = simulate_cells_for_age_fixed(ground, Em, samples_per_age=3, age_steps=100, cell_num=cell_num, deviate_ground=False)

regr, stats = pred_and_plot(samples=samples,
                                   ages=ages,
                                   samples2=samples_2,
                                   ages2=ages_2,
                                   outname=f'{plot_path}{o}',
                                   xlab=xlab,
                                   ylab=ylab,
                                   fontsize=fontsize,
                                   height=height,
                                   tick_step=25, savepic=True)

#Sup5D

young16y = 'GSM1007467' 
meta = pd.read_csv(f'{input_path}GSE41037_meta.csv', index_col=0, sep='\t')
meta = meta.T
meta = meta[meta['!Sample_source_name_ch1'].str.contains('control')]
meta.columns = ['Name', 'Gender', 'Age', 'Disease']
meta.Age = meta.Age.astype(int)
meta = meta[['Gender', 'Age']]
dat = pd.read_csv(f'{input_path}GSE41037.csv', sep='\t', index_col=0)
dat = dat.iloc[:-1]
dat = dat.dropna()
dat = dat[meta.index]
dat = dat.T
dat = dat.join(meta)
dat = dat.sort_values(by='Age') 
blood = dat.iloc[:,:-2].T
ground_size = 2000
ground_sites = np.random.choice(blood[young16y].index, replace=False, size=ground_size)
ground_sites = np.sort(ground_sites)
ground = blood[blood.index.isin(ground_sites)].loc[:,young16y].values

ground = [0.5]*ground_size
o=f'Sup5d_All_0.5_empiricalEm_groundsize{ground_size}'
Em_df_new = get_noise_Em_all_new(blood, old88y, Em_lim=Em_lim, Ed_lim=Ed_lim)
Em_df_new = Em_df_new[Em_df_new.Site.isin(ground_sites)]
samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new, samples_per_age=3, age_steps=100, cell_num=cell_num, deviate_ground=False)
samples_emp_noquantile_2, ages_emp_noquantile_2 = simulate_for_age_empirical_noquantile(ground, Em_df_new, samples_per_age=3, age_steps=100, cell_num=cell_num, deviate_ground=False)
regr, stats = pred_and_plot(samples=samples_emp_noquantile,
                                   ages=ages_emp_noquantile,
                                   samples2=samples_emp_noquantile_2,
                                   ages2=ages_emp_noquantile_2,
                                   outname=f'{plot_path}{o}',
                                   xlab=xlab,
                                   ylab=ylab,
                                   fontsize=fontsize,
                                   height=height,
                                   tick_step=25, savepic=True)
#Sup5E
o=f'Sup5e_SingleCell_EmpiricalEm_Ed'
old88 = 'GSM1007832'
Em_lim=0.95
Ed_lim=0.23 
Em_df_new = get_noise_Em_all_new(blood, old88, Em_lim=Em_lim, Ed_lim=Ed_lim)
Em_df_new = Em_df_new[Em_df_new.Site.isin(ground_sites)]
samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new, samples_per_age=3, age_steps=100)
scatter_size=1
xlab='Ground state values'
ylab = 'Ground state values +\n100x single-cell noise (empirical)'
tick_step=0.25
sns.set(font='Times New Roman', style='white')
g=sns.jointplot(x=ground, y=samples_emp_noquantile[-1], height=height, kind='scatter',s=4, color=dot_color)
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc = plticker.MultipleLocator(base=tick_step)
g.ax_joint.xaxis.set_major_locator(loc)
g.ax_joint.yaxis.set_major_locator(loc)
g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
plt.savefig(f'{plot_path}{o}_100xNoise_fontsize{fontsize}_height{height}{filetype}')
plt.close()


#Sup5F
young16y = 'GSM1007467' 
old88 = 'GSM1007832'
size=2000
cell_num = 1000
ground_sites = np.random.choice(blood[young16y].index, replace=False, size=size)
ground_sites = np.sort(ground_sites)
ground = blood[blood.index.isin(ground_sites)].loc[:,young16y].values

o=f'Sup5f_SingleCell_EmpiricalEm_Ed'
Em_lim=0.95
Ed_lim=0.23
Em_df_new = get_noise_Em_all_new(blood, old88, Em_lim=Em_lim, Ed_lim=Ed_lim)
Em_df_new = Em_df_new[Em_df_new.Site.isin(ground_sites)]
samples, ages = simulate_for_age_empirical_noquantile(ground, Em_df_new, samples_per_age=3, age_steps=100)


regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], n_jobs=n_jobs)
regr.fit(samples, ages)

xlab='Ground state values'
ylab='Elastic net regression coefficients'
sns.set(font='Times New Roman', style='white')
g = sns.jointplot(x=ground, y=regr.coef_, kind='reg', height=height, scatter_kws={'s':1}, color=dot_color, joint_kws={'line_kws':{'color':line_color}})  
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
loc_x = plticker.MultipleLocator(base=0.25)
loc_y = plticker.MultipleLocator(base=5)
g.ax_joint.xaxis.set_major_locator(loc_x)
g.ax_joint.yaxis.set_major_locator(loc_y)
g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}{o}_RegressionToTheMean_fontsize{fontsize}_height{height}{filetype}')
plt.close()

'''
Supplement Figure 6
'''
# Sup6a see Code for Figure 4B


# Sup Fig 6B
meta = pd.read_csv(f'{input_path}GSE41037_meta.csv', index_col=0, sep='\t')
meta = meta.T
meta = meta[meta['!Sample_source_name_ch1'].str.contains('control')]
meta.columns = ['Name', 'Gender', 'Age', 'Disease']
meta.Age = meta.Age.astype(int)
meta = meta[['Gender', 'Age']]
dat = pd.read_csv(f'{input_path}GSE41037.csv', sep='\t', index_col=0)
dat = dat.iloc[:-1]
cell_num=1000
young16y = 'GSM1007467'
old88y = 'GSM1007832'
age_steps=74
kind='scatter'
# 4A with RANDOM Em and Ed within the limits
d = {}
d['Em_lim'] = []
d['Ed_lim'] = []
d['Pearson r'] = []
d['Pearson p'] = []
d['R2'] = []
horvath = f'{input_path}horvath_clock_coef.csv'
horvath_cpgs, horvath_model = get_clock(clock_csv_file=horvath)
Em_lim = 0.97
Ed_lim=0.05
for _ in range(30):
    o = f'S6b_Horvath_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells_Emlim{Em_lim}_Edlim{Ed_lim}_RANDOM'

    samples_horvath, ages_horvath, data = get_samples_random(dat,
                                                      cpg_sites=horvath_cpgs,
                                                      age_steps=74,
                                                      cell_num=cell_num,
                                                      Em_lim=Em_lim,
                                                      Ed_lim=Ed_lim,
                                                      young16y=young16y,
                                                      old88y=old88y)
    samples_horvath = samples_horvath
    ages_horvath = [i + 15 for i in ages_horvath]
    pear, spear, r2, pred = get_prediction(samples_horvath,
                                           ages_horvath,
                                           data,
                                           clock_model=horvath_model,
                                           outname=o,
                                           scatter_size=scatter_size,
                                           tick_step=25,
                                           fontsize=fontsize,
                                           height=height, kind=kind)

    d['Em_lim'].append(Em_lim)
    d['Ed_lim'].append(Ed_lim)
    d['Pearson r'].append(pear[0])
    d['Pearson p'].append(pear[1])
    d['R2'].append(r2)
df = pd.DataFrame(d)
df.to_csv(f'{plot_path}S6b_horvath_limits_comp_RANDOM.csv')

fig = plt.figure(figsize=(height, height))
ax = sns.boxplot(x='Em_lim', y='Pearson r', data=df, color='white')
ax = sns.swarmplot(x='Em_lim', y='Pearson r', data=df, dodge=True, s=2, color='black')
plt.xlabel('Random Em and Ed', fontsize=fontsize)
plt.ylabel('Pearson correlation (Horvath)', fontsize=fontsize)
plt.ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([0, 0.5, 1], fontsize=fontsize)
plt.tight_layout()

plt.savefig(f'{plot_path}S6b_horvath_limits_comp_RANDOM_Pearson.pdf')
plt.close()


# Sup6c
horvath = f'{input_path}horvath_clock_coef.csv'
horvath_cpgs, horvath_model = get_clock(clock_csv_file=horvath)

d = {}
d['Em'] = []
d['Ed'] = []
d['Eu'] = []
d['R2'] = []
d['Pearson r'] = []
d['Pearson p'] = []
d['Spearman r'] = []
d['Spearman p'] = []
for Em in [0.9, 0.95, 0.99, 0.999, 0.9995]:
    for _ in range(5):
        print(Em)
        Ed = 1 - Em

        o = f'Sup6c_Horvath_FixedEm{Em}_FixedEd{Ed}_{age_steps}NoiseCycles_{cell_num}SimulatedCells'
        samples_horvath, ages_horvath, data = get_samples_fixed(dat,
                                                                cpg_sites=horvath_cpgs,
                                                                Em=Em,
                                                                Ed=Ed,
                                                                age_steps=age_steps,
                                                                cell_num=cell_num,
                                                                young16y=young16y,
                                                                old88y=old88y)
        ages_horvath = [i + 15 for i in ages_horvath]

        pear, spear, r2, pred = get_prediction(samples_horvath,
                                         ages_horvath,
                                         data,
                                         clock_model=horvath_model,
                                         outname=o,
                                         scatter_size=scatter_size,
                                         tick_step=25,
                                         fontsize=fontsize,
                                         height=height, kind=kind)

        d['Em'].append(Em)
        d['Ed'].append(Ed)
        d['Eu'].append(Em)
        d['R2'].append(r2)
        d['Pearson r'].append(pear[0])
        d['Pearson p'].append(pear[1])
        d['Spearman r'].append(spear[0])
        d['Spearman p'].append(spear[1])

df = pd.DataFrame(d)
fig = plt.figure(figsize=(height, height))
ax = sns.swarmplot(x='Em', y='Pearson r', data=df, color='black', s=2)
ax = sns.boxplot(x='Em', y='Pearson r', data=df, color='white')
ax.set_xlabel('Methylation maintenance efficiency (%)', fontsize=fontsize)
ax.set_ylabel('Pearson correlation', fontsize=fontsize)
ax.set_xticklabels([90, 95, 99, 99.9, 99.95], fontsize=fontsize)
plt.ylim((0, 1))
loc = plticker.MultipleLocator(base=0.5)
ax.yaxis.set_major_locator(loc)
ax.set_yticklabels([-0.5, 0.0, 0.5, 1.0], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}Sup6c_Horvath_Em_vs_R2.pdf')
plt.close()

# Sup6d see Code for Figure 4E


# Sup Fig 6E
# PHENOAGE RANDOM
d = {}
d['Em_lim'] = []
d['Ed_lim'] = []
d['Pearson r'] = []
d['Pearson p'] = []
d['R2'] = []
phenoage = f'{input_path}phenoage_clock_coef.csv'
phenoage_cpgs, phenoage_model = get_clock(clock_csv_file=phenoage)
Em_lim = 0.97
Ed_lim = 0.05
for _ in range(30):
    print(Em_lim, Ed_lim)
    o = f'S6e_Phenoage_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells_Emlim{Em_lim}_Edlim{Ed_lim}_RANDOM'

    samples_phenoage, ages_phenoage, data = get_samples_random(dat,
                                                               cpg_sites=phenoage_cpgs,
                                                               age_steps=74,
                                                               cell_num=cell_num,
                                                               Em_lim=Em_lim,
                                                               Ed_lim=Ed_lim,
                                                               young16y=young16y,
                                                               old88y=old88y)

    ages_phenoage = [i + 15 for i in ages_phenoage]
    pear, spear, r2, pred = get_prediction(samples_phenoage,
                                           ages_phenoage,
                                           data,
                                           clock_model=phenoage_model,
                                           outname=o,
                                           scatter_size=scatter_size,
                                           tick_step=25,
                                           fontsize=fontsize,
                                           height=height, kind=kind)

    d['Em_lim'].append(Em_lim)
    d['Ed_lim'].append(Ed_lim)
    d['Pearson r'].append(pear[0])
    d['Pearson p'].append(pear[1])
    d['R2'].append(r2)
df = pd.DataFrame(d)
df.to_csv(f'{plot_path}S6e_phenoage_limits_comp_RANDOM.csv')


fig = plt.figure(figsize=(height, height))
ax = sns.boxplot(x='Em_lim', y='Pearson r', data=df, color='white')
ax = sns.swarmplot(x='Em_lim', y='Pearson r', data=df, dodge=True, s=2, color='black')
plt.xlabel('Random Em and Ed', fontsize=fontsize)
plt.ylabel('Pearson correlation (PhenoAge)', fontsize=fontsize)
plt.ylim(-1, 1)
ax.set_xticks([])
plt.ylim(-1, 1)
ax.set_yticks([-1,-0.5,0, 0.5, 1])
ax.set_yticklabels([-1,-0.5,0, 0.5, 1], fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}S6e_phenoage_limits_comp_RANDOM_Pearson.pdf')
plt.close()



# Sup6f

phenoage = f'{input_path}phenoage_clock_coef.csv'
phenoage_cpgs, phenoage_model = get_clock(clock_csv_file = phenoage)
age_steps=74
d = {}
d['Em'] = []
d['Ed'] = []
d['Eu'] = []
d['R2'] = []
d['Pearson r'] = []
d['Pearson p'] = []
d['Spearman r'] = []
d['Spearman p'] = []
for Em in [0.9, 0.95, 0.99, 0.999, 0.9995]:
    for _ in range(5):
        Ed = 1 - Em
        o = f'Sup6f_PhenoAge_FixedEm{Em}_FixedEd{Ed}_{age_steps}NoiseCycles_{cell_num}SimulatedCells'
        samples, ages, data = get_samples_fixed(dat, 
                                                            cpg_sites=phenoage_cpgs, 
                                                            Em=Em, 
                                                            Ed=Ed, 
                                                            age_steps=age_steps, 
                                                            cell_num=cell_num, 
                                                            young16y = young16y, 
                                                            old88y = old88y)
        ages = [i + 15 for i in ages]
    
        pear, spear, r2, pred = get_prediction(samples, 
                   ages, 
                   data, 
                   clock_model=phenoage_model, 
                   outname=o, 
                   scatter_size=scatter_size, 
                   tick_step=25, 
                   fontsize=fontsize, 
                   height=height, kind=kind)
    
    
    
        d['Em'].append(Em)
        d['Ed'].append(Ed)
        d['Eu'].append(Em)
        d['R2'].append(r2)
        d['Pearson r'].append(pear[0])
        d['Pearson p'].append(pear[1])
        d['Spearman r'].append(spear[0])
        d['Spearman p'].append(spear[1])

df =pd.DataFrame(d)
fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Em', y='Pearson r', data=df, color='black', s=2)
ax=sns.boxplot(x='Em', y='Pearson r', data=df, color='white')
ax.set_xlabel('Methylation maintenance efficiency (%)',fontsize=fontsize)
ax.set_ylabel('Pearson correlation', fontsize=fontsize)
ax.set_xticklabels([90, 95, 99, 99.9, 99.95], fontsize=fontsize)
plt.ylim((0,1))
loc = plticker.MultipleLocator(base=0.5)
ax.yaxis.set_major_locator(loc)
ax.set_yticklabels([-0.5,0.0,0.5,1.0], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}Sup6f_PhenoAge_Em_vs_R2.pdf')
plt.close()

# Sup6g
Em_lim=0.97
Ed_lim= 0.05
age_steps=100
o = f'Sup6g_Horvath_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells_Emlim{Em_lim}_Edlim{Ed_lim}_testage16_limax'
horvath = f'{input_path}horvath_clock_coef.csv'
horvath_cpgs, horvath_model = get_clock(clock_csv_file = horvath)
xlab='Simulated age\n(estimated maintenance efficiencies)'
ylab='Predicted age (Horvath)'
samples_horvath, ages_horvath, data =  get_samples(dat, 
            cpg_sites=horvath_cpgs, 
            age_steps=100, 
            cell_num=cell_num,
            Em_lim=Em_lim, 
            Ed_lim=Ed_lim, 
            young16y = young16y, 
            old88y = old88y)

pear, spear, r2, pred=get_prediction(samples_horvath, 
               ages_horvath, 
               data, 
               clock_model=horvath_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, lim_ax=True, xlab='Simulated age\n(estimated maintenance efficiencies)', ylab='Predicted age (Horvath)')
sns.set(font='Times New Roman', style='white')

g = sns.jointplot(x=ages_horvath, y=pred[1:], kind=kind, height=height, s=4, color='grey')
g.ax_joint.set_ylim([0, 80])
lims = [0, 80]
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
g.ax_joint.set_yticks([0,20,40,60,80])
g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}{o}_Prediction.pdf')
plt.close()


# Sup6h
Em_lim=0.97
Ed_lim= 0.05
o = f'Sup6h_Horvath_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells_Emlim{Em_lim}_Edlim{Ed_lim}_testage37_limax'
horvath = f'{input_path}horvath_clock_coef.csv'
horvath_cpgs, horvath_model = get_clock(clock_csv_file = horvath)

samples_horvath, ages_horvath, data =  get_samples(dat, 
            cpg_sites=horvath_cpgs, 
            age_steps=100, 
            cell_num=cell_num,
            Em_lim=Em_lim, 
            Ed_lim=Ed_lim, 
            young16y = 'GSM1007384', 
            old88y = old88y)
pear, spear, r2, pred=get_prediction(samples_horvath, 
               ages_horvath, 
               data, 
               clock_model=horvath_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, lim_ax=True, xlab='Simulated age\n(estimated maintenance efficiencies)', ylab='Predicted age (Horvath)')
sns.set(font='Times New Roman', style='white')
g = sns.jointplot(x=ages_horvath, y=pred[1:], kind=kind, height=height, s=4, color='grey')
g.ax_joint.set_ylim([0, 80])
lims = [0, 80] 
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
g.ax_joint.set_yticks([0,20,40,60,80])
g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}{o}_Prediction.pdf')
plt.close()


# Sup6i
Em_lim=0.97
Ed_lim= 0.05
o = f'Sup6i_Horvath_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells_Emlim{Em_lim}_Edlim{Ed_lim}_testage81_limax'
horvath = f'{input_path}horvath_clock_coef.csv'
horvath_cpgs, horvath_model = get_clock(clock_csv_file = horvath)

samples_horvath, ages_horvath, data =  get_samples(dat, 
            cpg_sites=horvath_cpgs, 
            age_steps=100, 
            cell_num=cell_num,
            Em_lim=Em_lim, 
            Ed_lim=Ed_lim, 
            young16y = 'GSM1007791', 
            old88y = old88y)

pear, spear, r2, pred=get_prediction(samples_horvath, 
               ages_horvath, 
               data, 
               clock_model=horvath_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, lim_ax=True, xlab='Simulated age\n(estimated maintenance efficiencies)', ylab='Predicted age (Horvath)')
sns.set(font='Times New Roman', style='white')

g = sns.jointplot(x=ages_horvath, y=pred[1:], kind=kind, height=height, s=4, color='grey')
g.ax_joint.set_ylim([0, 80])
lims = [0, 80]  
g.set_axis_labels(xlab, ylab, fontsize=fontsize)
g.ax_joint.set_yticks([0,20,40,60,80])
g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}{o}_Prediction.pdf')
plt.close()




'''
Supplement Figure 7
'''

meta = pd.read_csv(f'{input_path}GSE41037_meta.csv', index_col=0, sep='\t')
meta = meta.T
meta = meta[meta['!Sample_source_name_ch1'].str.contains('control')]
meta.columns = ['Name', 'Gender', 'Age', 'Disease']
meta.Age = meta.Age.astype(int)
meta = meta[['Gender', 'Age']]
dat = pd.read_csv(f'{input_path}GSE41037.csv', sep='\t', index_col=0)
dat = dat.iloc[:-1]

age_steps=74
cell_num=1000
young16y = 'GSM1007467'
old88y = 'GSM1007832'

Em=0.99
Ed = 1-Em

Em_lim=0.97
Ed_lim=0.05

# sup 7a

o = f'Sup7a_Vidal_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells'
vidal = f'{input_path}VidalBralo_weights.csv'
vidal_cpgs, vidal_model = get_clock(clock_csv_file = vidal)


samples, ages, data =  get_samples(dat, 
            cpg_sites=vidal_cpgs, 
            age_steps=age_steps, 
            cell_num=cell_num,
            Em_lim=Em_lim, 
            Ed_lim=Ed_lim, 
            young16y = young16y, 
            old88y = old88y)
ages = [i + 15 for i in ages]
get_prediction(samples, 
               ages, 
               data, 
               clock_model=vidal_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, tight=False, xlab='Simulated age\n(estimated maintenance efficiencies)', ylab='Predicted age (Vidal-Bralo)')

# sup 7b
o = f'Sup7b_Vidal_FixedEm{Em}_FixedEd{Ed}_{age_steps}NoiseCycles_{cell_num}SimulatedCells'
samples, ages, data = get_samples_fixed(dat, 
                                                        cpg_sites=vidal_cpgs, 
                                                        Em=Em, 
                                                        Ed=Ed, 
                                                        age_steps=age_steps, 
                                                        cell_num=cell_num, 
                                                        young16y = young16y, 
                                                        old88y = old88y)
ages = [i + 15 for i in ages]
get_prediction(samples, 
               ages, 
               data, 
               clock_model=vidal_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, tight=False, xlab='Simulated age\n(universal maintenance efficiency)', ylab='Predicted age (Vidal-Bralo)')

# sup 7c

o = f'Sup7c_Lin_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells'
lin = f'{input_path}Lin_Weights.csv'
lin_cpgs, lin_model = get_clock(clock_csv_file = lin)
samples, ages, data =  get_samples(dat, 
            cpg_sites=lin_cpgs, 
            age_steps=age_steps, 
            cell_num=cell_num,
            Em_lim=Em_lim, 
            Ed_lim=Ed_lim, 
            young16y = young16y, 
            old88y = old88y)
ages = [i + 15 for i in ages]
get_prediction(samples, 
               ages, 
               data, 
               clock_model=lin_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, tight=False, xlab='Simulated age\n(estimated maintenance efficiencies)', ylab='Predicted age (Lin)')

# sup 7d
o = f'Sup7d_Lin_FixedEm{Em}_FixedEd{Ed}_{age_steps}NoiseCycles_{cell_num}SimulatedCells'
samples, ages, data = get_samples_fixed(dat, 
                                                        cpg_sites=lin_cpgs, 
                                                        Em=Em, 
                                                        Ed=Ed, 
                                                        age_steps=age_steps, 
                                                        cell_num=cell_num, 
                                                        young16y = young16y, 
                                                        old88y = old88y)
ages = [i + 15 for i in ages]
get_prediction(samples, 
               ages, 
               data, 
               clock_model=lin_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, xlab='Simulated age\n(universal maintenance efficiency)', ylab='Predicted age (Lin)')

# sup 7e
o = f'Sup7e_Weidner_BiologicalEm_Ed_{age_steps}NoiseCycles_{cell_num}SimulatedCells'
weidner = f'{input_path}WeidnerAge.csv'
weidner_cpgs, weidner_model = get_clock(clock_csv_file = weidner, sep='\t')


samples, ages, data =  get_samples(dat, 
            cpg_sites=weidner_cpgs, 
            age_steps=age_steps, 
            cell_num=cell_num,
            Em_lim=Em_lim, 
            Ed_lim=Ed_lim, 
            young16y = young16y, 
            old88y = old88y)
ages = [i + 15 for i in ages]
get_prediction(samples, 
               ages, 
               data, 
               clock_model=weidner_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, tight=False, xlab='Simulated age\n(estimated maintenance efficiencies)', ylab='Predicted age (Weidner)')


# sup 7f
o = f'Sup7f_Weidner_FixedEm{Em}_FixedEd{Ed}_{age_steps}NoiseCycles_{cell_num}SimulatedCells'
samples, ages, data = get_samples_fixed(dat, 
                                                        cpg_sites=weidner_cpgs, 
                                                        Em=Em, 
                                                        Ed=Ed, 
                                                        age_steps=age_steps, 
                                                        cell_num=cell_num, 
                                                        young16y = young16y, 
                                                        old88y = old88y)
ages = [i + 15 for i in ages]
get_prediction(samples, 
               ages, 
               data, 
               clock_model=weidner_model, 
               outname=o, 
               scatter_size=scatter_size, 
               tick_step=25, 
               fontsize=fontsize, 
               height=height, kind=kind, tight=False, xlab='Simulated age\n(universal maintenance efficiency)', ylab='Predicted age (Weidner)')



'''
Supplement Figure 8
'''
cell_num=1000
young16y = 'GSM1007467'
old88y = 'GSM1007832'

Em=0.99
Ed = 1-Em
age_steps=74
kind='scatter'
meta = pd.read_csv(f'{input_path}GSE41037_meta.csv', index_col=0, sep='\t')
meta = meta.T
meta = meta[meta['!Sample_source_name_ch1'].str.contains('control')]
meta.columns = ['Name', 'Gender', 'Age', 'Disease']
meta.Age = meta.Age.astype(int)
meta = meta[['Gender', 'Age']]
dat = pd.read_csv(f'{input_path}GSE41037.csv', sep='\t', index_col=0)
dat = dat.iloc[:-1]
dat = dat.dropna() 
dat = dat[meta.index]
dat = dat.T
dat = dat.join(meta)
dat = dat.sort_values(by='Age') 
blood = dat.iloc[:,:-2].T
horvath = f'{input_path}horvath_clock_coef.csv'
horvath_cpgs, horvath_model = get_clock(clock_csv_file = horvath)
size=len(horvath_cpgs)
ground_sites = np.sort(horvath_cpgs)
ground = blood[blood.index.isin(ground_sites)].loc[:, young16y].values
old88 = 'GSM1007832'
Em_lim = 0.97
Ed_lim = 0.05  
cell_num = 1000
Em_df_new = get_noise_Em_all_new(blood, old88, Em_lim=Em_lim, Ed_lim=Ed_lim)
Em_df_new = Em_df_new[Em_df_new.Site.isin(ground_sites)]
samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new,
                                                                                    samples_per_age=1, age_steps=74,
                                                                                    cell_num=cell_num)


tmp_ages = ages_emp_noquantile
rescaled_ages = ((tmp_ages - np.min(tmp_ages))/(np.max(tmp_ages) - np.min(tmp_ages))*400)-120

regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    alphas=[1])  
regr.fit(samples_emp_noquantile, rescaled_ages)
pred = regr.predict(blood[blood.index.isin(ground_sites)].T)
pred = pd.DataFrame(pred)
pred.columns = ['Predicted']
pred['Sample'] = blood.columns
pred = pred.set_index('Sample')
pred = pred.join(meta)
pred['Disease'] = 'Healthy'
pred = pred[~pred.index.isin([young16y, old88])]  # exclude the starting point and end point

pear = pearsonr(pred.Predicted, pred.Age)
spear = spearmanr(pred.Predicted, pred.Age)

g = sns.jointplot(x=pred.Age, y=pred.Predicted, kind='reg', height=height, scatter_kws={'s': scatter_size},
                       color=dot_color, joint_kws={'line_kws': {'color': line_color}})
g.set_axis_labels('Chronological age', 'Predicted age (Stochastic data-based)', fontsize=fontsize)
g.ax_joint.set_xticks([20,40,60,80])
g.ax_joint.set_yticks([0,20,40,60,80,100])
g.ax_joint.set_xticklabels([20,40,60,80], fontsize=fontsize)
g.ax_joint.set_yticklabels([0,20,40,60,80,100], fontsize=fontsize)
plt.tight_layout()
plt.savefig(f'{plot_path}S8A_RandomClock_Horvath_new.pdf')
plt.close()


# Test cell-type heterogeneity via regression
# GSE41037_estCellTypes was calculated with epiDISH as shown in the code for Figure 5
est = pd.read_csv(f'{input_path}GSE41037_estCellTypes.csv', index_col=0)
pred = pred.join(est)
import statsmodels.formula.api as smf
model = smf.ols(formula='Age ~ Predicted + B + NK + CD4T + CD8T + Mono + Neutro + Eosino', data=pred).fit()
model.summary2()

# Horvath's clock cell-type heterogeneity test
meta = pd.read_csv(f'{input_path}GSE41037_meta.csv', index_col=0, sep='\t')
meta = meta.T
meta = meta[meta['!Sample_source_name_ch1'].str.contains('control')]
meta.columns = ['Name', 'Gender', 'Age', 'Disease']
meta.Age = meta.Age.astype(int)
meta = meta[['Gender', 'Age']]
dat = pd.read_csv(f'{input_path}GSE41037.csv', sep='\t', index_col=0)
dat = dat.iloc[:-1]
dat = dat.fillna(0) #fillna
dat = dat[meta.index]
dat = dat.T
dat = dat.join(meta)
dat = dat.sort_values(by='Age') 
horvath = f'{input_path}horvath_clock_coef.csv'
horvath_cpgs, horvath_model = get_clock(clock_csv_file = horvath)
pred = anti_transform_age(horvath_model.predict(dat[sorted(list(set(dat.columns)&set(horvath_cpgs)))]))
pred = pd.DataFrame(pred)
pred.columns = ['Predicted']
pred['Sample'] = blood.columns
pred = pred.set_index('Sample')
pred = pred.join(meta)
pred['Disease'] = 'Healthy'
pred = pred[~pred.index.isin([young16y, old88])]  # exclude the starting point and end point
est = pd.read_csv(f'{input_path}GSE41037_estCellTypes.csv', index_col=0)
pred = pred.join(est)
import statsmodels.formula.api as smf
model = smf.ols(formula='Age ~ Predicted + B + NK + CD4T + CD8T + Mono + Neutro + Eosino', data=pred).fit()
model.summary2()


# sup 8b
##
c = {}
c['Size'] = []
c['Pearson r'] = []
c['Pearson p'] = []
c['Spearman r'] = []
c['Spearman p'] = []
for size in [500,1000,2000,3000,4000,5000]:
    for _ in range(5):
        ground_sites = np.random.choice(blood[young16y].index, replace=False, size=size)
        ground_sites = np.sort(ground_sites)
        ground = blood[blood.index.isin(ground_sites)].loc[:,young16y].values
        old88 = 'GSM1007832'
        Em_lim=0.97
        Ed_lim=0.05  
        cell_num=1000
        Em_df_new = get_noise_Em_all_new(blood, old88, Em_lim=Em_lim, Ed_lim=Ed_lim)
        Em_df_new = Em_df_new[Em_df_new.Site.isin(ground_sites)]
        samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new, samples_per_age=1, age_steps=74,cell_num=cell_num)
        
        tmp_ages = ages_emp_noquantile
        rescaled_ages = ((tmp_ages - np.min(tmp_ages))/(np.max(tmp_ages) - np.min(tmp_ages))*400)-120

        regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],alphas=[1]) # 
        regr.fit(samples_emp_noquantile, rescaled_ages)
        pred = regr.predict(blood[blood.index.isin(ground_sites)].T)
        pred = pd.DataFrame(pred)
        pred.columns = ['Predicted']
        pred['Sample'] = blood.columns
        pred = pred.set_index('Sample')
        pred = pred.join(meta)
        pred=pred[~pred.index.isin([young16y, old88])] #exclude the starting point and end point
        pear = pearsonr(pred.Predicted, pred.Age)
        spear = spearmanr(pred.Predicted, pred.Age)
        c['Size'].append(size)
        c['Pearson r'].append(pear[0])
        c['Pearson p'].append(pear[1])
        c['Spearman r'].append(spear[0])
        c['Spearman p'].append(spear[1]) 
df = pd.DataFrame(c)

fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Size', y='Pearson r', data=df, color='black', s=2)
ax=sns.boxplot(x='Size', y='Pearson r', data=df, color='white')
ax.set_xlabel('Feature size',fontsize=fontsize)
ax.set_ylabel('Pearson correlation', fontsize=fontsize)
ax.set_xticklabels([500,1000,2000,3000,4000,5000], fontsize=fontsize)
plt.ylim((0,1))
loc = plticker.MultipleLocator(base=0.5)
ax.yaxis.set_major_locator(loc)
ax.set_yticklabels([-0.5,0.0,0.5,1.0], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}Sup8b_RandomClock_vs_bio_Feature_size_stats_excludeyoungold.pdf')
plt.close()


# supplement 8c 
c = {}
c['Size'] = []
c['Pearson r'] = []
c['Pearson p'] = []
c['Spearman r'] = []
c['Spearman p'] = []

for size in [500,1000,2000,3000,4000,5000]:
    for _ in range(5):
        ground_sites = np.random.choice(blood[young16y].index, replace=False, size=size)
        ground_sites = np.sort(ground_sites)
        ground = blood[blood.index.isin(ground_sites)].loc[:,young16y].values
        old88 = 'GSM1007832'
        Em_lim=0.97
        Ed_lim=0.05 
        cell_num=1000
        Em_df_new = get_noise_Em_all_new(blood, old88, Em_lim=Em_lim, Ed_lim=Ed_lim)
        Em_df_new = Em_df_new[Em_df_new.Site.isin(ground_sites)]
        samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new, samples_per_age=1, age_steps=74,cell_num=cell_num)

        
        tmp_ages = ages_emp_noquantile
        rescaled_ages = ((tmp_ages - np.min(tmp_ages))/(np.max(tmp_ages) - np.min(tmp_ages))*400)-120
        
        regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],alphas=[1]) # 
        regr.fit(samples_emp_noquantile, ages_emp_noquantile)
        pred = regr.predict(blood[blood.index.isin(ground_sites)].T)
        pred = pd.DataFrame(pred)
        pred.columns = ['Predicted']
        pred['Sample'] = blood.columns
        pred = pred.set_index('Sample')
        pred = pred.join(meta)
        pred=pred[~pred.index.isin([young16y, old88])] #exclude the starting point and end point
        pred.Age = np.random.permutation(pred.Age.values) # permutate the age to check accuracy
        pear = pearsonr(pred.Predicted, pred.Age)
        spear = spearmanr(pred.Predicted, pred.Age)
        c['Size'].append(size)
        c['Pearson r'].append(pear[0])
        c['Pearson p'].append(pear[1])
        c['Spearman r'].append(spear[0])
        c['Spearman p'].append(spear[1])
df = pd.DataFrame(c)

fig = plt.figure(figsize=(height,height))
ax=sns.swarmplot(x='Size', y='Pearson r', data=df, color='black', s=2)
ax=sns.boxplot(x='Size', y='Pearson r', data=df, color='white')
ax.set_xlabel('Feature size',fontsize=fontsize)
ax.set_ylabel('Pearson correlation', fontsize=fontsize)
ax.set_xticklabels([500,1000,2000,3000,4000,5000], fontsize=fontsize)
plt.ylim((-0.5,1))
loc = plticker.MultipleLocator(base=0.5)
ax.yaxis.set_major_locator(loc)
ax.set_yticklabels([-1,-0.5,0.0,0.5,1.0], fontsize=fontsize)
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'{plot_path}Sup8c_RandomClock_vs_bio_Feature_size_stats_excludeyoungold_RandomAge.pdf')
plt.close()


'''
Supplement Figure 9-10 see Code for Figure 5
'''




