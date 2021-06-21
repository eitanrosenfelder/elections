# Functions for Loading and analysis of election data
# Modified from a python notebook from Harel Kein
# Call libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats.stats import pearsonr as per
from scipy.stats.stats import spearmanr as sp
import scipy
import statsmodels.api as sm
import seaborn as sns

na = pd.read_csv("ballot a.csv",encoding="iso8859-8")
nb = pd.read_csv("votes per ballot 2019b.csv",encoding="iso8859-8")

# na = [na["שם ישוב"] != "מעטפות חיצוניות"]
# nb = [nb["שם ישוב"] != "מעטפות חיצוניות"]

parties_dict ={'אמת' : "עבודה גשר", 'ג' : "יהדות התורה", 'ודעם'  : "הרשימה המשותפת", 'טב'  : "ימינה", 'כף'  : "עוצמה יהודית",
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", "פה": "כחול לבן", 'מרצ'  : "המחנה הדמוקרטי", 'שס'  : "שס"}

parties_dict_2019a ={'אמת' : "עבודה",  'נר'  : "גשר", 'ג' : "יהדות התורה", 'דעם'  : "רעם בלד", 'ום'  : "חדש תעל", 'טב'  : "איחוד מפלגות הימין",  'נ'  : "ימין חדש",  'ז'  : "זהות",
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", 'מרצ'  : "מרצ", 'פה'  : "כחול לבן",  'כ'  : "כולנו", 'שס'  : "שס"}

big_parties_b = big_parties_2019b = parties_dict.keys() # [n for n in parties_dict.keys()]
big_parties_2019a = parties_dict_2019a.keys() # [n for n in parties_dict.keys()]
big_parties_namesa = [parties_dict_2019a[n][::-1] for n in parties_dict_2019a]
big_parties_namesb = [parties_dict[n][::-1] for n in parties_dict]

def adapt_df(df, parties, include_no_vote=False, ballot_number_field_name=None):
    df['ballot_id'] = df['סמל ישוב'].astype(str) + '__' + df[ballot_number_field_name].astype(str)
    # df_yeshuv = df.index  # new: keep yeshuv
    df = df.set_index('ballot_id')
    eligible_voters = df['בזב']
    total_voters = df['מצביעים']
    df = df[parties]
    # df['ישוב'] = df_yeshuv  # new: keep yeshuv
    # df = df.reindex(sorted(df.columns), axis=1)
    if include_no_vote:
        df['לא הצביע'] = eligible_voters - total_voters
    return df

na = adapt_df(na, parties_dict_2019a.keys(), include_no_vote=False, ballot_number_field_name='מספר קלפי')
nb = adapt_df(nb, parties_dict, include_no_vote=False, ballot_number_field_name='קלפי')

shared_ballots = nb.index.intersection(na.index)
na = na.loc[shared_ballots]
nb = nb.loc[shared_ballots]

na_double = np.dot(na.transpose(), na)
inverse = np.linalg.inv(na_double)
nba = np.dot(nb.transpose(), na)
new = np.dot(nba, inverse)
new_t = np.dot(nba, inverse).transpose()
m = np.linalg.norm(new, ord = 2)

#plotting time
fig, ax = plt.subplots()
im = ax.imshow(new, cmap=plt.get_cmap('viridis'))

# Add parties names using set_xtick, set_xticklabels
ax.set_title("Vote change between elections")

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)  # **cbar_kw)
cbar.ax.set_ylabel('transfering between parties', va="bottom")

ax.set_xticks(np.arange(len(big_parties_namesa)))
ax.set_yticks(np.arange(len(big_parties_namesb)))
ax.set_xticklabels(big_parties_namesa , rotation=90)
ax.set_yticklabels(big_parties_namesb)

#b
new_rounded = pd.DataFrame(new.round(4))
new_norm = new_rounded.div(new_rounded.sum(axis=1), axis=0)

#plotting time
fig, ax = plt.subplots()
im = ax.imshow(new_norm, cmap=plt.get_cmap('viridis'))

# Add parties names using set_xtick, set_xticklabels
ax.set_title("Vote change between elections, normalised")

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)  # **cbar_kw)
cbar.ax.set_ylabel('transfering between parties', va="bottom")

ax.set_xticks(np.arange(len(big_parties_namesa)))
ax.set_yticks(np.arange(len(big_parties_namesb)))
ax.set_xticklabels(big_parties_namesa, rotation=90)
ax.set_yticklabels(big_parties_namesb)

#q2
na2 = pd.read_csv("ballot a.csv",encoding="iso8859-8")
nb2 = pd.read_csv("votes per ballot 2019b.csv",encoding="iso8859-8")

na2 = adapt_df(na2, parties_dict_2019a.keys(), include_no_vote=True, ballot_number_field_name='מספר קלפי')
nb2 = adapt_df(nb2, parties_dict, include_no_vote=True, ballot_number_field_name='קלפי')

shared_ballots = nb2.index.intersection(na.index)
na2 = na2.loc[shared_ballots]
nb2 = nb2.loc[shared_ballots]

na2_double = np.dot(na2.transpose(), na2)
inverse2 = np.linalg.inv(na2_double)
nba2 = np.dot(nb2.transpose(), na2)
new2 = np.dot(nba2, inverse2)
new2_rounded = pd.DataFrame(new2.round(4))
new2_norm = new2_rounded.div(new2_rounded.sum(axis=1), axis=0)
new_t2 = np.dot(nba2, inverse2).transpose()
m2 = np.linalg.norm(new2, ord = 2)

names_with_not_votinga = big_parties_namesa
names_with_not_votinga.append('ועיבצה אל')
names_with_not_votingb = big_parties_namesb
names_with_not_votingb.append('ועיבצה אל')

#plotting time
fig, ax = plt.subplots()
im = ax.imshow(new2_norm, cmap=plt.get_cmap('viridis'))

# Add parties names using set_xtick, set_xticklabels
ax.set_title("Vote change with non voters")

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)  # **cbar_kw)
cbar.ax.set_ylabel('transfering between parties', va="bottom")

ax.set_xticks(np.arange(len(names_with_not_votinga)))
ax.set_yticks(np.arange(len(names_with_not_votingb)))
ax.set_xticklabels(names_with_not_votinga, rotation=90)
ax.set_yticklabels(names_with_not_votingb)

#b
difference = new2_norm - new_norm

#plotting time
fig, ax = plt.subplots()
im = ax.imshow(difference, cmap=plt.get_cmap('plasma'))

# Add parties names using set_xtick, set_xticklabels
ax.set_title("How do non voters change the map?")

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)  # **cbar_kw)
cbar.ax.set_ylabel('transfering between parties', va="bottom")

ax.set_xticks(np.arange(len(names_with_not_votinga)))
ax.set_yticks(np.arange(len(names_with_not_votingb)))
ax.set_xticklabels(names_with_not_votinga, rotation=90)
ax.set_yticklabels(names_with_not_votingb)

#3
nm = np.dot(na2, new2.transpose())
m_df = pd.DataFrame(new)
m3 = m_df.clip(lower = 0)
new3_norm = m3.div(m3.sum(axis=1), axis=0)


#plotting time
fig, ax = plt.subplots()
im = ax.imshow(new3_norm, cmap=plt.get_cmap('viridis'))

# Add parties names using set_xtick, set_xticklabels
ax.set_title("Vote change - by nnls")

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)  # **cbar_kw)
cbar.ax.set_ylabel('transfering between parties', va="bottom")

big_parties_namesa = [parties_dict_2019a[n][::-1] for n in parties_dict_2019a]
big_parties_namesb = [parties_dict[n][::-1] for n in parties_dict]

ax.set_xticks(np.arange(len(big_parties_namesa)))
ax.set_yticks(np.arange(len(big_parties_namesb)))
ax.set_xticklabels(big_parties_namesa , rotation=90)
ax.set_yticklabels(big_parties_namesb)

#b
gap = m3 - new_norm
fig, ax = plt.subplots()
im = ax.imshow(gap, cmap=plt.get_cmap('plasma'))

# Add parties names using set_xtick, set_xticklabels
ax.set_title("How different are results by nnls")

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)  # **cbar_kw)
cbar.ax.set_ylabel('transfering between parties', va="bottom")

ax.set_xticks(np.arange(len(big_parties_namesa)))
ax.set_yticks(np.arange(len(big_parties_namesb)))
ax.set_xticklabels(big_parties_namesa , rotation=90)
ax.set_yticklabels(big_parties_namesb)

#4
na2 = pd.read_csv("ballot a.csv",encoding="iso8859-8")
nb2 = pd.read_csv("votes per ballot 2019b.csv",encoding="iso8859-8")

na2 = adapt_df(na2, parties_dict_2019a.keys(), include_no_vote=True, ballot_number_field_name='מספר קלפי')
nb2 = adapt_df(nb2, parties_dict, include_no_vote=True, ballot_number_field_name='קלפי')

shared_ballots = nb2.index.intersection(na.index)
na2 = na2.loc[shared_ballots]
nb2 = nb2.loc[shared_ballots]

na2_double = np.dot(na2.transpose(), na2)
inverse2 = np.linalg.inv(na2_double)
nba2 = np.dot(nb2.transpose(), na2)
new2 = np.dot(nba2, inverse2)
new2_rounded = pd.DataFrame(new2.round(4))
new2_norm = new2_rounded.div(new2_rounded.sum(axis=1), axis=0)
new_t2 = np.dot(nba2, inverse2).transpose()
m2 = np.linalg.norm(new2, ord = 2)

first = np.dot(na2, new2_norm.transpose())
res = first - nb2
# mean = pd.DataFrame(res.mean(axis=0))
mean = res.mean(axis=0)
mean_d = pd.DataFrame(mean)


fig, ax = plt.subplots()

x = list(range(12))
dct = {}
for i in mean_d.index:
    dct[i[::-1]] = (mean_d.loc[i][0])**2

a = plt.bar(x=dct.keys(),height=dct.values(),alpha = 0.5)
plt.grid()
plt.xlabel("Parties")
plt.ylabel("The size of residual")
plt.title("The mean of residual matrix")
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width(), 1.05*height,
                '%d' % float(height),
                ha='center', va='bottom')

autolabel(a)
plt.savefig("The mean of squared residual matrix", dpi=None, facecolor='w', edgecolor='w')

plt.show()
