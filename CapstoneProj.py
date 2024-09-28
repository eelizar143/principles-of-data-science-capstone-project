# -*- coding: utf-8 -*-
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.decomposition import PCA

# Using unique number for random seeding
n_number = 17971719
random.seed(n_number)

# Read in data
data = pd.read_csv('spotify52kData.csv',delimiter=',')

# Getting features into one matrix:
# 10 features = duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence and tempo.

feats = data.iloc[:, [5, 7, 8, 10, 12, 13, 14, 15, 16, 17]]
num_col = feats.shape[1]


#Plotted each of the features with normal distribution overlay separately 
#since they all need different labels for x axes

dur = feats.iloc[:, 0]

plt.hist(dur, bins=500, density=True, alpha=0.6)
mu, std = np.mean(dur), np.std(dur)
xmin, xmax = np.min(dur), np.max(dur)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Duration")
plt.ylabel("Frequency")
plt.show()

dan = feats.iloc[:, 1]

plt.hist(dan, bins=500, density=True, alpha=0.6)
mu, std = np.mean(dan), np.std(dan)
xmin, xmax = np.min(dan), np.max(dan)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Danceability")
plt.ylabel("Frequency")
plt.show()

en = feats.iloc[:, 2]

plt.hist(en, bins=500, density=True, alpha=0.6)
mu, std = np.mean(en), np.std(en)
xmin, xmax = np.min(en), np.max(en)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.show()

loud = feats.iloc[:, 3]

plt.hist(loud, bins=500, density=True, alpha=0.6)
mu, std = np.mean(loud), np.std(loud)
xmin, xmax = np.min(loud), np.max(loud)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Loudness")
plt.ylabel("Frequency")
plt.show()

sp = feats.iloc[:, 4]

plt.hist(sp, bins=500, density=True, alpha=0.6)
mu, std = np.mean(sp), np.std(sp)
xmin, xmax = np.min(sp), np.max(sp)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Speechiness")
plt.ylabel("Frequency")
plt.show()

ac = feats.iloc[:, 5]

plt.hist(ac, bins=500, density=True, alpha=0.6)
mu, std = np.mean(ac), np.std(ac)
xmin, xmax = np.min(ac), np.max(ac)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Acousticness")
plt.ylabel("Frequency")
plt.show()

ins = feats.iloc[:, 6]

plt.hist(ins, bins=500, density=True, alpha=0.6)
mu, std = np.mean(ins), np.std(ins)
xmin, xmax = np.min(ins), np.max(ins)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Instrumentalness")
plt.ylabel("Frequency")
plt.show()

li = feats.iloc[:, 7]

plt.hist(li, bins=500, density=True, alpha=0.6)
mu, std = np.mean(li), np.std(li)
xmin, xmax = np.min(li), np.max(li)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Liveness")
plt.ylabel("Frequency")
plt.show()

val = feats.iloc[:, 8]

plt.hist(val, bins=500, density=True, alpha=0.6)
mu, std = np.mean(val), np.std(val)
xmin, xmax = np.min(val), np.max(val)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Valence")
plt.ylabel("Frequency")
plt.show()


tem = feats.iloc[:, 9]

plt.hist(tem, bins=500, density=True, alpha=0.6)
mu, std = np.mean(tem), np.std(tem)
xmin, xmax = np.min(tem), np.max(tem)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Overlay")
plt.xlabel("Tempo")
plt.ylabel("Frequency")
plt.show()

# Is there a relationship between song length and popularity? -------------------------------------------------------------

len_pop = data.iloc[:, [5, 4]]
length = len_pop.iloc[:, 0]
popularity = len_pop.iloc[:, 1]

plt.scatter(length, popularity)
plt.title('Relationship between Song Length and Popularity')
plt.xlabel('Song Length')
plt.ylabel('Popularity')
plt.show()

corr_coef = np.corrcoef(length, popularity) 
print("Correlation coefficient:", corr_coef)
print('')


exp_pop = data.iloc[:, [6, 4]]

exp = exp_pop.iloc[:, 0]
pop = exp_pop.iloc[:, 1]

# Are explicitly rated songs more popular than songs that are not explicit? -------------------------------------------------------------

popularity_explicit = []
popularity_non_explicit= []
for i in range (len(exp)):
    if str(exp.iloc[i]) == 'True':
        popularity_explicit.append(pop.iloc[i])
    if str(exp.iloc[i]) == 'False':
        popularity_non_explicit.append(pop.iloc[i])


print("Pop Exp",np.mean(popularity_explicit))
print("Pop non-exp",np.mean(popularity_non_explicit))

t_stat, p = stats.ttest_ind(popularity_explicit, popularity_non_explicit, equal_var=False)
print("T-statistic:", t_stat)
print("p-value:", p)
print('')


# Are songs in major key more popular than songs in minor key? -------------------------------------------------------------

mode_pop = data.iloc[:, [11, 4]]

mode = mode_pop.iloc[:, 0]
pop = mode_pop.iloc[:, 1]

popularity_maj = []
popularity_min = []
for i in range (len(mode)):
    if mode.iloc[i] == 1:
        popularity_maj.append(pop.iloc[i])
    if mode.iloc[i] == 0:
        popularity_min.append(pop.iloc[i])



print("Mean popularity for songs in Maj key",np.mean(popularity_maj))
print("Mean popularity for songs in Min key",np.mean(popularity_min))

t_stat, p = stats.ttest_ind(popularity_maj, popularity_min, equal_var=False)
print("T-statistic:", t_stat)
print("p-value:", p) 
print('')



#QUESTION 5 Does energy largely reflect the "loudness" of a song? Using a scatterplot. --------------------------------------------------------

en_loud = data.iloc[:, [8, 10]]
ins = feats.iloc[:, 6]


pop = data.iloc[:, 4]

energy = en_loud.iloc[:, 0]
loudness = en_loud.iloc[:, 1]


plt.scatter(loudness, energy)
plt.title('Relationship between Song Energy and Loudness')
plt.xlabel('Loudness')
plt.ylabel('Energy')
plt.show()

corr_coef = np.corrcoef(energy, loudness)
  
print("Correlation coefficient:", corr_coef)
print('')


# Which of the 10 individual (single) song features predicts popularity best?


for i in range(feats.shape[1]):
    X = feats.iloc[:, i]
    X = X[~np.isnan(X)]

    X = X.values.reshape(-1, 1)

    y = data.iloc[:, 4] 

    model = LinearRegression()
    model.fit(X, y)
    rSq = model.score(X, y)
    print("Rsquared for feature",i+1,":",rSq)

print('')

# Building a model that uses *all* of the 10 song features, how well can I predict popularity now? ---------------------------------------

'''The n_number for this question and 
all future questions with a train/test split 
is my random seed from the beginning'''

X = feats.values
y = data.iloc[:, 4]
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=n_number)

# Define the number of bootstrap iterations
iterations = 1000

# Initialize an array to store the coefficients for each bootstrap iteration
coefficients = np.zeros((iterations, X_train.shape[1]))

# Perform bootstrapping
for i in range(iterations):
    # Resample the training data with replacement
    X_resampled, y_resampled = resample(X_train, y_train, replace=True, random_state=n_number)
    
    # Fit linear regression model to the resampled data
    model = LinearRegression()
    model.fit(X_resampled, y_resampled)
    
    # Store the coefficients
    coefficients[i] = model.coef_

mean_coefficients = np.mean(coefficients, axis=0)

final_model = LinearRegression()
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#print("Root Mean Squared Error on test set:", rmse)
#print("Score", final_model.score(X_test, y_test))

# When considering the 10 song features above, how many meaningful principal components can I extract? -----------------------------------------

feats_z = stats.zscore(feats)
feats_pca = PCA().fit(feats_z)
feats_eig = feats_pca.explained_variance_
n_components_kaiser = np.sum(feats_eig > 1)

variance_y = np.var(data.iloc[:, 4])
feats_selected_pca = feats_pca.components_[:n_components_kaiser]
variance_projected = np.sum(np.var(feats_selected_pca, axis=0))
proportion_variance_explained = variance_projected / variance_y
 
 
print("Number of principal components using Kaiser criterion:", n_components_kaiser)

print("Proportion of variance explained:", proportion_variance_explained)
print('')


# Can I predict whether a song is in major or minor key from valence? How good is this prediction and is there a better predictor? ------------
X = data.iloc[:,16].values.reshape(-1, 1) 
y = data.iloc[:,11]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=n_number)

model = LogisticRegression() 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Plot data points
plt.scatter(X, y, color='blue')

# Plot sigmoid curve
X_range = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
sigmoid_values = model.predict_proba(X_range)[:, 1]

plt.plot(X_range, sigmoid_values, color='red')

plt.xlabel('Valence')
plt.ylabel('Major or minor')
plt.title('Logistic Regression w/ Sigmoid Curve')
plt.show()
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("AUC for valence and mode:", auc)


z = data.iloc[:, 7].values.reshape(-1, 1)

z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.25, random_state=n_number)

model = LogisticRegression() 
model.fit(z_train, y_train)

y_pred = model.predict(z_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for danceability and mode:", accuracy)


# Plot data points
plt.scatter(z, y, color='blue')

# Plot sigmoid curve
z_range = np.linspace(np.min(z), np.max(z), 100).reshape(-1, 1)
sigmoid_values = model.predict_proba(z_range)[:, 1]

plt.plot(z_range, sigmoid_values, color='red')

plt.xlabel('Danceability')
plt.ylabel('Major or minor')
plt.title('Logistic Regression w/ Sigmoid Curve')
plt.show() 
y_prob = model.predict_proba(z_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("AUC for danceability and mode:", auc)  
print('')


# Which is a better predictor of whether a song is classical music – duration or the principal components you extracted  earlier? --------------

genre = data.iloc[:, 19]
binary_encoded = [1 if label == 'classical' else 0 for label in genre]


X = data.iloc[:,0].values.reshape(-1, 1) 
y = binary_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=n_number)

model = LogisticRegression() 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for duration and genre:", accuracy)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("AUC for duration and genre:", auc)


X = data.iloc[:, 9].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=n_number)


model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

auc = roc_auc_score(y_test, y_prob)
print(auc)
