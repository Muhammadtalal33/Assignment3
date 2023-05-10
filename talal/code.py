#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 00:54:52 2023

@author: talal
"""


import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
import scipy.optimize as opt


def logistics(t, n0, g, t0):
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

directory_folder = os.path.dirname(__file__)
milk_data = pd.read_csv(os.path.join(directory_folder, 'milk_data.csv'))


# Correlation map using seabor

plt.figure(figsize = [20, 10], clear = True, facecolor = "white")
sns.heatmap(milk_data.corr(), annot = True, linewidths = 1);

# Seaborn pairplot
sns.pairplot(milk_data)


# Clustring
fig, ax = plt.subplots(1, 2, figsize = (20, 12))
ax = ax.flatten()
sns.kdeplot(ax=ax[0], x=milk_data['pH'], 
            y=milk_data['Colour'], hue='Grade',data=milk_data);
sns.scatterplot(ax = ax[1], x = "pH", y = "Colour",
                hue = "Grade", sizes=(20, 100), 
                legend="full", data = milk_data);


x = milk_data['Temprature']
y = milk_data['Colour']
kmeans = KMeans(n_clusters=3)
milk_data['cluster'] = kmeans.fit_predict(milk_data[['Temprature', 
                                                 'Colour']])
# get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]
## add to df
milk_data['cen_x'] = milk_data.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2],
                                            })
milk_data['cen_y'] = milk_data.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
# define and map colors

colors = ['r', 'g', 'b',]
milk_data['colors'] = milk_data.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})

fig, ax = plt.subplots()

ax.scatter(x, y, 
            c=milk_data.colors, alpha = 0.4, s=7)
plt.xlabel("Temprature")
plt.ylabel("Colour")
plt.scatter(milk_data['cen_x'], milk_data['cen_y'], 10, "purple", marker="d",)
plt.show()













