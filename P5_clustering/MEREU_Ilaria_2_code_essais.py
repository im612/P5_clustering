#!/opt/anaconda3/bin/python3.9
#Inizializzazione
import pandas as pd
import commun

# from pandas import Series, DataFrame
import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing
import datetime, os
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_samples, silhouette_score

import pickle

# Écriture di stdout and stderr dans un fichier extérieur
# sys.stdout = open('out-silh-22-6.log', 'w')
# sys.stderr = sys.stdout

#1. Importation des dataframes
import commun

# Elbow test kmeans
def k_means_elbow(j, label):
    strj = str(j)
    print(strj)
    strj_tex = strj.replace('_',' ')

    x_matrix = commun.create_x_matrix(j)

    # print(x_matrix)

    inertia = []

    K_range = range(3,8)

    for k in K_range:
        print(k)
        model = KMeans(n_clusters=k, init='k-means++').fit(x_matrix)
        inertia.append(model.inertia_)

    print(inertia)
    plt.plot(K_range, inertia)
    plt.grid()
    plt.title(
        'Cout du modele k_means++ (inertia) \n mod. %s\n\n' % j[0])
    plt.xlabel('nombre de clusters')
    plt.ylabel('Cout du modele (inertia)')
    nom = 'k_means_pp/%s/elbow_%s.pdf' % (j[2], label)
    plt.savefig(nom, bbox_inches='tight', pad_inches=1.5)   #pdf
    plt.close
# Fin elbow test kmeans

def k_means_silhouette(j, label):
    print("Begin k_means")
    strj = str(j)
    print('strj', strj)
    strj_tex = strj.replace('_', ' ')

    x_matrix = commun.create_x_matrix(j)

    # N_lines = 2000
    # x_matrix = x_matrix[1:N_lines]

    K_range = range(3, 8)

    silhouette_avg_l = []

    for k in K_range:
            # Création clusters ( + pickle dump )
            clusterer = KMeans(n_clusters=k)
            # Creates a K-means model to search
            # for K different centroids.
            # We need to fit these centroids to inputted data.
            cluster_labels = clusterer.fit_predict(x_matrix)
            # Executes K-means on inputted data using
            # an initialized KMeans object.
            # The returned clusters array contains
            # clusters IDs ranging from 0 to k
            commun.mark_time('end of clustering')

            # Pickle dump
            nom = 'k_means/%s/dumps/km__%s_%s_.pickle' % (j[2], label, k)
            with open(nom, 'wb') as f:
                pickle.dump(cluster_labels, f)
            commun.mark_time('end of pickle dump - cluster_labels')

            # Pickle load
            # nom = 'k_means/%s/dumps/km__%s_%s_.pickle' % (j[2], label, k)
            # with open(nom, 'rb') as f:
            #     cluster_labels = pickle.load(f)
            # commun.mark_time('end of pickle load - cluster_labels')

            # print(cluster_labels)

            # The silhouette_score gives the average value for
            # all the samples.
            # This gives a perspective into the density and separation
            # of the formed clusters
            commun.mark_time('begin of silh_avg calculation')
            silhouette_avg = silhouette_score(x_matrix, cluster_labels)
            # long calcul d'une seule valeur
            commun.mark_time('end of silh_avg')
            silhouette_avg_l.append(silhouette_avg)
            print("silhouette_avg_l", silhouette_avg_l)

            # # Compute the silhouette scores for each sample
            commun.mark_time('begin of silh_values')
            sample_silhouette_values \
                = silhouette_samples(x_matrix, cluster_labels)
            # long calculation
            print(len(sample_silhouette_values), sample_silhouette_values)

            nom = 'k_means/%s/dumps/silh_values__%s_%s_.pickle' \
                  % ( j[2], label, k )
            with open(nom, 'wb') as f:
                pickle.dump(sample_silhouette_values, f)
            commun.mark_time('end of silh_values dump')

            # with open(nom, 'rb') as f:
            #     sample_silhouette_values = pickle.load(f)
            # commun.mark_time('end of pickle load')

            y_lower = 10
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            fig, ax1 = plt.subplots(1, 1)
            # ax1 = plt.subplots(1, 2)
            # ax1 = plt.plot()
            fig.set_size_inches(9, 7)

            # The silhouette coefficient can range from -1
            ax1.set_xlim([-0.2, 1])

            # The (n_clusters+1)*10 is for inserting blank space
            # between silhouette
            # plots of individual clusters, to demarcate them clearly.
            # ax1.set_ylim([0, len(x_matrix) + (k + 1) * 15])

            for ii in range(k):
                # Aggregate the silhouette scores for samples belonging
                # to cluster i, and sort them
                ith_cluster_silhouette_values \
                    = sample_silhouette_values[cluster_labels == ii]
                print("Cluster %s has %i elements"
                      % (ii, len(ith_cluster_silhouette_values)))
                if len(ith_cluster_silhouette_values) < 500:
                    print("!!!")

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # color = cm.nipy_spectral(float(ii) / k)
                # color = cm.plasma(float(ii) / k)
                color = cm.tab20(float(ii) / 8)

                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with
                # their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(ii))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("The silhouette plot "
                              "for the various clusters.")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                # The vertical line for average silhouette score
                # of all the values
                ax1.axvline(x=silhouette_avg, color="red",
                            linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                namefig = "k_means/%s/silh/silhouette_%s_%s.pdf" \
                          % (j[2], label, k)
                plt.savefig( namefig, bbox_inches='tight',
                             pad_inches=1.5)   #pdf
                print(namefig)

            plt.close()

def k_means_silhouette_pp(j, label):
    print("Begin k_means ++")
    strj = str(j)
    print('strj', strj)
    strj_tex = strj.replace('_', ' ')

    x_matrix = commun.create_x_matrix(j)

    # N_lines = 2000
    # x_matrix = x_matrix[1:N_lines]

    K_range = range(3, 8)

    silhouette_avg_l = []

    for k in K_range:
            clusterer = KMeans(n_clusters=k, init='k-means++')
            cluster_labels = clusterer.fit_predict(x_matrix)
            # Executes K-means on inputted data using an initialized
            # KMeans object. The returned clusters array contains
            # clusters IDs ranging from 0 to K
            commun.mark_time('end of clustering')

            # # Pickle dump
            nom = 'k_means_pp/%s/dumps/km__%s_%s_.pickle' \
                  % (j[2], label, k)
            with open(nom, 'wb') as f:
                pickle.dump(cluster_labels, f)
            commun.mark_time('end of pickle dump - cluster_labels')

            # Pickle load
            # nom = 'k_means_pp/%s/dumps/km__%s_%s_.pickle' % (j[2], label, k)
            # with open(nom, 'rb') as f:
            #     cluster_labels = pickle.load(f)
            # commun.mark_time('end of pickle load - cluster_labels')
            #
            # print(cluster_labels)

            # The silhouette_score gives the average value
            # for all the samples.
            # This gives a perspective into the density
            # and separation of the formed clusters
            commun.mark_time('begin of silh_avg calculation')
            silhouette_avg = silhouette_score(x_matrix, cluster_labels)
            # lungo calcolo di un solo valore
            commun.mark_time('end of silh_avg')

            print("For n_clusters =", k,
                  "The average silhouette_score is :", silhouette_avg)

            silhouette_avg_l.append(silhouette_avg)
            print("silhouette_avg_l", silhouette_avg_l)

            # Compute the silhouette scores for each sample
            commun.mark_time('begin of silh_values')
            sample_silhouette_values \
                = silhouette_samples(x_matrix, cluster_labels)
            # long calculation
            print(len(sample_silhouette_values),
                  sample_silhouette_values)

            nom = 'k_means_pp/%s/dumps/silh_values__%s_%s_.pickle' \
                  % ( j[2], label, k )
            with open(nom, 'wb') as f:
                pickle.dump(sample_silhouette_values, f)
            commun.mark_time('end of silh_values dump')

            # with open(nom, 'rb') as f:
            #     sample_silhouette_values = pickle.load(f)
            # commun.mark_time('end of pickle load')

            y_lower = 10
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            fig, ax1 = plt.subplots(1, 1)
            # ax1 = plt.subplots(1, 2)
            # ax1 = plt.plot()
            fig.set_size_inches(9, 7)

            # The silhouette coefficient can range from -1, 1
            ax1.set_xlim([-0.2, 1])

            # The (n_clusters+1)*10 is for inserting blank space
            # between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(x_matrix) + (k + 1) * 15])

            for ii in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values \
                    = sample_silhouette_values[cluster_labels == ii]
                print("Cluster %s has %i elements"
                      % (ii, len(ith_cluster_silhouette_values)))
                if len(ith_cluster_silhouette_values) < 500:
                    print("!!!")

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # color = cm.nipy_spectral(float(ii) / k)
                # color = cm.plasma(float(ii) / k)
                color = cm.tab20(float(ii) / 8)

                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots
                # with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i,
                         str(ii))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("The silhouette plot "
                              "for the various clusters.")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                # The vertical line for average silhouette score
                # of all the values
                ax1.axvline(x=silhouette_avg, color="red",
                            linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                namefig = "k_means_pp/%s/silh/silhouette_%s_%s.pdf" \
                          % (j[2], label, k)
                plt.savefig( namefig, bbox_inches='tight',
                             pad_inches=1.5)   #pdf
                print(namefig)

            plt.close()

###### Autres algorithmes ######
def DBS_epsilon(j, label):
    strj = str(j)
    print(strj)
    strj_tex = strj.replace('_', ' ')

    x_matrix = commun.create_x_matrix(j)

    N_lines = 20000
    x_matrix = x_matrix[1:N_lines]

    K_range = range(3, 8)

    silhouette_avg_l = []

    x_matrix_scaled = StandardScaler().fit_transform(x_matrix)

    eps_array = np.arange(0.01, 2, 0.05)
    print(eps_array)
    min_samples = 100
    for i in eps_array:
        db = DBSCAN(eps=i, min_samples=min_samples).fit(x_matrix_scaled)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Epsilon= %f . Estimated number of clusters: %d . "
              "Estimated number of noise points: %d / %d ( %.2f )"
              % (i, n_clusters_, n_noise_, N_lines, n_noise_/N_lines))

def DBS_epsilon_automatised(j, label):
    strj = str(j)
    print(strj)
    strj_tex = strj.replace('_', ' ')

    x_matrix = commun.create_x_matrix(j)

    N_lines = 20000
    x_matrix = x_matrix[1:N_lines]

    K_range = range(3, 8)

    silhouette_avg_l = []

    x_matrix_scaled = StandardScaler().fit_transform(x_matrix)

    eps_array = np.arange(0.01, 2, 0.05)
    print(eps_array)
    min_samples = 100
    for i in eps_array:
        db = DBSCAN(eps=i, min_samples=min_samples).fit(x_matrix_scaled)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        perc_noise = float(int(n_noise_)/N_lines)

        print("Epsilon= %f . Estimated number of clusters: %d . "
              "Estimated number of noise points: %d / %d ( %.5f )"
              % (i, n_clusters_, n_noise_, N_lines, perc_noise))

        if perc_noise <= 0.01 :
            print("Chosen epsilon: %.2f" % i)
            return

def DBS_funzione(j, label):
    strj = str(j)
    print(strj)
    strj_tex = strj.replace('_', ' ')

    x_matrix = commun.create_x_matrix(j)

    min_samples = 100

    N_lines = x_matrix.shape[0]
    # N_lines = 20000
    # x_matrix = x_matrix[1:N_lines]

    x_matrix_scaled = StandardScaler().fit_transform(x_matrix)

    clusterer = DBSCAN(eps=eps_best[label], min_samples=min_samples) # Creates a K-means model to search for K different centroids. We need to fit these centroids to inputted data.
    # cluster_labels = clusterer.fit_predict(x_matrix_scaled)  # Executes K-means on inputted data using an initialized KMeans object. The returned clusters array contains clusters IDs ranging from 0 to K
    clusterer.fit(x_matrix_scaled)  # Executes K-means on inputted data using an initialized KMeans object. The returned clusters array contains clusters IDs ranging from 0 to K

    cluster_labels = clusterer.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise_ = list(cluster_labels).count(-1)

    print("# Epsilon= %f . Number of clusters is %s and number of noise points is %d / %d ( %.2f )" % (eps_best[label], n_clusters_, n_noise_, N_lines, n_noise_ / N_lines))

    if n_clusters_ > 10 or n_clusters_ < 3:
        print("Number of clusters is %s - min_samples = %s" % (n_clusters_, min_samples))
        return

    if n_noise_ > int(N_lines*0.5):
        print("Number of noise points is %s ()- min_samples = %s" % (n_noise_, min_samples))
        return

    commun.mark_time('end of clustering')

    # # PICKLE https://stackoverflow.com/questions/49441280/means-to-save-a-python-kmodes-clustering-model-to-disk#49892171
    # # Pickle dump
    nom = 'dbscan/%s/dumps/km__%s_%s_.pickle' % (j[2], label, n_clusters_)
    with open(nom, 'wb') as f:
        pickle.dump(cluster_labels, f)
    commun.mark_time('end of pickle dump - cluster_labels')

    # Pickle load
    # nom = 'k_means/%s/dumps/km__%s_%s_.pickle' % (j[2], label, k)
    # with open(nom, 'rb') as f:
    #     cluster_labels = pickle.load(f)
    # commun.mark_time('end of pickle load - cluster_labels')

    # print(cluster_labels)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    commun.mark_time('begin of silh_avg calculation')
    silhouette_avg = silhouette_score(x_matrix, cluster_labels)  # lungo calcolo di un solo valore
    commun.mark_time('end of silh_avg')

    # silhouette_avg_l.append(silhouette_avg)
    # print("silhouette_avg_l", silhouette_avg_l)

    # # Compute the silhouette scores for each sample
    commun.mark_time('begin of silh_values')
    sample_silhouette_values = silhouette_samples(x_matrix, cluster_labels)  # long calculation
    print(len(sample_silhouette_values), sample_silhouette_values)

    nom = 'dbscan/%s/dumps/silh_values__%s_%s_.pickle' % (j[2], label, n_clusters_)
    with open(nom, 'wb') as f:
        pickle.dump(sample_silhouette_values, f)
    commun.mark_time('end of silh_values dump')

    # with open(nom, 'rb') as f:
    #     sample_silhouette_values = pickle.load(f)
    # commun.mark_time('end of pickle load')

    y_lower = 10
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax1 = plt.subplots(1, 1)
    # ax1 = plt.subplots(1, 2)
    # ax1 = plt.plot()
    fig.set_size_inches(9, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.2, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(x_matrix) + (n_clusters_ + 1) * 15])

    for ii in range(n_clusters_):
        #     # Aggregate the silhouette scores for samples belonging to
        #     # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == ii]
        print("Cluster %s has %i elements" % (ii, len(ith_cluster_silhouette_values)))
        if len(ith_cluster_silhouette_values) < 500:
            print("!!!")

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # color = cm.nipy_spectral(float(ii) / k)
        # color = cm.plasma(float(ii) / k)
        color = cm.tab20(float(ii) / 8)

        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(ii))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        namefig = "dbscan/%s/silh/silhouette_%s_%s.pdf" % (j[2], label, n_clusters_)
        plt.savefig(namefig, bbox_inches='tight', pad_inches=1.5)  # pdf
        print(namefig)

    plt.close()

def agglo(j, label):
    strj = str(j)
    print(strj)
    strj_tex = strj.replace('_', ' ')

    x_matrix = commun.create_x_matrix(j)

    # N_lines = x_matrix.shape[0]
    N_lines = 20000
    x_matrix = x_matrix[1:N_lines]

    x_matrix_scaled = StandardScaler().fit_transform(x_matrix)

    K_range = range(3, 8)

    for k in K_range:
        print('clusters:', k)
        clusterer = AgglomerativeClustering(n_clusters=k)  # Creates a K-means model to search for K different centroids. We need to fit these centroids to inputted data.
        clusterer.fit(x_matrix_scaled)  # Executes K-means on inputted data using an initialized KMeans object. The returned clusters array contains clusters IDs ranging from 0 to K

        cluster_labels = clusterer.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print('n_clusters:', n_clusters_)

        commun.mark_time('end of clustering')

        # # PICKLE https://stackoverflow.com/questions/49441280/means-to-save-a-python-kmodes-clustering-model-to-disk#49892171
        # # Pickle dump
        nom = 'agglo/%s/dumps/km__%s_%s_.pickle' % (j[2], label, n_clusters_)
        with open(nom, 'wb') as f:
            pickle.dump(cluster_labels, f)
        commun.mark_time('end of pickle dump - cluster_labels')

        # Pickle load
        # nom = 'k_means/%s/dumps/km__%s_%s_.pickle' % (j[2], label, k)
        # with open(nom, 'rb') as f:
        #     cluster_labels = pickle.load(f)
        # commun.mark_time('end of pickle load - cluster_labels')

        # print(cluster_labels)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        commun.mark_time('begin of silh_avg calculation')
        silhouette_avg = silhouette_score(x_matrix, cluster_labels)  # lungo calcolo di un solo valore
        commun.mark_time('end of silh_avg')

        # silhouette_avg_l.append(silhouette_avg)
        # print("silhouette_avg_l", silhouette_avg_l)

        # # Compute the silhouette scores for each sample
        commun.mark_time('begin of silh_values')
        sample_silhouette_values = silhouette_samples(x_matrix, cluster_labels)  # long calculation
        print(len(sample_silhouette_values), sample_silhouette_values)

        nom = 'agglo/%s/dumps/silh_values__%s_%s_.pickle' % (j[2], label, n_clusters_)
        with open(nom, 'wb') as f:
            pickle.dump(sample_silhouette_values, f)
        commun.mark_time('end of silh_values dump')

        # with open(nom, 'rb') as f:
        #     sample_silhouette_values = pickle.load(f)
        # commun.mark_time('end of pickle load')

        y_lower = 10

        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(9, 7)

        ax1.set_xlim([-0.2, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(x_matrix) + (n_clusters_ + 1) * 15])

        for ii in range(n_clusters_):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == ii]
            print("Cluster %s has %i elements" % (ii, len(ith_cluster_silhouette_values)))
            if len(ith_cluster_silhouette_values) < 500:
                print("!!!")

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.tab20(float(ii) / 8)

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(ii))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            namefig = "agglo/%s/silh/silhouette_%s_%s.pdf" % (j[2], label, n_clusters_)
            plt.savefig(namefig, bbox_inches='tight', pad_inches=1.5)  # pdf
            print(namefig)

        plt.close()

    return

###### Analyse cluster ########

def analyse_cluster_joint_with_rating(j, label):
    for dd in ('k_means', 'k_means_pp'):
        plt.close()
        strj = str(j)
        print(strj)
        strj_tex = strj.replace('_', ' ')

        # x_matrix = create_x_matrix(j)
        x_matrix = commun.create_x_matrix(j)
        print('x_matrix', x_matrix[0:2])

        K_range = range(3, 8)

        for k in K_range:
            # Pickle load
            nom = '%s/%s/dumps/km__%s_%s_.pickle' \
                  % ( dd, folders_name[label], label, k)
            with open(nom, 'rb') as f:
                cluster_labels = pickle.load(f)

            print("Cluster %s %s loaded" % (label, k))
            print(cluster_labels)

            print(np.shape(x_matrix))
            print(np.shape(cluster_labels))
            colonne = np.shape(x_matrix)[1]
            x_matrix_one = np.split(x_matrix, colonne, axis=1)
            print('x_matrix_one', x_matrix_one)

            print('x_matrix before', x_matrix[1:5])

            x_matrix_df = pd.DataFrame(x_matrix)
            x_matrix_df.columns = j[0]
            x_matrix_df['ind'] = x_matrix_df.index
            print(x_matrix_df.columns)
            merge_0 = j[0].copy()
            merge_0.append('ind')
            print(merge_0)

            cluster_labels_ind = pd.DataFrame(cluster_labels.copy())
            cluster_labels_ind.columns = ['cluster']
            cluster_labels_ind['ind'] = cluster_labels_ind.index
            print(cluster_labels_ind.columns)
            merge_1 = ['cluster', 'ind']

            l_df = [x_matrix_df, cluster_labels_ind]
            l_merge = [[[0, merge_0], [1, merge_1], 'ind']]

            df = commun.create_df_from_multiple_dfs(l_df, l_merge)
            df.drop('ind', axis=1, inplace=True)
            # https://stackoverflow.com/questions/13411544/
            # delete-a-column-from-a-pandas-dataframe

            df_means = pd.DataFrame(df.groupby('cluster').mean())
            print('df_means :\n', df_means)
            values_massimi = \
                pd.DataFrame(df_means.max()).values.flatten().tolist()
            print('values_massimi :\n', values_massimi)
            values_minimi = \
                pd.DataFrame(df_means.min()).values.flatten().tolist()
            print('values_minimi :\n', values_massimi)

            df_short = df.groupby('cluster').count()['payment_value']
            values_clients \
                = pd.DataFrame(df_short).values.flatten().tolist()
            print('values_clients :\n', values_clients)
            total_clients = sum(values_clients)
            df_short2 = df.groupby('cluster')['payment_value']
            values_recettes \
                = pd.DataFrame(df_short2.sum()).values.flatten().tolist()
            print('values_recettes :\n', values_recettes)
            total_recettes = sum(values_recettes)
            print('total_recettes', total_recettes)

            print('indexes/number of clusters:',
                  df_means.index, df_means.shape)

            x = df_means.values  # returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df_means_scaled = pd.DataFrame(x_scaled)

            print(df_means_scaled.head())

            #https://www.python-graph-gallery.com/390-basic-radar-chart

            categories = j[0]
            N = len(categories)
            print(categories, N)

            n_features = df_means.shape[0]

            # We are going to plot the first line of the data frame.
            # But we need to repeat the first value to close
            # the circular graph:
            corri_su = list(df.index)
            for p in range(df_means.shape[0]):
                plt.close()
                # color = cm.plasma(float(p) / N)
                color = cm.tab20(float(p) / 8)

                # Valeurs des segments
                values \
                    = df_means_scaled.loc[p].values.flatten().tolist()
                values_non_normalises \
                    = df_means.loc[p].values.flatten().tolist()

                print(values)
                values += values[:1]
                print(values)
                # values

                # What will be the angle of each axis in the plot?
                # (we divide the plot / number of variable)
                angles = [n / float(N) * 2 * pi for n in range(N)]
                angles += angles[:1]

                # Initialise the spider plot
                fig = plt.figure()
                ax = plt.subplot(211, polar=True)

                # Draw one axe per variable + add labels
                plt.xticks(angles[:-1], categories,
                           color='grey', size=8)

                # Draw ylabels
                ax.set_rlabel_position(0)
                plt.yticks([0, 0.33, 0.67], ["0", "0.33", "0.67"],
                           color="grey", size=5)
                plt.ylim(-0.2, 1)

                radar_g_title = 'Cluster %d' % (p)
                ax.set_title(radar_g_title)
                # Plot data
                ax.plot(angles, values, color=color,
                        linewidth=1, linestyle='solid')

                # Fill area
                ax.fill(angles, values, color=color, alpha=0.1)

                # Show the graph
                graph_name \
                    = '%s/%s/charts/' \
                      'radar_clustering_%d_cluster_%d_over_%d.pdf' \
                      % (dd, folders_name[label], label, p, k)
                print(graph_name)

                resume_c = 'Pourcentage des clients: '
                percentage_c \
                    = float(values_clients[p] * 100 / total_clients)
                resume_c = resume_c + '{:.1f}%'.format(percentage_c)
                resume_c = resume_c + ' (%d)\n' % values_clients[p]

                resume_c = resume_c + 'Poids financier du cluster: '
                percentage_f \
                    = float(values_recettes[p] * 100 / total_recettes)
                resume_c = resume_c + '{:.0f}%\n'.format(percentage_f)
                print('values_recettes[p]', values_recettes[p],
                      'total_recettes', total_recettes, '--- %i'
                      % float(values_recettes[p]*100/total_recettes) )
                fig.text(0.3, 0.35, resume_c)

                resume = ''
                rating = ''

                for f in j[0]:
                    ind = j[0].index(f)
                    print(f, ind)

                    val = float(values_non_normalises[ind])
                    val_plot = float(values[ind])
                    print('val plot type', type(val_plot))
                    cfr_val_max = float(values_massimi[ind])
                    cfr_val_min = float(values_minimi[ind])

                    if f == 'orders':
                        if ((val >= cfr_val_min)
                                and (val < (cfr_val_min
                                            + (cfr_val_max-cfr_val_min)*0.25))):
                            rating = rating + 'D'
                            resume = resume + \
                                     'Bas nombre de commandes ' \
                                     '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                    + (cfr_val_max - cfr_val_min) * 0.25))
                                and (val < (cfr_val_min
                                    + (cfr_val_max - cfr_val_min) * 0.5))):
                            rating = rating + 'C'
                            resume = resume \
                                     + 'Nombre modéré de commandes ' \
                                              '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max - cfr_val_min) * 0.5))
                              and (val < (cfr_val_min
                                       + (cfr_val_max - cfr_val_min) * 0.75))):
                            rating = rating + 'B'
                            resume = resume \
                                     + 'Haut nombre de commandes ' \
                                       '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max - cfr_val_min) * 0.75))
                              and (val <= cfr_val_max)):
                            rating = rating + 'A'
                            resume = resume + 'Très haut nombre de commandes ' \
                                              '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)
                        else:
                            resume = resume \
                                     + 'Something is wrong (%s) ' \
                                       '(%.2f - [%.2f - %.2f])\n' \
                                     % ( f, val, cfr_val_min, cfr_val_max)

                    if f == 'payment_value':
                        if ((val >= cfr_val_min)
                                and (val
                                     < (cfr_val_min
                                        + (cfr_val_max-cfr_val_min)*0.25))):
                            rating = rating + 'D'
                            resume = resume + 'Dépenses réduites ' \
                                              '(%.2f BRL - ' \
                                              '[%.2f - %.2f] BRL)\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max - cfr_val_min) * 0.25))
                              and (val
                                   < (cfr_val_min
                                      + (cfr_val_max - cfr_val_min) * 0.5))):
                            resume = resume + 'Dépenses modérées ' \
                                              '(%.2f BRL - ' \
                                              '[%.2f - %.2f] BRL)\n' \
                                     % (val, cfr_val_min, cfr_val_max)
                            rating = rating + 'C'

                        elif ((val
                               >= (cfr_val_min
                                   + (cfr_val_max - cfr_val_min) * 0.5))
                              and (val
                                   < (cfr_val_min
                                      + (cfr_val_max - cfr_val_min) * 0.75))):
                            rating = rating + 'B'
                            resume = resume + 'Dépenses hautes ' \
                                              '(%.2f BRL - ' \
                                              '[%.2f - %.2f] BRL)\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val
                               >= (cfr_val_min
                                   + (cfr_val_max - cfr_val_min) * 0.75))
                              and (val <= cfr_val_max)):
                            rating = rating + 'A'
                            resume = resume + 'Très hautes dépenses ' \
                                              '(%.2f BRL ' \
                                              '- [%.2f - %.2f] BRL)\n' \
                                     % (val, cfr_val_min, cfr_val_max)
                        else:
                            resume = resume + 'Something is wrong (%s) ' \
                                              '(%.2f ' \
                                              '- [%.2f - %.2f])\n'\
                                     % ( f, val, cfr_val_min, cfr_val_max)

                    if f == 'review_score':
                        if ((val >= cfr_val_min)
                                and (val
                                     < (cfr_val_min
                                        + (cfr_val_max
                                           -cfr_val_min)*0.25))):
                            rating = rating + 'D'
                            resume = resume + 'Clientèle insatisfaite ' \
                                              '(%.2f ' \
                                              '- [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max
                                          - cfr_val_min) * 0.25))
                              and (val < (cfr_val_min
                                          + (cfr_val_max
                                             - cfr_val_min) * 0.5))):
                            rating = rating + 'C'
                            resume = resume + \
                                     'Clientèle assez insatisfaite ' \
                                     '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val
                               >= (cfr_val_min
                                   + (cfr_val_max
                                      - cfr_val_min) * 0.5))
                              and (val
                                   < (cfr_val_min
                                          + (cfr_val_max
                                             - cfr_val_min) * 0.75))):
                            rating = rating + 'B'
                            resume = resume + \
                                     'Clientèle assez satisfaite ' \
                                     '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val
                               >= (cfr_val_min
                                   + (cfr_val_max
                                      - cfr_val_min) * 0.75))
                              and (val <= cfr_val_max)):
                            rating = rating + 'A'
                            resume = resume + 'Clientèle satisfaite ' \
                                              '(%.2f ' \
                                              '- [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)
                        else:
                            resume = resume + \
                                     'Something is wrong (%s) ' \
                                     '(%.2f - [%.2f - %.2f])\n' \
                                     % ( f, val, cfr_val_min, cfr_val_max)
                    if f == 'rapidite_livraison':
                        if ((val >= cfr_val_min)
                                and (val < (cfr_val_min
                                            + (cfr_val_max
                                               -cfr_val_min)*0.25))):
                            rating = rating + 'D'
                            resume = resume + 'Livraison très lente ' \
                                              '(%.2f ' \
                                              '- [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max
                                          - cfr_val_min) * 0.25))
                              and (val < (cfr_val_min
                                          + (cfr_val_max
                                             - cfr_val_min) * 0.5))):
                            rating = rating + 'C'
                            resume = resume + 'Livraison lente ' \
                                              '(%.2f ' \
                                              '- [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max
                                          - cfr_val_min) * 0.5))
                              and (val < (cfr_val_min
                                          + (cfr_val_max
                                             - cfr_val_min) * 0.75))):
                            rating = rating + 'B'
                            resume = resume + 'Livraison assez rapide ' \
                                              '(%.2f ' \
                                              '- [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max
                                          - cfr_val_min) * 0.75))
                              and (val <= cfr_val_max)):
                            rating = rating + 'A'
                            resume = resume + 'Livraison rapide ' \
                                              '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)
                        else:
                            resume = resume + 'Something is wrong (%s)' \
                                              ' (%.2f ' \
                                              '- [%.2f - %.2f])\n' \
                                     % ( f, val, cfr_val_min,
                                         cfr_val_max)

                    if f == 'recency_crescent':
                        if ((val >= cfr_val_min)
                                and (val < (cfr_val_min
                                            + (cfr_val_max
                                               -cfr_val_min)*0.25))):
                            rating = rating + 'D'
                            resume = resume + 'Dernier achat ancien ' \
                                              '(%.2f ' \
                                              '- [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max
                                          - cfr_val_min) * 0.25))
                              and (val < (cfr_val_min
                                          + (cfr_val_max
                                             - cfr_val_min) * 0.5))):
                            rating = rating + 'C'
                            resume = resume + \
                                     'Dernier achat assez ancien ' \
                                     '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max
                                          - cfr_val_min) * 0.5))
                              and (val < (cfr_val_min
                                          + (cfr_val_max
                                             - cfr_val_min) * 0.75))):
                            rating = rating + 'B'
                            resume = resume + \
                                     'Dernier achat assez récent ' \
                                     '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)

                        elif ((val >= (cfr_val_min
                                       + (cfr_val_max
                                          - cfr_val_min) * 0.75))
                              and (val <= cfr_val_max)):
                            rating = rating + 'A'
                            resume = resume + 'Dernier achat récent ' \
                                              '(%.2f - [%.2f - %.2f])\n' \
                                     % (val, cfr_val_min, cfr_val_max)
                        else:
                            resume = resume + 'Something is wrong (%s) ' \
                                              '(%.2f - [%.2f - %.2f])\n' \
                                     % ( f, val, cfr_val_min, cfr_val_max)

                resume = resume + '\nRating: %s ' \
                                  '(ratio f/n: %.1f)\n' \
                         % (rating, percentage_f/percentage_c)

                print('# Cluster %s de %s (model %s - %s): %s, %s \n'
                      % (p, k, label, dd, resume_c, resume))

                fig.text(0.25, 0.04, resume)

                plt.savefig(graph_name, bbox_inches='tight',
                            pad_inches=1.5)   #pdf
                plt.show()

#CRÉATION DE la liste des listes des INSTANCES FEATURES
list_of_instances_olist = commun.list_of_instances_olist

eps_best = ['', 0.81, 0.71, 0.71, 0.86, 0.71, 0.71 ]

# Paramètres des modèles
modeles_lineaires1 = commun.modeles_lineaires1
modeles_lineaires2 = commun.modeles_lineaires2
modeles_lineaires3 = commun.modeles_lineaires3

n = 1

liste_modeles = (modeles_lineaires1, modeles_lineaires2)

print('liste_modeles', len(liste_modeles), liste_modeles)

folders_name = commun.folders_name

###### Appel aux fonctions #####

# os.system("mkdir dbscan")
os.system("mkdir agglo")
os.system("mkdir k_means_pp")

# for i in (modeles_lineaires1, modeles_lineaires2):
for i in liste_modeles:
    print(n, i)
    mi = liste_modeles.index(i)

    list_of_instances_olist = commun.list_of_instances_olist[mi]
    # On assume que la lista de modeles a indice 0 correspond
    # à la liste des instances à indice 0, etc...
    for j in i:
            commun.mark_time("Model %s - %s" % (n, j) )
            print(n, j)

            DBS_epsilon_automatised(j, n)

            # os.system("mkdir k_means/%s k_means/%s/dumps "
            #           "k_means/%s/silh" % (j[2], j[2], j[2]))
            # k_means_silhouette(j, n)
            # os.system("mkdir k_means_pp/%s k_means_pp/%s/dumps "
            #           "k_means_pp/%s/silh" % (j[2], j[2], j[2]))
            # k_means_silhouette_pp(j, n)
            #
            # os.system("mkdir agglo/%s agglo/%s/dumps "
            #           "agglo/%s/silh" % (j[2], j[2], j[2]))
            # agglo(j, n)
            #
            # analyse_cluster_joint_with_rating(j, n)

            n += 1

