#!/usr/bin/python

import pandas as pd
import numpy as np
import os
from functools import reduce
import itertools
import math
import folium
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn import datasets, tree, metrics, model_selection, ensemble, neural_network
from sklearn.externals.six import StringIO
import pydotplus
import seaborn as sns

# extract feature data from CSV files
def extractdata(requested, norm=None, geo="LC Postcode Sector"):

    # set paths
    suffix = "_BULK_OUTPUT.csv"
    #datapath = "/data/old/data/work/research/teaching/P1/source"
    datapath = "https://www2.ph.ed.ac.uk/~awashbro/PDAML/source"

    features = []
    for table, cols in requested.items():

        # load in data
        tablename = datapath + '/' + geo + '/' + table + suffix
        tablename = tablename.replace(' ','%20')
        t = pd.read_csv(tablename)

        # rename ONS column and reindex
        t.rename(columns={ 'Unnamed: 0':'ONSCode'}, inplace=True)
        t.set_index('ONSCode', inplace=True)

        # normalise against provided code
        if norm is not None:
            for c in cols:
                allcode = norm[table][c]
                t[table + c] /= t[table + allcode]

        # extract chosen columns from dataframe
        cols = [table + c for c in cols]
        df = t[cols]

        # append to df list
        features.append(df)

    # merge into single df
    result = reduce(lambda x, y: pd.merge(x, y, on='ONSCode'), features)

    # drop all of scotland result
    result.drop('S92000003', inplace=True)

    return result

# generate classification score based on weighted sum of features
# TODO - extractdata extended
def genclfscore(requested, weights, tiers, norm=None, geo="LC Postcode Sector"):

    # get data
    sample = extractdata(requested, norm, geo)

    # sum all rows
    scount = sample.sum(axis=1)

    # transform to weighted average
    for table, cols in requested.items():
        for c in cols:
            code = table + c
            sample[code] *= weights[code]

    # calculate score
    score = sample.sum(axis=1) / scount

    # get quantiles
    quantiles = score.quantile(tiers)

    # simple ranking mechanism
    clf = []
    for s in score:
        t = 0
        for q in quantiles:
            if s < q:
                clf.append(t + 1)
                break
            t += 1
        if (t == len(quantiles)):
            clf.append(t + 1)

    # append scores and tier values
    sample['Score'] = score
    sample['Class'] = clf

    # return scores and tier values
    return sample[['Score', 'Class']]

# generate classification based on single feature values
def genclfsingle(feature, tiers, norm=None, geo="LC Postcode Sector"):

    # To improve - not very clever way of splitting feature and norm strings into key notation expected by extract data
    requested = { feature.split('SC',)[0] + 'SC' : [ feature.split('SC',)[1] ] }
    norm = { norm.split('SC',)[0] + 'SC' : { feature.split('SC',)[1] : norm.split('SC',)[1] } }

    # get data
    sample = extractdata(requested, norm, geo)

    # get quantiles
    quantiles = sample[feature].quantile(tiers)

    # simple ranking mechanism
    clf = []
    for s in sample[feature]:
        #print(s)
        t = 0
        for q in quantiles:
            #print(q)
            if s < q:
                clf.append(t + 1)
                break
            t += 1
        if (t == len(quantiles)):
            clf.append(t + 1)

    # append scores and tier values
    sample['Class'] = clf

    # return scores and tier values
    return sample[['Class']]

# generate series of 2D feature plots from pair-wise permutations
def featureplot(data, target, classes, flabels, clabels):

    plt_colors = "rybgcm"
    n_classes = len(classes)

    # just a safety feature - can work if more there are more than 6 features
    if (len(flabels) > 6):
        print("Number of features is too high to plot")
        return

    # get pair list of permutations and get unique set
    n_features = data.shape[1]
    x = [sorted(i) for i in itertools.permutations(np.arange(n_features), r=2)]
    x.sort()
    pairs = list(k for k,_ in itertools.groupby(x))

    # set subplot layout
    sub_y = math.ceil(len(pairs)/4.)
    full_y = sub_y * 3.5

    # set figure size
    plt.figure(1, figsize=(15, full_y))

    # enumerate over combinations
    for pairidx, pair in enumerate(pairs):

        # extract data for pair
        datapair = data[:, pair]

        # define new plot
        plt.subplot(sub_y, 4, pairidx + 1)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        # add labels
        plt.xlabel(flabels[pair[0]])
        plt.ylabel(flabels[pair[1]])

        # Plot the points
        for i, color in zip(range(n_classes), plt_colors):

            idx = np.where(target == classes[i])

            plt.scatter(datapair[idx, 0], datapair[idx, 1], c=color, label=clabels[i],
                            cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")

    return

# generate choropleth map
# TODO - provide json file
def genchoropleth(df=None, feature=None, mopts_custom=None, copts_custom=None, geo="LC Postcode Sector"):

    # simple checks
    if df is None:
        print("need input data")
        return
    if feature is None:
        print("need input feature")
        return

    # geo check
    if (geo != "LC Postcode Sector"):
        print("Sorry, only LC Postcode Sector geo is available")
        return
    if (df.shape[0] != 1012):
        print("Incorrect number of entries found - are you using LC Postcode Sector geo?")
        return

    # workaround - need to re-add ONS column for choropleth
    df['ONS'] = df.index.values
    cols = ['ONS', feature]

    # map options
    mopts = { "location" : [57.5, -3.5],
             "zoom_start" : 6.5,
             "width" : "60%",
             "height" : "100%" }
    if mopts_custom:
         mopts.update(mopts_custom)

    # chloropleth options
    copts = { "geo_data" : "../maps/LC.geojson",
             "name" : "choropleth",
             "legend_name" : feature,
             "data" : df,
             "columns" : cols,
             "key_on" : "feature.id",
             "fill_color" : 'BuPu',
             "fill_opacity" : 0.7,
             "line_opacity" : 0.2 }
    if copts_custom:
         copts.update(copts_custom)


    # define map
    m = folium.Map(**mopts)

    # overlay choropleth
    folium.Choropleth(**copts).add_to(m)

    return m


# plot decision tree using sklearn export_graphviz
# See: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
# from command line "run dot -Tpng iris.dot -o tree.png"
# Inline logic adapted from https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176
# note: requires pydotplus package
def plotDT(fit, flabels, clabels, fname=None):

    if (fname is not None):
        tree.export_graphviz(fit, out_file=fname, filled=True, rounded=True,
                             special_characters=True,
                             feature_names=flabels,
                             class_names=clabels)
        graph = 0

    else:
        dot_data = StringIO()
        tree.export_graphviz(fit, out_file=dot_data, filled=True, rounded=True,
                             special_characters=True,
                             feature_names=flabels,
                             class_names=clabels)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    return graph


# pretty print confusion matrix
# orginial: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_cm(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return

# heat map for confusion matrices and parameter scans
# adapted from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def heatmap(d, classes=None, labels=None, title=None,
            palette="Green",
            normalize=False,
            annot=True,
            size=None):

    if normalize:
        d = d.astype('float') / d.sum(axis=1)[:, np.newaxis]

    # figure size
    if (size == 'large'):
        plt.figure(figsize=(20.0, 10.0))

    # round down numbers
    d = np.around(d, decimals=2)

    ax = plt.subplot()

    # define colour map
    my_cmap = sns.light_palette(palette, as_cmap=True)

    # plot heatmap
    sns.heatmap(d, annot=True, ax=ax, cmap=my_cmap, fmt='g')

    # labels, title and ticks
    if (labels is not None):
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    if (title is not None):
        ax.set_title('Confusion Matrix')
    if (classes is not None):
        ax.xaxis.set_ticklabels(classes[0])
        ax.yaxis.set_ticklabels(classes[1])

    return

# Rerun fit and prediction steps for later examples
def runML(clf, d):

    # get training and test data and targets
    train_data, test_data, train_target, test_target = d

    # fit classifier with data
    fit = clf.fit(train_data, train_target)

    # define expected and predicted
    expected = test_target
    predicted = clf.predict(test_data)

    # return results
    return [expected, predicted]


# plot loss curve over iterations
# based on http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
def lossplot(loss, scale='linear'):

    plt.figure(figsize=(10.0, 5.0))

    if (scale == 'log'):
        plt.yscale('log')
    else:
        plt.yscale('linear')

    plt.plot(loss)
    plt.title('Value of loss function across training epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid()

    return

# Note - expect convergence warning at small training sizes
def compare_traintest(data, target, params, split=0, scale='linear'):

    # define 0.01 - 0.1, 0.1 - 0.9, 0.91 - 0.99 sample if split array not defined
    if (split == 0):
        split = np.concatenate((np.linspace(0.01,0.09,9), np.linspace(0.1,0.9,9), np.linspace(0.91,0.99,9)), axis=None)

    print("NN parameters")
    print(params)

    print("Split sample:")
    print(split)

    train_scores = []
    test_scores = []

    for s in split:

        print("Running with test size of: %0.2f" % s)

        # get train/test for this split
        d = model_selection.train_test_split(data, target,
                                             test_size=s, random_state=0)

        # define classifer
        if params is not None:
            clf = neural_network.MLPClassifier(**params)
        else:
            clf = neural_network.MLPClassifier()

        # run classifer
        e, p = runML(clf, d)

        # get training and test scores for fit and prediction
        train_scores.append(clf.score(d[0], d[2]))
        test_scores.append(clf.score(d[1], d[3]))

    # plot results
    plt.figure(figsize=(15.0, 5.0))
    if (scale == 'log'):
        plt.yscale('log')
    else:
        plt.yscale('linear')
    plt.plot(split, train_scores, label='Training accuracy', marker='o')
    plt.plot(split, test_scores, label='Testing accuracy', marker='o')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Test sample proportion')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, 1.0, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim([min(split),max(split)])
    plt.ylim([0,1.01])
    plt.grid()
    plt.legend()

    return

## EXTRA

# Generate dataframe containing all table and category codes
# NOTE - only run if you have all the data locally! Load in from file instead using pd.read_pickle('/path/to/lookuptable.pkl')
def genlookup():

    # TODO
    # - change file paths to wget

    # set dir path
    lookuppath = "/data/old/data/work/research/teaching/P1/source/Lookup"
    # load table descriptions
    xlfile = "/data/old/data/work/research/teaching/P1/source/Cell Reference Data.xls"
    xls = pd.ExcelFile(xlfile)

    # read table description from each sheet
    sheets = xls.sheet_names[1:]
    tdesc = {}
    sep = ' - '
    for s in sheets:
        t = pd.read_excel(xls, s)
        desc = t.loc[0][0].split(sep, 1)[1]
        tdesc[s] = desc

    # get lookup file list
    csvfiles = [x for x in os.listdir(lookuppath) if x.endswith('csv')]

    # table codes
    lookup = []

    for f in csvfiles:

        lfile = lookuppath + '/' + f
        ltable = f.split('_LOOKUP.csv')[0]
        #print(lfile, ltable)

        # read in table
        l = pd.read_csv(lfile, names=['Code', 'Description'])

        # FIX - check if ltable exists in lookup
        if ltable not in tdesc:
            td = tdesc[ltable.split('SC')[0] + 'SC']
        else:
            td = tdesc[ltable]

        # generate dataframe for table
        df = pd.DataFrame(
                {'TableCode': [ltable for x in range(l.shape[0])],
                    'TableDescription': [td for x in range(l.shape[0])],
                    'UniqueCatCode': l['Code'],
                    'CatCode': [x[-4:] for x in l['Code']],
                    'CatDescription': l['Description']}
        )


        # append to df list
        lookup.append(df)

    lookup = pd.concat(lookup)

    # change df to reindex on UniqueCatCode
    lookup.set_index('UniqueCatCode', inplace=True)

    return lookup
