import ntplib
from time import ctime
import streamlit as st
import datetime
valid = datetime.datetime(2024, 1, 30, 12, 00, 00)
ntp_client = ntplib.NTPClient()

try:
    response = ntp_client.request('pool.ntp.org')               # sometimes it may be out of air
    now = datetime.datetime.strptime(ctime(response.tx_time), "%a %b %d %H:%M:%S %Y")
    if now > valid:
        st.write('This module has expired. Please, contact Thiago to renew your plan.')
        st.stop()
except:
    pass

import sqlite3
import pandas as pd
import numpy as np

from sys import getsizeof

pd.set_option("display.max_colwidth", None)
pd.set_option('display.width', -1)
st.set_page_config(layout="wide")

conn = sqlite3.connect('orkideon.sqlite')
df_filtered = pd.read_sql('select * from df_filtered', conn)
df_filtered_columns = list(pd.read_sql('select * from df_filtered_columns', conn).stack().values)
df_filtered_columns_num = list(pd.read_sql('select * from df_filtered_columns_num', conn).stack().values)
df_filtered_columns_int = list(pd.read_sql('select * from df_filtered_columns_int', conn).stack().values)
df_filtered_columns_float = list(pd.read_sql('select * from df_filtered_columns_float', conn).stack().values)
df_filtered_columns_cat = list(pd.read_sql('select * from df_filtered_columns_cat', conn).stack().values)
conn.close()
del conn

# Do not sort!!!! To keep the columns hierarchy, which will drive the tree.
# df_filtered_columns.sort()
# df_filtered_columns_int.sort()
# df_filtered_columns_int.sort()
# df_filtered_columns_float.sort()
# df_filtered_columns_cat.sort()

df_filtered[df_filtered_columns_num] = df_filtered[df_filtered_columns_num].apply(pd.to_numeric, errors='coerce')

try:
    df_filtered = df_filtered.sort_values(by=['Group', 'Period'], ascending=[True, True])
except:
    try:
        df_filtered = df_filtered.sort_values(by='Group', ascending=True)
    except:
        pass

try:
    df_filtered['Period'] = pd.to_datetime(df_filtered['Period'])
except:
    pass

df_filtered = df_filtered.fillna(value=np.nan)

# Charts coding may require min, max, >, etc, and these don't work with nan. So delete them from the original but don't save.
df_filtered.dropna(inplace=True, how='all', axis=1)
df_filtered.dropna(inplace=True, how='any', axis=0)


# Exclusive here -----------------------------------------------------------------------------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.api.types import is_float_dtype, is_integer_dtype
from PIL import Image
import matplotlib.cm as cm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.stats import spearmanr, pearsonr, kendalltau
from itertools import cycle
from pandas.api.types import is_integer_dtype
# --------------------------------------------------------------------------------------------------------------------------


st.title("Analytics")

sns.reset_defaults()
sns.set()
sns.set(rc={
    'axes.facecolor':'#0e1117', 
    'figure.facecolor':'#0e1117',
    "figure.figsize":(6, 2),
})
sns.set_context(rc={
    "font.size":5,
    "axes.titlesize":7,
    "axes.labelsize":5, 
    "xtick.labelsize":5, 
    "ytick.labelsize":5, 
    "legend.fontsize":5, 
    "legend.title_fontsize":5,
    "axes.linewidth": 0.5, 
    "grid.linewidth": 0.3,
    "lines.linewidth": 0.5, 
    "lines.markersize": 2,
})
sns.set_style({                                         # https://stackoverflow.com/questions/60878196/seaborn-rc-parameters-for-set-context-and-set-style
    'font.family': 'sans serif',                        # https://jonathansoma.com/lede/data-studio/matplotlib/list-all-fonts-available-in-matplotlib-plus-samples/
    'axes.grid': True,
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'xtick.top': False,
    'ytick.right': False,
    'text.color': 'white',
    'axes.edgecolor': 'white',
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.right': False,
    'axes.spines.top': False,   

    "grid.color": 'grey', 
    "grid.linestyle": ':', 
})



dashboard_concept, dashboard_correlation, dashboard_intracorrelation, dashboard_autocorrelation, dashboard_significance, dashboard_conditional, dashboard_tree, dashboard_boxplot, dashboard_cluster, dashboard_univariate, dashboard_multicollinearity, dashboard_permutation_importances, dashboard_admin = st.tabs(["Concept", "Correlation", "Intracorrelation", "Autocorrelation", "Significance", "Conditional Probalility", "Decision Tree", "Spread", "Hierarchical Clustering", "Univariate Analysis", "Multicollinearity", "Feature Importances", "Admin"])

with dashboard_admin:
    dashboard_head = st.slider('Dtaframe head:', 1, 1000, 100)

with dashboard_concept:
    try:
        image = Image.open('concept.png') 
        st.image(image, use_column_width=True)
    except:
        pass

with dashboard_permutation_importances:
    st.markdown("Not ported to this version yet.")

with dashboard_conditional:

    # https://jdatascientist.medium.com/data-science-with-the-penguins-data-set-conditional-propability-bd998bfedd35
    # https://jdatascientist.medium.com/data-science-with-the-penguins-data-set-bayesian-networks-b6b694b9a5b
    # https://allendowney.github.io/BiteSizeBayes/10_joint.html
    # https://www.kaggle.com/code/johnoliverjones/naive-bayesian-network-with-7-features/notebook
    # https://stackoverflow.com/questions/72259986/how-to-visualize-a-bayesian-network-model-constructed-with-pomegranate

    dashboard_conditional1, dashboard_conditional2, dashboard_conditional3, dashboard_conditional4 = st.columns(4, gap='large')

    with dashboard_conditional1:
        cross_index = st.multiselect("Conditional rows:", df_filtered_columns, default=df_filtered_columns[0])

    with dashboard_conditional2:
        df_filtered_columns_ = list(set(df_filtered_columns) - set(cross_index))
        cross_columns = st.multiselect("Conditional columns:", df_filtered_columns_, default=df_filtered_columns_[0])

    with dashboard_conditional3:
        bins = st.slider('Conditional bins:', 1, 5, 2, 1)

    with dashboard_conditional4:
        if st.button('Calculate Conditional Probabilities'):

            df0 = df_filtered.copy()

            for col in df_filtered_columns_num:
                df0[col] = pd.cut(x=df0[col], bins=bins, precision=1)


            def JP(df0, cross_index, cross_columns):
                all_cols = cross_index + cross_columns
                n = df0.shape[0]
                joint_counts = pd.pivot_table(df0[all_cols], index=cross_index, columns=cross_columns, aggfunc='size').replace(np.nan,0)
                joint_prob = np.round(joint_counts/n, 3)
                return joint_prob

            def CPD(joint_prob):
                cpd = joint_prob
                col_totals = joint_prob.sum(axis=0)
                
                for col in col_totals.index:
                    cpd[col] = cpd[col] / col_totals.loc[col]
                    
                # cpd.columns = [f'b{i+1} = {x}' for i,x in enumerate(cpd.columns)]
                # cpd.index = [f'a{i+1} = {x}' for i,x in enumerate(cpd.index)]

                # cpd.columns = [f'{x}' for i,x in enumerate(cpd.columns)]

                return cpd.round(3)

            jp = JP(df0, cross_index, cross_columns)
            cpd = CPD(JP(df0, cross_index, cross_columns))
            pp = CPD(JP(df0, cross_index, cross_columns).T)

            # cpd.reset_index(drop=False, inplace=True)
            # jp.reset_index(drop=False, inplace=True)
            # df0 = jp

    try:
        st.write(f"""##### Joint Probabilities""")
        st.dataframe(jp)

        st.write(f"""##### Conditional Probabilities""")
        st.dataframe(cpd)

        st.write(f"""##### Posterior Probabilities""")
        st.dataframe(pp)
    except:
        pass



with dashboard_tree:

    dashboard_tree1, dashboard_tree2, dashboard_tree3, dashboard_tree4 = st.columns(4, gap='large')

    with dashboard_tree1:
        target = st.selectbox('Tree target:', df_filtered_columns_cat)

    with dashboard_tree2:
        max_depth = st.slider('Tree maximum depth:', 1, 5, 3, 1)

    with dashboard_tree3:
        
        def get_fill_color(value, colors):
            max_value_index = np.argmax(value)
            color = colors[max_value_index]
            return color

        if st.button('Draw Decision Tree'):

            try:
                os.remove("decision_tree.png")
            except:
                pass

            df_encoded = pd.get_dummies(df_filtered.drop(target, axis=1))

            try:
                df_encoded.drop(['Period'], axis=1, inplace=True)
            except:
                pass

            X = df_encoded
            y = df_filtered[target]

            clf = DecisionTreeClassifier(max_depth=max_depth)
            clf.fit(X, y)

            n_classes = len(np.unique(y))
            colors = [cm.cividis(i / n_classes) for i in range(n_classes)]

            plt.figure(figsize=(15, 10))
            feature_names = list(X.columns)
            class_names = df_filtered[target].unique()
            
            plt.figtext(0.1, 0.9, "Decision Tree", fontsize=18, ha='left', va='center', color='white')

            artists = plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=13, max_depth=max_depth, proportion=True)

            for artist, impurity, value in zip(artists, clf.tree_.impurity, clf.tree_.value):
                fill_color = get_fill_color(value, colors)
                artist.get_bbox_patch().set_facecolor(fill_color)
                artist.get_bbox_patch().set_edgecolor('black')

            plt.savefig("decision_tree.png", dpi=300)


    try:
        image = Image.open('decision_tree.png')
        st.image(image, use_column_width=True)
    except:
        pass



with dashboard_boxplot:

    dashboard_boxplot1, dashboard_boxplot2, dashboard_boxplot3, dashboard_boxplot4, dashboard_boxplot5, dashboard_boxplot6 = st.columns(6, gap='medium')

    with dashboard_boxplot1:
        x = st.selectbox('Spread focus:', df_filtered_columns_num)

    with dashboard_boxplot2:
        y = st.selectbox('Spread group by:', df_filtered_columns_cat)

    with dashboard_boxplot3:
        hue = st.selectbox('Spread hue:', df_filtered_columns_cat)

    with dashboard_boxplot4:
        if st.checkbox('Spread show mean'):
            showmeans = True
        else:
            showmeans = False

        if st.checkbox('Spread show median'):
            show_median = True
        else:
            show_median = False

    with dashboard_boxplot5:
        if st.checkbox('Spread show jitter & rug'):
            jitter = True
        else:
            jitter = False

        if st.checkbox('Spread dodge'):
            dodge = True
        else:
            dodge = False

        if st.checkbox('Spread fill'):
            fill = True
        else:
            fill = False

    with dashboard_boxplot6:
        if st.checkbox('Spread log scale'):
            log_scale = True
        else:
            log_scale = False

        if st.checkbox('Spread draw grid'):
            add_grid = True
        else:
            add_grid = False

        data = df_filtered[df_filtered_columns]

        if st.button('Draw Spread'):

            try:
                os.remove("boxplot_grid.png")
            except:
                pass

            plt.subplot(1, 2, 1)
            g = sns.boxplot(
                data = data, 
                # x = x,
                # y = y, 
                # hue = hue, 
                linecolor = 'w', 
                fill = True,                        # fill = True demands linewidth
                linewidth = 0.5,
                legend = False,
                log_scale = True,
                palette = 'viridis',
                notch = False,
                saturation = 1,
                gap = 0.4,
                
                # linewidth = 0.9,
                boxprops = dict(alpha = 1),
                # patch_artist = True, 
                showmeans = showmeans,
                meanprops={"marker":"v","markerfacecolor":"none", "markeredgecolor":"white", "markersize": 2}, 
                orient='h',                   
            )

            for line in g.lines:
                if line.get_markerfacecolor() == 'none':
                    line.set_markeredgewidth(0.5)

            # g.tick_params(axis='x', labelrotation=90)
            # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            plt.title('Population')
            # plt.savefig('outlier1.png', dpi=300, bbox_inches='tight')
            # plt.clf()

            plt.subplot(1, 2, 2)

            def SelectedBoxplot(data, x, y, hue, log_scale, dodge, showmeans, show_median, jitter, fill, **kwargs):

                # Label outliers: https://stackoverflow.com/questions/61734304/label-outliers-in-a-boxplot-python
                # Scatter instead of jitter: https://stats.stackexchange.com/questions/299017/outliers-in-boxplots
                # Median labels: https://sharkcoder.com/data-visualization/seaborn-boxplot

                g2 = sns.boxplot(
                    data = data, 
                    x = x,
                    y = y, 
                    hue = hue, 
                    linecolor = 'w', 
                    legend = True,
                    log_scale = log_scale,
                    # palette = 'viridis',
                    notch = False,
                    saturation = 1,
                    gap = 0.4,
                    dodge = dodge,

                    fill = fill,                        # fill = True demands linewidth
                    linewidth = 0.5,

                    boxprops = dict(alpha = 1),
                    # patch_artist = True, 
                    showmeans = showmeans,
                    meanprops={"marker":"v","markerfacecolor":"none", "markeredgecolor":"white", "markersize": 2},  
                )

                for line in g2.lines:
                    if line.get_markerfacecolor() == 'none':
                        line.set_markeredgewidth(0.5)  # Set the edge thickness

                # try:
                #     medians = df_filtered.groupby([y,hue])[x].median().reset_index()
                # except:
                #     medians = df_filtered.groupby([y])[x].median().reset_index()
                # for idx, row in medians.iterrows():
                #     y_val = row[y]
                #     hue_val = row[hue]
                #     median_value = row[x]
                #     color = 'grey'          
                #     if show_median:
                #         g2.axvline(median_value, color=color, linewidth=0.5, linestyle='-', clip_on=True)
                #         g2.text(median_value, -0.62, f'{median_value:.1f}', color=color, ha='center', va='top', rotation=0)
                if show_median:
                    global_median = df_filtered[x].median()
                    g2.axvline(x=global_median, color='white', linestyle='--', linewidth=0.5, label='median')
                    # g2.text(global_median, -0.62, f'{global_median:.1f}', color='white', ha='center', va='top', rotation=0)

                # if show_label:
                # g2.axvline(x = df_filtered[x].median(), color='white', linestyle='--', linewidth = 0.5, label = 'median')
                # g2.text(df_filtered[x].median(), 1, f'{df_filtered[x].median().round(1)}', color='white', ha='center', va='bottom', rotation=0)

                # g2.set_yticklabels([])
                # g2.set_yticks([])
                # g2.spines['left'].set_visible(False)

                if jitter:
                  
                    sns.stripplot(data=data, x=x, y=y, hue=hue, dodge=dodge, legend=False, size=2, jitter=True, ax=plt.gca()) # , size=4, markersize=2
                    
                    sns.rugplot(data=data, x=x, hue=hue, ax=plt.gca())

                    # sns.swarmplot(data=data, x=x, y=y, hue=hue, linewidth=0, ax=plt.gca())
                    # sns.violinplot(data=data, x=x, y=y, hue=hue, dodge=dodge, ax=plt.gca())
                    # sns.pointplot(data=data, x=x, y=y, hue=hue, ax=plt.gca())
                    # sns.barplot(data=data, x=x, y=y, hue=hue, dodge=dodge, estimator=np.mean, ax=plt.gca())
                    # sns.lineplot(data=data, x=x, y=y, ax=plt.gca())
                    # sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=plt.gca())

                    if fill:
                        for collection in g2.collections:
                            # hue_values = collection.get_offsets()[:, -1]
                            collection.set_edgecolor('white')  # Customize the edge color
                            collection.set_linewidth(0.5)  # Set the edge thickness
                            # collection.set_facecolor('none')

                # sns.despine(offset=1, trim=True)
                # g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
                g2.tick_params(axis='x', labelrotation=90)
                
                try:
                    plt.ticklabel_format(style='plain', axis='x')
                except:
                    pass

                g2.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

                return g2
            

            g2 = SelectedBoxplot(data, x, y, hue, log_scale, dodge, showmeans, show_median, jitter, fill)

            plt.title('Focus')

            # plt.tight_layout(pad=3)
            plt.subplots_adjust(left=0.1, right=0.9, wspace=0.8)
            plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
            plt.clf()

            # IMPORTANT: You must create the facet grid before the chart. If the chart has too much code, you better encapsulate it.

            if add_grid:
                grid = sns.FacetGrid(data=data, col=hue, hue=hue, col_wrap=3, despine=True, legend_out=True, sharex=False, sharey=False, height=2, aspect=1)                    # sharex=False make individual scales for each item
                grid.map_dataframe(SelectedBoxplot, data=data, x=x, y=y, hue=y, log_scale=log_scale, dodge=False, showmeans=showmeans, show_median=show_median, jitter=jitter, fill=fill, facet_kws={'sharex': False, 'sharey': False})
                grid.set_titles(col_template='{col_name}', row_template='{hue_name}')
                grid.set_xticklabels(rotation=90)
                grid.fig.suptitle(f"{x} x {y}")
                grid.fig.subplots_adjust(hspace=1)
                # g.add_legend()                                                            # will show empty if fill = False
                plt.tight_layout()
                plt.savefig('boxplot_grid.png', dpi=300, bbox_inches='tight')
                plt.clf()

            # st.rerun()


    try:
        image = Image.open('boxplot.png')
        st.image(image, use_column_width=True)
    except:
        pass

    try:
        image = Image.open('boxplot_grid.png')
        st.image(image, use_column_width=True)
    except:
        pass




with dashboard_cluster:

    # methods: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    # metrics: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
    # These are all formidable tutorials:
    # https://vitalflux.com/hierarchical-clustering-explained-with-python-example/
    # https://discdown.org/dataanalytics/hierarchical-clustering.html                           <--- tutorial nice to follow!
    # https://towardsdatascience.com/all-you-need-to-know-about-seaborn-6678a02f31ff
    # https://www.nxn.se/valent/extract-cluster-elements-by-color-in-python
    # https://blog.finxter.com/how-to-visualize-a-cluster-in-python-dendrogram/
    # https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python
    # https://www.learndatasci.com/glossary/hierarchical-clustering/
    # https://refactored.ai/microcourse/notebook?path=content%2F07-Unsupervised_Models%2F06-Heirarchical_Clustering%2Fagglomerative-clustering.ipynb
    # https://biit.cs.ut.ee/clustvis/
    # https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    # https://www.youtube.com/watch?v=HXjnDIgGDuI&list=PLmNPvQr9Tf-ZSDLwOzxpvY-HrE0yv-8Fy

    dashboard_cluster1, dashboard_cluster2, dashboard_cluster3, dashboard_cluster4, dashboard_cluster5, dashboard_cluster6 = st.columns(6, gap='medium')

    with dashboard_cluster1:
        selected_index = st.selectbox('Cluster index:', df_filtered_columns_cat)
   
    with dashboard_cluster2:
        metric = st.selectbox('Cluster metric:', ['euclidean', 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'])       # cityblock is the manhatan

    with dashboard_cluster3:
        method = st.selectbox('Cluster method:', ['average', 'single', 'complete', 'weighted', 'centroid', 'median', 'ward'])

    with dashboard_cluster4:
        scaling = st.radio('Cluster scaling:', ['Standard Scale', 'Z-Score'])
        
    with dashboard_cluster5:
        if scaling == 'Z-Score':
            
            z_score2 = st.radio('Cluster z-score:', ['Yes', 'No'])
  
            if z_score2 == 'Yes':
                z_score = 1
                center = 0
                vmin = -4 
                vmax = 4
            else:
                z_score = None
                center = None
                vmin = None 
                vmax = None

            standard_scale = None

        else:
            
            standard_scale2 = st.radio('Cluster standard scale:', ['Yes', 'No'])
            
            if standard_scale2 == 'Yes':
                standard_scale = 1
                center = 0.5
                vmin = 0 
                vmax = 1

            else:
                standard_scale = 0
                center = None
                vmin = None 
                vmax = None

            z_score = None

    with dashboard_cluster6:

        # sample_size = st.slider("Clu Sample size:", min_value=1, max_value=len(df_filtered.index), value=len(df_filtered.index)//100, step=len(df_filtered.index)//100)               # https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
        sample_size = st.slider("Cluster sample size:", min_value=1, max_value=len(df_filtered.index), value=len(df_filtered.index)//20, step=max(1, len(df_filtered.index)//20))       # gpt

        if st.button('Plot Clustermap'):

            try:
                os.remove("clustermap.png")
            except:
                pass

            def create_color_mapping(series, palette_name):
                counts = series.value_counts()
                n_categories = len(counts)
                palette = sns.color_palette(palette_name, n_colors=n_categories)
                return {value: palette[i] for i, value in enumerate(counts.index)}

            def create_row_colors(df, index_column, palette_name):
                return df[index_column].map(create_color_mapping(df[index_column], palette_name))

            def create_col_colors(df, col_column, palette_name):
                return df[col_column].map(create_color_mapping(df[col_column], palette_name))

            row_colors = create_row_colors(df_filtered.sample(sample_size, random_state=42), selected_index, 'Spectral')
            col_colors = create_col_colors(df_filtered.sample(sample_size, random_state=42), selected_index, 'Spectral')

            # row_counts = df_filtered[selected_index].value_counts()
            # n_categories = len(row_counts)
            # viridis_palette = sns.color_palette("viridis", n_colors=n_categories)
            # index_colors = {index: viridis_palette[i] for i, index in enumerate(row_counts.index)}
            # row_colors = df_filtered[selected_index].map(index_colors)

            data = df_filtered.set_index([selected_index], drop=True)
            data = data[df_filtered_columns_num] 

            data = data.sample(sample_size, random_state=42)

            # row_linkage = sns.clustermap(data, row_colors=row_colors, method=method).dendrogram_row.linkage

            try:
                c = sns.clustermap(
                    data = data,
                    standard_scale = standard_scale,
                    z_score = z_score,                              # ValueError: Cannot perform both z-scoring and standard-scaling on data
                    center = center,
                    vmin = vmin, 
                    vmax = vmax,
                    metric = metric,
                    method = method,                                # method means linkage method
                    row_cluster = True,
                    col_cluster = True,
                    cmap = 'cividis',                               # viridis
                    row_colors = [row_colors],
                    col_colors = [col_colors],
                    linewidth = 0.1,
                    linecolor = 'black',
                    figsize = (6, 6),
                    annot = True,
                    fmt = ".1f",
                    # robust = True,
                    # tree_kws = {'colors':'white'},
                    tree_kws = {'colors': row_colors},
                    # row_linkage = row_linkage,
                )
            except:
                st.markdown("Please, increase sample size.")
                st.stop()

            for a in c.ax_row_dendrogram.collections:
                a.set_linewidth(0.5)
            for a in c.ax_col_dendrogram.collections:
                a.set_linewidth(0.5)
            
            plt.title('Population')
            c.fig.suptitle(f"Hierarchical Clustering", size=10)
            # plt.tight_layout(pad=3)
            plt.savefig('clustermap.png', dpi=300, bbox_inches='tight')
            plt.clf()

            del data

    try:
        image = Image.open('clustermap.png')
        st.image(image, use_column_width=True)
    except:
        pass



with dashboard_correlation:

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6313491/
    # Correlation helps to identify which independent variables have stronger impacts on the dependent variables, therefore leading to a better forecast of a dependent variable. 
    # With a strong relationship, the independent variable can be considered a strong predictor for the dependent variable. 
    # The limitation of this analysis is that, in the case of two independent variables, it cannot determine the causality between the two.

    dashboard_correlation1, dashboard_correlation2, dashboard_correlation3, dashboard_correlation4 = st.columns(4, gap='large')

    with dashboard_correlation1:
        correlation_target = st.selectbox('Correlation target:', df_filtered_columns_num)

    with dashboard_correlation2:
        correlation_method = st.selectbox('Correlation method:', ['spearman', 'pearson', 'kendall'])

    with dashboard_correlation3:
        correlation_min = st.slider('Correlation minimum', 0.1, 0.9, 0.3, 0.1)

    with dashboard_correlation4:
        correlation_button = st.button('Calculate Correlation')

    try:
        image = Image.open('correlations.png') 
        st.image(image, use_column_width=True)
    except:
        pass


    if correlation_button:

        try:
            os.remove("correlations.png")
        except:
            pass

        df_correlations = df_filtered[df_filtered_columns_num].corr(method=correlation_method)
        df_correlations[np.abs(df_correlations) == 1] = np.nan
        df_correlations[np.abs(df_correlations) < correlation_min] = np.nan
        df_correlations_selected = df_correlations.dropna(how='all', axis=1)
        df_correlations_selected = df_correlations_selected.dropna(how='all', axis=0)

        title = f"""{correlation_target}: Relevant Correlations ({correlation_method})"""

        # df_correlations_selected.sort_values(by=[correlation_target], ascending=True, inplace=True)
        # Use `tril` instead of `triu` if the lower triangular matrix is needed
        # Use `np.bool_` instead of `np.bool` if you using NumPy >= 1.20
        mask = np.tril(np.ones_like(df_correlations_selected, dtype=np.bool_), 0)
        g = sns.heatmap(df_correlations_selected, annot=True, fmt=".2f", linewidth=1, linecolor='#0e1117', mask=mask, vmin=-1, vmax=1, cmap='summer', annot_kws={'size': 8})     # , cmap='RdYlGn_r', copper , cmap=['blue', '#0e1117', 'blue']
        # g.set_title(title, fontdict={'fontsize':10}, pad=10)
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=8, labelbottom = False, bottom=False, top = False, labeltop=True)

        plt.title(title, fontsize=14, loc='left')
        # plt.tight_layout(pad=3)
        plt.savefig('correlations.png', dpi=300, bbox_inches='tight')
        plt.clf()

        # Relevant to the target
        try:
            st.write(f"""##### {correlation_target}: Revelant Correlations ({correlation_method})""")
            st.dataframe(df_correlations_selected[correlation_target].sort_values(ascending=False).dropna(how='all', axis=0))
        except:
            st.markdown('There are no relevant correlations with the target.')

        st.write(f"##### Correlation Matrix ({correlation_method})")
        st.write(df_filtered[df_filtered_columns_num].corr(method=correlation_method))

        del df_correlations, df_correlations_selected, correlation_target, correlation_method, title

        st.rerun()


with dashboard_intracorrelation:

    dashboard_intracorrelation1, dashboard_intracorrelation2, dashboard_intracorrelation3, dashboard_intracorrelation4, dashboard_intracorrelation5, dashboard_intracorrelation6 = st.columns(6, gap='small')

    with dashboard_intracorrelation1:
        period = st.selectbox('Intracorrelation period:', ['No Period', 'Y', 'M', '2M', '3M', 'W', '2W', 'D', '2D', '3D', 'S', 'B', 'Q', 'A'])

    with dashboard_intracorrelation2:
        id_vars = st.selectbox('Intracorrelation labels:', df_filtered_columns_cat)

    with dashboard_intracorrelation3:
        df_filtered_columns_ = list(set(df_filtered_columns_num) - set(id_vars))
        value_vars = st.multiselect('Intracorrelation values:', df_filtered_columns_)

    with dashboard_intracorrelation4:
        aggregation_func = st.selectbox('Intracorrelation function:', ['mean', 'sum', 'min', 'max', 'count'])

    with dashboard_intracorrelation5:
        intracorrelation_method = st.selectbox('Intracorrelation method:', ['spearman', 'pearson', 'kendall'])

    with dashboard_intracorrelation6:
        intracorrelation_min = st.slider('Intracorrelation minimum', 0.1, 0.9, 0.3, 0.1)

        intracorrelation_button = st.button('Calculate Intracorrelation')

    try:
        image = Image.open('intracorrelations.png') 
        st.image(image, use_column_width=True)
    except:
        pass

    if intracorrelation_button:

        try:
            os.remove("intracorrelations.png")
        except:
            pass

        df_filtered.set_index('Period', inplace=True, drop=False)
        # correlations = (
        #     df_model.groupby([id_vars, pd.Grouper(freq='M')])[value_vars]
        #     .mean()
        #     .unstack(id_vars)
        #     .corr(method=intracorrelation_method)
        # )
        # st.dataframe(correlations)


        if aggregation_func == 'mean':
            correlations = (
                df_filtered.groupby([id_vars, pd.Grouper(freq=period)])[value_vars]
                .mean()
                .unstack(id_vars)
                # .corr(method=intracorrelation_method)
            )
        elif aggregation_func == 'sum':
            correlations = (
                df_filtered.groupby([id_vars, pd.Grouper(freq=period)])[value_vars]
                .sum()
                .unstack(id_vars)
                # .corr(method=intracorrelation_method)
            )
        elif aggregation_func == 'min':
            correlations = (
                df_filtered.groupby([id_vars, pd.Grouper(freq=period)])[value_vars]
                .min()
                .unstack(id_vars)
                # .corr(method=intracorrelation_method)
            )
        elif aggregation_func == 'max':
            correlations = (
                df_filtered.groupby([id_vars, pd.Grouper(freq=period)])[value_vars]
                .max()
                .unstack(id_vars)
                # .corr(method=intracorrelation_method)
            )
        elif aggregation_func == 'count':
            correlations = (
                df_filtered.groupby([id_vars, pd.Grouper(freq=period)])[value_vars]
                .count()
                .unstack(id_vars)
                # .corr(method=intracorrelation_method)
            )

        df_intracorrelations = correlations.corr(method=correlation_method)

        df_intracorrelations[np.abs(df_intracorrelations) == 1] = np.nan
        df_intracorrelations[np.abs(df_intracorrelations) < intracorrelation_min] = np.nan
        df_intracorrelations_selected = df_intracorrelations.dropna(how='all', axis=1)
        df_intracorrelations_selected = df_intracorrelations_selected.dropna(how='all', axis=0)

        title = f"""{id_vars}: Relevant Intracorrelations ({intracorrelation_method})"""

        mask = np.tril(np.ones_like(df_intracorrelations_selected, dtype=np.bool_), 0)
        g = sns.heatmap(df_intracorrelations_selected, annot=True, fmt=".2f", linewidth=1, linecolor='#0e1117', mask=mask, vmin=-1, vmax=1, cmap='summer', annot_kws={'size': 8})     # , cmap='RdYlGn_r', copper , cmap=['blue', '#0e1117', 'blue']
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=8, labelbottom = False, bottom=False, top = False, labeltop=True)

        plt.title(title, fontsize=14, loc='left')
        plt.savefig('intracorrelations.png', dpi=300, bbox_inches='tight')
        plt.clf()

        try:
            st.write(f"""##### {id_vars}: Revelant Intracorrelations ({intracorrelation_method})""")
            st.dataframe(df_intracorrelations_selected[id_vars].sort_values(ascending=False).dropna(how='all', axis=0))
        except:
            st.markdown('There are no relevant intracorrelations available.')

        del df_intracorrelations, df_intracorrelations_selected, id_vars, intracorrelation_method, title

        st.rerun()


with dashboard_significance:

    # https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Introductory_Statistics_(OpenStax)/12%3A_Linear_Regression_and_Correlation/12.05%3A_Testing_the_Significance_of_the_Correlation_Coefficient
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    # https://www.askpython.com/python/examples/spearman-correlation-python
    # https://www.statology.org/spearman-correlation-python/
    # https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/
    # https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
    # https://realpython.com/numpy-scipy-pandas-correlation-python/
    # https://stackoverflow.com/questions/57226054/seaborn-correlation-matrix-with-p-values-with-python

    dashboard_significance1, dashboard_significance2, dashboard_significance3, dashboard_significance4 = st.columns(4, gap='large')

    with dashboard_significance1:
        significance_target = st.selectbox('Significance target:', df_filtered_columns_num)

    with dashboard_significance2:
        significance_method = st.selectbox('Significance method:', ['spearman', 'pearson', 'kendall'])

    with dashboard_significance3:
        significance_min = st.slider('Significance minimum', 0.1, 0.9, 0.3, 0.1)

    with dashboard_significance4:
        significance_button = st.button('Calculate Significance')

    try:
        image = Image.open('significances.png') 
        st.image(image, use_column_width=True)
    except:
        pass

    if significance_button:

        try:
            os.remove("significances.png")
        except:
            pass

        cols = pd.DataFrame(columns=df_filtered_columns_num)
        p = cols.transpose().join(cols, how='outer')
        for r in df_filtered_columns_num:
            for c in df_filtered_columns_num:
                tmp = df_filtered[df_filtered[r].notnull() & df_filtered[c].notnull()]

                if significance_method == 'spearman':
                    p[r][c] = round(spearmanr(tmp[r], tmp[c])[1], 4)

                if significance_method == 'pearson':
                    p[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)

                if significance_method == 'kendall':
                    p[r][c] = round(kendalltau(tmp[r], tmp[c])[1], 4)

        p = p[df_filtered_columns_num]
        p.dropna(inplace=True, how='all', axis=1)
        p.dropna(inplace=True, how='all', axis=0)

        df_significances = df_filtered[df_filtered_columns_num].corr(method=significance_method)
        df_significances[np.abs(df_significances) == 1] = np.nan
        df_significances[np.abs(df_significances) < significance_min] = np.nan
        df_significances_selected = df_significances.dropna(how='all', axis=1)
        df_significances_selected = df_significances_selected.dropna(how='all', axis=0)

        title = f"""{significance_target}: Relevant Significances ({significance_method})"""

        mask = np.tril(np.ones_like(df_significances_selected, dtype=np.bool_), 0)
        g = sns.heatmap(df_significances_selected, annot=True, fmt=".2f", linewidth=1, linecolor='#0e1117', mask=mask, vmin=-1, vmax=1, cmap='summer', annot_kws={'size': 8})     # , cmap='RdYlGn_r', copper , cmap=['blue', '#0e1117', 'blue']
        # g.set_title(title, fontdict={'fontsize':10}, pad=10)
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=8, labelbottom = False, bottom=False, top = False, labeltop=True)
        plt.title(title, fontsize=14, loc='left')
        plt.savefig('significances.png', dpi=300, bbox_inches='tight')

        try:
            # Relevant to the target
            st.write(f"""##### {significance_target}: Relevant Significances ({significance_method})""")
            st.dataframe(df_significances_selected[significance_target].sort_values(ascending=False).dropna(how='all', axis=0))
        except:
            st.markdown('There are no relevant significances with the target.')

        st.write(f"##### Significance Matrix ({significance_method})")
        st.dataframe(p)
        # # dfi.export(p, "table_significance.png", fontsize=8, max_rows=27, max_cols=6, dpi=300, table_conversion="matplotlib")
        
        del p, r, c, tmp, cols, significance_target, significance_method, title

        st.rerun()








with dashboard_univariate:

    # small bar overlaps big bar: https://stackoverflow.com/questions/76167841/set-bar-with-lower-value-to-foreground-in-histplot
    # double slided bar: https://discuss.streamlit.io/t/is-there-a-double-sided-slider/11947
    # https://agilescientific.com/blog/tag/colour
    # chart size: https://www.tutorialspoint.com/how-do-we-adjust-the-size-of-the-plot-in-seaborn#:~:text=Seaborn%20provides%20a%20convenient%20function,figsize'%20parameter.
    # https://towardsdatascience.com/all-you-need-to-know-about-seaborn-6678a02f31ff

    dashboard_univariate1, dashboard_univariate2, dashboard_univariate3, dashboard_univariate4, dashboard_univariate5, dashboard_univariate6 = st.columns(6, gap='medium')

    with dashboard_univariate1:
        stat = st.selectbox('Uni statistics:', ['count', 'density', 'percent', 'probability', 'frequency'])        # https://seaborn.pydata.org/generated/seaborn.histplot.html
        element = st.selectbox('Uni element:', ['bars', 'step', 'poly'])
        multiple = st.selectbox('Uni multiple:', ['layer', 'dodge', 'stack', 'fill'])


    with dashboard_univariate2:

        df_filtered_columns_ = list(set(df_filtered_columns) - set(['Group', 'Period']))

        x = st.selectbox('Uni focus:', df_filtered_columns_)

        if df_filtered[x].dtypes == object:
            x_options = list(df_filtered[x].unique())
            x_options = [x for x in x_options if str(x) != 'nan']
            # x_options = ["All"] + x_options
            bins = len(x_options)
            x_options.sort()
            x_selected = st.multiselect("Distribution x filter:", x_options, default=x_options)

        elif np.issubdtype(df_filtered[x].dtype, np.longlong):
            x_options = list(df_filtered[x].unique())
            x_options.sort()
            x_selected = st.multiselect("Distribution x filter:", x_options, default=x_options)
            bins = len(x_selected)

        # elif np.issubdtype(df_filtered[x].dtype, np.datetime64):
        #     x_values = st.slider(
        #         f'{x} range:',
        #         df_filtered[x].min().timestamp(),
        #         df_filtered[x].max().timestamp(),
        #         (df_filtered[x].min().timestamp(), df_filtered[x].max().timestamp())
        #     )

        else:
            x_values = st.slider(f'{x} range:', df_filtered[x].min(), df_filtered[x].max(), (df_filtered[x].min(), df_filtered[x].max()))
            bins = st.slider(f"{x} bins:", 1, 20, 1)
            df_filtered['cut'] = pd.cut(x=df_filtered[x], bins=bins)
            # x_options = df.groupby('cut')[x].transform('mean').unique().round(2)          # mean option, slower
            x_options = list(df_filtered['cut'].apply(lambda x: x.mid).unique())                        # mid option, faster
            # x_options = list(df['cut'].unique())
            df_filtered.drop(['cut'], axis=1, inplace=True)


    with dashboard_univariate3:

        hue = st.selectbox('Uni hue:', df_filtered_columns_cat)
        
        if hue == x:
            hue_options = x_selected
        else:
            hue_options = list(df_filtered[hue].unique())

        try:
            hue_options.sort()
        except:
            pass

        hue_selected = st.multiselect(f"{hue} filter:", hue_options, default=hue_options)


    with dashboard_univariate4:
        st.markdown('Second axis')


    with dashboard_univariate5:
        columns = [hue] + [x]
        columns = np.unique(columns)
        
        df0 = df_filtered[columns]

        if st.checkbox('Fill'):
            fill = True
        else:
            fill = False

        if st.checkbox('Cumulative'):
            cumulative = True
        else:
            cumulative = False

        if element == 'bars':
            show_label = st.checkbox('Label')
        else:
            show_label = None
            

    with dashboard_univariate6:

        # if st.checkbox('Smooth Line'):
        #     if len(x_selected) > 1:
        #         kde = True                        # disabling kde because it is far off, polygon is better
        #     else:
        #         kde = False
        # else:
        #     kde = False

        if stat == 'probability' or stat == 'proportion' or stat == 'percent' or stat == 'density':
            show_normalize = st.checkbox('Normalize')
            if show_normalize:
                common_norm = True
            else:
                common_norm = False
        else:
            common_norm = False

        if is_float_dtype(df0[x]):
            if st.checkbox('Log scale'):
                log_scale = True
            else:
                log_scale = False
        else:
            log_scale = False


        if st.button('Plot Univariate'):

            try:
                os.remove("histplot.png")
            except:
                pass

            df3 = pd.DataFrame(columns = df0.columns)
    
            if df0[x].dtypes == object:
                for i in x_selected:
                    for j in hue_selected:
                        df1 = df0[df0[x] == i]
                        df2 = df1[df1[hue] == j]
                        df3 = pd.concat([df3, df2])
                        del df1, df2
           
            elif np.issubdtype(df0[x].dtype, np.longlong):
                for i in x_selected:
                    for j in hue_selected:
                        df1 = df0[df0[x] == i]
                        df2 = df1[df1[hue] == j]
                        df3 = pd.concat([df2, df3])
                        del df1, df2
                df3[x] = df3[x].astype('int')

            else:
                df0 = df0[df0[x] >= x_values[0]]
                df0 = df0[df0[x] <= x_values[1]]
                for j in hue_selected:
                    df2 = df0[df0[hue] == j]
                    df3 = pd.concat([df3, df2])
                    del df2

            g = sns.histplot(
                data = df3, 
                x = x,
                bins = bins,
                stat = stat,
                hue = hue,
                alpha = 1,
                multiple = multiple,
                kde = False,                                                                # kde = kde,
                line_kws={'zorder': 10, 'lw': 0.9, 'ls': '-'},
                legend = True,
                # element="step",
                cumulative = cumulative,
                fill = fill,
                element = element,
                common_norm = common_norm,
                shrink = .8,
                # discrete = True,                                                          # screws bins, bars centered on their data points
                log_scale = log_scale,
                pmax = 0.5,
                # cbar = True,                                                              # why is it not showing?
                # cbar_kws=dict(shrink=.75),
                linewidth = 0.9,
                palette = 'viridis',     # viridis, cividis, Spectral, bone, pink, copper, gist_ncar, terrain, summer, Blues_r, gnuplot2
            )                               # nipy_spectral, turbo, gist_earth, gist_rainbow_r, ocean, gist_stern, cubehelix, YlGnBu_r
                                            # magma

            num_groups = len(x_options)                                                     # to put smaller bars in front
            for bars in zip(*g.containers):
                order = np.argsort(np.argsort([b.get_height() for b in bars]))
                for bar, bar_order in zip(bars, order):
                    bar.set_zorder(2 + num_groups - bar_order)

            if show_label:
                for container in g.containers:
                    g.bar_label(container, label_type='edge', fontsize=5, rotation='vertical', padding=3)

            patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*", "|", "\\"]
            pattern_cycle = cycle(patterns)

            for container in g.containers:
                for bar in container.patches:
                    bar.set_hatch(next(pattern_cycle))


            if len(x) > 15:
                g.tick_params(axis='x', labelrotation=90)

            if is_integer_dtype(df3[x]):
                unique_xticks = df3[x].unique()
                plt.xticks(unique_xticks)


            if cumulative:
                if common_norm:
                    y_label = f"Cumulative Normalized {stat}"
                else:
                    y_label = f"Cumulative {stat}"
            else:
                if common_norm:
                    y_label = f"Normalized {stat}"
                else:
                    y_label = f"{stat}"
            plt.ylabel(y_label)

            sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, 0.85), ncol=8, title=None, frameon=False)
            plt.subplots_adjust(top=1)                                          # to increase title padding
            title = f"""{x}: Univariate Analysis ({element}, {multiple}, {bins} bins)"""
            plt.title(title, fontsize=10, loc='left')

            # g.axvline(x = df3[x].median(), color='gray', linestyle='--', linewidth = 1, label = 'median')
            # plt.grid()


            # g2.tick_params(axis='x', labelrotation=90)
            # try:
            #     plt.ticklabel_format(style='plain', axis='x')
            # except:
            #     pass
            # g2.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
            # g2 = SelectedBoxplot(data, x, y, hue, log_scale, dodge, showmeans, show_median, jitter, fill)
            # plt.title('Focus')
            # # plt.tight_layout(pad=3)
            # plt.subplots_adjust(left=0.1, right=0.9, wspace=0.8)
            # plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
            # plt.clf()
            # if add_grid:
            #     grid = sns.FacetGrid(data=data, col=hue, hue=hue, col_wrap=3, despine=True, legend_out=True, sharex=False, sharey=False, height=2, aspect=1)                    # sharex=False make individual scales for each item
            #     grid.map_dataframe(SelectedBoxplot, data=data, x=x, y=y, hue=y, log_scale=log_scale, dodge=False, showmeans=showmeans, show_median=show_median, jitter=jitter, fill=fill, facet_kws={'sharex': False, 'sharey': False})
            #     grid.set_titles(col_template='{col_name}', row_template='{hue_name}')
            #     grid.set_xticklabels(rotation=90)
            #     grid.fig.suptitle(f"{x} x {y}")
            #     # g.add_legend()                                                                  # will show empty if fill = False
            #     plt.tight_layout()
            #     plt.savefig('boxplot_grid.png', dpi=300, bbox_inches='tight')
            #     plt.clf()


            plt.savefig('histplot.png', dpi=300, bbox_inches='tight')
            plt.clf()

            del df3

    try:
        image = Image.open('histplot.png')
        st.image(image, use_column_width=True)
    except:
        pass



    del df0








with dashboard_multicollinearity:
    st.markdown("""##### Multicollinearity""")
    df_num = df_filtered[df_filtered_columns_num].dropna()

    vif_scores = [variance_inflation_factor(df_num.values, feature) for feature in range(len(df_num.columns))]       
    multicollinearity_df = pd.DataFrame({'Feature': df_num.columns, 'Variance Inflation Factor (>10 is bad)': vif_scores})
    multicollinearity_df.sort_values(by='Variance Inflation Factor (>10 is bad)', inplace=True, ascending=True)
    st.dataframe(multicollinearity_df)
    # dfi.export(multicollinearity_df, "table_multicollinearity.png", fontsize=8, max_rows=27, max_cols=6, dpi=300, table_conversion="matplotlib")
    del df_num, vif_scores, multicollinearity_df


with dashboard_autocorrelation:
    st.markdown("""##### Autocorrelation""")
    st.markdown("Not ported to this version yet.")



with dashboard_admin:

    del df_filtered, df_filtered_columns, df_filtered_columns_cat, df_filtered_columns_num

    st.markdown("""##### In-Scope Variables""")
    in_scope_variables = dir()                                                                  # https://stackoverflow.com/questions/633127/viewing-all-defined-variables
    st.markdown(f"Active variables: {in_scope_variables}")
    st.markdown(f"Memory usage: {getsizeof(in_scope_variables)/1000} kb")                       # https://stackoverflow.com/questions/14372006/variables-memory-size-in-python
    # local_variables = locals()
    # global_variables = globals()
    # st.markdown(local_variables)
    # st.markdown(global_variables)


# no report escolher quais graficos vao no rel e montar o rel so c os escolhidos, pode ser com checkboxes.