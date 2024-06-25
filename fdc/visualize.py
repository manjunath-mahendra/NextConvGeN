import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plotCluster(data, clusterName="cluster", xName="FDC_1", yName="FDC_2", stroke=20):
  colors_set = [
    'lightgray', 'lightcoral', 'cornflowerblue', 'orange','mediumorchid', 'lightseagreen'
    , 'olive', 'chocolate', 'steelblue', 'paleturquoise',  'lightgreen'
    , 'burlywood', 'lightsteelblue']

  customPalette_set = sns.set_palette(sns.color_palette(colors_set))
  sns.set_style('whitegrid',{'axes.grid': False})

  sns.lmplot(
    x=xName
    , y=yName
    , data=data
    , fit_reg=False
    , legend=True
    , hue=clusterName
    , scatter_kws={"s": stroke}
    , palette=customPalette_set
    )
  plt.show()



def plotMapping(data, xName="UMAP_0", yName="UMAP_1"):
  colors_set1 = [
    "lightcoral", "lightseagreen", "mediumorchid", "orange", "burlywood"
    , "cornflowerblue", "plum", "yellowgreen"]

  customPalette_set1 = sns.set_palette(sns.color_palette(colors_set1))
  sns.set_style('whitegrid',{'axes.grid': False})


  sns.lmplot(x=xName
    , y=yName
    , data=data
    , fit_reg=False
    , legend=False
    , scatter_kws={"s": 3}
    , palette=customPalette_set1)
  plt.show()



def vizx(feature_list, cluster_df_list, main_data, umap_data, cont_features, rev_dict, xName="FDC_1", yName="FDC_2"):
  vizlimit = 15
  plt.rcParams["figure.figsize"] = (12, 6)
  
  col = sns.color_palette("Set2")
  
  rows = 3
  columns = 3
  
  for feature in feature_list:
    print('Feature name:', feature.upper())
    print('\n')
  
    if len(main_data[feature].value_counts()) <= vizlimit:
      for cluster_counter, cluster in enumerate(cluster_df_list):
        print('Cluster '+ str(cluster_counter + 1) + ' frequency distribution')
        if feature in rev_dict:
          r = rev_dict[feature]
          print(cluster.replace({feature:r})[feature].value_counts())
        else:
          print(cluster[feature].value_counts())
        print('\n')
    
      print('\n')
      print('\n')
    
      cluster_bar = []
      for cluster in cluster_df_list:
        if feature in rev_dict:
          y = np.array(cluster.replace({feature:r})[feature].value_counts())
          x = np.array(cluster.replace({feature:r})[feature].value_counts().index)
          cluster_bar.append([x,y])
        else:
          y = np.array(cluster[feature].value_counts().sort_index())
          x = np.array(cluster[feature].value_counts().sort_index().index)
          cluster_bar.append([x,y])
          
      cluster_bar = np.array(cluster_bar)
    
      figx, ax = plt.subplots(rows, columns)
      figx.set_size_inches(10.5, 28.5)
      cluster_in_subplot_axis_dict = np.array(
        [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[1,1],[2,2]])
      c = 0
      for i in range(rows):
        for j in range(columns):
          if c >= len(cluster_df_list):
            break
          ax[i,j].bar(cluster_bar[c,0], cluster_bar[c,1], color=col)
          ax[i,j].tick_params(axis='x', which='major', labelsize=8, rotation=90)
          ax[i,j].set_title('Cluster: ' + str(c + 1))
          c += 1
      
    means = []
    sds = []
    cluster_labels = []
    for cluster_counter, cluster in enumerate(cluster_df_list):
      if feature in cont_features:
        print('Cluster '+ str(cluster_counter + 1) + ' summary statistics')
        print('\n')
        cm = cluster[feature].mean()
        cs = cluster[feature].std()
        print('feature mean:', cm)
        print('feature standard deviation:', cs)
        print('feature median:', cluster[feature].median())
        print('\n')
        means.append(cm)
        sds.append(cs)
        cluster_labels.append('C' + str(cluster_counter + 1))
        
    means = np.array(means)
    sds = np.array(sds)
    cluster_labels = np.array(cluster_labels)
    
    print('\n')  
    
    print('Distribution of feature across clusters')
    if feature in cont_features:   
      fig, ax7 = plt.subplots()
      ax7.bar(cluster_labels, means, yerr=sds, color=sns.color_palette("Set3"))
      ax7.tick_params(axis='both', which='major', labelsize=10)
      plt.xlabel(feature, fontsize=15)
      plt.show()
    
    print('\n')
    print('\n')
    
    customPalette_set = sns.set_palette(sns.color_palette(
      [ 'lightgray', 'lightcoral', 'cornflowerblue', 'orange', 'mediumorchid'
      , 'lightseagreen', 'olive', 'chocolate', 'steelblue', 'paleturquoise'
      , 'lightgreen', 'burlywood','lightsteelblue'
      ]))
    
    if feature not in cont_features:
      print('Feature distribution in UMAP embedding')
      if feature in rev_dict:
        r = rev_dict[feature]
        umap_data[feature] = np.array(main_data.replace({feature:r})[feature])
      else:
        umap_data[feature] = np.array(main_data[feature])
      sns.lmplot(x=xName, y=yName,
        data=umap_data, 
        fit_reg=False, 
        legend=True,
        hue=feature, # color by cluster
        scatter_kws={"s": 20},
        palette=customPalette_set) # specify the point size
      plt.show()
    
    print('\n')
    print('\n')
        
