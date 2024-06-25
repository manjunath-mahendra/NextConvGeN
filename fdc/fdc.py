# --[ Known to be used ]----
import numpy as np
from numba import jit
import umap.umap_ as umap

from fdc.tools import Timing

# --[ Known to be used but can we avoid it? ]----
import pandas as pd
from fdc.visualize import plotMapping


def value(v, defaultValue):
    if v is None:
        return defaultValue
    else:
        return v


def feature_clustering(UMAP_neb, min_dist_UMAP, metric, data, visual=False):
    data_embedded = Clustering(metric, UMAP_neb, min_dist_UMAP).fit(data)

    result = pd.DataFrame(data=data_embedded, columns=['UMAP_0', 'UMAP_1'])
    
    if visual:
        plotMapping(result)

    return result



@jit(nopython=True)
def canberra_modified(a,b):
    return np.sqrt(np.sum(np.array(
        [np.abs(1.0 - x) / (1.0 + np.abs(x)) for x in (np.abs(a-b) + 1.0)]
        )))



class Clustering:
    def __init__(self, metric='euclidian', UMAP_neb=30, min_dist_UMAP=0.1, max_components=2):
        self.metric = metric
        self.UMAP_neb = UMAP_neb
        self.min_dist_UMAP = min_dist_UMAP
        self.max_components = max_components

    def normalize(self, x):
        return (x - np.mean(x)) / np.std(x)

    def fit(self, data):
        np.random.seed(42)

        # ensure that the data is a 2d array.
        if len(data.shape) < 2:
            data = data.reshape((data.shape[0], 1))

        # do UMAP if needed (e.g. data has more than 2 features)
        data_embedded = umap.UMAP(
            n_neighbors=self.UMAP_neb
            , min_dist=self.min_dist_UMAP
            , n_components=self.max_components
            , metric=self.metric
            , random_state=42
            ).fit_transform(data)

        # normalize the data
        for n in range(data_embedded.shape[1]):
            data_embedded[:, n] = self.normalize(data_embedded[:, n])
        
        return data_embedded



class FDC:
    def __init__(self,
                 clustering_cont=None, clustering_ord=None, clustering_nom=None,
                 visual=False,
                 with_2d_embedding=False,
                 use_pandas_output=False
                 ):
        # used clusterings
        self.clustering_cont = value(clustering_cont, Clustering('euclidean', 30, 0.1))
        self.clustering_ord = value(clustering_ord, Clustering(canberra_modified, 30, 0.1))
        self.clustering_nom = value(clustering_nom, Clustering('hamming', 30, 0.1, max_components=1))

        # Control of data output
        self.use_pandas_output = use_pandas_output
        self.with_2d_embedding = with_2d_embedding

        # Control if a graph is shown
        self.visual = visual

        # Lists to select columns for continueous, nomial and ordinal data.
        self.cont_list = None
        self.nom_list = None
        self.ord_list = None
        
    def selectFeatures(self, continueous=None, nomial=None, ordinal=None):
        self.cont_list = continueous
        self.nom_list = nomial
        self.ord_list = ordinal

    def normalize(self, data,
                  cont_list=None, nom_list=None, ord_list=None,
                  with_2d_embedding=None,
                  visual=None
                  ):

        timing = Timing("FDC.normalize")

        # Take instance value if parameter was not given.
        visual = value(visual, self.visual)
        with_2d_embedding = value(with_2d_embedding, self.with_2d_embedding)
        
        # Initialize data. 
        np.random.seed(42)
        concat_column_names = []
        concat_lists = []
        
        timing.step("init")

        # Reducing features into 2dim or 1dim
        actions = [
            ("CONT", self.clustering_cont, value(cont_list, self.cont_list))
            , ("ORD", self.clustering_ord, value(ord_list, self.ord_list))
            , ("NOM", self.clustering_nom, value(nom_list, self.nom_list))
            ]

        for (name, clustering, column_list) in actions:
            if column_list is not None:
                if str(type(data)) == "<class 'numpy.ndarray'>":
                    part = data[:, column_list]
                else:
                    part = data[column_list]
                emb = clustering.fit(part)
                concat_lists.append(emb)
                for n in range(emb.shape[1]):
                    concat_column_names.append(f"{name}_UMAP_{n}")
            timing.step(f"clustering {name}")

        # Merge results
        if concat_lists == []:
            raise ValueError("Expected at least one non empty column list.") 

        result_concat = np.concatenate(concat_lists, axis=1)
        timing.step("concat")

        # Create 2d embedding from 5d embedding
        if with_2d_embedding or visual:
            result_reduced = umap.UMAP(
                n_neighbors=30
                , min_dist=0.001
                , n_components=2
                , metric='euclidean'
                , random_state=42
                ).fit_transform(result_concat)

            timing.step("umap 5 -> 2")
        
            if self.use_pandas_output:
                result_reduced = pd.DataFrame(
                    data=result_reduced, columns=['UMAP_0', 'UMAP_1'])
                timing.step("array -> DataFrame")

            # Show mapping if needed
            if visual:
                if self.use_pandas_output:
                    plotMapping(result_reduced)
                else:
                    plotMapping(pd.DataFrame(
                        data=result_reduced, columns=['UMAP_0', 'UMAP_1']))
                timing.step("plotting")

        # Transform to pandas DataFrame if needed.
        if self.use_pandas_output:
            result_concat = pd.DataFrame(
                data=result_concat, columns=concat_column_names)
            timing.step("array -> DataFrame")

        timing.step("total")

        if with_2d_embedding:
            #returns both 5D and 2D embeddings
            return result_concat, result_reduced
        else:
            #returns 5D embedding only
            return result_concat
