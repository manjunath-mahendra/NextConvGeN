import json

import pandas as pd
import numpy as np

from fdc.tools import *
from fdc.visualize import plotCluster
from fdc.missingValues import fix_missing_values


class DataSheet:

  # ---------------------------------------------------------------------------
  # Data I/O
  # ---------------------------------------------------------------------------
  def __init__(self, file_name=None, index_col=0, dataFrame=None):
    if file_name is not None:
      self.data = pd.read_csv(file_name, index_col=0).sample(frac=1)
    elif dataFrame is not None:
      self.data = dataFrame
    else:
      self.data = pd.DataFrame(np.array([[0.0]]), "dummy")

    self.value_dict = {}
    self.cols_cont = []
    self.cols_ord = []
    self.cols_nom = []

    for k in self.data.dtypes.keys():
      t = str(self.data.dtypes[k])
      if t[:3] == "int":
        self.cols_ord.append(k)
      elif t == "object":
        self.cols_nom.append(k)
      else:
        self.cols_cont.append(k)

    self.has_missing_values = False
    self.updateMissingValuesState()
    self.detectMapping()

  def saveTable(self, file_name):
    self.data.to_csv(file_name)

  # ---------------------------------------------------------------------------
  # Data mapping
  # ---------------------------------------------------------------------------
  def saveMapping(self, file_name):
    with open(file_name, "w") as f:
      json.dump(self.value_dict, f)

  def loadMapping(self, file_name):
    with open(file_name) as f:
      pass #json.dump(tb.value_dict, f)

  def detectMapping(self):
    columnsToFix = []
    for k in self.data.dtypes.keys():
      if str(self.data.dtypes[k]) == "object":
        columnsToFix.append(k)

    self.value_dict = {}
    for c in columnsToFix:
      histogram = self.data[c].value_counts()
      self.value_dict[c] = { n : k for n, k in enumerate(histogram.keys()) }

  def useMapping(self, mapping=None):
    if mapping is None:
      mapping = { c: { k: v for v, k in self.value_dict[c].items() } for c in self.value_dict.keys() }

    if len(mapping.keys()) > 0:
      self.data.replace(mapping, inplace=True)

  # ---------------------------------------------------------------------------
  # Statistics
  # ---------------------------------------------------------------------------
  def showStatistic(self):
    print(f"Fratures: {self.data.shape[1]}")
    print(f"Points:   {self.data.shape[0]}")
    print(f"Columns:")

    for k in self.data.dtypes.keys():
      t = str(self.data.dtypes[k])
      e = " c"
      if k in self.cols_ord:
        e = " o"
      if k in self.cols_nom:
        e = " n"
      indentPair(k, t, e)
    print()
    print(f"Missing values:")

    n = 0
    d = self.data.isna().sum()
    for k in d.keys():
      if d[k] > 0:
        indentPair(k, str(d[k]))
        n += 1
    if n == 0:
      print("  none")

  def updateMissingValuesState(self):
    self.has_missing_values = False
    for k in self.data.isna().sum():
      if k > 0:
        self.has_missing_values = True
        break


  # ---------------------------------------------------------------------------
  # Automatic fixing tools
  # ---------------------------------------------------------------------------
  def fixDatatypes(self):
    self.detectMapping()
    self.useMapping() 

  def fix_missing_values(self):
    self.data = fix_missing_values(self.data, 4)
    self.updateMissingValuesState()

