#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 23:05:26 2020

@author: anmol
"""

# Purely all me no Udemy in this
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#to do someday
