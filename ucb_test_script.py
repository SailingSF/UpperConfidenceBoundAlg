from UpperConfAlg import Confidence
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

Confidence.UCB(dataset)