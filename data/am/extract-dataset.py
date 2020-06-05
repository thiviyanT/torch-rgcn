import rdflib as rdf
import gzip
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

g = rdf.Graph()

with gzip.open('am-combined.nt.gz', 'rb') as input:
    g.parse(input, format='nt')

print 'loaded graph'

rel = rdf.term.URIRef("http://purl.org/collections/nl/am/objectCategory")

# Get the labels
labels = set()
num = 0
for _, _, label in g.triples((None, rel, None)):
    labels.add(str(label))
    num += 1

enc = LabelEncoder()
enc.fit(list(labels))

print 'encoded labels'

df = pd.DataFrame(columns={'instance' : [], 'label_original': [], 'label' : []}, index=range(num))

for i, (s, _, o) in enumerate(g.triples((None, rel, None))):
    row = {'instance' : str(s), 'label_original': str(o), 'label' : enc.transform([str(o)])[0]}
    df.ix[i] = row

g.close()

df.to_csv('completeDataset.all.tsv', sep='\t')
print 'created dataframe'

# * Split test and train sets

# fixed seed for deterministic output
np.random.seed(0)

train_size = int(0.8 * len(df))
test_size = len(df) - train_size

bin = np.concatenate( (np.ones((train_size,)), np.zeros((test_size,))), axis=0)
np.random.shuffle(bin)
msk = bin > 0.5

print msk

train = df[msk]
train.to_csv('trainSet.all.tsv', sep='\t')

test = df[~msk]
test.to_csv('testSet.all.tsv', sep='\t')

print 'created text/train split'







