import pandas as pd


dfm = pd.read_csv('h3.bed', sep='\t', header=None, index_col=None)

dfm.columns = ['chrom', 'start', 'end']

dfm['length'] = dfm['end'] - dfm['start']

dfm.to_csv('h3.tsv', sep='\t', index=None)