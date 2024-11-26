# coding: utf-8
import pandas as pd
from pathlib import Path
get_ipython().run_line_magic('run', '-i ~/path_scripts/pandas_print_options.py')
get_ipython().run_line_magic('run', '-i ~/path_scripts/my_helping_functions')
imgpaths = find_images('images/')
dictionary = [{'name': p.name, 'path': str(p), 'type': p.parent.stem} for p in find_images('images/')]
df = pd.DataFrame.from_dict(dictionary)
df['type'].unique()
get_ipython().run_line_magic('pwd', '')
df.to_csv('classes.csv')
df_ref = df.copy()
df_hyp = pd.read_csv('/Users/mohamedfayed/important_repos/GraphIngestionEngineFall2024/GIE-graph-classifier/predictions/model1.keras/icpr22/output.csv')
len(df_hyp)
len(df_ref)
import sklearn.metrics as metrics
df_ref.iloc[3]
df_hyp.iloc[3]
df_hyp['name'] = df_hyp['image_index'].apply(lambda x: Path(x).name)
df = pd.merge(df_ref, df_hyp, on=['name'])
df.columns
df['type'].unique()
df['type_pred'].unique()
df['type'].replace({'vertical-bar': 'BarGraph', 'horizontal-bar': 'BarGraph', 'line': 'LineGraph', 'scatter': 'ScatterGraph'}).unique()
df['type_true'] = df['type'].replace({'vertical-bar': 'BarGraph', 'horizontal-bar': 'BarGraph', 'line': 'LineGraph', 'scatter': 'ScatterGraph'})
metrics.accuracy_score(df['type_true'], df['type_pred'])
df2 = df[df['type_true'] != 'vertical-box']
len(df2)
metrics.accuracy_score(df['type_true'], df['type_pred'])
metrics.accuracy_score(df2['type_true'], df2['type_pred'])
get_ipython().run_line_magic('save', '/Users/mohamedfayed/important_repos/GraphIngestionEngineFall2024/GIE-graph-classifier/ipy_scripts/compute_accuracy_for_classifier_on_icpr.py  1-30')
