# coding: utf-8
sample_pred = sorted(list(pred_dir.iterdir()))[0]
df_pred = pd.read_csv(pred_dir / sample_pred.name)
dict_true = read_json_file(true_dir / (sample_pred.stem.replace('.jpg', '') + '.json'))
df_true = extract_dataframe_from_icpr_dictionary(dict_true)
compute_rms(df_pred, df_true)
