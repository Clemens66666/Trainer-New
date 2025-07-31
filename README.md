(ml-env) PS C:\trainers> python train.py --cfg config.yaml --type hybrid --ssl-weights ft_ssl.pt
▶  Starte hybrid_longtrend_trainer.HybridLongTrendTrainer mit C:\trainers\config.yaml
C:\trainers\utils\features.py:99: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="bfill", inplace=True)
C:\trainers\utils\features.py:100: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="ffill", inplace=True)
C:\trainers\utils\dataset.py:24: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.        


  df["label"].fillna(0.0, inplace=True)
[I 2025-07-31 17:06:36,165] A new study created in memory with name: no-name-8d6a680e-556b-4621-8318-dbcc285dea5c
The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
C:\trainers\ml-env\Lib\site-packages\transformers\training_args.py:1604: FutureWarning: using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. Use `use_cpu` instead
  warnings.warn(
{'loss': 0.1947, 'grad_norm': 0.020521439611911774, 'learning_rate': 1.6784889014109723e-06, 'epoch': 0.07}
{'loss': 0.1958, 'grad_norm': 0.02060028910636902, 'learning_rate': 1.5475615300060603e-06, 'epoch': 0.14}
{'loss': 0.1942, 'grad_norm': 0.029195768758654594, 'learning_rate': 1.4166341586011481e-06, 'epoch': 0.22}
{'loss': 0.1945, 'grad_norm': 0.022220784798264503, 'learning_rate': 1.2857067871962361e-06, 'epoch': 0.29}
{'loss': 0.1916, 'grad_norm': 0.01952018029987812, 'learning_rate': 1.1547794157913241e-06, 'epoch': 0.36}
{'loss': 0.1929, 'grad_norm': 0.04115518182516098, 'learning_rate': 1.023852044386412e-06, 'epoch': 0.43}
{'loss': 0.1941, 'grad_norm': 0.036632515490055084, 'learning_rate': 8.929246729815e-07, 'epoch': 0.51}
{'loss': 0.1941, 'grad_norm': 0.02738921158015728, 'learning_rate': 7.619973015765881e-07, 'epoch': 0.58}
{'loss': 0.1936, 'grad_norm': 0.03614040091633797, 'learning_rate': 6.31069930171676e-07, 'epoch': 0.65}
 69%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                        | 474/690 [28:39<12:54,  3.59s/it] 


 durchlauf noch am laufen 
