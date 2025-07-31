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
{'loss': 0.1967, 'grad_norm': 0.0307643860578537, 'learning_rate': 5.001425587667639e-07, 'epoch': 0.72}
{'loss': 0.1941, 'grad_norm': 0.02459217980504036, 'learning_rate': 3.692151873618519e-07, 'epoch': 0.8}
{'loss': 0.1924, 'grad_norm': 0.020896486937999725, 'learning_rate': 2.3828781595693988e-07, 'epoch': 0.87}
{'loss': 0.1946, 'grad_norm': 0.0240284726023674, 'learning_rate': 1.0736044455202786e-07, 'epoch': 0.94}
{'eval_loss': 0.1951095312833786, 'eval_runtime': 91.9217, 'eval_samples_per_second': 59.997, 'eval_steps_per_second': 7.506, 'epoch': 1.0}
{'train_runtime': 2604.4521, 'train_samples_per_second': 8.469, 'train_steps_per_second': 0.265, 'train_loss': 0.1940563215725664, 'epoch': 1.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 690/690 [43:24<00:00,  3.77s/it]
[I 2025-07-31 17:53:46,627] Trial 0 finished with value: 0.7207729816436768 and parameters: {'lr': 1.806797725387786e-06, 'n_blocks': 6}. Best is trial 0 with value: 0.7207729816436768.
🟢  FT-best: {'lr': 1.806797725387786e-06, 'n_blocks': 6}
C:\trainers\ml-env\Lib\site-packages\transformers\training_args.py:1604: FutureWarning: using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. Use `use_cpu` instead
  warnings.warn(
{'loss': 0.1526, 'grad_norm': 0.012603125534951687, 'learning_rate': 1.7040911261488052e-06, 'epoch': 0.06}
  8%|██████████████▌                                                                                                                                                                       | 69/862 [03:24<39:58,  3.02s/it] 
