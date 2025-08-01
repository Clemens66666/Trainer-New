(ml-env) PS C:\trainers> python train.py --cfg config.yaml --type hybrid --ssl-weights ft_ssl.pt
▶  Starte hybrid_longtrend_trainer.HybridLongTrendTrainer mit C:\trainers\config.yaml
C:\trainers\utils\features.py:99: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
▶  Starte hybrid_longtrend_trainer.HybridLongTrendTrainer mit C:\trainers\config.yaml
C:\trainers\utils\features.py:99: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="bfill", inplace=True)
C:\trainers\utils\features.py:100: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="ffill", inplace=True)
C:\trainers\utils\dataset.py:24: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

▶  Starte hybrid_longtrend_trainer.HybridLongTrendTrainer mit C:\trainers\config.yaml
C:\trainers\utils\features.py:99: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="bfill", inplace=True)
C:\trainers\utils\features.py:100: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="ffill", inplace=True)
C:\trainers\utils\dataset.py:24: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.        
▶  Starte hybrid_longtrend_trainer.HybridLongTrendTrainer mit C:\trainers\config.yaml
C:\trainers\utils\features.py:99: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="bfill", inplace=True)
C:\trainers\utils\features.py:100: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="ffill", inplace=True)
C:\trainers\utils\dataset.py:24: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

  df.fillna(method="bfill", inplace=True)
C:\trainers\utils\features.py:100: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="ffill", inplace=True)
C:\trainers\utils\dataset.py:24: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.        


For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.        


For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.        





  df["label"].fillna(0.0, inplace=True)
[I 2025-08-01 00:10:48,346] A new study created in memory with name: no-name-5e5c8b4b-b048-41d7-b006-6eb059e7572f
The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
[I 2025-08-01 00:10:48,346] A new study created in memory with name: no-name-5e5c8b4b-b048-41d7-b006-6eb059e7572f
The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
C:\trainers\ml-env\Lib\site-packages\transformers\training_args.py:1604: FutureWarning: using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. Use `use_cpu` instead
  warnings.warn(
{'loss': 0.1946, 'grad_norm': 0.021127495914697647, 'learning_rate': 7.99043405228915e-06, 'epoch': 0.07}
C:\trainers\ml-env\Lib\site-packages\transformers\training_args.py:1604: FutureWarning: using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. Use `use_cpu` instead
  warnings.warn(
{'loss': 0.1946, 'grad_norm': 0.021127495914697647, 'learning_rate': 7.99043405228915e-06, 'epoch': 0.07}
{'loss': 0.1946, 'grad_norm': 0.021127495914697647, 'learning_rate': 7.99043405228915e-06, 'epoch': 0.07}
{'loss': 0.1955, 'grad_norm': 0.021615341305732727, 'learning_rate': 7.367155265059107e-06, 'epoch': 0.14}
{'loss': 0.1936, 'grad_norm': 0.03425659239292145, 'learning_rate': 6.743876477829063e-06, 'epoch': 0.22}
{'loss': 0.1934, 'grad_norm': 0.0251360684633255, 'learning_rate': 6.12059769059902e-06, 'epoch': 0.29}
{'loss': 0.1903, 'grad_norm': 0.02215576171875, 'learning_rate': 5.497318903368978e-06, 'epoch': 0.36}
{'loss': 0.1912, 'grad_norm': 0.04964413493871689, 'learning_rate': 4.874040116138935e-06, 'epoch': 0.43}
{'loss': 0.192, 'grad_norm': 0.04905476048588753, 'learning_rate': 4.250761328908892e-06, 'epoch': 0.51}
{'loss': 0.1915, 'grad_norm': 0.03332998976111412, 'learning_rate': 3.6274825416788496e-06, 'epoch': 0.58}
{'loss': 0.1906, 'grad_norm': 0.05232324078679085, 'learning_rate': 3.0042037544488066e-06, 'epoch': 0.65}
{'loss': 0.1934, 'grad_norm': 0.041438810527324677, 'learning_rate': 2.3809249672187636e-06, 'epoch': 0.72}
{'loss': 0.1904, 'grad_norm': 0.03159739449620247, 'learning_rate': 1.7576461799887208e-06, 'epoch': 0.8}
{'loss': 0.1887, 'grad_norm': 0.026969019323587418, 'learning_rate': 1.1343673927586779e-06, 'epoch': 0.87}
{'loss': 0.1909, 'grad_norm': 0.02962564304471016, 'learning_rate': 5.110886055286352e-07, 'epoch': 0.94}
{'eval_loss': 0.18996812403202057, 'eval_runtime': 78.7174, 'eval_samples_per_second': 70.061, 'eval_steps_per_second': 8.766, 'epoch': 1.0}
{'train_runtime': 2296.7065, 'train_samples_per_second': 9.604, 'train_steps_per_second': 0.3, 'train_loss': 0.1918546275816102, 'epoch': 1.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 690/690 [38:16<00:00,  3.33s/it] 
[I 2025-08-01 00:51:30,030] Trial 0 finished with value: 0.7181212306022644 and parameters: {'lr': 8.601247263774591e-06, 'n_blocks': 6}. Best is trial 0 with value: 0.7181212306022644.
🟢  FT-best: {'lr': 8.601247263774591e-06, 'n_blocks': 6}
C:\trainers\ml-env\Lib\site-packages\transformers\training_args.py:1604: FutureWarning: using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. Use `use_cpu` instead
  warnings.warn(
{'loss': 0.1525, 'grad_norm': 0.012561550363898277, 'learning_rate': 8.112313254580908e-06, 'epoch': 0.06}
{'loss': 0.152, 'grad_norm': 0.017893288284540176, 'learning_rate': 7.613401000301639e-06, 'epoch': 0.12}
{'loss': 0.1497, 'grad_norm': 0.01649777591228485, 'learning_rate': 7.114488746022371e-06, 'epoch': 0.17}
{'loss': 0.1513, 'grad_norm': 0.028929997235536575, 'learning_rate': 6.615576491743102e-06, 'epoch': 0.23}
{'loss': 0.1508, 'grad_norm': 0.018830960616469383, 'learning_rate': 6.1166642374638335e-06, 'epoch': 0.29}
{'loss': 0.1504, 'grad_norm': 0.013582208193838596, 'learning_rate': 5.617751983184565e-06, 'epoch': 0.35}
{'loss': 0.1508, 'grad_norm': 0.022435473278164864, 'learning_rate': 5.118839728905295e-06, 'epoch': 0.41}
{'loss': 0.1524, 'grad_norm': 0.01844612881541252, 'learning_rate': 4.619927474626028e-06, 'epoch': 0.46}
{'loss': 0.1499, 'grad_norm': 0.014798521995544434, 'learning_rate': 4.121015220346759e-06, 'epoch': 0.52}
{'loss': 0.1507, 'grad_norm': 0.013310804031789303, 'learning_rate': 3.62210296606749e-06, 'epoch': 0.58}
{'loss': 0.1491, 'grad_norm': 0.01641390100121498, 'learning_rate': 3.1231907117882216e-06, 'epoch': 0.64}
{'loss': 0.151, 'grad_norm': 0.014274735003709793, 'learning_rate': 2.624278457508953e-06, 'epoch': 0.7}
{'loss': 0.1515, 'grad_norm': 0.01402992568910122, 'learning_rate': 2.1253662032296843e-06, 'epoch': 0.75}
{'loss': 0.1513, 'grad_norm': 0.01338890939950943, 'learning_rate': 1.626453948950416e-06, 'epoch': 0.81}
{'loss': 0.153, 'grad_norm': 0.014131077565252781, 'learning_rate': 1.127541694671147e-06, 'epoch': 0.87}
{'loss': 0.149, 'grad_norm': 0.022401638329029083, 'learning_rate': 6.286294403918785e-07, 'epoch': 0.93}
{'loss': 0.1512, 'grad_norm': 0.013312422670423985, 'learning_rate': 1.2971718611260984e-07, 'epoch': 0.99}
{'train_runtime': 2373.4886, 'train_samples_per_second': 11.617, 'train_steps_per_second': 0.363, 'train_loss': 0.1509037281963499, 'epoch': 1.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 862/862 [39:33<00:00,  2.75s/it]
[LightGBM] [Info] Number of positive: 13608, number of negative: 13964
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.022148 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 49656
[LightGBM] [Info] Number of data points in the train set: 27572, number of used features: 240
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.493544 -> initscore=-0.025825
[LightGBM] [Info] Start training from score -0.025825
[LightGBM] [Info] Number of positive: 13513, number of negative: 14059
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.020013 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 49656
[LightGBM] [Info] Number of data points in the train set: 27572, number of used features: 240
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.490099 -> initscore=-0.039611
[LightGBM] [Info] Start training from score -0.039611
[LightGBM] [Info] Number of positive: 13606, number of negative: 13966
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.019312 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 49656
[LightGBM] [Info] Number of data points in the train set: 27572, number of used features: 240
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.493472 -> initscore=-0.026115
[LightGBM] [Info] Start training from score -0.026115
C:\trainers\ml-env\Lib\site-packages\transformers\training_args.py:1604: FutureWarning: using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. Use `use_cpu` instead
  warnings.warn(
  0%|                                                                                                                                                                                               | 0/132 [00:00<?, ?it/s]Traceback (most recent call last):
  File "C:\trainers\train.py", line 67, in <module>
    main()
  File "C:\trainers\train.py", line 63, in main
    trainer.run()
  File "C:\trainers\trainers\base.py", line 67, in run
    study       = self.optimize(X_feat, y)
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\trainers\hybrid_longtrend_trainer.py", line 540, in optimize
    trainer.train()
  File "C:\trainers\ml-env\Lib\site-packages\transformers\trainer.py", line 2206, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\ml-env\Lib\site-packages\transformers\trainer.py", line 2502, in _inner_training_loop
    batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\ml-env\Lib\site-packages\transformers\trainer.py", line 5300, in get_batch_samples
    batch_samples.append(next(epoch_iterator))
                         ^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\ml-env\Lib\site-packages\accelerate\data_loader.py", line 567, in __iter__
    current_batch = next(dataloader_iter)
                    ^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\ml-env\Lib\site-packages\torch\utils\data\dataloader.py", line 733, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "C:\trainers\ml-env\Lib\site-packages\torch\utils\data\dataloader.py", line 789, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\ml-env\Lib\site-packages\torch\utils\data\_utils\fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\ml-env\Lib\site-packages\transformers\trainer_utils.py", line 872, in __call__
    return self.data_collator(features)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\utils\collate.py", line 26, in meta_collate
    preds  = torch.tensor([x["preds"]  for x in batch], dtype=torch.float32)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\utils\collate.py", line 26, in <listcomp>
    preds  = torch.tensor([x["preds"]  for x in batch], dtype=torch.float32)
                           ~^^^^^^^^^
KeyError: 'preds'
  0%|                                                                                                                                                                                               | 0/132 [00:00<?, ?it/s]
(ml-env) PS C:\trainers> 
