(ml-env) PS C:\trainers> python train.py --cfg config.yaml --type hybrid --ssl-weights ft_ssl.pt
>>
▶  Starte hybrid_longtrend_trainer.HybridLongTrendTrainer mit C:\trainers\config.yaml
C:\trainers\utils\features.py:99: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="bfill", inplace=True)
C:\trainers\utils\features.py:100: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="ffill", inplace=True)
C:\trainers\utils\dataset.py:24: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.        


  df["label"].fillna(0.0, inplace=True)
[I 2025-08-02 18:45:58,158] A new study created in memory with name: no-name-cb10deb1-d563-48b5-94c5-213448bdd9b0
The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
C:\trainers\ml-env\Lib\site-packages\transformers\training_args.py:1604: FutureWarning: using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. Use `use_cpu` instead
  warnings.warn(
{'loss': 0.1502, 'grad_norm': 0.011129430495202541, 'learning_rate': 1.7224039638699973e-05, 'epoch': 0.07}
{'loss': 0.1499, 'grad_norm': 0.010617414489388466, 'learning_rate': 1.5880510805728055e-05, 'epoch': 0.14}
{'loss': 0.1496, 'grad_norm': 0.0057229455560445786, 'learning_rate': 1.4536981972756136e-05, 'epoch': 0.22}
{'loss': 0.1518, 'grad_norm': 0.009565943852066994, 'learning_rate': 1.319345313978422e-05, 'epoch': 0.29}
{'loss': 0.1538, 'grad_norm': 0.008295751176774502, 'learning_rate': 1.1849924306812303e-05, 'epoch': 0.36}
{'loss': 0.152, 'grad_norm': 0.009106618352234364, 'learning_rate': 1.0506395473840388e-05, 'epoch': 0.43}
{'loss': 0.1534, 'grad_norm': 0.010157187469303608, 'learning_rate': 9.16286664086847e-06, 'epoch': 0.51}
{'loss': 0.1507, 'grad_norm': 0.012108191847801208, 'learning_rate': 7.819337807896555e-06, 'epoch': 0.58}
{'loss': 0.1521, 'grad_norm': 0.011580160818994045, 'learning_rate': 6.475808974924639e-06, 'epoch': 0.65}
{'loss': 0.1516, 'grad_norm': 0.009370746091008186, 'learning_rate': 5.132280141952721e-06, 'epoch': 0.72}
{'loss': 0.1512, 'grad_norm': 0.00992507766932249, 'learning_rate': 3.7887513089808053e-06, 'epoch': 0.8}
{'loss': 0.1505, 'grad_norm': 0.00802541058510542, 'learning_rate': 2.445222476008888e-06, 'epoch': 0.87}
{'loss': 0.1509, 'grad_norm': 0.014746108092367649, 'learning_rate': 1.1016936430369716e-06, 'epoch': 0.94}
{'eval_loss': 0.15009209513664246, 'eval_runtime': 66.9493, 'eval_samples_per_second': 82.376, 'eval_steps_per_second': 10.306, 'epoch': 1.0}
{'train_runtime': 1719.4327, 'train_samples_per_second': 12.828, 'train_steps_per_second': 0.401, 'train_loss': 0.15143377193506213, 'epoch': 1.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 690/690 [28:39<00:00,  2.49s/it]
[I 2025-08-02 19:17:34,646] Trial 0 finished with value: 0.7093477845191956 and parameters: {'lr': 1.854069789501245e-05, 'n_blocks': 5}. Best is trial 0 with value: 0.7093477845191956.
🟢  FT-best: {'lr': 1.854069789501245e-05, 'n_blocks': 5}
C:\trainers\ml-env\Lib\site-packages\transformers\training_args.py:1604: FutureWarning: using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. Use `use_cpu` instead
  warnings.warn(
{'loss': 0.1455, 'grad_norm': 0.00863359309732914, 'learning_rate': 1.7486760311653275e-05, 'epoch': 0.06}
{'loss': 0.1481, 'grad_norm': 0.00930135976523161, 'learning_rate': 1.641131379802146e-05, 'epoch': 0.12}
{'loss': 0.1466, 'grad_norm': 0.01141196209937334, 'learning_rate': 1.5335867284389648e-05, 'epoch': 0.17}
{'loss': 0.1495, 'grad_norm': 0.015210215002298355, 'learning_rate': 1.4260420770757837e-05, 'epoch': 0.23}
{'loss': 0.148, 'grad_norm': 0.010364619083702564, 'learning_rate': 1.3184974257126022e-05, 'epoch': 0.29}
{'loss': 0.1474, 'grad_norm': 0.014036393724381924, 'learning_rate': 1.210952774349421e-05, 'epoch': 0.35}
{'loss': 0.1485, 'grad_norm': 0.019459588453173637, 'learning_rate': 1.1034081229862397e-05, 'epoch': 0.41}
{'loss': 0.1477, 'grad_norm': 0.009766368195414543, 'learning_rate': 9.958634716230585e-06, 'epoch': 0.46}
{'loss': 0.1494, 'grad_norm': 0.010259459726512432, 'learning_rate': 8.883188202598772e-06, 'epoch': 0.52}
{'loss': 0.1487, 'grad_norm': 0.012038699351251125, 'learning_rate': 7.807741688966959e-06, 'epoch': 0.58}
{'loss': 0.148, 'grad_norm': 0.01870327815413475, 'learning_rate': 6.732295175335147e-06, 'epoch': 0.64}
{'loss': 0.1492, 'grad_norm': 0.00972810760140419, 'learning_rate': 5.656848661703335e-06, 'epoch': 0.7}
{'loss': 0.1481, 'grad_norm': 0.011989228427410126, 'learning_rate': 4.581402148071521e-06, 'epoch': 0.75}
{'loss': 0.1488, 'grad_norm': 0.008928313851356506, 'learning_rate': 3.5059556344397094e-06, 'epoch': 0.81}
{'loss': 0.1504, 'grad_norm': 0.012650073505938053, 'learning_rate': 2.4305091208078965e-06, 'epoch': 0.87}
{'loss': 0.1512, 'grad_norm': 0.013402874581515789, 'learning_rate': 1.355062607176084e-06, 'epoch': 0.93}
{'loss': 0.1467, 'grad_norm': 0.010893856175243855, 'learning_rate': 2.7961609354427126e-07, 'epoch': 0.99}
{'train_runtime': 2094.7581, 'train_samples_per_second': 13.162, 'train_steps_per_second': 0.412, 'train_loss': 0.14839547070439066, 'epoch': 1.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 862/862 [34:54<00:00,  2.43s/it] 
[LightGBM] [Info] Number of positive: 13608, number of negative: 13964
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.027495 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 49656
[LightGBM] [Info] Number of data points in the train set: 27572, number of used features: 240
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.493544 -> initscore=-0.025825
[LightGBM] [Info] Start training from score -0.025825
[LightGBM] [Info] Number of positive: 13513, number of negative: 14059
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.028068 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 49656
[LightGBM] [Info] Number of data points in the train set: 27572, number of used features: 240
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.490099 -> initscore=-0.039611
[LightGBM] [Info] Start training from score -0.039611
[LightGBM] [Info] Number of positive: 13606, number of negative: 13966
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.025207 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 49656
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.493472 -> initscore=-0.026115
[LightGBM] [Info] Start training from score -0.026115
C:\trainers\ml-env\Lib\site-packages\transformers\training_args.py:1604: FutureWarning: using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. Use `use_cpu` instead
  warnings.warn(
{'loss': 0.3108, 'grad_norm': 0.15799970924854279, 'learning_rate': 0.0006287878787878788, 'epoch': 1.14}
{'loss': 0.2528, 'grad_norm': 0.311767578125, 'learning_rate': 0.00025, 'epoch': 2.27}
{'train_runtime': 3.4371, 'train_samples_per_second': 4821.549, 'train_steps_per_second': 38.405, 'train_loss': 0.27287301511475537, 'epoch': 3.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 132/132 [00:03<00:00, 34.58it/s] 
✅  Modelle gespeichert in models\hybrid_longtrend_20250802_184344
✅  Training abgeschlossen.
