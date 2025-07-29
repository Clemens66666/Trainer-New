{'loss': 0.1283, 'grad_norm': 0.04298216477036476, 'learning_rate': 2.0101634294592037e-07, 'epoch': 7.89}
{'loss': 0.1299, 'grad_norm': 0.04545285925269127, 'learning_rate': 9.739967132431193e-08, 'epoch': 7.95}
{'train_runtime': 168518.2218, 'train_samples_per_second': 1.309, 'train_steps_per_second': 0.041, 'train_loss': 0.12215324849097746, 'epoch': 8.0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6896/6896 [46:48:38<00:00, 24.44s/it] 
[LightGBM] [Info] Number of positive: 13608, number of negative: 13964
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.021607 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 49656
[LightGBM] [Info] Number of data points in the train set: 27572, number of used features: 240
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.493544 -> initscore=-0.025825
[LightGBM] [Info] Start training from score -0.025825
[LightGBM] [Info] Number of positive: 13513, number of negative: 14059
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.019920 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 49656
[LightGBM] [Info] Number of data points in the train set: 27572, number of used features: 240
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.490099 -> initscore=-0.039611
[LightGBM] [Info] Start training from score -0.039611
[LightGBM] [Info] Number of positive: 13606, number of negative: 13966
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.020095 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 49656
[LightGBM] [Info] Number of data points in the train set: 27572, number of used features: 240
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.493472 -> initscore=-0.026115
[LightGBM] [Info] Start training from score -0.026115
Traceback (most recent call last):
  File "C:\trainers\train.py", line 67, in <module>
    main()
  File "C:\trainers\train.py", line 63, in main
    trainer.run()
  File "C:\trainers\trainers\base.py", line 59, in run
    study         = self.optimize(X_feat, y)
                    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\trainers\hybrid_longtrend_trainer.py", line 489, in optimize
    self._predict_ft_logits(X_val_flat),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\trainers\hybrid_longtrend_trainer.py", line 277, in _predict_ft_logits
    device=self.device         # self.device ist cpu | cuda
           ^^^^^^^^^^^
AttributeError: 'HybridLongTrendTrainer' object has no attribute 'device'
(ml-env) PS C:\trainers>      
