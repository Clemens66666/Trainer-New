(ml-env) PS C:\trainers> python train.py --cfg config.yaml --type hybrid
▶  Starte hybrid_longtrend_trainer.HybridLongTrendTrainer mit C:\trainers\config.yaml
C:\trainers\utils\features.py:99: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="bfill", inplace=True)
C:\trainers\utils\features.py:100: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df.fillna(method="ffill", inplace=True)
C:\trainers\utils\dataset.py:24: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.        


  df["label"].fillna(0.0, inplace=True)

=== RF: Optuna + Final-Training ===
[I 2025-08-13 10:32:18,122] A new study created in memory with name: rf_study
[I 2025-08-13 10:32:59,842] Trial 0 finished with value: 0.33298508009891326 and parameters: {'n_estimators': 382, 'max_depth': 5, 'min_samples_split': 8, 'max_features': None}. Best is trial 0 with value: 0.33298508009891326.
  0%|                                                                                                                                                                                                 | 0/1 [00:41<?, ?it/s][10:32:59] rf_study trial#0 value=0.33298508009891326 params={'n_estimators': 382, 'max_depth': 5, 'min_samples_split': 8, 'max_features': None}
Best trial: 0. Best value: 0.332985: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:41<00:00, 41.72s/it] 
=== RF: fertig ===

=== LGB: Optuna startet ===
[I 2025-08-13 11:07:29,271] A new study created in memory with name: lgb_study
[I 2025-08-13 11:07:37,490] Trial 0 finished with value: 0.3204109757825252 and parameters: {'learning_rate': 0.012073035028913043, 'num_leaves': 91, 'feature_fraction': 0.7156735293415233, 'bagging_fraction': 0.8276289885201258, 'num_boost_round': 575}. Best is trial 0 with value: 0.3204109757825252.
  0%|                                                                                                                                                                                                 | 0/1 [00:08<?, ?it/s][11:07:37] lgb_study trial#0 value=0.3204109757825252 params={'learning_rate': 0.012073035028913043, 'num_leaves': 91, 'feature_fraction': 0.7156735293415233, 'bagging_fraction': 0.8276289885201258, 'num_boost_round': 575}Best trial: 0. Best value: 0.320411: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.22s/it] 
=== LGB: fertig ===

=== XGB: Optuna startet ===
[I 2025-08-13 11:08:19,981] A new study created in memory with name: xgb_study
  0%|                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s[ 
0]      val-logloss:0.66338
[10]    val-logloss:0.47967
[20]    val-logloss:0.39629
[30]    val-logloss:0.35592
[40]    val-logloss:0.33529
[50]    val-logloss:0.32523
[60]    val-logloss:0.32103
[70]    val-logloss:0.31916
[80]    val-logloss:0.31839
[90]    val-logloss:0.31910
[100]   val-logloss:0.32034
[110]   val-logloss:0.32110
[112]   val-logloss:0.32171
[I 2025-08-13 11:08:28,778] Trial 0 finished with value: 0.31866227814608433 and parameters: {'eta': 0.05345223405557697, 'max_depth': 10, 'subsample': 0.6882988928928874, 'colsample_bytree': 0.655490794726417, 'lambda_l2': 0.47324270519990147}. Best is trial 0 with value: 0.31866227814608433.
  0%|                                                                                                                                                                                                 | 0/1 [00:08<?, ?it/s][11:08:28] xgb_study trial#0 value=0.31866227814608433 params={'eta': 0.05345223405557697, 'max_depth': 10, 'subsample': 0.6882988928928874, 'colsample_bytree': 0.655490794726417, 'lambda_l2': 0.47324270519990147}
Best trial: 0. Best value: 0.318662: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.80s/it] 
=== XGB: fertig ===

=== CNN: Optuna startet ===
[I 2025-08-13 11:09:57,258] A new study created in memory with name: cnn_study
  0%|                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/sC 
:\trainers\ml-env\Lib\site-packages\torch\nn\modules\module.py:1158: UserWarning: expandable_segments not supported on this platform (Triggered internally at ..\c10\cuda\CUDACachingAllocator.cpp:803.)
  return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
[I 2025-08-13 11:09:59,510] Trial 0 finished with value: 7.926386295514214 and parameters: {'lr': 0.0008641745350886424, 'n_filters': 41}. Best is trial 0 with value: 7.926386295514214.
  0%|                                                                                                                                                                                                 | 0/1 [00:02<?, ?it/s][11:09:59] cnn_study trial#0 value=7.926386295514214 params={'lr': 0.0008641745350886424, 'n_filters': 41}
Best trial: 0. Best value: 7.92639: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.25s/it] 
=== CNN: fertig ===

=== FT: Optuna startet ===
[I 2025-08-13 11:09:59,891] A new study created in memory with name: ft_study
{'loss': 0.1789, 'grad_norm': 0.013345381245017052, 'learning_rate': 0.00022991724332038115, 'epoch': 0.04}
{'loss': 0.1853, 'grad_norm': 0.028482532128691673, 'learning_rate': 0.0002278144699225361, 'epoch': 0.07}
{'loss': 0.1816, 'grad_norm': 0.03705849498510361, 'learning_rate': 0.000225711696524691, 'epoch': 0.11}
{'loss': 0.1841, 'grad_norm': 0.020613903179764748, 'learning_rate': 0.00022360892312684594, 'epoch': 0.15}
{'loss': 0.1849, 'grad_norm': 0.01011725515127182, 'learning_rate': 0.00022150614972900085, 'epoch': 0.18}
{'loss': 0.1825, 'grad_norm': 0.026951229199767113, 'learning_rate': 0.00021940337633115576, 'epoch': 0.22}
{'loss': 0.1797, 'grad_norm': 0.009750276803970337, 'learning_rate': 0.00021730060293331067, 'epoch': 0.25}
{'loss': 0.1738, 'grad_norm': 0.014644986018538475, 'learning_rate': 0.00021519782953546558, 'epoch': 0.29}
{'loss': 0.1769, 'grad_norm': 0.017589464783668518, 'learning_rate': 0.00021309505613762052, 'epoch': 0.33}
{'loss': 0.1778, 'grad_norm': 0.027829326689243317, 'learning_rate': 0.00021099228273977543, 'epoch': 0.36}
{'loss': 0.1758, 'grad_norm': 0.029317660257220268, 'learning_rate': 0.00020888950934193037, 'epoch': 0.4}
{'loss': 0.173, 'grad_norm': 0.0346447229385376, 'learning_rate': 0.00020678673594408528, 'epoch': 0.44}
{'loss': 0.1765, 'grad_norm': 0.03214106336236, 'learning_rate': 0.0002046839625462402, 'epoch': 0.47}
{'loss': 0.1792, 'grad_norm': 0.015108542516827583, 'learning_rate': 0.00020258118914839513, 'epoch': 0.51}
{'loss': 0.1778, 'grad_norm': 0.030650140717625618, 'learning_rate': 0.00020047841575055, 'epoch': 0.54}
{'loss': 0.1772, 'grad_norm': 0.02781517431139946, 'learning_rate': 0.00019837564235270495, 'epoch': 0.58}
{'loss': 0.1758, 'grad_norm': 0.02021060511469841, 'learning_rate': 0.00019627286895485986, 'epoch': 0.62}
{'loss': 0.1768, 'grad_norm': 0.04730759188532829, 'learning_rate': 0.0001941700955570148, 'epoch': 0.65}
{'loss': 0.1756, 'grad_norm': 0.025725986808538437, 'learning_rate': 0.0001920673221591697, 'epoch': 0.69}
{'loss': 0.1781, 'grad_norm': 0.03688499704003334, 'learning_rate': 0.00018996454876132462, 'epoch': 0.73}
{'loss': 0.1782, 'grad_norm': 0.021809369325637817, 'learning_rate': 0.00018786177536347955, 'epoch': 0.76}
{'loss': 0.1762, 'grad_norm': 0.02962137572467327, 'learning_rate': 0.00018575900196563446, 'epoch': 0.8}
{'loss': 0.1806, 'grad_norm': 0.04943295195698738, 'learning_rate': 0.0001836562285677894, 'epoch': 0.83}
{'loss': 0.1792, 'grad_norm': 0.031092416495084763, 'learning_rate': 0.00018155345516994429, 'epoch': 0.87}
{'loss': 0.1788, 'grad_norm': 0.035038385540246964, 'learning_rate': 0.00017945068177209922, 'epoch': 0.91}
{'loss': 0.1798, 'grad_norm': 0.06566005200147629, 'learning_rate': 0.00017734790837425413, 'epoch': 0.94}
{'loss': 0.1772, 'grad_norm': 0.025382624939084053, 'learning_rate': 0.00017524513497640904, 'epoch': 0.98}
{'eval_loss': 0.17078366875648499, 'eval_runtime': 2.4985, 'eval_samples_per_second': 2207.354, 'eval_steps_per_second': 69.242, 'epoch': 1.0}
{'loss': 0.1779, 'grad_norm': 0.03067781589925289, 'learning_rate': 0.00017314236157856398, 'epoch': 1.02}
{'loss': 0.1774, 'grad_norm': 0.047614529728889465, 'learning_rate': 0.0001710395881807189, 'epoch': 1.05}
{'loss': 0.1781, 'grad_norm': 0.039284974336624146, 'learning_rate': 0.00016893681478287383, 'epoch': 1.09}
{'loss': 0.1837, 'grad_norm': 0.03945275396108627, 'learning_rate': 0.00016683404138502874, 'epoch': 1.12}
{'loss': 0.1837, 'grad_norm': 0.030255405232310295, 'learning_rate': 0.00016473126798718365, 'epoch': 1.16}
{'loss': 0.1824, 'grad_norm': 0.028198452666401863, 'learning_rate': 0.00016262849458933856, 'epoch': 1.2}
{'loss': 0.1841, 'grad_norm': 0.02391853928565979, 'learning_rate': 0.00016052572119149347, 'epoch': 1.23}
{'loss': 0.1795, 'grad_norm': 0.040995415300130844, 'learning_rate': 0.0001584229477936484, 'epoch': 1.27}
{'loss': 0.1787, 'grad_norm': 0.03391001746058464, 'learning_rate': 0.00015632017439580332, 'epoch': 1.31}
{'loss': 0.1772, 'grad_norm': 0.026919906958937645, 'learning_rate': 0.00015421740099795826, 'epoch': 1.34}
{'loss': 0.179, 'grad_norm': 0.048711467534303665, 'learning_rate': 0.00015211462760011317, 'epoch': 1.38}
{'loss': 0.1782, 'grad_norm': 0.04408733546733856, 'learning_rate': 0.00015001185420226808, 'epoch': 1.41}
{'loss': 0.1848, 'grad_norm': 0.028464514762163162, 'learning_rate': 0.00014790908080442302, 'epoch': 1.45}
{'loss': 0.1843, 'grad_norm': 0.0438697375357151, 'learning_rate': 0.0001458063074065779, 'epoch': 1.49}
{'loss': 0.1819, 'grad_norm': 0.04062449932098389, 'learning_rate': 0.00014370353400873284, 'epoch': 1.52}
{'loss': 0.1801, 'grad_norm': 0.021493591368198395, 'learning_rate': 0.00014160076061088775, 'epoch': 1.56}
{'loss': 0.1795, 'grad_norm': 0.0456719733774662, 'learning_rate': 0.00013949798721304266, 'epoch': 1.6}
{'loss': 0.1793, 'grad_norm': 0.04268608242273331, 'learning_rate': 0.0001373952138151976, 'epoch': 1.63}
{'loss': 0.1816, 'grad_norm': 0.035512540489435196, 'learning_rate': 0.0001352924404173525, 'epoch': 1.67}
{'loss': 0.1795, 'grad_norm': 0.04033079370856285, 'learning_rate': 0.00013318966701950745, 'epoch': 1.7}
{'loss': 0.1823, 'grad_norm': 0.07269236445426941, 'learning_rate': 0.00013108689362166236, 'epoch': 1.74}
{'loss': 0.1776, 'grad_norm': 0.05373634397983551, 'learning_rate': 0.0001289841202238173, 'epoch': 1.78}
{'loss': 0.1732, 'grad_norm': 0.037162765860557556, 'learning_rate': 0.00012688134682597218, 'epoch': 1.81}
{'loss': 0.1718, 'grad_norm': 0.03893067315220833, 'learning_rate': 0.0001247785734281271, 'epoch': 1.85}
{'loss': 0.1769, 'grad_norm': 0.05159313604235649, 'learning_rate': 0.00012267580003028203, 'epoch': 1.89}
{'loss': 0.1818, 'grad_norm': 0.06476900726556778, 'learning_rate': 0.00012057302663243695, 'epoch': 1.92}
{'loss': 0.1821, 'grad_norm': 0.03542570769786835, 'learning_rate': 0.00011847025323459186, 'epoch': 1.96}
{'loss': 0.1791, 'grad_norm': 0.04780969023704529, 'learning_rate': 0.00011636747983674679, 'epoch': 1.99}
{'eval_loss': 0.17689114809036255, 'eval_runtime': 2.5277, 'eval_samples_per_second': 2181.83, 'eval_steps_per_second': 68.442, 'epoch': 2.0}
{'loss': 0.1818, 'grad_norm': 0.04534081369638443, 'learning_rate': 0.00011426470643890171, 'epoch': 2.03}
{'loss': 0.1812, 'grad_norm': 0.067215695977211, 'learning_rate': 0.00011216193304105662, 'epoch': 2.07}
{'loss': 0.1802, 'grad_norm': 0.04119114577770233, 'learning_rate': 0.00011005915964321154, 'epoch': 2.1}
{'loss': 0.1824, 'grad_norm': 0.022871049121022224, 'learning_rate': 0.00010795638624536647, 'epoch': 2.14}
{'loss': 0.1812, 'grad_norm': 0.039161767810583115, 'learning_rate': 0.00010585361284752138, 'epoch': 2.18}
{'loss': 0.1835, 'grad_norm': 0.04834091290831566, 'learning_rate': 0.00010375083944967629, 'epoch': 2.21}
{'loss': 0.178, 'grad_norm': 0.04491353780031204, 'learning_rate': 0.00010164806605183121, 'epoch': 2.25}
{'loss': 0.181, 'grad_norm': 0.049159422516822815, 'learning_rate': 9.954529265398614e-05, 'epoch': 2.28}
{'loss': 0.1773, 'grad_norm': 0.03737832233309746, 'learning_rate': 9.744251925614106e-05, 'epoch': 2.32}
{'loss': 0.1811, 'grad_norm': 0.07700033485889435, 'learning_rate': 9.533974585829599e-05, 'epoch': 2.36}
{'loss': 0.1816, 'grad_norm': 0.03962216153740883, 'learning_rate': 9.323697246045088e-05, 'epoch': 2.39}
{'loss': 0.182, 'grad_norm': 0.052292607724666595, 'learning_rate': 9.113419906260581e-05, 'epoch': 2.43}
{'loss': 0.1757, 'grad_norm': 0.058614809066057205, 'learning_rate': 8.903142566476073e-05, 'epoch': 2.47}
{'loss': 0.1779, 'grad_norm': 0.04249607399106026, 'learning_rate': 8.692865226691566e-05, 'epoch': 2.5}
{'loss': 0.1767, 'grad_norm': 0.023921657353639603, 'learning_rate': 8.482587886907057e-05, 'epoch': 2.54}
{'loss': 0.1788, 'grad_norm': 0.06293678283691406, 'learning_rate': 8.272310547122549e-05, 'epoch': 2.57}
{'loss': 0.1776, 'grad_norm': 0.03478106111288071, 'learning_rate': 8.06203320733804e-05, 'epoch': 2.61}
{'loss': 0.1836, 'grad_norm': 0.03848040848970413, 'learning_rate': 7.851755867553533e-05, 'epoch': 2.65}
{'loss': 0.1796, 'grad_norm': 0.03946920484304428, 'learning_rate': 7.641478527769025e-05, 'epoch': 2.68}
{'loss': 0.1827, 'grad_norm': 0.04939498007297516, 'learning_rate': 7.431201187984516e-05, 'epoch': 2.72}
{'loss': 0.1841, 'grad_norm': 0.06885991245508194, 'learning_rate': 7.220923848200008e-05, 'epoch': 2.76}
{'loss': 0.1844, 'grad_norm': 0.053312983363866806, 'learning_rate': 7.010646508415501e-05, 'epoch': 2.79}
{'loss': 0.1832, 'grad_norm': 0.04640611261129379, 'learning_rate': 6.800369168630993e-05, 'epoch': 2.83}
{'loss': 0.1784, 'grad_norm': 0.04630404710769653, 'learning_rate': 6.590091828846483e-05, 'epoch': 2.86}
{'loss': 0.1814, 'grad_norm': 0.04997804015874863, 'learning_rate': 6.379814489061975e-05, 'epoch': 2.9}
{'loss': 0.181, 'grad_norm': 0.031434912234544754, 'learning_rate': 6.169537149277468e-05, 'epoch': 2.94}
{'loss': 0.1791, 'grad_norm': 0.031871188431978226, 'learning_rate': 5.95925980949296e-05, 'epoch': 2.97}
{'eval_loss': 0.17926999926567078, 'eval_runtime': 2.5163, 'eval_samples_per_second': 2191.722, 'eval_steps_per_second': 68.752, 'epoch': 3.0}
{'train_runtime': 94.0529, 'train_samples_per_second': 938.068, 'train_steps_per_second': 58.648, 'train_loss': 0.1796991641261539, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [01:34<00:31, 43.99it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:02<00:00, 68.79it/s]
[I 2025-08-13 11:11:37,284] Trial 0 finished with value: 0.17078366875648499 and parameters: {'lr': 0.00023197796125026934, 'n_blocks': 4}. Best is trial 0 with value: 0.17078366875648499.
  0%|                                                                                                                                                                                                 | 0/1 [01:37<?, ?it/s][11:11:37] ft_study trial#0 value=0.17078366875648499 params={'lr': 0.00023197796125026934, 'n_blocks': 4}
Best trial: 0. Best value: 0.170784: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:37<00:00, 97.39s/it] 
{'loss': 0.1399, 'grad_norm': 0.06311099976301193, 'learning_rate': 0.00022358490711280672, 'epoch': 0.29}
{'loss': 0.1559, 'grad_norm': 0.03591396287083626, 'learning_rate': 0.00021517503322757365, 'epoch': 0.58}
{'loss': 0.1598, 'grad_norm': 0.07641571015119553, 'learning_rate': 0.00020676515934234057, 'epoch': 0.87}
{'loss': 0.163, 'grad_norm': 0.04499112069606781, 'learning_rate': 0.0001983552854571075, 'epoch': 1.16}
{'loss': 0.1652, 'grad_norm': 0.08143437653779984, 'learning_rate': 0.00018994541157187439, 'epoch': 1.45}
{'loss': 0.1695, 'grad_norm': 0.07159923017024994, 'learning_rate': 0.0001815355376866413, 'epoch': 1.74}
{'loss': 0.1725, 'grad_norm': 0.0677449181675911, 'learning_rate': 0.00017312566380140823, 'epoch': 2.03}
{'loss': 0.1694, 'grad_norm': 0.04938913881778717, 'learning_rate': 0.00016471578991617515, 'epoch': 2.32}
{'loss': 0.1748, 'grad_norm': 0.06731779128313065, 'learning_rate': 0.00015630591603094207, 'epoch': 2.61}
{'loss': 0.1755, 'grad_norm': 0.04360582306981087, 'learning_rate': 0.000147896042145709, 'epoch': 2.9}
{'loss': 0.175, 'grad_norm': 0.06915118545293808, 'learning_rate': 0.00013948616826047592, 'epoch': 3.19}
{'loss': 0.174, 'grad_norm': 0.07800295203924179, 'learning_rate': 0.00013107629437524284, 'epoch': 3.48}
{'loss': 0.1739, 'grad_norm': 0.05458209291100502, 'learning_rate': 0.00012266642049000974, 'epoch': 3.77}
{'loss': 0.1768, 'grad_norm': 0.042463839054107666, 'learning_rate': 0.00011425654660477666, 'epoch': 4.06}
{'loss': 0.1802, 'grad_norm': 0.07560218125581741, 'learning_rate': 0.00010584667271954357, 'epoch': 4.35}
{'loss': 0.1739, 'grad_norm': 0.06104797124862671, 'learning_rate': 9.743679883431049e-05, 'epoch': 4.64}
{'loss': 0.1732, 'grad_norm': 0.051287341862916946, 'learning_rate': 8.902692494907741e-05, 'epoch': 4.93}
{'loss': 0.174, 'grad_norm': 0.045744720846414566, 'learning_rate': 8.061705106384433e-05, 'epoch': 5.22}
{'loss': 0.178, 'grad_norm': 0.08308804780244827, 'learning_rate': 7.220717717861124e-05, 'epoch': 5.51}
{'loss': 0.1783, 'grad_norm': 0.04887527972459793, 'learning_rate': 6.379730329337816e-05, 'epoch': 5.8}
{'loss': 0.1711, 'grad_norm': 0.0726473331451416, 'learning_rate': 5.538742940814508e-05, 'epoch': 6.09}
{'loss': 0.1778, 'grad_norm': 0.06607899814844131, 'learning_rate': 4.6977555522911994e-05, 'epoch': 6.38}
{'loss': 0.1747, 'grad_norm': 0.05776311457157135, 'learning_rate': 3.856768163767892e-05, 'epoch': 6.67}
{'loss': 0.1736, 'grad_norm': 0.037868682295084, 'learning_rate': 3.015780775244583e-05, 'epoch': 6.96}
{'loss': 0.1739, 'grad_norm': 0.06535971164703369, 'learning_rate': 2.174793386721275e-05, 'epoch': 7.25}
{'loss': 0.1756, 'grad_norm': 0.07853063195943832, 'learning_rate': 1.3338059981979668e-05, 'epoch': 7.54}
{'loss': 0.176, 'grad_norm': 0.06747207790613174, 'learning_rate': 4.928186096746586e-06, 'epoch': 7.83}
{'train_runtime': 283.9071, 'train_samples_per_second': 776.93, 'train_steps_per_second': 48.579, 'train_loss': 0.1714268599198202, 'epoch': 8.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13792/13792 [04:43<00:00, 48.58it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 3.47}
{'train_runtime': 18.0839, 'train_samples_per_second': 1016.815, 'train_steps_per_second': 63.703, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1152/1152 [00:18<00:00, 63.70it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 0.87}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 2.61}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 3.48}
{'train_runtime': 34.9515, 'train_samples_per_second': 1051.973, 'train_steps_per_second': 65.806, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2300/2300 [00:34<00:00, 65.80it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 0.58}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 1.16}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 2.32}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 2.9}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 3.48}
{'train_runtime': 52.5342, 'train_samples_per_second': 1049.755, 'train_steps_per_second': 65.633, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3448/3448 [00:52<00:00, 65.63it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 0.44}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 0.87}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 1.31}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 2.18}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 2.61}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 3.05}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 3.92}
{'train_runtime': 70.2988, 'train_samples_per_second': 1045.935, 'train_steps_per_second': 65.378, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4596/4596 [01:10<00:00, 65.38it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 0.35}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 0.7}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 1.04}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 1.39}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 2.09}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 2.44}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 2.78}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 3.13}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00023197796125026934, 'epoch': 3.83}
{'train_runtime': 89.1422, 'train_samples_per_second': 1031.027, 'train_steps_per_second': 64.481, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [01:29<00:00, 64.47it/s]
=== FT: fertig ===
[I 2025-08-13 11:21:59,336] A new study created in memory with name: meta_study
[W 2025-08-13 11:21:59,792] Trial 0 failed with parameters: {'d_token': 118, 'dropout': 0.03829238272134328, 'lr': 0.0001452567789056763} because of the following error: RuntimeError('torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\nMany models use a sigmoid layer right before the binary cross entropy layer.\nIn this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\nor torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\nsafe to autocast.').
Traceback (most recent call last):
  File "C:\trainers\ml-env\Lib\site-packages\optuna\study\_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "C:\trainers\trainers\hybrid_longtrend_trainer.py", line 156, in _safe_objective
    val = objective_fn(trial)
          ^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\trainers\hybrid_longtrend_trainer.py", line 551, in <lambda>
    lambda t: self._meta_objective_seq(t, H_tr, C_tr, y_tr, H_va, C_va, y_va),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\trainers\hybrid_longtrend_trainer.py", line 517, in _meta_objective_seq
    bce    = nn.functional.binary_cross_entropy(p_hat, y_s)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\trainers\ml-env\Lib\site-packages\torch\nn\functional.py", line 3122, in binary_cross_entropy
    return torch._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
Many models use a sigmoid layer right before the binary cross entropy layer.
In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits
or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are
safe to autocast.
[W 2025-08-13 11:21:59,792] Trial 0 failed with value None.
  0%|                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s] 
⚠️  Meta-Training fehlgeschlagen: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
Many models use a sigmoid layer right before the binary cross entropy layer.
In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits
or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are
safe to autocast.. Fallback auf einfache LogReg.
