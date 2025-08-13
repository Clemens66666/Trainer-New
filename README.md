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
[I 2025-08-13 14:23:02,170] A new study created in memory with name: rf_study
[I 2025-08-13 14:23:07,945] Trial 0 finished with value: 0.33074134764072916 and parameters: {'n_estimators': 339, 'max_depth': 12, 'min_samples_split': 10, 'max_features': 'sqrt'}. Best is trial 0 with value: 0.33074134764072916.
  0%|                                                                                                                                                                                                 | 0/1 [00:05<?, ?it/s][14:23:07] rf_study trial#0 value=0.33074134764072916 params={'n_estimators': 339, 'max_depth': 12, 'min_samples_split': 10, 'max_features': 'sqrt'}
Best trial: 0. Best value: 0.330741: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.78s/it] 
=== RF: fertig ===

=== LGB: Optuna startet ===
[I 2025-08-13 14:24:02,210] A new study created in memory with name: lgb_study
[I 2025-08-13 14:24:09,638] Trial 0 finished with value: 0.31693893561476666 and parameters: {'learning_rate': 0.014516523739139702, 'num_leaves': 107, 'feature_fraction': 0.97477217382406, 'bagging_fraction': 0.57882550149857, 'num_boost_round': 347}. Best is trial 0 with value: 0.31693893561476666.
  0%|                                                                                                                                                                                                 | 0/1 [00:07<?, ?it/s][14:24:09] lgb_study trial#0 value=0.31693893561476666 params={'learning_rate': 0.014516523739139702, 'num_leaves': 107, 'feature_fraction': 0.97477217382406, 'bagging_fraction': 0.57882550149857, 'num_boost_round': 347}  
Best trial: 0. Best value: 0.316939: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:07<00:00,  7.43s/it] 
=== LGB: fertig ===

=== XGB: Optuna startet ===
[I 2025-08-13 14:24:48,270] A new study created in memory with name: xgb_study
  0%|                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s[ 
0]      val-logloss:0.68685
[10]    val-logloss:0.62828
[20]    val-logloss:0.58085
[30]    val-logloss:0.54166
[40]    val-logloss:0.50889
[50]    val-logloss:0.48137
[60]    val-logloss:0.45826
[70]    val-logloss:0.43846
[80]    val-logloss:0.42175
[90]    val-logloss:0.40731
[100]   val-logloss:0.39499
[110]   val-logloss:0.38430
[120]   val-logloss:0.37519
[130]   val-logloss:0.36720
[140]   val-logloss:0.36028
[150]   val-logloss:0.35437
[160]   val-logloss:0.34944
[170]   val-logloss:0.34504
[180]   val-logloss:0.34122
[190]   val-logloss:0.33791
[200]   val-logloss:0.33509
[210]   val-logloss:0.33256
[220]   val-logloss:0.33052
[230]   val-logloss:0.32859
[240]   val-logloss:0.32691
[250]   val-logloss:0.32530
[260]   val-logloss:0.32377
[270]   val-logloss:0.32242
[280]   val-logloss:0.32158
[290]   val-logloss:0.32070
[300]   val-logloss:0.31985
[310]   val-logloss:0.31928
[320]   val-logloss:0.31845
[330]   val-logloss:0.31783
[340]   val-logloss:0.31735
[350]   val-logloss:0.31683
[360]   val-logloss:0.31642
[370]   val-logloss:0.31601
[380]   val-logloss:0.31576
[390]   val-logloss:0.31552
[400]   val-logloss:0.31525
[410]   val-logloss:0.31487
[420]   val-logloss:0.31466
[430]   val-logloss:0.31450
[440]   val-logloss:0.31425
[450]   val-logloss:0.31444
[460]   val-logloss:0.31401
[470]   val-logloss:0.31405
[480]   val-logloss:0.31388
[490]   val-logloss:0.31375
[499]   val-logloss:0.31378
[I 2025-08-13 14:24:54,647] Trial 0 finished with value: 0.31375168638477335 and parameters: {'eta': 0.012236370837720197, 'max_depth': 5, 'subsample': 0.6766886369747924, 'colsample_bytree': 0.8812256615492199, 'lambda_l2': 0.16113894772022203}. Best is trial 0 with value: 0.31375168638477335.
  0%|                                                                                                                                                                                                 | 0/1 [00:06<?, ?it/s][14:24:54] xgb_study trial#0 value=0.31375168638477335 params={'eta': 0.012236370837720197, 'max_depth': 5, 'subsample': 0.6766886369747924, 'colsample_bytree': 0.8812256615492199, 'lambda_l2': 0.16113894772022203}        
Best trial: 0. Best value: 0.313752: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.38s/it] 
=== XGB: fertig ===

=== CNN: Optuna startet ===
[I 2025-08-13 14:25:26,386] A new study created in memory with name: cnn_study
  0%|                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\module.py:1158: UserWarning: expandable_segments not supported on this platform (Triggered internally at ..\c10\cuda\CUDACachingAllocator.cpp:803.)
  return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
[I 2025-08-13 14:25:27,150] Trial 0 finished with value: 3.9668169240436324 and parameters: {'lr': 0.003476232468808119, 'n_filters': 30}. Best is trial 0 with value: 3.9668169240436324.
  0%|                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s][14:25:27] cnn_study trial#0 value=3.9668169240436324 params={'lr': 0.003476232468808119, 'n_filters': 30}
Best trial: 0. Best value: 3.96682: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.31it/s] 
=== CNN: fertig ===

=== FT: Optuna startet ===
[I 2025-08-13 14:25:27,416] A new study created in memory with name: ft_study
{'loss': 0.2261, 'grad_norm': 0.004429894499480724, 'learning_rate': 0.00018663579168441775, 'epoch': 0.04}
{'loss': 0.2276, 'grad_norm': 0.016915133222937584, 'learning_rate': 0.0001849288610855114, 'epoch': 0.07}
{'loss': 0.2191, 'grad_norm': 0.018487831577658653, 'learning_rate': 0.0001832219304866051, 'epoch': 0.11}
{'loss': 0.2151, 'grad_norm': 0.014862538315355778, 'learning_rate': 0.00018151499988769876, 'epoch': 0.15}
{'loss': 0.2111, 'grad_norm': 0.020637163892388344, 'learning_rate': 0.00017980806928879245, 'epoch': 0.18}
{'loss': 0.202, 'grad_norm': 0.05404770001769066, 'learning_rate': 0.0001781011386898861, 'epoch': 0.22}
{'loss': 0.1943, 'grad_norm': 0.015003501437604427, 'learning_rate': 0.00017639420809097977, 'epoch': 0.25}
{'loss': 0.1979, 'grad_norm': 0.014108679257333279, 'learning_rate': 0.00017468727749207346, 'epoch': 0.29}
{'loss': 0.1952, 'grad_norm': 0.024708792567253113, 'learning_rate': 0.00017298034689316712, 'epoch': 0.33}
{'loss': 0.197, 'grad_norm': 0.020974893122911453, 'learning_rate': 0.0001712734162942608, 'epoch': 0.36}
{'loss': 0.1962, 'grad_norm': 0.01692037284374237, 'learning_rate': 0.00016956648569535447, 'epoch': 0.4}
{'loss': 0.1943, 'grad_norm': 0.03679809346795082, 'learning_rate': 0.00016785955509644816, 'epoch': 0.44}
{'loss': 0.1896, 'grad_norm': 0.023053936660289764, 'learning_rate': 0.00016615262449754182, 'epoch': 0.47}
{'loss': 0.1917, 'grad_norm': 0.025877829641103745, 'learning_rate': 0.0001644456938986355, 'epoch': 0.51}
{'loss': 0.1918, 'grad_norm': 0.02093193493783474, 'learning_rate': 0.00016273876329972917, 'epoch': 0.54}
{'loss': 0.1891, 'grad_norm': 0.024240443482995033, 'learning_rate': 0.00016103183270082283, 'epoch': 0.58}
{'loss': 0.1957, 'grad_norm': 0.02050587348639965, 'learning_rate': 0.00015932490210191652, 'epoch': 0.62}
{'loss': 0.1939, 'grad_norm': 0.03569320961833, 'learning_rate': 0.00015761797150301018, 'epoch': 0.65}
{'loss': 0.1996, 'grad_norm': 0.015869446098804474, 'learning_rate': 0.00015591104090410387, 'epoch': 0.69}
{'loss': 0.1979, 'grad_norm': 0.02277517504990101, 'learning_rate': 0.00015420411030519753, 'epoch': 0.73}
{'loss': 0.1971, 'grad_norm': 0.01619407720863819, 'learning_rate': 0.00015249717970629122, 'epoch': 0.76}
{'loss': 0.1981, 'grad_norm': 0.023974964395165443, 'learning_rate': 0.0001507902491073849, 'epoch': 0.8}
{'loss': 0.2029, 'grad_norm': 0.05419335141777992, 'learning_rate': 0.00014908331850847857, 'epoch': 0.83}
{'loss': 0.198, 'grad_norm': 0.0223961491137743, 'learning_rate': 0.00014737638790957223, 'epoch': 0.87}
{'loss': 0.1944, 'grad_norm': 0.030561713501811028, 'learning_rate': 0.0001456694573106659, 'epoch': 0.91}
{'loss': 0.1947, 'grad_norm': 0.057625338435173035, 'learning_rate': 0.00014396252671175958, 'epoch': 0.94}
{'loss': 0.188, 'grad_norm': 0.023631595075130463, 'learning_rate': 0.00014225559611285324, 'epoch': 0.98}
{'eval_loss': 0.1813015341758728, 'eval_runtime': 2.6757, 'eval_samples_per_second': 2061.15, 'eval_steps_per_second': 64.656, 'epoch': 1.0}
{'loss': 0.1924, 'grad_norm': 0.019444676116108894, 'learning_rate': 0.00014054866551394693, 'epoch': 1.02}
{'loss': 0.1885, 'grad_norm': 0.029582418501377106, 'learning_rate': 0.00013884173491504061, 'epoch': 1.05}
{'loss': 0.1915, 'grad_norm': 0.041891541332006454, 'learning_rate': 0.00013713480431613428, 'epoch': 1.09}
{'loss': 0.1985, 'grad_norm': 0.0347626768052578, 'learning_rate': 0.00013542787371722796, 'epoch': 1.12}
{'loss': 0.1955, 'grad_norm': 0.026603665202856064, 'learning_rate': 0.00013372094311832163, 'epoch': 1.16}
{'loss': 0.1921, 'grad_norm': 0.03263406082987785, 'learning_rate': 0.0001320140125194153, 'epoch': 1.2}
{'loss': 0.194, 'grad_norm': 0.018892696127295494, 'learning_rate': 0.00013030708192050895, 'epoch': 1.23}
{'loss': 0.1911, 'grad_norm': 0.030518868938088417, 'learning_rate': 0.00012860015132160264, 'epoch': 1.27}
{'loss': 0.191, 'grad_norm': 0.029868556186556816, 'learning_rate': 0.00012689322072269633, 'epoch': 1.31}
{'loss': 0.1839, 'grad_norm': 0.025685925036668777, 'learning_rate': 0.00012518629012378999, 'epoch': 1.34}
{'loss': 0.1933, 'grad_norm': 0.05146321654319763, 'learning_rate': 0.00012347935952488367, 'epoch': 1.38}
{'loss': 0.1882, 'grad_norm': 0.02952968329191208, 'learning_rate': 0.00012177242892597734, 'epoch': 1.41}
{'loss': 0.1903, 'grad_norm': 0.03663218766450882, 'learning_rate': 0.00012006549832707101, 'epoch': 1.45}
{'loss': 0.1961, 'grad_norm': 0.060082629323005676, 'learning_rate': 0.00011835856772816467, 'epoch': 1.49}
{'loss': 0.1944, 'grad_norm': 0.05692325904965401, 'learning_rate': 0.00011665163712925835, 'epoch': 1.52}
{'loss': 0.1909, 'grad_norm': 0.034321561455726624, 'learning_rate': 0.00011494470653035202, 'epoch': 1.56}
{'loss': 0.1849, 'grad_norm': 0.04183581843972206, 'learning_rate': 0.0001132377759314457, 'epoch': 1.6}
{'loss': 0.1917, 'grad_norm': 0.06986969709396362, 'learning_rate': 0.00011153084533253937, 'epoch': 1.63}
{'loss': 0.1876, 'grad_norm': 0.04129866883158684, 'learning_rate': 0.00010982391473363305, 'epoch': 1.67}
{'loss': 0.1905, 'grad_norm': 0.0467582568526268, 'learning_rate': 0.00010811698413472672, 'epoch': 1.7}
{'loss': 0.1874, 'grad_norm': 0.04933348670601845, 'learning_rate': 0.0001064100535358204, 'epoch': 1.74}
{'loss': 0.1865, 'grad_norm': 0.042765114456415176, 'learning_rate': 0.00010470312293691408, 'epoch': 1.78}
{'loss': 0.1808, 'grad_norm': 0.033005110919475555, 'learning_rate': 0.00010299619233800773, 'epoch': 1.81}
{'loss': 0.1817, 'grad_norm': 0.04864885285496712, 'learning_rate': 0.0001012892617391014, 'epoch': 1.85}
{'loss': 0.188, 'grad_norm': 0.07118485122919083, 'learning_rate': 9.958233114019508e-05, 'epoch': 1.89}
{'loss': 0.1888, 'grad_norm': 0.07381638139486313, 'learning_rate': 9.787540054128876e-05, 'epoch': 1.92}
{'loss': 0.1828, 'grad_norm': 0.03565515577793121, 'learning_rate': 9.616846994238243e-05, 'epoch': 1.96}
{'loss': 0.1865, 'grad_norm': 0.05956994742155075, 'learning_rate': 9.44615393434761e-05, 'epoch': 1.99}
{'eval_loss': 0.1813322901725769, 'eval_runtime': 2.6986, 'eval_samples_per_second': 2043.628, 'eval_steps_per_second': 64.107, 'epoch': 2.0}
{'loss': 0.1831, 'grad_norm': 0.04468066990375519, 'learning_rate': 9.27546087445698e-05, 'epoch': 2.03}
{'loss': 0.1845, 'grad_norm': 0.05234445631504059, 'learning_rate': 9.104767814566346e-05, 'epoch': 2.07}
{'loss': 0.1866, 'grad_norm': 0.03718015179038048, 'learning_rate': 8.934074754675713e-05, 'epoch': 2.1}
{'loss': 0.1862, 'grad_norm': 0.04761454835534096, 'learning_rate': 8.76338169478508e-05, 'epoch': 2.14}
{'loss': 0.1865, 'grad_norm': 0.03894756734371185, 'learning_rate': 8.592688634894448e-05, 'epoch': 2.18}
{'loss': 0.183, 'grad_norm': 0.038056693971157074, 'learning_rate': 8.421995575003814e-05, 'epoch': 2.21}
{'loss': 0.1853, 'grad_norm': 0.05046474188566208, 'learning_rate': 8.251302515113182e-05, 'epoch': 2.25}
{'loss': 0.1872, 'grad_norm': 0.08462359011173248, 'learning_rate': 8.08060945522255e-05, 'epoch': 2.28}
{'loss': 0.1816, 'grad_norm': 0.07471771538257599, 'learning_rate': 7.909916395331918e-05, 'epoch': 2.32}
{'loss': 0.184, 'grad_norm': 0.08493302762508392, 'learning_rate': 7.739223335441285e-05, 'epoch': 2.36}
{'loss': 0.1823, 'grad_norm': 0.04034212604165077, 'learning_rate': 7.568530275550651e-05, 'epoch': 2.39}
{'loss': 0.1824, 'grad_norm': 0.06116552650928497, 'learning_rate': 7.397837215660019e-05, 'epoch': 2.43}
{'loss': 0.1825, 'grad_norm': 0.1041119322180748, 'learning_rate': 7.227144155769386e-05, 'epoch': 2.47}
{'loss': 0.1779, 'grad_norm': 0.04690788686275482, 'learning_rate': 7.056451095878754e-05, 'epoch': 2.5}
{'loss': 0.1781, 'grad_norm': 0.0535496361553669, 'learning_rate': 6.88575803598812e-05, 'epoch': 2.54}
{'loss': 0.1795, 'grad_norm': 0.07394973188638687, 'learning_rate': 6.715064976097489e-05, 'epoch': 2.57}
{'loss': 0.1811, 'grad_norm': 0.043131597340106964, 'learning_rate': 6.544371916206856e-05, 'epoch': 2.61}
{'loss': 0.1821, 'grad_norm': 0.0640927255153656, 'learning_rate': 6.373678856316224e-05, 'epoch': 2.65}
{'loss': 0.1841, 'grad_norm': 0.06145460531115532, 'learning_rate': 6.202985796425591e-05, 'epoch': 2.68}
{'loss': 0.1811, 'grad_norm': 0.06676965206861496, 'learning_rate': 6.0322927365349575e-05, 'epoch': 2.72}
{'loss': 0.1828, 'grad_norm': 0.09072937816381454, 'learning_rate': 5.861599676644325e-05, 'epoch': 2.76}
{'loss': 0.1858, 'grad_norm': 0.05474850535392761, 'learning_rate': 5.6909066167536924e-05, 'epoch': 2.79}
{'loss': 0.1859, 'grad_norm': 0.04487064108252525, 'learning_rate': 5.5202135568630606e-05, 'epoch': 2.83}
{'loss': 0.1807, 'grad_norm': 0.04827384650707245, 'learning_rate': 5.349520496972427e-05, 'epoch': 2.86}
{'loss': 0.1865, 'grad_norm': 0.04911255091428757, 'learning_rate': 5.178827437081794e-05, 'epoch': 2.9}
{'loss': 0.1832, 'grad_norm': 0.035611432045698166, 'learning_rate': 5.008134377191162e-05, 'epoch': 2.94}
{'loss': 0.183, 'grad_norm': 0.0614934079349041, 'learning_rate': 4.83744131730053e-05, 'epoch': 2.97}
{'eval_loss': 0.1808575689792633, 'eval_runtime': 2.7016, 'eval_samples_per_second': 2041.349, 'eval_steps_per_second': 64.035, 'epoch': 3.0}
{'loss': 0.1845, 'grad_norm': 0.055401138961315155, 'learning_rate': 4.6667482574098966e-05, 'epoch': 3.01}
{'loss': 0.1813, 'grad_norm': 0.06131391227245331, 'learning_rate': 4.496055197519264e-05, 'epoch': 3.05}
{'loss': 0.1826, 'grad_norm': 0.0248765479773283, 'learning_rate': 4.3253621376286316e-05, 'epoch': 3.08}
{'loss': 0.185, 'grad_norm': 0.056590624153614044, 'learning_rate': 4.154669077737999e-05, 'epoch': 3.12}
{'loss': 0.1818, 'grad_norm': 0.06034496799111366, 'learning_rate': 3.983976017847366e-05, 'epoch': 3.15}
{'loss': 0.1801, 'grad_norm': 0.06134390830993652, 'learning_rate': 3.8132829579567333e-05, 'epoch': 3.19}
{'loss': 0.1779, 'grad_norm': 0.04574226960539818, 'learning_rate': 3.642589898066101e-05, 'epoch': 3.23}
{'loss': 0.1806, 'grad_norm': 0.031809233129024506, 'learning_rate': 3.471896838175468e-05, 'epoch': 3.26}
{'loss': 0.1807, 'grad_norm': 0.052777621895074844, 'learning_rate': 3.301203778284835e-05, 'epoch': 3.3}
{'loss': 0.1811, 'grad_norm': 0.0372450053691864, 'learning_rate': 3.1305107183942026e-05, 'epoch': 3.34}
{'loss': 0.1793, 'grad_norm': 0.09947455674409866, 'learning_rate': 2.9598176585035704e-05, 'epoch': 3.37}
{'loss': 0.181, 'grad_norm': 0.051322150975465775, 'learning_rate': 2.7891245986129375e-05, 'epoch': 3.41}
{'loss': 0.1794, 'grad_norm': 0.038184598088264465, 'learning_rate': 2.618431538722305e-05, 'epoch': 3.44}
{'loss': 0.1775, 'grad_norm': 0.03392966091632843, 'learning_rate': 2.447738478831672e-05, 'epoch': 3.48}
{'loss': 0.1803, 'grad_norm': 0.052270177751779556, 'learning_rate': 2.2770454189410396e-05, 'epoch': 3.52}
{'loss': 0.1792, 'grad_norm': 0.05412457510828972, 'learning_rate': 2.106352359050407e-05, 'epoch': 3.55}
{'loss': 0.18, 'grad_norm': 0.05872975289821625, 'learning_rate': 1.9356592991597743e-05, 'epoch': 3.59}
{'loss': 0.18, 'grad_norm': 0.039455633610486984, 'learning_rate': 1.7649662392691417e-05, 'epoch': 3.63}
{'loss': 0.1799, 'grad_norm': 0.12051897495985031, 'learning_rate': 1.594273179378509e-05, 'epoch': 3.66}
{'loss': 0.1791, 'grad_norm': 0.06972835212945938, 'learning_rate': 1.4235801194878764e-05, 'epoch': 3.7}
{'loss': 0.1782, 'grad_norm': 0.037383776158094406, 'learning_rate': 1.2528870595972437e-05, 'epoch': 3.73}
{'loss': 0.185, 'grad_norm': 0.04528043046593666, 'learning_rate': 1.082193999706611e-05, 'epoch': 3.77}
{'loss': 0.181, 'grad_norm': 0.048771802335977554, 'learning_rate': 9.115009398159785e-06, 'epoch': 3.81}
{'loss': 0.1785, 'grad_norm': 0.04386158660054207, 'learning_rate': 7.408078799253457e-06, 'epoch': 3.84}
{'loss': 0.1846, 'grad_norm': 0.06577583402395248, 'learning_rate': 5.70114820034713e-06, 'epoch': 3.88}
{'loss': 0.1807, 'grad_norm': 0.06722797453403473, 'learning_rate': 3.994217601440804e-06, 'epoch': 3.92}
{'loss': 0.1756, 'grad_norm': 0.08009912073612213, 'learning_rate': 2.2872870025344775e-06, 'epoch': 3.95}
{'loss': 0.1811, 'grad_norm': 0.033215709030628204, 'learning_rate': 5.80356403628151e-07, 'epoch': 3.99}
{'eval_loss': 0.17717307806015015, 'eval_runtime': 2.7416, 'eval_samples_per_second': 2011.587, 'eval_steps_per_second': 63.101, 'epoch': 4.0}
{'train_runtime': 133.8687, 'train_samples_per_second': 659.063, 'train_steps_per_second': 41.205, 'train_loss': 0.1882504811314589, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5516/5516 [02:13<00:00, 41.20it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:02<00:00, 63.45it/s]
[I 2025-08-13 14:27:44,488] Trial 0 finished with value: 0.17717307806015015 and parameters: {'lr': 0.00018830858367134594, 'n_blocks': 4}. Best is trial 0 with value: 0.17717307806015015.
  0%|                                                                                                                                                                                                 | 0/1 [02:17<?, ?it/s][14:27:44] ft_study trial#0 value=0.17717307806015015 params={'lr': 0.00018830858367134594, 'n_blocks': 4}
Best trial: 0. Best value: 0.177173: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:17<00:00, 137.07s/it] 
{'loss': 0.175, 'grad_norm': 0.038536570966243744, 'learning_rate': 0.00018149550483926926, 'epoch': 0.29}
{'loss': 0.1781, 'grad_norm': 0.01920286938548088, 'learning_rate': 0.00017466877254259923, 'epoch': 0.58}
{'loss': 0.1798, 'grad_norm': 0.030893728137016296, 'learning_rate': 0.00016784204024592923, 'epoch': 0.87}
{'loss': 0.1807, 'grad_norm': 0.017900127917528152, 'learning_rate': 0.0001610153079492592, 'epoch': 1.16}
{'loss': 0.179, 'grad_norm': 0.03161802142858505, 'learning_rate': 0.00015418857565258916, 'epoch': 1.45}
{'loss': 0.1824, 'grad_norm': 0.05158458650112152, 'learning_rate': 0.00014736184335591913, 'epoch': 1.74}
{'loss': 0.1813, 'grad_norm': 0.04348459839820862, 'learning_rate': 0.0001405351110592491, 'epoch': 2.03}
{'loss': 0.1785, 'grad_norm': 0.03418709710240364, 'learning_rate': 0.0001337083787625791, 'epoch': 2.32}
{'loss': 0.1816, 'grad_norm': 0.05459701269865036, 'learning_rate': 0.00012688164646590906, 'epoch': 2.61}
{'loss': 0.1819, 'grad_norm': 0.0399433970451355, 'learning_rate': 0.00012005491416923905, 'epoch': 2.9}
{'loss': 0.1815, 'grad_norm': 0.051138099282979965, 'learning_rate': 0.00011322818187256903, 'epoch': 3.19}
{'loss': 0.1814, 'grad_norm': 0.04071744158864021, 'learning_rate': 0.000106401449575899, 'epoch': 3.48}
{'loss': 0.1785, 'grad_norm': 0.0502961203455925, 'learning_rate': 9.957471727922897e-05, 'epoch': 3.77}
{'loss': 0.1822, 'grad_norm': 0.041250333189964294, 'learning_rate': 9.274798498255895e-05, 'epoch': 4.06}
{'loss': 0.1843, 'grad_norm': 0.05188198387622833, 'learning_rate': 8.592125268588892e-05, 'epoch': 4.35}
{'loss': 0.1791, 'grad_norm': 0.07360602915287018, 'learning_rate': 7.90945203892189e-05, 'epoch': 4.64}
{'loss': 0.1778, 'grad_norm': 0.07315852493047714, 'learning_rate': 7.226778809254888e-05, 'epoch': 4.93}
{'loss': 0.178, 'grad_norm': 0.03444605693221092, 'learning_rate': 6.544105579587885e-05, 'epoch': 5.22}
{'loss': 0.1809, 'grad_norm': 0.07972709834575653, 'learning_rate': 5.8614323499208824e-05, 'epoch': 5.51}
{'loss': 0.1835, 'grad_norm': 0.047864168882369995, 'learning_rate': 5.17875912025388e-05, 'epoch': 5.8}
{'loss': 0.1758, 'grad_norm': 0.06342171132564545, 'learning_rate': 4.496085890586878e-05, 'epoch': 6.09}
{'loss': 0.1791, 'grad_norm': 0.1166735365986824, 'learning_rate': 3.813412660919875e-05, 'epoch': 6.38}
{'loss': 0.1791, 'grad_norm': 0.07105131447315216, 'learning_rate': 3.130739431252873e-05, 'epoch': 6.67}
{'loss': 0.1782, 'grad_norm': 0.02180059812963009, 'learning_rate': 2.4480662015858703e-05, 'epoch': 6.96}
{'loss': 0.1774, 'grad_norm': 0.05901414901018143, 'learning_rate': 1.7653929719188682e-05, 'epoch': 7.25}
{'loss': 0.1781, 'grad_norm': 0.09112218022346497, 'learning_rate': 1.0827197422518657e-05, 'epoch': 7.54}
{'loss': 0.1785, 'grad_norm': 0.06517618894577026, 'learning_rate': 4.000465125848634e-06, 'epoch': 7.83}
{'train_runtime': 312.4662, 'train_samples_per_second': 705.919, 'train_steps_per_second': 44.139, 'train_loss': 0.17967415575084997, 'epoch': 8.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13792/13792 [05:12<00:00, 44.14it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.47}
{'train_runtime': 19.9345, 'train_samples_per_second': 922.421, 'train_steps_per_second': 57.789, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1152/1152 [00:19<00:00, 57.79it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 0.87}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 2.61}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'train_runtime': 39.0194, 'train_samples_per_second': 942.3, 'train_steps_per_second': 58.945, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2300/2300 [00:39<00:00, 58.94it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 0.58}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 1.16}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 2.32}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 2.9}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'train_runtime': 57.5382, 'train_samples_per_second': 958.459, 'train_steps_per_second': 59.925, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3448/3448 [00:57<00:00, 59.93it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 0.44}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 0.87}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 1.31}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 2.18}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 2.61}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.05}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.92}
{'train_runtime': 77.0773, 'train_samples_per_second': 953.951, 'train_steps_per_second': 59.628, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4596/4596 [01:17<00:00, 59.63it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 0.35}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 0.7}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 1.04}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 1.39}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 2.09}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 2.44}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 2.78}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.13}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [01:36<00:00, 59.68it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [01:36<00:00, 59.68it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [01:36<00:00, 59.68it/s] 
=== FT: fertig ===
[I 2025-08-13 14:39:09,813] A new study created in memory with name: meta_study
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [01:36<00:00, 59.68it/s] 
=== FT: fertig ===
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [01:36<00:00, 59.68it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [01:36<00:00, 59.68it/s] 
=== FT: fertig ===
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [01:36<00:00, 59.68it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018830858367134594, 'epoch': 3.83}
{'train_runtime': 96.312, 'train_samples_per_second': 954.274, 'train_steps_per_second': 59.681, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [01:36<00:00, 59.68it/s] 
=== FT: fertig ===
[I 2025-08-13 14:39:09,813] A new study created in memory with name: meta_study
[I 2025-08-13 14:39:25,371] Trial 0 finished with value: 0.7593175691787885 and parameters: {'d_token': 96, 'dropout': 0.11890408620122697, 'lr': 0.0001034394944889572}. Best is trial 0 with value: 0.7593175691787885.    
  0%|                                                                                                                                                                                                 | 0/1 [00:15<?, ?it/s][14:39:25] meta_study trial#0 value=0.7593175691787885 params={'d_token': 96, 'dropout': 0.11890408620122697, 'lr': 0.0001034394944889572}
Best trial: 0. Best value: 0.759318: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:15<00:00, 15.56s/it] 
✅ Modelle gespeichert → models\hybrid_longtrend_20250813_142301
✅  Training abgeschlossen.
(ml-env) PS C:\trainers> 
