also wir kommen jetzt endlich bis hier hin(siehe log im anschluss) allerdings läuft (warscheinlich in der train_meta phase) der arbeitsspeicher voll und der computer friert ein selbst nach 3 stunden keine besserung wir müssen hier die train batches verkleinern damit der arbeitsspeicher nicht voll läuft oder fällt dir was besseres ein? recherchire online und mach einen diff patch in dem du die arbeitspeichersparende train_meta variante einbaust







{'loss': 0.116, 'grad_norm': 0.02127949520945549, 'learning_rate': 5.8664387890512585e-06, 'epoch': 2.86}
{'loss': 0.1197, 'grad_norm': 0.017748316749930382, 'learning_rate': 5.6792518461970385e-06, 'epoch': 2.9}
{'loss': 0.1247, 'grad_norm': 0.02008732594549656, 'learning_rate': 5.4920649033428185e-06, 'epoch': 2.94}
{'loss': 0.1216, 'grad_norm': 0.01747085154056549, 'learning_rate': 5.304877960488599e-06, 'epoch': 2.97}
{'eval_loss': 0.1311730146408081, 'eval_runtime': 6.5201, 'eval_samples_per_second': 845.841, 'eval_steps_per_second': 26.533, 'epoch': 3.0}
{'train_runtime': 225.3153, 'train_samples_per_second': 391.576, 'train_steps_per_second': 24.481, 'train_loss': 0.11015076910438013, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [03:45<01:15, 18.36it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:06<00:00, 25.38it/s]
[I 2025-08-12 16:24:42,075] Trial 10 finished with value: 0.10707330703735352 and parameters: {'lr': 2.0650463535677565e-05, 'n_blocks': 3}. Best is trial 10 with value: 0.10707330703735352.
Best trial: 6. Best value: 0.109823:  50%|██████████████████████████████████████████████████████████████████████▌                                                                      | 10/20 [1:17:19<1:19:47, 478.73s/it][16:24:42] ft_study trial#10 value=0.10707330703735352 params={'lr': 2.0650463535677565e-05, 'n_blocks': 3}
{'loss': 0.1698, 'grad_norm': 0.003823998384177685, 'learning_rate': 2.6673035135963194e-05, 'epoch': 0.04}
{'loss': 0.1674, 'grad_norm': 0.0013281560968607664, 'learning_rate': 2.642908932348868e-05, 'epoch': 0.07}
{'loss': 0.1685, 'grad_norm': 0.014045720919966698, 'learning_rate': 2.618514351101417e-05, 'epoch': 0.11}
{'loss': 0.1755, 'grad_norm': 0.0015973179833963513, 'learning_rate': 2.5941197698539656e-05, 'epoch': 0.15}
{'loss': 0.1667, 'grad_norm': 0.0019628119189292192, 'learning_rate': 2.5697251886065145e-05, 'epoch': 0.18}
{'loss': 0.1641, 'grad_norm': 0.003292680252343416, 'learning_rate': 2.545330607359063e-05, 'epoch': 0.22}
{'loss': 0.1685, 'grad_norm': 0.002810648176819086, 'learning_rate': 2.5209360261116117e-05, 'epoch': 0.25}
{'loss': 0.1623, 'grad_norm': 0.00125778466463089, 'learning_rate': 2.4965414448641607e-05, 'epoch': 0.29}
{'loss': 0.1684, 'grad_norm': 0.0010421654442325234, 'learning_rate': 2.4721468636167093e-05, 'epoch': 0.33}
{'loss': 0.1686, 'grad_norm': 0.0025985599495470524, 'learning_rate': 2.4477522823692583e-05, 'epoch': 0.36}
{'loss': 0.1666, 'grad_norm': 0.003423956921324134, 'learning_rate': 2.423357701121807e-05, 'epoch': 0.4}
{'loss': 0.169, 'grad_norm': 0.0020499685779213905, 'learning_rate': 2.3989631198743558e-05, 'epoch': 0.44}
{'loss': 0.1644, 'grad_norm': 0.0022462152410298586, 'learning_rate': 2.3745685386269044e-05, 'epoch': 0.47}
{'loss': 0.1681, 'grad_norm': 0.005004641599953175, 'learning_rate': 2.3501739573794534e-05, 'epoch': 0.51}
{'loss': 0.1618, 'grad_norm': 0.008818875066936016, 'learning_rate': 2.3257793761320017e-05, 'epoch': 0.54}
{'loss': 0.1721, 'grad_norm': 0.0032470794394612312, 'learning_rate': 2.3013847948845506e-05, 'epoch': 0.58}
{'loss': 0.1667, 'grad_norm': 0.0037902272306382656, 'learning_rate': 2.2769902136370992e-05, 'epoch': 0.62}
{'loss': 0.1665, 'grad_norm': 0.006932093296200037, 'learning_rate': 2.2525956323896482e-05, 'epoch': 0.65}
{'loss': 0.1655, 'grad_norm': 0.0044281757436692715, 'learning_rate': 2.2282010511421968e-05, 'epoch': 0.69}
{'loss': 0.1663, 'grad_norm': 0.00884674396365881, 'learning_rate': 2.2038064698947458e-05, 'epoch': 0.73}
{'loss': 0.1659, 'grad_norm': 0.00488657783716917, 'learning_rate': 2.1794118886472944e-05, 'epoch': 0.76}
{'loss': 0.1744, 'grad_norm': 0.010033167898654938, 'learning_rate': 2.1550173073998433e-05, 'epoch': 0.8}
{'loss': 0.1674, 'grad_norm': 0.006027637980878353, 'learning_rate': 2.130622726152392e-05, 'epoch': 0.83}
{'loss': 0.1688, 'grad_norm': 0.003909028600901365, 'learning_rate': 2.1062281449049406e-05, 'epoch': 0.87}
{'loss': 0.1644, 'grad_norm': 0.004436292219907045, 'learning_rate': 2.0818335636574892e-05, 'epoch': 0.91}
{'loss': 0.1699, 'grad_norm': 0.01610715501010418, 'learning_rate': 2.057438982410038e-05, 'epoch': 0.94}
{'loss': 0.1623, 'grad_norm': 0.004431666806340218, 'learning_rate': 2.0330444011625867e-05, 'epoch': 0.98}
{'eval_loss': 0.17384466528892517, 'eval_runtime': 7.1322, 'eval_samples_per_second': 773.249, 'eval_steps_per_second': 24.256, 'epoch': 1.0}
{'loss': 0.168, 'grad_norm': 0.006458386313170195, 'learning_rate': 2.0086498199151357e-05, 'epoch': 1.02}
{'loss': 0.1674, 'grad_norm': 0.007191732991486788, 'learning_rate': 1.9842552386676847e-05, 'epoch': 1.05}
{'loss': 0.1741, 'grad_norm': 0.006484450306743383, 'learning_rate': 1.9598606574202333e-05, 'epoch': 1.09}
{'loss': 0.1658, 'grad_norm': 0.007162653375416994, 'learning_rate': 1.9354660761727822e-05, 'epoch': 1.12}
{'loss': 0.1714, 'grad_norm': 0.005263091530650854, 'learning_rate': 1.911071494925331e-05, 'epoch': 1.16}
{'loss': 0.1668, 'grad_norm': 0.011574916541576385, 'learning_rate': 1.8866769136778794e-05, 'epoch': 1.2}
{'loss': 0.1746, 'grad_norm': 0.0033962884917855263, 'learning_rate': 1.862282332430428e-05, 'epoch': 1.23}
{'loss': 0.1652, 'grad_norm': 0.0060912612825632095, 'learning_rate': 1.837887751182977e-05, 'epoch': 1.27}
{'loss': 0.1653, 'grad_norm': 0.0094583909958601, 'learning_rate': 1.8134931699355256e-05, 'epoch': 1.31}
{'loss': 0.1705, 'grad_norm': 0.007320611272007227, 'learning_rate': 1.7890985886880746e-05, 'epoch': 1.34}
{'loss': 0.169, 'grad_norm': 0.011483502574265003, 'learning_rate': 1.7647040074406232e-05, 'epoch': 1.38}
{'loss': 0.1669, 'grad_norm': 0.0062040844932198524, 'learning_rate': 1.740309426193172e-05, 'epoch': 1.41}
{'loss': 0.1702, 'grad_norm': 0.011128270998597145, 'learning_rate': 1.7159148449457208e-05, 'epoch': 1.45}
{'loss': 0.1668, 'grad_norm': 0.009892698377370834, 'learning_rate': 1.6915202636982694e-05, 'epoch': 1.49}
{'loss': 0.1671, 'grad_norm': 0.008944308385252953, 'learning_rate': 1.667125682450818e-05, 'epoch': 1.52}
{'loss': 0.164, 'grad_norm': 0.012524101883172989, 'learning_rate': 1.642731101203367e-05, 'epoch': 1.56}
{'loss': 0.1681, 'grad_norm': 0.006805729120969772, 'learning_rate': 1.6183365199559156e-05, 'epoch': 1.6}
{'loss': 0.1651, 'grad_norm': 0.0077547235414385796, 'learning_rate': 1.5939419387084645e-05, 'epoch': 1.63}
{'loss': 0.1683, 'grad_norm': 0.007617291063070297, 'learning_rate': 1.569547357461013e-05, 'epoch': 1.67}
{'loss': 0.1676, 'grad_norm': 0.005941971205174923, 'learning_rate': 1.545152776213562e-05, 'epoch': 1.7}
{'loss': 0.1701, 'grad_norm': 0.009003055281937122, 'learning_rate': 1.5207581949661109e-05, 'epoch': 1.74}
{'loss': 0.1669, 'grad_norm': 0.01388612762093544, 'learning_rate': 1.4963636137186597e-05, 'epoch': 1.78}
{'loss': 0.1729, 'grad_norm': 0.012940037995576859, 'learning_rate': 1.4719690324712081e-05, 'epoch': 1.81}
{'loss': 0.1666, 'grad_norm': 0.007988519966602325, 'learning_rate': 1.4475744512237569e-05, 'epoch': 1.85}
{'loss': 0.1732, 'grad_norm': 0.012711605057120323, 'learning_rate': 1.4231798699763057e-05, 'epoch': 1.89}
{'loss': 0.1674, 'grad_norm': 0.013744529336690903, 'learning_rate': 1.3987852887288545e-05, 'epoch': 1.92}
{'loss': 0.1703, 'grad_norm': 0.00975741259753704, 'learning_rate': 1.3743907074814032e-05, 'epoch': 1.96}
{'loss': 0.1713, 'grad_norm': 0.009277832694351673, 'learning_rate': 1.349996126233952e-05, 'epoch': 1.99}
{'eval_loss': 0.17513160407543182, 'eval_runtime': 6.2906, 'eval_samples_per_second': 876.705, 'eval_steps_per_second': 27.501, 'epoch': 2.0}
{'loss': 0.1674, 'grad_norm': 0.00849941000342369, 'learning_rate': 1.3256015449865008e-05, 'epoch': 2.03}
{'loss': 0.1685, 'grad_norm': 0.007187620736658573, 'learning_rate': 1.3012069637390494e-05, 'epoch': 2.07}
{'loss': 0.1718, 'grad_norm': 0.009485996328294277, 'learning_rate': 1.2768123824915982e-05, 'epoch': 2.1}
{'loss': 0.1699, 'grad_norm': 0.008734784089028835, 'learning_rate': 1.252417801244147e-05, 'epoch': 2.14}
{'loss': 0.1722, 'grad_norm': 0.0068236589431762695, 'learning_rate': 1.2280232199966958e-05, 'epoch': 2.18}
{'loss': 0.1706, 'grad_norm': 0.009424284100532532, 'learning_rate': 1.2036286387492444e-05, 'epoch': 2.21}
{'loss': 0.1629, 'grad_norm': 0.022921953350305557, 'learning_rate': 1.1792340575017932e-05, 'epoch': 2.25}
{'loss': 0.1695, 'grad_norm': 0.006109787151217461, 'learning_rate': 1.154839476254342e-05, 'epoch': 2.28}
{'loss': 0.1643, 'grad_norm': 0.006130622699856758, 'learning_rate': 1.1304448950068907e-05, 'epoch': 2.32}
{'loss': 0.1667, 'grad_norm': 0.01234290562570095, 'learning_rate': 1.1060503137594395e-05, 'epoch': 2.36}
{'loss': 0.1725, 'grad_norm': 0.006783014629036188, 'learning_rate': 1.0816557325119883e-05, 'epoch': 2.39}
{'loss': 0.1693, 'grad_norm': 0.012811644934117794, 'learning_rate': 1.0572611512645371e-05, 'epoch': 2.43}
{'loss': 0.1724, 'grad_norm': 0.011238830164074898, 'learning_rate': 1.0328665700170859e-05, 'epoch': 2.47}
{'loss': 0.1736, 'grad_norm': 0.009137644432485104, 'learning_rate': 1.0084719887696347e-05, 'epoch': 2.5}
{'loss': 0.1668, 'grad_norm': 0.005485036410391331, 'learning_rate': 9.840774075221833e-06, 'epoch': 2.54}
{'loss': 0.1727, 'grad_norm': 0.012972933240234852, 'learning_rate': 9.59682826274732e-06, 'epoch': 2.57}
{'loss': 0.1695, 'grad_norm': 0.007868612185120583, 'learning_rate': 9.352882450272808e-06, 'epoch': 2.61}
{'loss': 0.1692, 'grad_norm': 0.007305152248591185, 'learning_rate': 9.108936637798296e-06, 'epoch': 2.65}
{'loss': 0.1699, 'grad_norm': 0.008151272311806679, 'learning_rate': 8.864990825323784e-06, 'epoch': 2.68}
{'loss': 0.1682, 'grad_norm': 0.007355810608714819, 'learning_rate': 8.62104501284927e-06, 'epoch': 2.72}
{'loss': 0.1684, 'grad_norm': 0.022511545568704605, 'learning_rate': 8.377099200374758e-06, 'epoch': 2.76}
{'loss': 0.1685, 'grad_norm': 0.03066411428153515, 'learning_rate': 8.133153387900246e-06, 'epoch': 2.79}
{'loss': 0.1713, 'grad_norm': 0.016781706362962723, 'learning_rate': 7.889207575425734e-06, 'epoch': 2.83}
{'loss': 0.1759, 'grad_norm': 0.009458056651055813, 'learning_rate': 7.64526176295122e-06, 'epoch': 2.86}
{'loss': 0.1745, 'grad_norm': 0.009080014191567898, 'learning_rate': 7.401315950476708e-06, 'epoch': 2.9}
{'loss': 0.1682, 'grad_norm': 0.007245362736284733, 'learning_rate': 7.157370138002196e-06, 'epoch': 2.94}
{'loss': 0.1737, 'grad_norm': 0.018835246562957764, 'learning_rate': 6.9134243255276835e-06, 'epoch': 2.97}
{'eval_loss': 0.17627213895320892, 'eval_runtime': 6.6862, 'eval_samples_per_second': 824.837, 'eval_steps_per_second': 25.874, 'epoch': 3.0}
{'train_runtime': 229.4755, 'train_samples_per_second': 384.477, 'train_steps_per_second': 24.037, 'train_loss': 0.16868259009457395, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [03:49<01:16, 18.03it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:07<00:00, 24.55it/s]
[I 2025-08-12 16:28:39,046] Trial 11 finished with value: 0.17384466528892517 and parameters: {'lr': 2.6912102032188216e-05, 'n_blocks': 3}. Best is trial 10 with value: 0.10707330703735352.
Best trial: 10. Best value: 0.107073:  55%|█████████████████████████████████████████████████████████████████████████████                                                               | 11/20 [1:21:16<1:00:30, 403.40s/it][16:28:39] ft_study trial#11 value=0.17384466528892517 params={'lr': 2.6912102032188216e-05, 'n_blocks': 3}
{'loss': 0.1698, 'grad_norm': 0.0038241061847656965, 'learning_rate': 2.76077541184118e-05, 'epoch': 0.04}
{'loss': 0.1674, 'grad_norm': 0.0013321583392098546, 'learning_rate': 2.735525956821597e-05, 'epoch': 0.07}
{'loss': 0.1685, 'grad_norm': 0.014060103334486485, 'learning_rate': 2.7102765018020143e-05, 'epoch': 0.11}
{'loss': 0.1755, 'grad_norm': 0.0016016956651583314, 'learning_rate': 2.6850270467824314e-05, 'epoch': 0.15}
{'loss': 0.1667, 'grad_norm': 0.0019713607616722584, 'learning_rate': 2.6597775917628488e-05, 'epoch': 0.18}
{'loss': 0.1641, 'grad_norm': 0.003316090675070882, 'learning_rate': 2.634528136743266e-05, 'epoch': 0.22}
{'loss': 0.1685, 'grad_norm': 0.002829756820574403, 'learning_rate': 2.609278681723683e-05, 'epoch': 0.25}
{'loss': 0.1623, 'grad_norm': 0.0012691837036982179, 'learning_rate': 2.5840292267041003e-05, 'epoch': 0.29}
{'loss': 0.1684, 'grad_norm': 0.0010506479302421212, 'learning_rate': 2.5587797716845177e-05, 'epoch': 0.33}
{'loss': 0.1686, 'grad_norm': 0.0026265850756317377, 'learning_rate': 2.5335303166649347e-05, 'epoch': 0.36}
{'loss': 0.1666, 'grad_norm': 0.003446929156780243, 'learning_rate': 2.508280861645352e-05, 'epoch': 0.4}
{'loss': 0.169, 'grad_norm': 0.0020747894886881113, 'learning_rate': 2.4830314066257692e-05, 'epoch': 0.44}
{'loss': 0.1644, 'grad_norm': 0.0022729611955583096, 'learning_rate': 2.4577819516061866e-05, 'epoch': 0.47}
{'loss': 0.1681, 'grad_norm': 0.00504842447116971, 'learning_rate': 2.432532496586604e-05, 'epoch': 0.51}
{'loss': 0.1618, 'grad_norm': 0.00891305785626173, 'learning_rate': 2.4072830415670207e-05, 'epoch': 0.54}
{'loss': 0.1721, 'grad_norm': 0.003289490006864071, 'learning_rate': 2.382033586547438e-05, 'epoch': 0.58}
{'loss': 0.1667, 'grad_norm': 0.003847916144877672, 'learning_rate': 2.3567841315278555e-05, 'epoch': 0.62}
{'loss': 0.1666, 'grad_norm': 0.007021484896540642, 'learning_rate': 2.3315346765082726e-05, 'epoch': 0.65}
{'loss': 0.1655, 'grad_norm': 0.004496174864470959, 'learning_rate': 2.30628522148869e-05, 'epoch': 0.69}
{'loss': 0.1663, 'grad_norm': 0.00900053046643734, 'learning_rate': 2.281035766469107e-05, 'epoch': 0.73}
{'loss': 0.166, 'grad_norm': 0.004931005649268627, 'learning_rate': 2.2557863114495244e-05, 'epoch': 0.76}
{'loss': 0.1744, 'grad_norm': 0.010258825495839119, 'learning_rate': 2.230536856429942e-05, 'epoch': 0.8}
{'loss': 0.1674, 'grad_norm': 0.006198802497237921, 'learning_rate': 2.205287401410359e-05, 'epoch': 0.83}
{'loss': 0.1689, 'grad_norm': 0.00398827763274312, 'learning_rate': 2.180037946390776e-05, 'epoch': 0.87}
{'loss': 0.1644, 'grad_norm': 0.004498797003179789, 'learning_rate': 2.1547884913711934e-05, 'epoch': 0.91}
{'loss': 0.1699, 'grad_norm': 0.01644468866288662, 'learning_rate': 2.1295390363516104e-05, 'epoch': 0.94}
{'loss': 0.1623, 'grad_norm': 0.004503154661506414, 'learning_rate': 2.1042895813320278e-05, 'epoch': 0.98}
{'eval_loss': 0.17387932538986206, 'eval_runtime': 7.0372, 'eval_samples_per_second': 783.689, 'eval_steps_per_second': 24.584, 'epoch': 1.0}
{'loss': 0.1681, 'grad_norm': 0.006585568655282259, 'learning_rate': 2.079040126312445e-05, 'epoch': 1.02}
{'loss': 0.1674, 'grad_norm': 0.007353146094828844, 'learning_rate': 2.0537906712928623e-05, 'epoch': 1.05}
{'loss': 0.1741, 'grad_norm': 0.006611280143260956, 'learning_rate': 2.0285412162732797e-05, 'epoch': 1.09}
{'loss': 0.1659, 'grad_norm': 0.007455141749233007, 'learning_rate': 2.0032917612536967e-05, 'epoch': 1.12}
{'loss': 0.1715, 'grad_norm': 0.0054252855479717255, 'learning_rate': 1.978042306234114e-05, 'epoch': 1.16}
{'loss': 0.1669, 'grad_norm': 0.01165090687572956, 'learning_rate': 1.9527928512145312e-05, 'epoch': 1.2}
{'loss': 0.1747, 'grad_norm': 0.0034899599850177765, 'learning_rate': 1.9275433961949482e-05, 'epoch': 1.23}
{'loss': 0.1653, 'grad_norm': 0.006221173796802759, 'learning_rate': 1.9022939411753656e-05, 'epoch': 1.27}
{'loss': 0.1654, 'grad_norm': 0.009759815409779549, 'learning_rate': 1.8770444861557827e-05, 'epoch': 1.31}
{'loss': 0.1706, 'grad_norm': 0.00757243949919939, 'learning_rate': 1.8517950311362e-05, 'epoch': 1.34}
{'loss': 0.1691, 'grad_norm': 0.011768573895096779, 'learning_rate': 1.8265455761166175e-05, 'epoch': 1.38}
{'loss': 0.167, 'grad_norm': 0.006406353320926428, 'learning_rate': 1.8012961210970346e-05, 'epoch': 1.41}
{'loss': 0.1703, 'grad_norm': 0.01140507310628891, 'learning_rate': 1.776046666077452e-05, 'epoch': 1.45}
{'loss': 0.1669, 'grad_norm': 0.010222329758107662, 'learning_rate': 1.750797211057869e-05, 'epoch': 1.49}
{'loss': 0.1672, 'grad_norm': 0.009191476739943027, 'learning_rate': 1.725547756038286e-05, 'epoch': 1.52}
{'loss': 0.1641, 'grad_norm': 0.013353921473026276, 'learning_rate': 1.7002983010187035e-05, 'epoch': 1.56}
{'loss': 0.1682, 'grad_norm': 0.007067216094583273, 'learning_rate': 1.6750488459991205e-05, 'epoch': 1.6}
{'loss': 0.1652, 'grad_norm': 0.008090750314295292, 'learning_rate': 1.649799390979538e-05, 'epoch': 1.63}
{'loss': 0.1684, 'grad_norm': 0.007862099446356297, 'learning_rate': 1.6245499359599553e-05, 'epoch': 1.67}
{'loss': 0.1678, 'grad_norm': 0.006291060242801905, 'learning_rate': 1.5993004809403724e-05, 'epoch': 1.7}
{'loss': 0.1702, 'grad_norm': 0.009308630600571632, 'learning_rate': 1.5740510259207898e-05, 'epoch': 1.74}
{'loss': 0.1671, 'grad_norm': 0.014437749981880188, 'learning_rate': 1.5488015709012072e-05, 'epoch': 1.78}
{'loss': 0.1731, 'grad_norm': 0.013518517836928368, 'learning_rate': 1.523552115881624e-05, 'epoch': 1.81}
{'loss': 0.1668, 'grad_norm': 0.008326271548867226, 'learning_rate': 1.4983026608620413e-05, 'epoch': 1.85}
{'loss': 0.1733, 'grad_norm': 0.013200410641729832, 'learning_rate': 1.4730532058424585e-05, 'epoch': 1.89}
{'loss': 0.1675, 'grad_norm': 0.01432901993393898, 'learning_rate': 1.4478037508228758e-05, 'epoch': 1.92}
{'loss': 0.1705, 'grad_norm': 0.01023170817643404, 'learning_rate': 1.422554295803293e-05, 'epoch': 1.96}
{'loss': 0.1715, 'grad_norm': 0.009597660042345524, 'learning_rate': 1.3973048407837104e-05, 'epoch': 1.99}
{'eval_loss': 0.17530736327171326, 'eval_runtime': 6.7982, 'eval_samples_per_second': 811.246, 'eval_steps_per_second': 25.448, 'epoch': 2.0}
{'loss': 0.1675, 'grad_norm': 0.008811663836240768, 'learning_rate': 1.3720553857641276e-05, 'epoch': 2.03}
{'loss': 0.1687, 'grad_norm': 0.007508180569857359, 'learning_rate': 1.3468059307445447e-05, 'epoch': 2.07}
{'loss': 0.172, 'grad_norm': 0.009723455645143986, 'learning_rate': 1.3215564757249619e-05, 'epoch': 2.1}
{'loss': 0.1701, 'grad_norm': 0.009010901674628258, 'learning_rate': 1.2963070207053793e-05, 'epoch': 2.14}
{'loss': 0.1724, 'grad_norm': 0.0070338621735572815, 'learning_rate': 1.2710575656857965e-05, 'epoch': 2.18}
{'loss': 0.1709, 'grad_norm': 0.009792778640985489, 'learning_rate': 1.2458081106662136e-05, 'epoch': 2.21}
{'loss': 0.1632, 'grad_norm': 0.023447636514902115, 'learning_rate': 1.2205586556466308e-05, 'epoch': 2.25}
{'loss': 0.1698, 'grad_norm': 0.006440466735512018, 'learning_rate': 1.1953092006270482e-05, 'epoch': 2.28}
{'loss': 0.1645, 'grad_norm': 0.006529440637677908, 'learning_rate': 1.1700597456074654e-05, 'epoch': 2.32}
{'loss': 0.1669, 'grad_norm': 0.01303196046501398, 'learning_rate': 1.1448102905878827e-05, 'epoch': 2.36}
{'loss': 0.1728, 'grad_norm': 0.007114199455827475, 'learning_rate': 1.1195608355682997e-05, 'epoch': 2.39}
{'loss': 0.1696, 'grad_norm': 0.013213497586548328, 'learning_rate': 1.0943113805487171e-05, 'epoch': 2.43}
{'loss': 0.1726, 'grad_norm': 0.011665496043860912, 'learning_rate': 1.0690619255291344e-05, 'epoch': 2.47}
{'loss': 0.1738, 'grad_norm': 0.00944163091480732, 'learning_rate': 1.0438124705095516e-05, 'epoch': 2.5}
{'loss': 0.1671, 'grad_norm': 0.005741126835346222, 'learning_rate': 1.0185630154899688e-05, 'epoch': 2.54}
{'loss': 0.1729, 'grad_norm': 0.013643588870763779, 'learning_rate': 9.93313560470386e-06, 'epoch': 2.57}
{'loss': 0.1698, 'grad_norm': 0.008251342922449112, 'learning_rate': 9.680641054508033e-06, 'epoch': 2.61}
{'loss': 0.1694, 'grad_norm': 0.0077118342742323875, 'learning_rate': 9.428146504312205e-06, 'epoch': 2.65}
{'loss': 0.1701, 'grad_norm': 0.008676649071276188, 'learning_rate': 9.175651954116377e-06, 'epoch': 2.68}
{'loss': 0.1684, 'grad_norm': 0.007653322070837021, 'learning_rate': 8.92315740392055e-06, 'epoch': 2.72}
{'loss': 0.1686, 'grad_norm': 0.02383018471300602, 'learning_rate': 8.670662853724722e-06, 'epoch': 2.76}
{'loss': 0.1688, 'grad_norm': 0.02988579496741295, 'learning_rate': 8.418168303528894e-06, 'epoch': 2.79}
{'loss': 0.1715, 'grad_norm': 0.017488548532128334, 'learning_rate': 8.165673753333066e-06, 'epoch': 2.83}
{'loss': 0.1761, 'grad_norm': 0.00982433557510376, 'learning_rate': 7.913179203137239e-06, 'epoch': 2.86}
{'loss': 0.1748, 'grad_norm': 0.009307028725743294, 'learning_rate': 7.660684652941411e-06, 'epoch': 2.9}
{'loss': 0.1685, 'grad_norm': 0.007551151793450117, 'learning_rate': 7.408190102745583e-06, 'epoch': 2.94}
{'loss': 0.174, 'grad_norm': 0.01934993453323841, 'learning_rate': 7.1556955525497565e-06, 'epoch': 2.97}
{'eval_loss': 0.17654810845851898, 'eval_runtime': 7.3363, 'eval_samples_per_second': 751.743, 'eval_steps_per_second': 23.581, 'epoch': 3.0}
{'train_runtime': 238.7576, 'train_samples_per_second': 369.53, 'train_steps_per_second': 23.103, 'train_loss': 0.16880917704736434, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [03:58<01:19, 17.33it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:07<00:00, 23.99it/s]
[I 2025-08-12 16:32:45,461] Trial 12 finished with value: 0.17387932538986206 and parameters: {'lr': 2.7855198777603708e-05, 'n_blocks': 3}. Best is trial 10 with value: 0.10707330703735352.
Best trial: 10. Best value: 0.107073:  60%|█████████████████████████████████████████████████████████████████████████████████████▏                                                        | 12/20 [1:25:23<47:02, 352.77s/it][16:32:45] ft_study trial#12 value=0.17387932538986206 params={'lr': 2.7855198777603708e-05, 'n_blocks': 3}
{'loss': 0.1698, 'grad_norm': 0.0038224668242037296, 'learning_rate': 1.0598349095170019e-05, 'epoch': 0.04}
{'loss': 0.1674, 'grad_norm': 0.0012626522220671177, 'learning_rate': 1.0501418885775745e-05, 'epoch': 0.07}
{'loss': 0.1685, 'grad_norm': 0.013796639628708363, 'learning_rate': 1.0404488676381469e-05, 'epoch': 0.11}
{'loss': 0.1755, 'grad_norm': 0.0015314474003389478, 'learning_rate': 1.0307558466987195e-05, 'epoch': 0.15}
{'loss': 0.1666, 'grad_norm': 0.001823199214413762, 'learning_rate': 1.021062825759292e-05, 'epoch': 0.18}
{'loss': 0.1641, 'grad_norm': 0.002907254733145237, 'learning_rate': 1.0113698048198644e-05, 'epoch': 0.22}
{'loss': 0.1684, 'grad_norm': 0.0024908604100346565, 'learning_rate': 1.001676783880437e-05, 'epoch': 0.25}
{'loss': 0.1623, 'grad_norm': 0.0010759361321106553, 'learning_rate': 9.919837629410094e-06, 'epoch': 0.29}
{'loss': 0.1683, 'grad_norm': 0.0009149129036813974, 'learning_rate': 9.82290742001582e-06, 'epoch': 0.33}
{'loss': 0.1685, 'grad_norm': 0.002116966526955366, 'learning_rate': 9.725977210621544e-06, 'epoch': 0.36}
{'loss': 0.1665, 'grad_norm': 0.003081246977671981, 'learning_rate': 9.62904700122727e-06, 'epoch': 0.4}
{'loss': 0.1689, 'grad_norm': 0.0016515797469764948, 'learning_rate': 9.532116791832994e-06, 'epoch': 0.44}
{'loss': 0.1643, 'grad_norm': 0.0017973160138353705, 'learning_rate': 9.43518658243872e-06, 'epoch': 0.47}
{'loss': 0.1679, 'grad_norm': 0.004278079140931368, 'learning_rate': 9.338256373044446e-06, 'epoch': 0.51}
{'loss': 0.1616, 'grad_norm': 0.007446652743965387, 'learning_rate': 9.241326163650169e-06, 'epoch': 0.54}
{'loss': 0.1719, 'grad_norm': 0.002585302572697401, 'learning_rate': 9.144395954255895e-06, 'epoch': 0.58}
{'loss': 0.1665, 'grad_norm': 0.0028897044248878956, 'learning_rate': 9.04746574486162e-06, 'epoch': 0.62}
{'loss': 0.1663, 'grad_norm': 0.005459375213831663, 'learning_rate': 8.950535535467345e-06, 'epoch': 0.65}
{'loss': 0.1652, 'grad_norm': 0.0033976531121879816, 'learning_rate': 8.853605326073071e-06, 'epoch': 0.69}
{'loss': 0.166, 'grad_norm': 0.006456276401877403, 'learning_rate': 8.756675116678796e-06, 'epoch': 0.73}
{'loss': 0.1656, 'grad_norm': 0.004307663533836603, 'learning_rate': 8.659744907284522e-06, 'epoch': 0.76}
{'loss': 0.1741, 'grad_norm': 0.006545566488057375, 'learning_rate': 8.562814697890246e-06, 'epoch': 0.8}
{'loss': 0.167, 'grad_norm': 0.003628920065239072, 'learning_rate': 8.465884488495972e-06, 'epoch': 0.83}
{'loss': 0.1685, 'grad_norm': 0.0026864842511713505, 'learning_rate': 8.368954279101696e-06, 'epoch': 0.87}
{'loss': 0.1638, 'grad_norm': 0.0035295747220516205, 'learning_rate': 8.27202406970742e-06, 'epoch': 0.91}
{'loss': 0.1694, 'grad_norm': 0.009381549432873726, 'learning_rate': 8.175093860313146e-06, 'epoch': 0.94}
{'loss': 0.1618, 'grad_norm': 0.0033549999352544546, 'learning_rate': 8.07816365091887e-06, 'epoch': 0.98}
{'eval_loss': 0.1735258400440216, 'eval_runtime': 6.9137, 'eval_samples_per_second': 797.691, 'eval_steps_per_second': 25.023, 'epoch': 1.0}
{'loss': 0.1676, 'grad_norm': 0.004295822232961655, 'learning_rate': 7.981233441524597e-06, 'epoch': 1.02}
{'loss': 0.1668, 'grad_norm': 0.0049254451878368855, 'learning_rate': 7.884303232130321e-06, 'epoch': 1.05}
{'loss': 0.1734, 'grad_norm': 0.003744279034435749, 'learning_rate': 7.787373022736047e-06, 'epoch': 1.09}
{'loss': 0.1651, 'grad_norm': 0.004138578660786152, 'learning_rate': 7.690442813341771e-06, 'epoch': 1.12}
{'loss': 0.1705, 'grad_norm': 0.002907993271946907, 'learning_rate': 7.593512603947497e-06, 'epoch': 1.16}
{'loss': 0.166, 'grad_norm': 0.00873018428683281, 'learning_rate': 7.496582394553222e-06, 'epoch': 1.2}
{'loss': 0.1736, 'grad_norm': 0.002020800020545721, 'learning_rate': 7.399652185158947e-06, 'epoch': 1.23}
{'loss': 0.1642, 'grad_norm': 0.004511980339884758, 'learning_rate': 7.302721975764672e-06, 'epoch': 1.27}
{'loss': 0.1641, 'grad_norm': 0.004882889334112406, 'learning_rate': 7.205791766370397e-06, 'epoch': 1.31}
{'loss': 0.1694, 'grad_norm': 0.0036617578007280827, 'learning_rate': 7.108861556976122e-06, 'epoch': 1.34}
{'loss': 0.1677, 'grad_norm': 0.007113135885447264, 'learning_rate': 7.011931347581847e-06, 'epoch': 1.38}
{'loss': 0.1657, 'grad_norm': 0.0035161578562110662, 'learning_rate': 6.9150011381875724e-06, 'epoch': 1.41}
{'loss': 0.1691, 'grad_norm': 0.0069579314440488815, 'learning_rate': 6.818070928793298e-06, 'epoch': 1.45}
{'loss': 0.1657, 'grad_norm': 0.0056109377183020115, 'learning_rate': 6.721140719399022e-06, 'epoch': 1.49}
{'loss': 0.1657, 'grad_norm': 0.004973667208105326, 'learning_rate': 6.624210510004747e-06, 'epoch': 1.52}
{'loss': 0.1629, 'grad_norm': 0.005088512785732746, 'learning_rate': 6.527280300610472e-06, 'epoch': 1.56}
{'loss': 0.1672, 'grad_norm': 0.0033859647810459137, 'learning_rate': 6.430350091216197e-06, 'epoch': 1.6}
{'loss': 0.1636, 'grad_norm': 0.004090358503162861, 'learning_rate': 6.3334198818219225e-06, 'epoch': 1.63}
{'loss': 0.1669, 'grad_norm': 0.005011168774217367, 'learning_rate': 6.236489672427648e-06, 'epoch': 1.67}
{'loss': 0.166, 'grad_norm': 0.002440267475321889, 'learning_rate': 6.139559463033373e-06, 'epoch': 1.7}
{'loss': 0.1688, 'grad_norm': 0.0053543029353022575, 'learning_rate': 6.042629253639099e-06, 'epoch': 1.74}
{'loss': 0.1657, 'grad_norm': 0.0072291698306798935, 'learning_rate': 5.945699044244824e-06, 'epoch': 1.78}
{'loss': 0.1711, 'grad_norm': 0.00626041553914547, 'learning_rate': 5.848768834850547e-06, 'epoch': 1.81}
{'loss': 0.1652, 'grad_norm': 0.003809775924310088, 'learning_rate': 5.7518386254562725e-06, 'epoch': 1.85}
{'loss': 0.1717, 'grad_norm': 0.006355782505124807, 'learning_rate': 5.6549084160619985e-06, 'epoch': 1.89}
{'loss': 0.1655, 'grad_norm': 0.006666179280728102, 'learning_rate': 5.557978206667724e-06, 'epoch': 1.92}
{'loss': 0.1686, 'grad_norm': 0.004497162066400051, 'learning_rate': 5.461047997273449e-06, 'epoch': 1.96}
{'loss': 0.1695, 'grad_norm': 0.004696046933531761, 'learning_rate': 5.364117787879174e-06, 'epoch': 1.99}
{'eval_loss': 0.1736229807138443, 'eval_runtime': 7.1227, 'eval_samples_per_second': 774.281, 'eval_steps_per_second': 24.288, 'epoch': 2.0}
{'loss': 0.1658, 'grad_norm': 0.004824776202440262, 'learning_rate': 5.267187578484899e-06, 'epoch': 2.03}
{'loss': 0.1665, 'grad_norm': 0.0035374562721699476, 'learning_rate': 5.170257369090623e-06, 'epoch': 2.07}
{'loss': 0.17, 'grad_norm': 0.005796546582132578, 'learning_rate': 5.0733271596963485e-06, 'epoch': 2.1}
{'loss': 0.1681, 'grad_norm': 0.005077902227640152, 'learning_rate': 4.976396950302074e-06, 'epoch': 2.14}
{'loss': 0.1702, 'grad_norm': 0.0037854197435081005, 'learning_rate': 4.8794667409078e-06, 'epoch': 2.18}
{'loss': 0.1682, 'grad_norm': 0.004659903235733509, 'learning_rate': 4.782536531513524e-06, 'epoch': 2.21}
{'loss': 0.1602, 'grad_norm': 0.014920873567461967, 'learning_rate': 4.685606322119249e-06, 'epoch': 2.25}
{'loss': 0.1672, 'grad_norm': 0.0025562376249581575, 'learning_rate': 4.588676112724974e-06, 'epoch': 2.28}
{'loss': 0.1619, 'grad_norm': 0.0023307285737246275, 'learning_rate': 4.491745903330699e-06, 'epoch': 2.32}
{'loss': 0.1643, 'grad_norm': 0.005365824326872826, 'learning_rate': 4.3948156939364245e-06, 'epoch': 2.36}
{'loss': 0.1698, 'grad_norm': 0.0031677219085395336, 'learning_rate': 4.29788548454215e-06, 'epoch': 2.39}
{'loss': 0.1667, 'grad_norm': 0.006933190859854221, 'learning_rate': 4.200955275147875e-06, 'epoch': 2.43}
{'loss': 0.1701, 'grad_norm': 0.0053141750395298, 'learning_rate': 4.1040250657536e-06, 'epoch': 2.47}
{'loss': 0.1723, 'grad_norm': 0.0051026297733187675, 'learning_rate': 4.007094856359325e-06, 'epoch': 2.5}
{'loss': 0.1644, 'grad_norm': 0.0021252695005387068, 'learning_rate': 3.910164646965049e-06, 'epoch': 2.54}
{'loss': 0.1701, 'grad_norm': 0.005840681027621031, 'learning_rate': 3.813234437570775e-06, 'epoch': 2.57}
{'loss': 0.1672, 'grad_norm': 0.004233148414641619, 'learning_rate': 3.7163042281765e-06, 'epoch': 2.61}
{'loss': 0.1668, 'grad_norm': 0.0029923703987151384, 'learning_rate': 3.619374018782225e-06, 'epoch': 2.65}
{'loss': 0.1678, 'grad_norm': 0.002964264713227749, 'learning_rate': 3.5224438093879508e-06, 'epoch': 2.68}
{'loss': 0.1654, 'grad_norm': 0.00397601118311286, 'learning_rate': 3.425513599993675e-06, 'epoch': 2.72}
{'loss': 0.1661, 'grad_norm': 0.008337439969182014, 'learning_rate': 3.3285833905994006e-06, 'epoch': 2.76}
{'loss': 0.1654, 'grad_norm': 0.038170166313648224, 'learning_rate': 3.2316531812051258e-06, 'epoch': 2.79}
{'loss': 0.1687, 'grad_norm': 0.008086919784545898, 'learning_rate': 3.134722971810851e-06, 'epoch': 2.83}
{'loss': 0.1741, 'grad_norm': 0.005117370747029781, 'learning_rate': 3.0377927624165756e-06, 'epoch': 2.86}
{'loss': 0.1719, 'grad_norm': 0.0060518463142216206, 'learning_rate': 2.9408625530223008e-06, 'epoch': 2.9}
{'loss': 0.1653, 'grad_norm': 0.003704852657392621, 'learning_rate': 2.843932343628026e-06, 'epoch': 2.94}
{'loss': 0.1707, 'grad_norm': 0.010336903855204582, 'learning_rate': 2.747002134233751e-06, 'epoch': 2.97}
{'eval_loss': 0.1736927032470703, 'eval_runtime': 6.6897, 'eval_samples_per_second': 824.401, 'eval_steps_per_second': 25.861, 'epoch': 3.0}
{'train_runtime': 234.846, 'train_samples_per_second': 375.685, 'train_steps_per_second': 23.488, 'train_loss': 0.16741859956095406, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [03:54<01:18, 17.62it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:06<00:00, 24.92it/s]
[I 2025-08-12 16:36:47,673] Trial 13 finished with value: 0.1735258400440216 and parameters: {'lr': 1.0693340700376408e-05, 'n_blocks': 3}. Best is trial 10 with value: 0.10707330703735352.
Best trial: 10. Best value: 0.107073:  65%|████████████████████████████████████████████████████████████████████████████████████████████▎                                                 | 13/20 [1:29:25<37:23, 320.55s/it][16:36:47] ft_study trial#13 value=0.1735258400440216 params={'lr': 1.0693340700376408e-05, 'n_blocks': 3}
{'loss': 0.1698, 'grad_norm': 0.0038274468388408422, 'learning_rate': 4.623096712757559e-05, 'epoch': 0.04}
{'loss': 0.1675, 'grad_norm': 0.001415449776686728, 'learning_rate': 4.580814869765447e-05, 'epoch': 0.07}
{'loss': 0.1685, 'grad_norm': 0.014357103034853935, 'learning_rate': 4.538533026773335e-05, 'epoch': 0.11}
{'loss': 0.1755, 'grad_norm': 0.00170180294662714, 'learning_rate': 4.4962511837812227e-05, 'epoch': 0.15}
{'loss': 0.1667, 'grad_norm': 0.002143052639439702, 'learning_rate': 4.4539693407891105e-05, 'epoch': 0.18}
{'loss': 0.1642, 'grad_norm': 0.0038139873649924994, 'learning_rate': 4.411687497796998e-05, 'epoch': 0.22}
{'loss': 0.1686, 'grad_norm': 0.003229074878618121, 'learning_rate': 4.3694056548048855e-05, 'epoch': 0.25}
{'loss': 0.1625, 'grad_norm': 0.001519558485597372, 'learning_rate': 4.327123811812773e-05, 'epoch': 0.29}
{'loss': 0.1685, 'grad_norm': 0.0012491293018683791, 'learning_rate': 4.2848419688206606e-05, 'epoch': 0.33}
{'loss': 0.1687, 'grad_norm': 0.003261962439864874, 'learning_rate': 4.2425601258285484e-05, 'epoch': 0.36}
{'loss': 0.1668, 'grad_norm': 0.0039703031070530415, 'learning_rate': 4.200278282836436e-05, 'epoch': 0.4}
{'loss': 0.1693, 'grad_norm': 0.0026100464165210724, 'learning_rate': 4.157996439844324e-05, 'epoch': 0.44}
{'loss': 0.1647, 'grad_norm': 0.0028213171754032373, 'learning_rate': 4.115714596852212e-05, 'epoch': 0.47}
{'loss': 0.1684, 'grad_norm': 0.005928691942244768, 'learning_rate': 4.0734327538601e-05, 'epoch': 0.51}
{'loss': 0.1622, 'grad_norm': 0.010734611190855503, 'learning_rate': 4.031150910867987e-05, 'epoch': 0.54}
{'loss': 0.1725, 'grad_norm': 0.004205591510981321, 'learning_rate': 3.988869067875875e-05, 'epoch': 0.58}
{'loss': 0.1672, 'grad_norm': 0.00512081990018487, 'learning_rate': 3.946587224883762e-05, 'epoch': 0.62}
{'loss': 0.1672, 'grad_norm': 0.008910308592021465, 'learning_rate': 3.90430538189165e-05, 'epoch': 0.65}
{'loss': 0.1662, 'grad_norm': 0.0058564720675349236, 'learning_rate': 3.862023538899538e-05, 'epoch': 0.69}
{'loss': 0.1673, 'grad_norm': 0.012337428517639637, 'learning_rate': 3.8197416959074256e-05, 'epoch': 0.73}
{'loss': 0.1669, 'grad_norm': 0.005965865217149258, 'learning_rate': 3.7774598529153134e-05, 'epoch': 0.76}
{'loss': 0.1752, 'grad_norm': 0.015473461709916592, 'learning_rate': 3.735178009923201e-05, 'epoch': 0.8}
{'loss': 0.1685, 'grad_norm': 0.010586113668978214, 'learning_rate': 3.692896166931089e-05, 'epoch': 0.83}
{'loss': 0.1699, 'grad_norm': 0.005803443491458893, 'learning_rate': 3.650614323938976e-05, 'epoch': 0.87}
{'loss': 0.166, 'grad_norm': 0.006072681862860918, 'learning_rate': 3.6083324809468635e-05, 'epoch': 0.91}
{'loss': 0.1712, 'grad_norm': 0.03013378195464611, 'learning_rate': 3.566050637954751e-05, 'epoch': 0.94}
{'loss': 0.164, 'grad_norm': 0.006190681364387274, 'learning_rate': 3.523768794962639e-05, 'epoch': 0.98}
{'eval_loss': 0.17542678117752075, 'eval_runtime': 6.7777, 'eval_samples_per_second': 813.7, 'eval_steps_per_second': 25.525, 'epoch': 1.0}
{'loss': 0.1696, 'grad_norm': 0.009097950533032417, 'learning_rate': 3.481486951970527e-05, 'epoch': 1.02}
{'loss': 0.1693, 'grad_norm': 0.010669773444533348, 'learning_rate': 3.439205108978415e-05, 'epoch': 1.05}
{'loss': 0.1763, 'grad_norm': 0.012817459180951118, 'learning_rate': 3.396923265986303e-05, 'epoch': 1.09}
{'loss': 0.1684, 'grad_norm': 0.016012663021683693, 'learning_rate': 3.3546414229941906e-05, 'epoch': 1.12}
{'loss': 0.1745, 'grad_norm': 0.010443437844514847, 'learning_rate': 3.312359580002078e-05, 'epoch': 1.16}
{'loss': 0.17, 'grad_norm': 0.015774494037032127, 'learning_rate': 3.2700777370099656e-05, 'epoch': 1.2}
{'loss': 0.1784, 'grad_norm': 0.006811127997934818, 'learning_rate': 3.227795894017853e-05, 'epoch': 1.23}
{'loss': 0.1689, 'grad_norm': 0.01068323478102684, 'learning_rate': 3.1855140510257406e-05, 'epoch': 1.27}
{'loss': 0.1694, 'grad_norm': 0.019672201946377754, 'learning_rate': 3.1432322080336285e-05, 'epoch': 1.31}
{'loss': 0.1744, 'grad_norm': 0.014381890185177326, 'learning_rate': 3.1009503650415164e-05, 'epoch': 1.34}
{'loss': 0.1733, 'grad_norm': 0.017819436267018318, 'learning_rate': 3.058668522049404e-05, 'epoch': 1.38}
{'loss': 0.1711, 'grad_norm': 0.012221762910485268, 'learning_rate': 3.0163866790572917e-05, 'epoch': 1.41}
{'loss': 0.174, 'grad_norm': 0.019236233085393906, 'learning_rate': 2.9741048360651796e-05, 'epoch': 1.45}
{'loss': 0.171, 'grad_norm': 0.01972152478992939, 'learning_rate': 2.9318229930730667e-05, 'epoch': 1.49}
{'loss': 0.172, 'grad_norm': 0.015098420903086662, 'learning_rate': 2.8895411500809546e-05, 'epoch': 1.52}
{'loss': 0.1679, 'grad_norm': 0.04877122864127159, 'learning_rate': 2.8472593070888424e-05, 'epoch': 1.56}
{'loss': 0.1715, 'grad_norm': 0.014825090765953064, 'learning_rate': 2.80497746409673e-05, 'epoch': 1.6}
{'loss': 0.1697, 'grad_norm': 0.017190126702189445, 'learning_rate': 2.7626956211046178e-05, 'epoch': 1.63}
{'loss': 0.1731, 'grad_norm': 0.013262676075100899, 'learning_rate': 2.7204137781125057e-05, 'epoch': 1.67}
{'loss': 0.173, 'grad_norm': 0.02243296056985855, 'learning_rate': 2.6781319351203935e-05, 'epoch': 1.7}
{'loss': 0.1746, 'grad_norm': 0.020277520641684532, 'learning_rate': 2.635850092128281e-05, 'epoch': 1.74}
{'loss': 0.1711, 'grad_norm': 0.029213344678282738, 'learning_rate': 2.593568249136169e-05, 'epoch': 1.78}
{'loss': 0.179, 'grad_norm': 0.026695910841226578, 'learning_rate': 2.551286406144056e-05, 'epoch': 1.81}
{'loss': 0.1713, 'grad_norm': 0.017727183178067207, 'learning_rate': 2.509004563151944e-05, 'epoch': 1.85}
{'loss': 0.1775, 'grad_norm': 0.027170665562152863, 'learning_rate': 2.4667227201598318e-05, 'epoch': 1.89}
{'loss': 0.1729, 'grad_norm': 0.027602070942521095, 'learning_rate': 2.4244408771677193e-05, 'epoch': 1.92}
{'loss': 0.1757, 'grad_norm': 0.02313118427991867, 'learning_rate': 2.382159034175607e-05, 'epoch': 1.96}
{'loss': 0.1764, 'grad_norm': 0.020030928775668144, 'learning_rate': 2.339877191183495e-05, 'epoch': 1.99}
{'eval_loss': 0.1797822117805481, 'eval_runtime': 6.8872, 'eval_samples_per_second': 800.761, 'eval_steps_per_second': 25.119, 'epoch': 2.0}
{'loss': 0.172, 'grad_norm': 0.016437798738479614, 'learning_rate': 2.2975953481913825e-05, 'epoch': 2.03}
{'loss': 0.1742, 'grad_norm': 0.017009848728775978, 'learning_rate': 2.25531350519927e-05, 'epoch': 2.07}
{'loss': 0.177, 'grad_norm': 0.01791139505803585, 'learning_rate': 2.213031662207158e-05, 'epoch': 2.1}
{'loss': 0.1754, 'grad_norm': 0.016244180500507355, 'learning_rate': 2.1707498192150457e-05, 'epoch': 2.14}
{'loss': 0.1775, 'grad_norm': 0.011141945607960224, 'learning_rate': 2.1284679762229336e-05, 'epoch': 2.18}
{'loss': 0.1772, 'grad_norm': 0.02104100212454796, 'learning_rate': 2.086186133230821e-05, 'epoch': 2.21}
{'loss': 0.1699, 'grad_norm': 0.03072505071759224, 'learning_rate': 2.0439042902387086e-05, 'epoch': 2.25}
{'loss': 0.1752, 'grad_norm': 0.014978794381022453, 'learning_rate': 2.0016224472465964e-05, 'epoch': 2.28}
{'loss': 0.1703, 'grad_norm': 0.0179408211261034, 'learning_rate': 1.9593406042544843e-05, 'epoch': 2.32}
{'loss': 0.1724, 'grad_norm': 0.030423881486058235, 'learning_rate': 1.9170587612623718e-05, 'epoch': 2.36}
{'loss': 0.1791, 'grad_norm': 0.015354715287685394, 'learning_rate': 1.8747769182702593e-05, 'epoch': 2.39}
{'loss': 0.1751, 'grad_norm': 0.023757975548505783, 'learning_rate': 1.8324950752781472e-05, 'epoch': 2.43}
{'loss': 0.1776, 'grad_norm': 0.02285836823284626, 'learning_rate': 1.790213232286035e-05, 'epoch': 2.47}
{'loss': 0.1765, 'grad_norm': 0.015800751745700836, 'learning_rate': 1.7479313892939225e-05, 'epoch': 2.5}
{'loss': 0.172, 'grad_norm': 0.013157506473362446, 'learning_rate': 1.70564954630181e-05, 'epoch': 2.54}
{'loss': 0.1783, 'grad_norm': 0.02902725711464882, 'learning_rate': 1.663367703309698e-05, 'epoch': 2.57}
{'loss': 0.174, 'grad_norm': 0.018330132588744164, 'learning_rate': 1.6210858603175858e-05, 'epoch': 2.61}
{'loss': 0.174, 'grad_norm': 0.01686445251107216, 'learning_rate': 1.5788040173254736e-05, 'epoch': 2.65}
{'loss': 0.1741, 'grad_norm': 0.017177322879433632, 'learning_rate': 1.536522174333361e-05, 'epoch': 2.68}
{'loss': 0.1735, 'grad_norm': 0.014148118905723095, 'learning_rate': 1.4942403313412486e-05, 'epoch': 2.72}
{'loss': 0.1727, 'grad_norm': 0.04892435297369957, 'learning_rate': 1.4519584883491365e-05, 'epoch': 2.76}
{'loss': 0.1754, 'grad_norm': 0.01813369058072567, 'learning_rate': 1.4096766453570242e-05, 'epoch': 2.79}
{'loss': 0.1762, 'grad_norm': 0.030162489041686058, 'learning_rate': 1.367394802364912e-05, 'epoch': 2.83}
{'loss': 0.1794, 'grad_norm': 0.016402844339609146, 'learning_rate': 1.3251129593727995e-05, 'epoch': 2.86}
{'loss': 0.1792, 'grad_norm': 0.013316105119884014, 'learning_rate': 1.2828311163806872e-05, 'epoch': 2.9}
{'loss': 0.1736, 'grad_norm': 0.0133871054276824, 'learning_rate': 1.2405492733885749e-05, 'epoch': 2.94}
{'loss': 0.179, 'grad_norm': 0.030921723693609238, 'learning_rate': 1.1982674303964628e-05, 'epoch': 2.97}
{'eval_loss': 0.17981819808483124, 'eval_runtime': 6.4541, 'eval_samples_per_second': 854.492, 'eval_steps_per_second': 26.805, 'epoch': 3.0}
{'train_runtime': 213.2792, 'train_samples_per_second': 413.674, 'train_steps_per_second': 25.863, 'train_loss': 0.1719962338592109, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [03:33<01:11, 19.40it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:06<00:00, 25.50it/s]
[I 2025-08-12 16:40:28,188] Trial 14 finished with value: 0.17542678117752075 and parameters: {'lr': 4.664532918889829e-05, 'n_blocks': 3}. Best is trial 10 with value: 0.10707330703735352.
Best trial: 10. Best value: 0.107073:  70%|███████████████████████████████████████████████████████████████████████████████████████████████████▍                                          | 14/20 [1:33:06<29:41, 296.89s/it][16:40:28] ft_study trial#14 value=0.17542678117752075 params={'lr': 4.664532918889829e-05, 'n_blocks': 3}
{'loss': 0.1189, 'grad_norm': 0.005240552127361298, 'learning_rate': 3.793313619445173e-06, 'epoch': 0.04}
{'loss': 0.1174, 'grad_norm': 0.00316711887717247, 'learning_rate': 3.75862079322014e-06, 'epoch': 0.07}
{'loss': 0.1194, 'grad_norm': 0.01637871004641056, 'learning_rate': 3.7239279669951062e-06, 'epoch': 0.11}
{'loss': 0.1188, 'grad_norm': 0.005295672453939915, 'learning_rate': 3.6892351407700726e-06, 'epoch': 0.15}
{'loss': 0.1141, 'grad_norm': 0.0023185417521744967, 'learning_rate': 3.654542314545039e-06, 'epoch': 0.18}
{'loss': 0.1145, 'grad_norm': 0.00815917644649744, 'learning_rate': 3.6198494883200052e-06, 'epoch': 0.22}
{'loss': 0.1189, 'grad_norm': 0.004570069257169962, 'learning_rate': 3.5851566620949716e-06, 'epoch': 0.25}
{'loss': 0.1157, 'grad_norm': 0.0033732582814991474, 'learning_rate': 3.550463835869938e-06, 'epoch': 0.29}
{'loss': 0.115, 'grad_norm': 0.0027639418840408325, 'learning_rate': 3.5157710096449047e-06, 'epoch': 0.33}
{'loss': 0.1164, 'grad_norm': 0.003657022025436163, 'learning_rate': 3.481078183419871e-06, 'epoch': 0.36}
{'loss': 0.1179, 'grad_norm': 0.005576276686042547, 'learning_rate': 3.4463853571948374e-06, 'epoch': 0.4}
{'loss': 0.1169, 'grad_norm': 0.005853637587279081, 'learning_rate': 3.4116925309698037e-06, 'epoch': 0.44}
{'loss': 0.114, 'grad_norm': 0.003521647769957781, 'learning_rate': 3.3769997047447705e-06, 'epoch': 0.47}
{'loss': 0.1182, 'grad_norm': 0.004456539172679186, 'learning_rate': 3.342306878519737e-06, 'epoch': 0.51}
{'loss': 0.1177, 'grad_norm': 0.004400113131850958, 'learning_rate': 3.3076140522947027e-06, 'epoch': 0.54}
{'loss': 0.119, 'grad_norm': 0.0033168888185173273, 'learning_rate': 3.272921226069669e-06, 'epoch': 0.58}
{'loss': 0.1144, 'grad_norm': 0.004211821127682924, 'learning_rate': 3.238228399844636e-06, 'epoch': 0.62}
{'loss': 0.1186, 'grad_norm': 0.011097232811152935, 'learning_rate': 3.203535573619602e-06, 'epoch': 0.65}
{'loss': 0.1172, 'grad_norm': 0.009114591404795647, 'learning_rate': 3.1688427473945685e-06, 'epoch': 0.69}
{'loss': 0.1167, 'grad_norm': 0.006178335286676884, 'learning_rate': 3.134149921169535e-06, 'epoch': 0.73}
{'loss': 0.1161, 'grad_norm': 0.003663974115625024, 'learning_rate': 3.0994570949445016e-06, 'epoch': 0.76}
{'loss': 0.1195, 'grad_norm': 0.008056160062551498, 'learning_rate': 3.064764268719468e-06, 'epoch': 0.8}
{'loss': 0.1154, 'grad_norm': 0.005547977983951569, 'learning_rate': 3.0300714424944343e-06, 'epoch': 0.83}
{'loss': 0.1144, 'grad_norm': 0.0034227382857352495, 'learning_rate': 2.9953786162694006e-06, 'epoch': 0.87}
{'loss': 0.1164, 'grad_norm': 0.005269040819257498, 'learning_rate': 2.960685790044367e-06, 'epoch': 0.91}
{'loss': 0.1142, 'grad_norm': 0.017199674621224403, 'learning_rate': 2.9259929638193333e-06, 'epoch': 0.94}
{'loss': 0.1119, 'grad_norm': 0.003562105353921652, 'learning_rate': 2.8913001375942996e-06, 'epoch': 0.98}
{'eval_loss': 0.12034811079502106, 'eval_runtime': 10.8324, 'eval_samples_per_second': 509.121, 'eval_steps_per_second': 15.971, 'epoch': 1.0}
{'loss': 0.1154, 'grad_norm': 0.005677653942257166, 'learning_rate': 2.8566073113692664e-06, 'epoch': 1.02}
{'loss': 0.118, 'grad_norm': 0.007033337838947773, 'learning_rate': 2.8219144851442327e-06, 'epoch': 1.05}
{'loss': 0.1163, 'grad_norm': 0.010999081656336784, 'learning_rate': 2.787221658919199e-06, 'epoch': 1.09}
{'loss': 0.1164, 'grad_norm': 0.0031364073511213064, 'learning_rate': 2.7525288326941654e-06, 'epoch': 1.12}
{'loss': 0.1168, 'grad_norm': 0.00454073678702116, 'learning_rate': 2.717836006469132e-06, 'epoch': 1.16}
{'loss': 0.1151, 'grad_norm': 0.004496036563068628, 'learning_rate': 2.683143180244098e-06, 'epoch': 1.2}
{'loss': 0.1148, 'grad_norm': 0.005483989603817463, 'learning_rate': 2.6484503540190644e-06, 'epoch': 1.23}
{'loss': 0.1186, 'grad_norm': 0.010139817371964455, 'learning_rate': 2.6137575277940308e-06, 'epoch': 1.27}
{'loss': 0.1141, 'grad_norm': 0.011096525005996227, 'learning_rate': 2.5790647015689975e-06, 'epoch': 1.31}
{'loss': 0.1152, 'grad_norm': 0.005934025626629591, 'learning_rate': 2.544371875343964e-06, 'epoch': 1.34}
{'loss': 0.1181, 'grad_norm': 0.006702817045152187, 'learning_rate': 2.50967904911893e-06, 'epoch': 1.38}
{'loss': 0.1162, 'grad_norm': 0.011285820975899696, 'learning_rate': 2.474986222893897e-06, 'epoch': 1.41}
{'loss': 0.1159, 'grad_norm': 0.00744537403807044, 'learning_rate': 2.4402933966688633e-06, 'epoch': 1.45}
{'loss': 0.1155, 'grad_norm': 0.00541946105659008, 'learning_rate': 2.4056005704438292e-06, 'epoch': 1.49}
{'loss': 0.1174, 'grad_norm': 0.009391587227582932, 'learning_rate': 2.3709077442187956e-06, 'epoch': 1.52}
{'loss': 0.1166, 'grad_norm': 0.012871098704636097, 'learning_rate': 2.3362149179937623e-06, 'epoch': 1.56}
{'loss': 0.1194, 'grad_norm': 0.010611712001264095, 'learning_rate': 2.3015220917687287e-06, 'epoch': 1.6}
{'loss': 0.1203, 'grad_norm': 0.008708273060619831, 'learning_rate': 2.266829265543695e-06, 'epoch': 1.63}
{'loss': 0.1187, 'grad_norm': 0.006886100862175226, 'learning_rate': 2.2321364393186613e-06, 'epoch': 1.67}
{'loss': 0.1207, 'grad_norm': 0.008069540373980999, 'learning_rate': 2.197443613093628e-06, 'epoch': 1.7}
{'loss': 0.1204, 'grad_norm': 0.007497469428926706, 'learning_rate': 2.1627507868685944e-06, 'epoch': 1.74}
{'loss': 0.118, 'grad_norm': 0.00910364929586649, 'learning_rate': 2.1280579606435608e-06, 'epoch': 1.78}
{'loss': 0.1132, 'grad_norm': 0.007033852860331535, 'learning_rate': 2.093365134418527e-06, 'epoch': 1.81}
{'loss': 0.1162, 'grad_norm': 0.009135556407272816, 'learning_rate': 2.0586723081934935e-06, 'epoch': 1.85}
{'loss': 0.1167, 'grad_norm': 0.0075691561214625835, 'learning_rate': 2.02397948196846e-06, 'epoch': 1.89}
{'loss': 0.1182, 'grad_norm': 0.007893914356827736, 'learning_rate': 1.989286655743426e-06, 'epoch': 1.92}
{'loss': 0.1174, 'grad_norm': 0.0054678646847605705, 'learning_rate': 1.954593829518393e-06, 'epoch': 1.96}
{'loss': 0.1173, 'grad_norm': 0.01571636274456978, 'learning_rate': 1.9199010032933592e-06, 'epoch': 1.99}
{'eval_loss': 0.12219860404729843, 'eval_runtime': 8.9606, 'eval_samples_per_second': 615.475, 'eval_steps_per_second': 19.307, 'epoch': 2.0}
{'loss': 0.1177, 'grad_norm': 0.00882367417216301, 'learning_rate': 1.8852081770683256e-06, 'epoch': 2.03}
{'loss': 0.1149, 'grad_norm': 0.009024540893733501, 'learning_rate': 1.850515350843292e-06, 'epoch': 2.07}
{'loss': 0.1182, 'grad_norm': 0.007965550757944584, 'learning_rate': 1.8158225246182583e-06, 'epoch': 2.1}
{'loss': 0.1154, 'grad_norm': 0.006657850928604603, 'learning_rate': 1.7811296983932248e-06, 'epoch': 2.14}
{'loss': 0.1193, 'grad_norm': 0.0070901354774832726, 'learning_rate': 1.7464368721681911e-06, 'epoch': 2.18}
{'loss': 0.1193, 'grad_norm': 0.011779649183154106, 'learning_rate': 1.7117440459431575e-06, 'epoch': 2.21}
{'loss': 0.1162, 'grad_norm': 0.009884088300168514, 'learning_rate': 1.6770512197181238e-06, 'epoch': 2.25}
{'loss': 0.1174, 'grad_norm': 0.005248096771538258, 'learning_rate': 1.6423583934930904e-06, 'epoch': 2.28}
{'loss': 0.1162, 'grad_norm': 0.008880042470991611, 'learning_rate': 1.607665567268057e-06, 'epoch': 2.32}
{'loss': 0.1194, 'grad_norm': 0.011046412400901318, 'learning_rate': 1.5729727410430233e-06, 'epoch': 2.36}
{'loss': 0.117, 'grad_norm': 0.005969533231109381, 'learning_rate': 1.5382799148179896e-06, 'epoch': 2.39}
{'loss': 0.1161, 'grad_norm': 0.012903901748359203, 'learning_rate': 1.503587088592956e-06, 'epoch': 2.43}
{'loss': 0.1151, 'grad_norm': 0.016855750232934952, 'learning_rate': 1.4688942623679225e-06, 'epoch': 2.47}
{'loss': 0.1169, 'grad_norm': 0.009715547785162926, 'learning_rate': 1.4342014361428888e-06, 'epoch': 2.5}
{'loss': 0.1178, 'grad_norm': 0.0025616739876568317, 'learning_rate': 1.3995086099178552e-06, 'epoch': 2.54}
{'loss': 0.1162, 'grad_norm': 0.005489982198923826, 'learning_rate': 1.3648157836928215e-06, 'epoch': 2.57}
{'loss': 0.118, 'grad_norm': 0.0053294962272048, 'learning_rate': 1.330122957467788e-06, 'epoch': 2.61}
{'loss': 0.1186, 'grad_norm': 0.007491092197597027, 'learning_rate': 1.2954301312427544e-06, 'epoch': 2.65}
{'loss': 0.1209, 'grad_norm': 0.004181768745183945, 'learning_rate': 1.260737305017721e-06, 'epoch': 2.68}
{'loss': 0.1219, 'grad_norm': 0.016620540991425514, 'learning_rate': 1.226044478792687e-06, 'epoch': 2.72}
{'loss': 0.1209, 'grad_norm': 0.017047585919499397, 'learning_rate': 1.1913516525676536e-06, 'epoch': 2.76}
{'loss': 0.1182, 'grad_norm': 0.03153868392109871, 'learning_rate': 1.1566588263426202e-06, 'epoch': 2.79}
{'loss': 0.1197, 'grad_norm': 0.014936701394617558, 'learning_rate': 1.1219660001175865e-06, 'epoch': 2.83}
{'loss': 0.1197, 'grad_norm': 0.010323376394808292, 'learning_rate': 1.0872731738925529e-06, 'epoch': 2.86}
{'loss': 0.1159, 'grad_norm': 0.013829387724399567, 'learning_rate': 1.0525803476675192e-06, 'epoch': 2.9}
{'loss': 0.1205, 'grad_norm': 0.007800382096320391, 'learning_rate': 1.0178875214424857e-06, 'epoch': 2.94}
{'loss': 0.1171, 'grad_norm': 0.008745045401155949, 'learning_rate': 9.83194695217452e-07, 'epoch': 2.97}
{'eval_loss': 0.1235562190413475, 'eval_runtime': 11.0024, 'eval_samples_per_second': 501.253, 'eval_steps_per_second': 15.724, 'epoch': 3.0}
{'train_runtime': 369.7945, 'train_samples_per_second': 238.587, 'train_steps_per_second': 14.916, 'train_loss': 0.11716843757878359, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [06:09<02:03, 11.19it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:10<00:00, 16.11it/s]
[I 2025-08-12 16:46:49,177] Trial 15 finished with value: 0.12034811079502106 and parameters: {'lr': 3.827312589145706e-06, 'n_blocks': 4}. Best is trial 10 with value: 0.10707330703735352.
Best trial: 10. Best value: 0.107073:  75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                   | 15/20 [1:39:26<22:49, 273.87s/it][16:46:49] ft_study trial#15 value=0.12034811079502106 params={'lr': 3.827312589145706e-06, 'n_blocks': 4}
{'loss': 0.1698, 'grad_norm': 0.0038276452105492353, 'learning_rate': 4.7172795949732687e-05, 'epoch': 0.04}
{'loss': 0.1675, 'grad_norm': 0.0014197215205058455, 'learning_rate': 4.674136375703347e-05, 'epoch': 0.07}
{'loss': 0.1685, 'grad_norm': 0.01438509114086628, 'learning_rate': 4.630993156433425e-05, 'epoch': 0.11}
{'loss': 0.1756, 'grad_norm': 0.001707485062070191, 'learning_rate': 4.587849937163503e-05, 'epoch': 0.15}
{'loss': 0.1667, 'grad_norm': 0.0021517183631658554, 'learning_rate': 4.544706717893581e-05, 'epoch': 0.18}
{'loss': 0.1642, 'grad_norm': 0.0038346885703504086, 'learning_rate': 4.501563498623659e-05, 'epoch': 0.22}
{'loss': 0.1686, 'grad_norm': 0.003249110421165824, 'learning_rate': 4.4584202793537366e-05, 'epoch': 0.25}
{'loss': 0.1625, 'grad_norm': 0.0015323122497648, 'learning_rate': 4.415277060083815e-05, 'epoch': 0.29}
{'loss': 0.1685, 'grad_norm': 0.0012592421844601631, 'learning_rate': 4.372133840813893e-05, 'epoch': 0.33}
{'loss': 0.1688, 'grad_norm': 0.0032957058865576982, 'learning_rate': 4.328990621543971e-05, 'epoch': 0.36}
{'loss': 0.1668, 'grad_norm': 0.003997476305812597, 'learning_rate': 4.285847402274049e-05, 'epoch': 0.4}
{'loss': 0.1693, 'grad_norm': 0.0026265722699463367, 'learning_rate': 4.2427041830041274e-05, 'epoch': 0.44}
{'loss': 0.1648, 'grad_norm': 0.0028473453130573034, 'learning_rate': 4.199560963734205e-05, 'epoch': 0.47}
{'loss': 0.1684, 'grad_norm': 0.005975694395601749, 'learning_rate': 4.156417744464283e-05, 'epoch': 0.51}
{'loss': 0.1622, 'grad_norm': 0.010827799327671528, 'learning_rate': 4.113274525194361e-05, 'epoch': 0.54}
{'loss': 0.1725, 'grad_norm': 0.004255599807947874, 'learning_rate': 4.070131305924439e-05, 'epoch': 0.58}
{'loss': 0.1672, 'grad_norm': 0.005190827883780003, 'learning_rate': 4.026988086654517e-05, 'epoch': 0.62}
{'loss': 0.1672, 'grad_norm': 0.009010281413793564, 'learning_rate': 3.9838448673845954e-05, 'epoch': 0.65}
{'loss': 0.1662, 'grad_norm': 0.005931778345257044, 'learning_rate': 3.940701648114673e-05, 'epoch': 0.69}
{'loss': 0.1673, 'grad_norm': 0.012541467323899269, 'learning_rate': 3.897558428844751e-05, 'epoch': 0.73}
{'loss': 0.167, 'grad_norm': 0.006029016803950071, 'learning_rate': 3.85441520957483e-05, 'epoch': 0.76}
{'loss': 0.1753, 'grad_norm': 0.015827255323529243, 'learning_rate': 3.8112719903049076e-05, 'epoch': 0.8}
{'loss': 0.1686, 'grad_norm': 0.010895615443587303, 'learning_rate': 3.7681287710349855e-05, 'epoch': 0.83}
{'loss': 0.17, 'grad_norm': 0.005991565994918346, 'learning_rate': 3.7249855517650634e-05, 'epoch': 0.87}
{'loss': 0.1662, 'grad_norm': 0.006154979579150677, 'learning_rate': 3.681842332495141e-05, 'epoch': 0.91}
{'loss': 0.1713, 'grad_norm': 0.03133312240242958, 'learning_rate': 3.638699113225219e-05, 'epoch': 0.94}
{'loss': 0.1641, 'grad_norm': 0.0063062808476388454, 'learning_rate': 3.595555893955298e-05, 'epoch': 0.98}
{'eval_loss': 0.1755860447883606, 'eval_runtime': 6.6002, 'eval_samples_per_second': 835.587, 'eval_steps_per_second': 26.212, 'epoch': 1.0}
{'loss': 0.1698, 'grad_norm': 0.009220661595463753, 'learning_rate': 3.5524126746853756e-05, 'epoch': 1.02}
{'loss': 0.1695, 'grad_norm': 0.01087817270308733, 'learning_rate': 3.5092694554154535e-05, 'epoch': 1.05}
{'loss': 0.1765, 'grad_norm': 0.013265042565762997, 'learning_rate': 3.4661262361455314e-05, 'epoch': 1.09}
{'loss': 0.1686, 'grad_norm': 0.01684308983385563, 'learning_rate': 3.42298301687561e-05, 'epoch': 1.12}
{'loss': 0.1748, 'grad_norm': 0.01084132120013237, 'learning_rate': 3.379839797605688e-05, 'epoch': 1.16}
{'loss': 0.1703, 'grad_norm': 0.0160196665674448, 'learning_rate': 3.336696578335766e-05, 'epoch': 1.2}
{'loss': 0.1787, 'grad_norm': 0.007036636583507061, 'learning_rate': 3.2935533590658436e-05, 'epoch': 1.23}
{'loss': 0.1692, 'grad_norm': 0.010997781530022621, 'learning_rate': 3.2504101397959215e-05, 'epoch': 1.27}
{'loss': 0.1697, 'grad_norm': 0.020167166367173195, 'learning_rate': 3.2072669205259994e-05, 'epoch': 1.31}
{'loss': 0.1746, 'grad_norm': 0.014788680709898472, 'learning_rate': 3.164123701256078e-05, 'epoch': 1.34}
{'loss': 0.1736, 'grad_norm': 0.01813383772969246, 'learning_rate': 3.120980481986156e-05, 'epoch': 1.38}
{'loss': 0.1714, 'grad_norm': 0.012566424906253815, 'learning_rate': 3.077837262716234e-05, 'epoch': 1.41}
{'loss': 0.1742, 'grad_norm': 0.0196805689483881, 'learning_rate': 3.034694043446312e-05, 'epoch': 1.45}
{'loss': 0.1713, 'grad_norm': 0.020336581394076347, 'learning_rate': 2.9915508241763895e-05, 'epoch': 1.49}
{'loss': 0.1723, 'grad_norm': 0.015412400476634502, 'learning_rate': 2.9484076049064677e-05, 'epoch': 1.52}
{'loss': 0.1682, 'grad_norm': 0.05115647241473198, 'learning_rate': 2.9052643856365456e-05, 'epoch': 1.56}
{'loss': 0.1717, 'grad_norm': 0.015317469835281372, 'learning_rate': 2.8621211663666238e-05, 'epoch': 1.6}
{'loss': 0.1699, 'grad_norm': 0.017650507390499115, 'learning_rate': 2.818977947096702e-05, 'epoch': 1.63}
{'loss': 0.1733, 'grad_norm': 0.013578688725829124, 'learning_rate': 2.77583472782678e-05, 'epoch': 1.67}
{'loss': 0.1733, 'grad_norm': 0.022837186232209206, 'learning_rate': 2.732691508556858e-05, 'epoch': 1.7}
{'loss': 0.1749, 'grad_norm': 0.02095162682235241, 'learning_rate': 2.689548289286936e-05, 'epoch': 1.74}
{'loss': 0.1713, 'grad_norm': 0.029857033863663673, 'learning_rate': 2.6464050700170143e-05, 'epoch': 1.78}
{'loss': 0.1793, 'grad_norm': 0.026990296319127083, 'learning_rate': 2.6032618507470918e-05, 'epoch': 1.81}
{'loss': 0.1715, 'grad_norm': 0.01817922852933407, 'learning_rate': 2.56011863147717e-05, 'epoch': 1.85}
{'loss': 0.1776, 'grad_norm': 0.027888961136341095, 'learning_rate': 2.516975412207248e-05, 'epoch': 1.89}
{'loss': 0.1732, 'grad_norm': 0.02816246822476387, 'learning_rate': 2.473832192937326e-05, 'epoch': 1.92}
{'loss': 0.1759, 'grad_norm': 0.023842573165893555, 'learning_rate': 2.430688973667404e-05, 'epoch': 1.96}
{'loss': 0.1766, 'grad_norm': 0.020536459982395172, 'learning_rate': 2.3875457543974822e-05, 'epoch': 1.99}
{'eval_loss': 0.17990291118621826, 'eval_runtime': 6.7387, 'eval_samples_per_second': 818.41, 'eval_steps_per_second': 25.673, 'epoch': 2.0}
{'loss': 0.1722, 'grad_norm': 0.016780823469161987, 'learning_rate': 2.34440253512756e-05, 'epoch': 2.03}
{'loss': 0.1744, 'grad_norm': 0.01771506480872631, 'learning_rate': 2.301259315857638e-05, 'epoch': 2.07}
{'loss': 0.1772, 'grad_norm': 0.01836608722805977, 'learning_rate': 2.2581160965877162e-05, 'epoch': 2.1}
{'loss': 0.1756, 'grad_norm': 0.016562996432185173, 'learning_rate': 2.214972877317794e-05, 'epoch': 2.14}
{'loss': 0.1777, 'grad_norm': 0.01128630992025137, 'learning_rate': 2.1718296580478723e-05, 'epoch': 2.18}
{'loss': 0.1774, 'grad_norm': 0.021396905183792114, 'learning_rate': 2.1286864387779502e-05, 'epoch': 2.21}
{'loss': 0.1701, 'grad_norm': 0.030868755653500557, 'learning_rate': 2.085543219508028e-05, 'epoch': 2.25}
{'loss': 0.1754, 'grad_norm': 0.015358987264335155, 'learning_rate': 2.0424000002381063e-05, 'epoch': 2.28}
{'loss': 0.1705, 'grad_norm': 0.018471308052539825, 'learning_rate': 1.9992567809681846e-05, 'epoch': 2.32}
{'loss': 0.1725, 'grad_norm': 0.031134970486164093, 'learning_rate': 1.9561135616982625e-05, 'epoch': 2.36}
{'loss': 0.1792, 'grad_norm': 0.015795711427927017, 'learning_rate': 1.9129703424283403e-05, 'epoch': 2.39}
{'loss': 0.1753, 'grad_norm': 0.024377688765525818, 'learning_rate': 1.8698271231584186e-05, 'epoch': 2.43}
{'loss': 0.1777, 'grad_norm': 0.023129554465413094, 'learning_rate': 1.8266839038884964e-05, 'epoch': 2.47}
{'loss': 0.1765, 'grad_norm': 0.016024067997932434, 'learning_rate': 1.7835406846185747e-05, 'epoch': 2.5}
{'loss': 0.1721, 'grad_norm': 0.013439981266856194, 'learning_rate': 1.7403974653486526e-05, 'epoch': 2.54}
{'loss': 0.1784, 'grad_norm': 0.0296255461871624, 'learning_rate': 1.6972542460787304e-05, 'epoch': 2.57}
{'loss': 0.1741, 'grad_norm': 0.018758287653326988, 'learning_rate': 1.6541110268088087e-05, 'epoch': 2.61}
{'loss': 0.1741, 'grad_norm': 0.016705496236681938, 'learning_rate': 1.6109678075388866e-05, 'epoch': 2.65}
{'loss': 0.1741, 'grad_norm': 0.0174848735332489, 'learning_rate': 1.5678245882689648e-05, 'epoch': 2.68}
{'loss': 0.1736, 'grad_norm': 0.014422407373785973, 'learning_rate': 1.5246813689990425e-05, 'epoch': 2.72}
{'loss': 0.1727, 'grad_norm': 0.04952980950474739, 'learning_rate': 1.4815381497291205e-05, 'epoch': 2.76}
{'loss': 0.1755, 'grad_norm': 0.017884371802210808, 'learning_rate': 1.4383949304591988e-05, 'epoch': 2.79}
{'loss': 0.1763, 'grad_norm': 0.030666053295135498, 'learning_rate': 1.3952517111892768e-05, 'epoch': 2.83}
{'loss': 0.1794, 'grad_norm': 0.016558829694986343, 'learning_rate': 1.3521084919193545e-05, 'epoch': 2.86}
{'loss': 0.1792, 'grad_norm': 0.013462048955261707, 'learning_rate': 1.3089652726494328e-05, 'epoch': 2.9}
{'loss': 0.1737, 'grad_norm': 0.013616745360195637, 'learning_rate': 1.2658220533795108e-05, 'epoch': 2.94}
{'loss': 0.179, 'grad_norm': 0.03135327622294426, 'learning_rate': 1.2226788341095889e-05, 'epoch': 2.97}
{'eval_loss': 0.17973652482032776, 'eval_runtime': 6.3701, 'eval_samples_per_second': 865.762, 'eval_steps_per_second': 27.158, 'epoch': 3.0}
{'train_runtime': 229.3605, 'train_samples_per_second': 384.67, 'train_steps_per_second': 24.049, 'train_loss': 0.17213422145018234, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [03:49<01:16, 18.04it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:06<00:00, 25.87it/s]
[I 2025-08-12 16:50:45,679] Trial 16 finished with value: 0.1755860447883606 and parameters: {'lr': 4.7595599498577924e-05, 'n_blocks': 3}. Best is trial 10 with value: 0.10707330703735352.
Best trial: 10. Best value: 0.107073:  80%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                            | 16/20 [1:43:23<20:24, 306.11s/it][16:50:45] ft_study trial#16 value=0.1755860447883606 params={'lr': 4.7595599498577924e-05, 'n_blocks': 3}
{'loss': 0.1538, 'grad_norm': 0.0005310293054208159, 'learning_rate': 1.293562775544905e-05, 'epoch': 0.04}
{'loss': 0.1569, 'grad_norm': 0.0006358601385727525, 'learning_rate': 1.2817321300762303e-05, 'epoch': 0.07}
{'loss': 0.1581, 'grad_norm': 0.0040433271788060665, 'learning_rate': 1.2699014846075554e-05, 'epoch': 0.11}
{'loss': 0.1585, 'grad_norm': 0.0010521318763494492, 'learning_rate': 1.2580708391388807e-05, 'epoch': 0.15}
{'loss': 0.1564, 'grad_norm': 0.0005400338559411466, 'learning_rate': 1.2462401936702058e-05, 'epoch': 0.18}
{'loss': 0.1561, 'grad_norm': 0.0008253469131886959, 'learning_rate': 1.2344095482015309e-05, 'epoch': 0.22}
{'loss': 0.1546, 'grad_norm': 0.0004477144102565944, 'learning_rate': 1.2225789027328562e-05, 'epoch': 0.25}
{'loss': 0.1581, 'grad_norm': 0.0004881001659668982, 'learning_rate': 1.2107482572641813e-05, 'epoch': 0.29}
{'loss': 0.1546, 'grad_norm': 0.0008242477779276669, 'learning_rate': 1.1989176117955065e-05, 'epoch': 0.33}
{'loss': 0.1573, 'grad_norm': 0.0005943109281361103, 'learning_rate': 1.1870869663268316e-05, 'epoch': 0.36}
{'loss': 0.156, 'grad_norm': 0.0004812637635041028, 'learning_rate': 1.1752563208581569e-05, 'epoch': 0.4}
{'loss': 0.155, 'grad_norm': 0.0008943293360061944, 'learning_rate': 1.163425675389482e-05, 'epoch': 0.44}
{'loss': 0.1576, 'grad_norm': 0.0011353585869073868, 'learning_rate': 1.1515950299208073e-05, 'epoch': 0.47}
{'loss': 0.1562, 'grad_norm': 0.0006859474233351648, 'learning_rate': 1.1397643844521325e-05, 'epoch': 0.51}
{'loss': 0.1559, 'grad_norm': 0.000591604330111295, 'learning_rate': 1.1279337389834575e-05, 'epoch': 0.54}
{'loss': 0.1553, 'grad_norm': 0.0006564960931427777, 'learning_rate': 1.1161030935147827e-05, 'epoch': 0.58}
{'loss': 0.154, 'grad_norm': 0.000631409406196326, 'learning_rate': 1.1042724480461078e-05, 'epoch': 0.62}
{'loss': 0.1555, 'grad_norm': 0.001249804045073688, 'learning_rate': 1.0924418025774331e-05, 'epoch': 0.65}
{'loss': 0.1558, 'grad_norm': 0.000681118865031749, 'learning_rate': 1.0806111571087584e-05, 'epoch': 0.69}
{'loss': 0.1559, 'grad_norm': 0.0013317398261278868, 'learning_rate': 1.0687805116400835e-05, 'epoch': 0.73}
{'loss': 0.1553, 'grad_norm': 0.0008085997542366385, 'learning_rate': 1.0569498661714087e-05, 'epoch': 0.76}
{'loss': 0.157, 'grad_norm': 0.001172361196950078, 'learning_rate': 1.0451192207027338e-05, 'epoch': 0.8}
{'loss': 0.1558, 'grad_norm': 0.000938587763812393, 'learning_rate': 1.0332885752340591e-05, 'epoch': 0.83}
{'loss': 0.1554, 'grad_norm': 0.0004679739067796618, 'learning_rate': 1.021457929765384e-05, 'epoch': 0.87}
{'loss': 0.1576, 'grad_norm': 0.0009219283820129931, 'learning_rate': 1.0096272842967093e-05, 'epoch': 0.91}
{'loss': 0.1545, 'grad_norm': 0.001482514082454145, 'learning_rate': 9.977966388280346e-06, 'epoch': 0.94}
{'loss': 0.157, 'grad_norm': 0.0005943455034866929, 'learning_rate': 9.859659933593597e-06, 'epoch': 0.98}
{'eval_loss': 0.16045856475830078, 'eval_runtime': 3.7842, 'eval_samples_per_second': 1457.391, 'eval_steps_per_second': 45.717, 'epoch': 1.0}
{'loss': 0.1556, 'grad_norm': 0.0011018735822290182, 'learning_rate': 9.74135347890685e-06, 'epoch': 1.02}
{'loss': 0.1571, 'grad_norm': 0.0008822968229651451, 'learning_rate': 9.6230470242201e-06, 'epoch': 1.05}
{'loss': 0.157, 'grad_norm': 0.0015371596673503518, 'learning_rate': 9.504740569533353e-06, 'epoch': 1.09}
{'loss': 0.1571, 'grad_norm': 0.0008082269923761487, 'learning_rate': 9.386434114846604e-06, 'epoch': 1.12}
{'loss': 0.1555, 'grad_norm': 0.0008396374178119004, 'learning_rate': 9.268127660159857e-06, 'epoch': 1.16}
{'loss': 0.1557, 'grad_norm': 0.0005001574754714966, 'learning_rate': 9.149821205473107e-06, 'epoch': 1.2}
{'loss': 0.1553, 'grad_norm': 0.0006723633268848062, 'learning_rate': 9.031514750786358e-06, 'epoch': 1.23}
{'loss': 0.1561, 'grad_norm': 0.0010233228094875813, 'learning_rate': 8.913208296099611e-06, 'epoch': 1.27}
{'loss': 0.1549, 'grad_norm': 0.0009622383513487875, 'learning_rate': 8.794901841412862e-06, 'epoch': 1.31}
{'loss': 0.1568, 'grad_norm': 0.0008170299115590751, 'learning_rate': 8.676595386726115e-06, 'epoch': 1.34}
{'loss': 0.1558, 'grad_norm': 0.0010038305772468448, 'learning_rate': 8.558288932039368e-06, 'epoch': 1.38}
{'loss': 0.1565, 'grad_norm': 0.0011943213175982237, 'learning_rate': 8.439982477352618e-06, 'epoch': 1.41}
{'loss': 0.1554, 'grad_norm': 0.0008764427620917559, 'learning_rate': 8.321676022665871e-06, 'epoch': 1.45}
{'loss': 0.1568, 'grad_norm': 0.0007728001801297069, 'learning_rate': 8.20336956797912e-06, 'epoch': 1.49}
{'loss': 0.1541, 'grad_norm': 0.0007759526488371193, 'learning_rate': 8.085063113292373e-06, 'epoch': 1.52}
{'loss': 0.1562, 'grad_norm': 0.007324470207095146, 'learning_rate': 7.966756658605624e-06, 'epoch': 1.56}
{'loss': 0.1576, 'grad_norm': 0.0010970287257805467, 'learning_rate': 7.848450203918877e-06, 'epoch': 1.6}
{'loss': 0.1571, 'grad_norm': 0.0007394492859020829, 'learning_rate': 7.73014374923213e-06, 'epoch': 1.63}
{'loss': 0.1558, 'grad_norm': 0.0012032793601974845, 'learning_rate': 7.6118372945453805e-06, 'epoch': 1.67}
{'loss': 0.1556, 'grad_norm': 0.0015686386032029986, 'learning_rate': 7.493530839858632e-06, 'epoch': 1.7}
{'loss': 0.1563, 'grad_norm': 0.001225635060109198, 'learning_rate': 7.375224385171885e-06, 'epoch': 1.74}
{'loss': 0.1597, 'grad_norm': 0.0014841905795037746, 'learning_rate': 7.256917930485137e-06, 'epoch': 1.78}
{'loss': 0.1531, 'grad_norm': 0.0009990812977775931, 'learning_rate': 7.138611475798387e-06, 'epoch': 1.81}
{'loss': 0.156, 'grad_norm': 0.0006542380433529615, 'learning_rate': 7.020305021111639e-06, 'epoch': 1.85}
{'loss': 0.1556, 'grad_norm': 0.000974342052359134, 'learning_rate': 6.901998566424891e-06, 'epoch': 1.89}
{'loss': 0.1564, 'grad_norm': 0.0015884236199781299, 'learning_rate': 6.783692111738143e-06, 'epoch': 1.92}
{'loss': 0.157, 'grad_norm': 0.0010228547034785151, 'learning_rate': 6.665385657051395e-06, 'epoch': 1.96}
{'loss': 0.1531, 'grad_norm': 0.0019116823095828295, 'learning_rate': 6.547079202364647e-06, 'epoch': 1.99}
{'eval_loss': 0.16059698164463043, 'eval_runtime': 3.8857, 'eval_samples_per_second': 1419.315, 'eval_steps_per_second': 44.522, 'epoch': 2.0}
{'loss': 0.156, 'grad_norm': 0.0014806651743128896, 'learning_rate': 6.428772747677899e-06, 'epoch': 2.03}
{'loss': 0.1537, 'grad_norm': 0.0018282340606674552, 'learning_rate': 6.31046629299115e-06, 'epoch': 2.07}
{'loss': 0.157, 'grad_norm': 0.001302298973314464, 'learning_rate': 6.192159838304402e-06, 'epoch': 2.1}
{'loss': 0.1559, 'grad_norm': 0.0007672425708733499, 'learning_rate': 6.0738533836176534e-06, 'epoch': 2.14}
{'loss': 0.1573, 'grad_norm': 0.0010513317538425326, 'learning_rate': 5.955546928930906e-06, 'epoch': 2.18}
{'loss': 0.1569, 'grad_norm': 0.0013532512821257114, 'learning_rate': 5.837240474244157e-06, 'epoch': 2.21}
{'loss': 0.157, 'grad_norm': 0.001320614363066852, 'learning_rate': 5.718934019557409e-06, 'epoch': 2.25}
{'loss': 0.1565, 'grad_norm': 0.0019403512123972178, 'learning_rate': 5.600627564870661e-06, 'epoch': 2.28}
{'loss': 0.1547, 'grad_norm': 0.0010761625599116087, 'learning_rate': 5.482321110183913e-06, 'epoch': 2.32}
{'loss': 0.157, 'grad_norm': 0.001373421517200768, 'learning_rate': 5.3640146554971644e-06, 'epoch': 2.36}
{'loss': 0.154, 'grad_norm': 0.0008307655807584524, 'learning_rate': 5.245708200810416e-06, 'epoch': 2.39}
{'loss': 0.1548, 'grad_norm': 0.0011739389738067985, 'learning_rate': 5.127401746123668e-06, 'epoch': 2.43}
{'loss': 0.1544, 'grad_norm': 0.001790322596207261, 'learning_rate': 5.00909529143692e-06, 'epoch': 2.47}
{'loss': 0.1552, 'grad_norm': 0.0007839623722247779, 'learning_rate': 4.890788836750172e-06, 'epoch': 2.5}
{'loss': 0.1567, 'grad_norm': 0.000817371706943959, 'learning_rate': 4.772482382063423e-06, 'epoch': 2.54}
{'loss': 0.1554, 'grad_norm': 0.001068132696673274, 'learning_rate': 4.6541759273766746e-06, 'epoch': 2.57}
{'loss': 0.1565, 'grad_norm': 0.0011188127100467682, 'learning_rate': 4.535869472689927e-06, 'epoch': 2.61}
{'loss': 0.1579, 'grad_norm': 0.001421657158061862, 'learning_rate': 4.417563018003179e-06, 'epoch': 2.65}
{'loss': 0.1563, 'grad_norm': 0.0006574337603524327, 'learning_rate': 4.299256563316431e-06, 'epoch': 2.68}
{'loss': 0.1557, 'grad_norm': 0.0017200239235535264, 'learning_rate': 4.180950108629682e-06, 'epoch': 2.72}
{'loss': 0.1579, 'grad_norm': 0.0019739565905183554, 'learning_rate': 4.062643653942934e-06, 'epoch': 2.76}
{'loss': 0.1572, 'grad_norm': 0.007919065654277802, 'learning_rate': 3.9443371992561856e-06, 'epoch': 2.79}
{'loss': 0.1567, 'grad_norm': 0.0010478930780664086, 'learning_rate': 3.826030744569437e-06, 'epoch': 2.83}
{'loss': 0.156, 'grad_norm': 0.0017748450627550483, 'learning_rate': 3.707724289882689e-06, 'epoch': 2.86}
{'loss': 0.1552, 'grad_norm': 0.0008287576492875814, 'learning_rate': 3.589417835195941e-06, 'epoch': 2.9}
{'loss': 0.1534, 'grad_norm': 0.0007748928037472069, 'learning_rate': 3.471111380509193e-06, 'epoch': 2.94}
{'loss': 0.1556, 'grad_norm': 0.0007278804550878704, 'learning_rate': 3.3528049258224447e-06, 'epoch': 2.97}
{'eval_loss': 0.16071860492229462, 'eval_runtime': 3.7031, 'eval_samples_per_second': 1489.274, 'eval_steps_per_second': 46.717, 'epoch': 3.0}
{'train_runtime': 137.2329, 'train_samples_per_second': 642.907, 'train_steps_per_second': 40.194, 'train_loss': 0.15601118814838605, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [02:17<00:45, 30.15it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:03<00:00, 45.10it/s]
[I 2025-08-12 16:53:07,197] Trial 17 finished with value: 0.16045856475830078 and parameters: {'lr': 1.3051568081042064e-05, 'n_blocks': 2}. Best is trial 10 with value: 0.10707330703735352.
Best trial: 10. Best value: 0.107073:  85%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                     | 17/20 [1:45:45<14:15, 285.18s/it][16:53:07] ft_study trial#17 value=0.16045856475830078 params={'lr': 1.3051568081042064e-05, 'n_blocks': 2}
{'loss': 0.1191, 'grad_norm': 0.006907169707119465, 'learning_rate': 5.9557807184250835e-05, 'epoch': 0.04}
{'loss': 0.1181, 'grad_norm': 0.005575005430728197, 'learning_rate': 5.901310435651852e-05, 'epoch': 0.07}
{'loss': 0.121, 'grad_norm': 0.026050694286823273, 'learning_rate': 5.8468401528786216e-05, 'epoch': 0.11}
{'loss': 0.122, 'grad_norm': 0.008938413113355637, 'learning_rate': 5.7923698701053904e-05, 'epoch': 0.15}
{'loss': 0.119, 'grad_norm': 0.006803914904594421, 'learning_rate': 5.73789958733216e-05, 'epoch': 0.18}
{'loss': 0.1209, 'grad_norm': 0.031196219846606255, 'learning_rate': 5.683429304558928e-05, 'epoch': 0.22}
{'loss': 0.1281, 'grad_norm': 0.021145176142454147, 'learning_rate': 5.628959021785697e-05, 'epoch': 0.25}
{'loss': 0.1294, 'grad_norm': 0.013797874562442303, 'learning_rate': 5.574488739012466e-05, 'epoch': 0.29}
{'loss': 0.1328, 'grad_norm': 0.012219354510307312, 'learning_rate': 5.520018456239235e-05, 'epoch': 0.33}
{'loss': 0.1367, 'grad_norm': 0.01707327738404274, 'learning_rate': 5.465548173466004e-05, 'epoch': 0.36}
{'loss': 0.1394, 'grad_norm': 0.02197483740746975, 'learning_rate': 5.411077890692773e-05, 'epoch': 0.4}
{'loss': 0.1392, 'grad_norm': 0.05722589045763016, 'learning_rate': 5.356607607919542e-05, 'epoch': 0.44}
{'loss': 0.1417, 'grad_norm': 0.020903104916214943, 'learning_rate': 5.302137325146311e-05, 'epoch': 0.47}
{'loss': 0.1439, 'grad_norm': 0.02482718974351883, 'learning_rate': 5.24766704237308e-05, 'epoch': 0.51}
{'loss': 0.1478, 'grad_norm': 0.03745199739933014, 'learning_rate': 5.1931967595998484e-05, 'epoch': 0.54}
{'loss': 0.1491, 'grad_norm': 0.023401232436299324, 'learning_rate': 5.138726476826618e-05, 'epoch': 0.58}
{'loss': 0.1432, 'grad_norm': 0.036261796951293945, 'learning_rate': 5.0842561940533865e-05, 'epoch': 0.62}
{'loss': 0.1486, 'grad_norm': 0.08383594453334808, 'learning_rate': 5.029785911280155e-05, 'epoch': 0.65}
{'loss': 0.145, 'grad_norm': 0.037596747279167175, 'learning_rate': 4.9753156285069246e-05, 'epoch': 0.69}
{'loss': 0.1474, 'grad_norm': 0.06136300787329674, 'learning_rate': 4.9208453457336934e-05, 'epoch': 0.73}
{'loss': 0.1491, 'grad_norm': 0.026197101920843124, 'learning_rate': 4.866375062960463e-05, 'epoch': 0.76}
{'loss': 0.1468, 'grad_norm': 0.04617845639586449, 'learning_rate': 4.8119047801872315e-05, 'epoch': 0.8}
{'loss': 0.1479, 'grad_norm': 0.17076168954372406, 'learning_rate': 4.757434497414001e-05, 'epoch': 0.83}
{'loss': 0.1467, 'grad_norm': 0.03546784073114395, 'learning_rate': 4.702964214640769e-05, 'epoch': 0.87}
{'loss': 0.1524, 'grad_norm': 0.04964607581496239, 'learning_rate': 4.6484939318675377e-05, 'epoch': 0.91}
{'loss': 0.1455, 'grad_norm': 0.1248733326792717, 'learning_rate': 4.594023649094307e-05, 'epoch': 0.94}
{'loss': 0.15, 'grad_norm': 0.024594629183411598, 'learning_rate': 4.539553366321076e-05, 'epoch': 0.98}
{'eval_loss': 0.16904176771640778, 'eval_runtime': 9.1741, 'eval_samples_per_second': 601.149, 'eval_steps_per_second': 18.857, 'epoch': 1.0}
{'loss': 0.1475, 'grad_norm': 0.030925238505005836, 'learning_rate': 4.485083083547845e-05, 'epoch': 1.02}
{'loss': 0.1538, 'grad_norm': 0.041170064359903336, 'learning_rate': 4.430612800774614e-05, 'epoch': 1.05}
{'loss': 0.148, 'grad_norm': 0.056852009147405624, 'learning_rate': 4.376142518001383e-05, 'epoch': 1.09}
{'loss': 0.1522, 'grad_norm': 0.04503636434674263, 'learning_rate': 4.321672235228152e-05, 'epoch': 1.12}
{'loss': 0.1553, 'grad_norm': 0.055863216519355774, 'learning_rate': 4.2672019524549214e-05, 'epoch': 1.16}
{'loss': 0.1555, 'grad_norm': 0.0648103728890419, 'learning_rate': 4.2127316696816895e-05, 'epoch': 1.2}
{'loss': 0.1563, 'grad_norm': 0.026365937665104866, 'learning_rate': 4.158261386908458e-05, 'epoch': 1.23}
{'loss': 0.1553, 'grad_norm': 0.05642728507518768, 'learning_rate': 4.1037911041352276e-05, 'epoch': 1.27}
{'loss': 0.147, 'grad_norm': 0.051251012831926346, 'learning_rate': 4.0493208213619963e-05, 'epoch': 1.31}
{'loss': 0.1471, 'grad_norm': 0.04143333435058594, 'learning_rate': 3.994850538588766e-05, 'epoch': 1.34}
{'loss': 0.1508, 'grad_norm': 0.04434764385223389, 'learning_rate': 3.9403802558155345e-05, 'epoch': 1.38}
{'loss': 0.1512, 'grad_norm': 0.05612935125827789, 'learning_rate': 3.885909973042304e-05, 'epoch': 1.41}
{'loss': 0.1483, 'grad_norm': 0.04592828452587128, 'learning_rate': 3.8314396902690726e-05, 'epoch': 1.45}
{'loss': 0.1498, 'grad_norm': 0.061548635363578796, 'learning_rate': 3.7769694074958406e-05, 'epoch': 1.49}
{'loss': 0.1543, 'grad_norm': 0.07053104788064957, 'learning_rate': 3.72249912472261e-05, 'epoch': 1.52}
{'loss': 0.147, 'grad_norm': 0.07652126997709274, 'learning_rate': 3.668028841949379e-05, 'epoch': 1.56}
{'loss': 0.1479, 'grad_norm': 0.046205874532461166, 'learning_rate': 3.613558559176148e-05, 'epoch': 1.6}
{'loss': 0.1519, 'grad_norm': 0.045915208756923676, 'learning_rate': 3.559088276402917e-05, 'epoch': 1.63}
{'loss': 0.1553, 'grad_norm': 0.027971724048256874, 'learning_rate': 3.504617993629686e-05, 'epoch': 1.67}
{'loss': 0.1547, 'grad_norm': 0.07183609157800674, 'learning_rate': 3.450147710856455e-05, 'epoch': 1.7}
{'loss': 0.1565, 'grad_norm': 0.0984317809343338, 'learning_rate': 3.3956774280832244e-05, 'epoch': 1.74}
{'loss': 0.1495, 'grad_norm': 0.052796170115470886, 'learning_rate': 3.341207145309993e-05, 'epoch': 1.78}
{'loss': 0.1509, 'grad_norm': 0.030243944376707077, 'learning_rate': 3.286736862536761e-05, 'epoch': 1.81}
{'loss': 0.1496, 'grad_norm': 0.02680211327970028, 'learning_rate': 3.2322665797635306e-05, 'epoch': 1.85}
{'loss': 0.1472, 'grad_norm': 0.06766975671052933, 'learning_rate': 3.177796296990299e-05, 'epoch': 1.89}
{'loss': 0.1539, 'grad_norm': 0.08691871166229248, 'learning_rate': 3.123326014217069e-05, 'epoch': 1.92}
{'loss': 0.1542, 'grad_norm': 0.04375525191426277, 'learning_rate': 3.0688557314438375e-05, 'epoch': 1.96}
{'loss': 0.1515, 'grad_norm': 0.06259069591760635, 'learning_rate': 3.0143854486706065e-05, 'epoch': 1.99}
{'eval_loss': 0.1639666110277176, 'eval_runtime': 6.4556, 'eval_samples_per_second': 854.294, 'eval_steps_per_second': 26.798, 'epoch': 2.0}
{'loss': 0.1508, 'grad_norm': 0.04341384395956993, 'learning_rate': 2.9599151658973756e-05, 'epoch': 2.03}
{'loss': 0.1488, 'grad_norm': 0.05654818192124367, 'learning_rate': 2.9054448831241443e-05, 'epoch': 2.07}
{'loss': 0.1511, 'grad_norm': 0.07315435260534286, 'learning_rate': 2.8509746003509134e-05, 'epoch': 2.1}
{'loss': 0.1543, 'grad_norm': 0.0362158827483654, 'learning_rate': 2.7965043175776824e-05, 'epoch': 2.14}
{'loss': 0.1573, 'grad_norm': 0.030760549008846283, 'learning_rate': 2.7420340348044515e-05, 'epoch': 2.18}
{'loss': 0.1548, 'grad_norm': 0.0741860419511795, 'learning_rate': 2.68756375203122e-05, 'epoch': 2.21}
{'loss': 0.1556, 'grad_norm': 0.06832625716924667, 'learning_rate': 2.633093469257989e-05, 'epoch': 2.25}
{'loss': 0.1545, 'grad_norm': 0.08415339887142181, 'learning_rate': 2.578623186484758e-05, 'epoch': 2.28}
{'loss': 0.1483, 'grad_norm': 0.06681852042675018, 'learning_rate': 2.524152903711527e-05, 'epoch': 2.32}
{'loss': 0.1538, 'grad_norm': 0.09245530515909195, 'learning_rate': 2.469682620938296e-05, 'epoch': 2.36}
{'loss': 0.1532, 'grad_norm': 0.04167932644486427, 'learning_rate': 2.415212338165065e-05, 'epoch': 2.39}
{'loss': 0.1506, 'grad_norm': 0.07325978577136993, 'learning_rate': 2.360742055391834e-05, 'epoch': 2.43}
{'loss': 0.1498, 'grad_norm': 0.093216173350811, 'learning_rate': 2.306271772618603e-05, 'epoch': 2.47}
{'loss': 0.148, 'grad_norm': 0.044852737337350845, 'learning_rate': 2.2518014898453717e-05, 'epoch': 2.5}
{'loss': 0.1519, 'grad_norm': 0.015793083235621452, 'learning_rate': 2.1973312070721405e-05, 'epoch': 2.54}
{'loss': 0.1488, 'grad_norm': 0.0734601691365242, 'learning_rate': 2.1428609242989095e-05, 'epoch': 2.57}
{'loss': 0.1488, 'grad_norm': 0.0445973239839077, 'learning_rate': 2.0883906415256786e-05, 'epoch': 2.61}
{'loss': 0.1537, 'grad_norm': 0.037213005125522614, 'learning_rate': 2.0339203587524476e-05, 'epoch': 2.65}
{'loss': 0.1501, 'grad_norm': 0.07471365481615067, 'learning_rate': 1.9794500759792167e-05, 'epoch': 2.68}
{'loss': 0.156, 'grad_norm': 0.07394533604383469, 'learning_rate': 1.9249797932059854e-05, 'epoch': 2.72}
{'loss': 0.1549, 'grad_norm': 0.10761819779872894, 'learning_rate': 1.8705095104327545e-05, 'epoch': 2.76}
{'loss': 0.1554, 'grad_norm': 0.09379156678915024, 'learning_rate': 1.8160392276595232e-05, 'epoch': 2.79}
{'loss': 0.1604, 'grad_norm': 0.050000790506601334, 'learning_rate': 1.7615689448862923e-05, 'epoch': 2.83}
{'loss': 0.1517, 'grad_norm': 0.04970264807343483, 'learning_rate': 1.707098662113061e-05, 'epoch': 2.86}
{'loss': 0.1532, 'grad_norm': 0.06864573061466217, 'learning_rate': 1.65262837933983e-05, 'epoch': 2.9}
{'loss': 0.1561, 'grad_norm': 0.04327084496617317, 'learning_rate': 1.598158096566599e-05, 'epoch': 2.94}
{'loss': 0.1549, 'grad_norm': 0.06062773987650871, 'learning_rate': 1.5436878137933682e-05, 'epoch': 2.97}
{'eval_loss': 0.16509336233139038, 'eval_runtime': 10.5083, 'eval_samples_per_second': 524.822, 'eval_steps_per_second': 16.463, 'epoch': 3.0}
{'loss': 0.1545, 'grad_norm': 0.05434630066156387, 'learning_rate': 1.489217531020137e-05, 'epoch': 3.01}
{'loss': 0.159, 'grad_norm': 0.06502946466207504, 'learning_rate': 1.434747248246906e-05, 'epoch': 3.05}
{'loss': 0.1558, 'grad_norm': 0.029232202097773552, 'learning_rate': 1.3802769654736749e-05, 'epoch': 3.08}
{'loss': 0.1588, 'grad_norm': 0.10376705229282379, 'learning_rate': 1.325806682700444e-05, 'epoch': 3.12}
{'loss': 0.1542, 'grad_norm': 0.036312803626060486, 'learning_rate': 1.2713363999272127e-05, 'epoch': 3.15}
{'loss': 0.1528, 'grad_norm': 0.06372179090976715, 'learning_rate': 1.2168661171539817e-05, 'epoch': 3.19}
{'loss': 0.1537, 'grad_norm': 0.06685247272253036, 'learning_rate': 1.1623958343807506e-05, 'epoch': 3.23}
{'loss': 0.1527, 'grad_norm': 0.02034807577729225, 'learning_rate': 1.1079255516075197e-05, 'epoch': 3.26}
{'loss': 0.1551, 'grad_norm': 0.05701436847448349, 'learning_rate': 1.0534552688342884e-05, 'epoch': 3.3}
{'loss': 0.15, 'grad_norm': 0.03135025501251221, 'learning_rate': 9.989849860610575e-06, 'epoch': 3.34}
{'loss': 0.1507, 'grad_norm': 0.07844045013189316, 'learning_rate': 9.445147032878266e-06, 'epoch': 3.37}
{'loss': 0.1516, 'grad_norm': 0.042888443917036057, 'learning_rate': 8.900444205145954e-06, 'epoch': 3.41}
{'loss': 0.1583, 'grad_norm': 0.042342688888311386, 'learning_rate': 8.355741377413643e-06, 'epoch': 3.44}
{'loss': 0.1591, 'grad_norm': 0.037472017109394073, 'learning_rate': 7.811038549681332e-06, 'epoch': 3.48}
{'loss': 0.1515, 'grad_norm': 0.046597957611083984, 'learning_rate': 7.266335721949023e-06, 'epoch': 3.52}
{'loss': 0.1551, 'grad_norm': 0.03285448998212814, 'learning_rate': 6.721632894216712e-06, 'epoch': 3.55}
{'loss': 0.1538, 'grad_norm': 0.04952247813344002, 'learning_rate': 6.176930066484402e-06, 'epoch': 3.59}
{'loss': 0.1537, 'grad_norm': 0.03870820626616478, 'learning_rate': 5.632227238752091e-06, 'epoch': 3.63}
{'loss': 0.1564, 'grad_norm': 0.13623088598251343, 'learning_rate': 5.0875244110197805e-06, 'epoch': 3.66}
{'loss': 0.1539, 'grad_norm': 0.0554799810051918, 'learning_rate': 4.542821583287469e-06, 'epoch': 3.7}
{'loss': 0.152, 'grad_norm': 0.022984085604548454, 'learning_rate': 3.998118755555159e-06, 'epoch': 3.73}
{'loss': 0.1569, 'grad_norm': 0.07613282650709152, 'learning_rate': 3.453415927822849e-06, 'epoch': 3.77}
{'loss': 0.1553, 'grad_norm': 0.027094904333353043, 'learning_rate': 2.9087131000905384e-06, 'epoch': 3.81}
{'loss': 0.1568, 'grad_norm': 0.03916410356760025, 'learning_rate': 2.3640102723582277e-06, 'epoch': 3.84}
{'loss': 0.1575, 'grad_norm': 0.08053787797689438, 'learning_rate': 1.8193074446259171e-06, 'epoch': 3.88}
{'loss': 0.155, 'grad_norm': 0.06106957793235779, 'learning_rate': 1.2746046168936067e-06, 'epoch': 3.92}
{'loss': 0.1527, 'grad_norm': 0.10212863981723785, 'learning_rate': 7.299017891612961e-07, 'epoch': 3.95}
{'loss': 0.1525, 'grad_norm': 0.04268335923552513, 'learning_rate': 1.851989614289856e-07, 'epoch': 3.99}
{'eval_loss': 0.1634370982646942, 'eval_runtime': 11.1334, 'eval_samples_per_second': 495.354, 'eval_steps_per_second': 15.539, 'epoch': 4.0}
{'train_runtime': 469.5536, 'train_samples_per_second': 187.898, 'train_steps_per_second': 11.747, 'train_loss': 0.1493627858213794, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5516/5516 [07:49<00:00, 11.75it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:11<00:00, 15.48it/s]
[I 2025-08-12 17:01:08,412] Trial 18 finished with value: 0.1634370982646942 and parameters: {'lr': 6.00916159554285e-05, 'n_blocks': 4}. Best is trial 10 with value: 0.10707330703735352.
Best trial: 10. Best value: 0.107073:  90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊              | 18/20 [1:53:46<08:04, 242.01s/it][17:01:08] ft_study trial#18 value=0.1634370982646942 params={'lr': 6.00916159554285e-05, 'n_blocks': 4}
{'loss': 0.1013, 'grad_norm': 0.002285267924889922, 'learning_rate': 3.126029372505408e-06, 'epoch': 0.04}
{'loss': 0.0973, 'grad_norm': 0.0015194586012512445, 'learning_rate': 3.097439383731808e-06, 'epoch': 0.07}
{'loss': 0.0991, 'grad_norm': 0.0053396024741232395, 'learning_rate': 3.0688493949582085e-06, 'epoch': 0.11}
{'loss': 0.0972, 'grad_norm': 0.0023663230240345, 'learning_rate': 3.0402594061846086e-06, 'epoch': 0.15}
{'loss': 0.0976, 'grad_norm': 0.0017154898960143328, 'learning_rate': 3.011669417411009e-06, 'epoch': 0.18}
{'loss': 0.1001, 'grad_norm': 0.00336571061052382, 'learning_rate': 2.9830794286374086e-06, 'epoch': 0.22}
{'loss': 0.0987, 'grad_norm': 0.0016051499405875802, 'learning_rate': 2.954489439863809e-06, 'epoch': 0.25}
{'loss': 0.103, 'grad_norm': 0.0011538441758602858, 'learning_rate': 2.925899451090209e-06, 'epoch': 0.29}
{'loss': 0.0979, 'grad_norm': 0.0008157997508533299, 'learning_rate': 2.897309462316609e-06, 'epoch': 0.33}
{'loss': 0.1003, 'grad_norm': 0.0007756354752928019, 'learning_rate': 2.8687194735430095e-06, 'epoch': 0.36}
{'loss': 0.0976, 'grad_norm': 0.001756690558977425, 'learning_rate': 2.8401294847694095e-06, 'epoch': 0.4}
{'loss': 0.0979, 'grad_norm': 0.002100331010296941, 'learning_rate': 2.81153949599581e-06, 'epoch': 0.44}
{'loss': 0.0996, 'grad_norm': 0.0010467276442795992, 'learning_rate': 2.78294950722221e-06, 'epoch': 0.47}
{'loss': 0.101, 'grad_norm': 0.0015288612339645624, 'learning_rate': 2.75435951844861e-06, 'epoch': 0.51}
{'loss': 0.1031, 'grad_norm': 0.0024432651698589325, 'learning_rate': 2.72576952967501e-06, 'epoch': 0.54}
{'loss': 0.1002, 'grad_norm': 0.0019974298775196075, 'learning_rate': 2.69717954090141e-06, 'epoch': 0.58}
{'loss': 0.0992, 'grad_norm': 0.0017580511048436165, 'learning_rate': 2.6685895521278105e-06, 'epoch': 0.62}
{'loss': 0.1, 'grad_norm': 0.00233367714099586, 'learning_rate': 2.6399995633542105e-06, 'epoch': 0.65}
{'loss': 0.0979, 'grad_norm': 0.001728188362903893, 'learning_rate': 2.611409574580611e-06, 'epoch': 0.69}
{'loss': 0.099, 'grad_norm': 0.002665087813511491, 'learning_rate': 2.582819585807011e-06, 'epoch': 0.73}
{'loss': 0.1006, 'grad_norm': 0.004630428738892078, 'learning_rate': 2.554229597033411e-06, 'epoch': 0.76}
{'loss': 0.0989, 'grad_norm': 0.003925323951989412, 'learning_rate': 2.5256396082598114e-06, 'epoch': 0.8}
{'loss': 0.099, 'grad_norm': 0.0024612797424197197, 'learning_rate': 2.4970496194862114e-06, 'epoch': 0.83}
{'loss': 0.1018, 'grad_norm': 0.0011455015046522021, 'learning_rate': 2.4684596307126114e-06, 'epoch': 0.87}
{'loss': 0.0973, 'grad_norm': 0.0014853961765766144, 'learning_rate': 2.4398696419390114e-06, 'epoch': 0.91}
{'loss': 0.0974, 'grad_norm': 0.0017460455419495702, 'learning_rate': 2.411279653165412e-06, 'epoch': 0.94}
{'loss': 0.0977, 'grad_norm': 0.002249350305646658, 'learning_rate': 2.382689664391812e-06, 'epoch': 0.98}
{'eval_loss': 0.09845146536827087, 'eval_runtime': 5.784, 'eval_samples_per_second': 953.491, 'eval_steps_per_second': 29.91, 'epoch': 1.0}
{'loss': 0.0986, 'grad_norm': 0.0016078707994893193, 'learning_rate': 2.354099675618212e-06, 'epoch': 1.02}
{'loss': 0.1009, 'grad_norm': 0.0021226464305073023, 'learning_rate': 2.3255096868446123e-06, 'epoch': 1.05}
{'loss': 0.0999, 'grad_norm': 0.00254792720079422, 'learning_rate': 2.2969196980710124e-06, 'epoch': 1.09}
{'loss': 0.0968, 'grad_norm': 0.0023429293651133776, 'learning_rate': 2.268329709297413e-06, 'epoch': 1.12}
{'loss': 0.0996, 'grad_norm': 0.0015684982063248754, 'learning_rate': 2.239739720523813e-06, 'epoch': 1.16}
{'loss': 0.0985, 'grad_norm': 0.001882717595435679, 'learning_rate': 2.2111497317502124e-06, 'epoch': 1.2}
{'loss': 0.0979, 'grad_norm': 0.001954102423042059, 'learning_rate': 2.182559742976613e-06, 'epoch': 1.23}
{'loss': 0.1018, 'grad_norm': 0.00317241414450109, 'learning_rate': 2.153969754203013e-06, 'epoch': 1.27}
{'loss': 0.0991, 'grad_norm': 0.004886454436928034, 'learning_rate': 2.1253797654294133e-06, 'epoch': 1.31}
{'loss': 0.1001, 'grad_norm': 0.003500038757920265, 'learning_rate': 2.0967897766558133e-06, 'epoch': 1.34}
{'loss': 0.1009, 'grad_norm': 0.0017401265213266015, 'learning_rate': 2.0681997878822133e-06, 'epoch': 1.38}
{'loss': 0.0988, 'grad_norm': 0.002421971643343568, 'learning_rate': 2.0396097991086138e-06, 'epoch': 1.41}
{'loss': 0.102, 'grad_norm': 0.002523901639506221, 'learning_rate': 2.0110198103350138e-06, 'epoch': 1.45}
{'loss': 0.0972, 'grad_norm': 0.002014156896620989, 'learning_rate': 1.982429821561414e-06, 'epoch': 1.49}
{'loss': 0.0987, 'grad_norm': 0.0022883161436766386, 'learning_rate': 1.953839832787814e-06, 'epoch': 1.52}
{'loss': 0.0975, 'grad_norm': 0.0140656977891922, 'learning_rate': 1.9252498440142142e-06, 'epoch': 1.56}
{'loss': 0.1004, 'grad_norm': 0.0027679167687892914, 'learning_rate': 1.8966598552406143e-06, 'epoch': 1.6}
{'loss': 0.1001, 'grad_norm': 0.002937422366812825, 'learning_rate': 1.8680698664670145e-06, 'epoch': 1.63}
{'loss': 0.1007, 'grad_norm': 0.002276219427585602, 'learning_rate': 1.8394798776934147e-06, 'epoch': 1.67}
{'loss': 0.103, 'grad_norm': 0.0019578190986067057, 'learning_rate': 1.810889888919815e-06, 'epoch': 1.7}
{'loss': 0.0999, 'grad_norm': 0.002311109099537134, 'learning_rate': 1.782299900146215e-06, 'epoch': 1.74}
{'loss': 0.0973, 'grad_norm': 0.0033808336593210697, 'learning_rate': 1.7537099113726152e-06, 'epoch': 1.78}
{'loss': 0.0983, 'grad_norm': 0.002296745078638196, 'learning_rate': 1.725119922599015e-06, 'epoch': 1.81}
{'loss': 0.0981, 'grad_norm': 0.0017533417558297515, 'learning_rate': 1.6965299338254152e-06, 'epoch': 1.85}
{'loss': 0.0984, 'grad_norm': 0.0020409070421010256, 'learning_rate': 1.6679399450518154e-06, 'epoch': 1.89}
{'loss': 0.1, 'grad_norm': 0.0035439757630228996, 'learning_rate': 1.6393499562782157e-06, 'epoch': 1.92}
{'loss': 0.0998, 'grad_norm': 0.0020310194231569767, 'learning_rate': 1.6107599675046157e-06, 'epoch': 1.96}
{'loss': 0.0987, 'grad_norm': 0.0038322526961565018, 'learning_rate': 1.582169978731016e-06, 'epoch': 1.99}
{'eval_loss': 0.09877973794937134, 'eval_runtime': 6.2661, 'eval_samples_per_second': 880.134, 'eval_steps_per_second': 27.609, 'epoch': 2.0}
{'loss': 0.1001, 'grad_norm': 0.004499278962612152, 'learning_rate': 1.5535799899574161e-06, 'epoch': 2.03}
{'loss': 0.1008, 'grad_norm': 0.0024329093284904957, 'learning_rate': 1.5249900011838162e-06, 'epoch': 2.07}
{'loss': 0.0995, 'grad_norm': 0.0019394466653466225, 'learning_rate': 1.4964000124102164e-06, 'epoch': 2.1}
{'loss': 0.0997, 'grad_norm': 0.00252286228351295, 'learning_rate': 1.4678100236366166e-06, 'epoch': 2.14}
{'loss': 0.0991, 'grad_norm': 0.002830798737704754, 'learning_rate': 1.4392200348630166e-06, 'epoch': 2.18}
{'loss': 0.1002, 'grad_norm': 0.001433710684068501, 'learning_rate': 1.4106300460894166e-06, 'epoch': 2.21}
{'loss': 0.0993, 'grad_norm': 0.005026262253522873, 'learning_rate': 1.3820400573158169e-06, 'epoch': 2.25}
{'loss': 0.0979, 'grad_norm': 0.002500007161870599, 'learning_rate': 1.353450068542217e-06, 'epoch': 2.28}
{'loss': 0.0988, 'grad_norm': 0.0022141628433018923, 'learning_rate': 1.3248600797686173e-06, 'epoch': 2.32}
{'loss': 0.0997, 'grad_norm': 0.002278719563037157, 'learning_rate': 1.2962700909950176e-06, 'epoch': 2.36}
{'loss': 0.0997, 'grad_norm': 0.0019859408494085073, 'learning_rate': 1.2676801022214176e-06, 'epoch': 2.39}
{'loss': 0.0986, 'grad_norm': 0.0018157653976231813, 'learning_rate': 1.2390901134478176e-06, 'epoch': 2.43}
{'loss': 0.0984, 'grad_norm': 0.0036961562000215054, 'learning_rate': 1.2105001246742178e-06, 'epoch': 2.47}
{'loss': 0.098, 'grad_norm': 0.001577572664245963, 'learning_rate': 1.181910135900618e-06, 'epoch': 2.5}
{'loss': 0.0992, 'grad_norm': 0.0010910463752225041, 'learning_rate': 1.153320147127018e-06, 'epoch': 2.54}
{'loss': 0.0967, 'grad_norm': 0.0025959971826523542, 'learning_rate': 1.1247301583534183e-06, 'epoch': 2.57}
{'loss': 0.1033, 'grad_norm': 0.003025081241503358, 'learning_rate': 1.0961401695798185e-06, 'epoch': 2.61}
{'loss': 0.0989, 'grad_norm': 0.0022556104231625795, 'learning_rate': 1.0675501808062185e-06, 'epoch': 2.65}
{'loss': 0.0999, 'grad_norm': 0.0028674262575805187, 'learning_rate': 1.0389601920326188e-06, 'epoch': 2.68}
{'loss': 0.103, 'grad_norm': 0.003205559216439724, 'learning_rate': 1.0103702032590188e-06, 'epoch': 2.72}
{'loss': 0.0967, 'grad_norm': 0.0053825369104743, 'learning_rate': 9.81780214485419e-07, 'epoch': 2.76}
{'loss': 0.0998, 'grad_norm': 0.0027789059095084667, 'learning_rate': 9.531902257118191e-07, 'epoch': 2.79}
{'loss': 0.1007, 'grad_norm': 0.0034200577065348625, 'learning_rate': 9.246002369382193e-07, 'epoch': 2.83}
{'loss': 0.099, 'grad_norm': 0.002968007232993841, 'learning_rate': 8.960102481646194e-07, 'epoch': 2.86}
{'loss': 0.099, 'grad_norm': 0.0029557307716459036, 'learning_rate': 8.674202593910196e-07, 'epoch': 2.9}
{'loss': 0.1007, 'grad_norm': 0.0019652817863970995, 'learning_rate': 8.388302706174197e-07, 'epoch': 2.94}
{'loss': 0.0987, 'grad_norm': 0.002330911112949252, 'learning_rate': 8.102402818438199e-07, 'epoch': 2.97}
{'eval_loss': 0.09902510792016983, 'eval_runtime': 7.0057, 'eval_samples_per_second': 787.214, 'eval_steps_per_second': 24.694, 'epoch': 3.0}
{'train_runtime': 228.1933, 'train_samples_per_second': 386.637, 'train_steps_per_second': 24.172, 'train_loss': 0.09938622304774841, 'epoch': 3.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 4137/5516 [03:48<01:16, 18.13it/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 173/173 [00:07<00:00, 24.23it/s]
[I 2025-08-12 17:05:04,183] Trial 19 finished with value: 0.09845146536827087 and parameters: {'lr': 3.154047561503536e-06, 'n_blocks': 3}. Best is trial 19 with value: 0.09845146536827087.
Best trial: 10. Best value: 0.107073:  95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉       | 19/20 [1:57:41<05:13, 313.85s/it][17:05:04] ft_study trial#19 value=0.09845146536827087 params={'lr': 3.154047561503536e-06, 'n_blocks': 3}
Best trial: 19. Best value: 0.0984515: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [1:57:41<00:00, 353.10s/it] 
{'loss': 0.1676, 'grad_norm': 0.0026905753184109926, 'learning_rate': 3.0399328766724554e-06, 'epoch': 0.29}
{'loss': 0.1684, 'grad_norm': 0.004208333790302277, 'learning_rate': 2.925589505098226e-06, 'epoch': 0.58}
{'loss': 0.1666, 'grad_norm': 0.0067200204357504845, 'learning_rate': 2.8112461335239973e-06, 'epoch': 0.87}
{'loss': 0.1677, 'grad_norm': 0.0037192697636783123, 'learning_rate': 2.696902761949768e-06, 'epoch': 1.16}
{'loss': 0.1677, 'grad_norm': 0.006045655347406864, 'learning_rate': 2.5825593903755388e-06, 'epoch': 1.45}
{'loss': 0.1686, 'grad_norm': 0.010022614151239395, 'learning_rate': 2.4682160188013095e-06, 'epoch': 1.74}
{'loss': 0.1679, 'grad_norm': 0.0019049098482355475, 'learning_rate': 2.3538726472270807e-06, 'epoch': 2.03}
{'loss': 0.1668, 'grad_norm': 0.004193390719592571, 'learning_rate': 2.2395292756528514e-06, 'epoch': 2.32}
{'loss': 0.1691, 'grad_norm': 0.004743210040032864, 'learning_rate': 2.1251859040786226e-06, 'epoch': 2.61}
{'loss': 0.1678, 'grad_norm': 0.004446667153388262, 'learning_rate': 2.0108425325043933e-06, 'epoch': 2.9}
{'loss': 0.1681, 'grad_norm': 0.004663076717406511, 'learning_rate': 1.8964991609301642e-06, 'epoch': 3.19}
{'loss': 0.1676, 'grad_norm': 0.005244527477771044, 'learning_rate': 1.7821557893559352e-06, 'epoch': 3.48}
{'loss': 0.1674, 'grad_norm': 0.004787179175764322, 'learning_rate': 1.6678124177817057e-06, 'epoch': 3.77}
{'loss': 0.1681, 'grad_norm': 0.004925755318254232, 'learning_rate': 1.5534690462074769e-06, 'epoch': 4.06}
{'loss': 0.1674, 'grad_norm': 0.009418497793376446, 'learning_rate': 1.4391256746332476e-06, 'epoch': 4.35}
{'loss': 0.1684, 'grad_norm': 0.0026359555777162313, 'learning_rate': 1.3247823030590186e-06, 'epoch': 4.64}
{'loss': 0.1691, 'grad_norm': 0.011230898089706898, 'learning_rate': 1.2104389314847895e-06, 'epoch': 4.93}
{'loss': 0.1666, 'grad_norm': 0.0018109414959326386, 'learning_rate': 1.0960955599105603e-06, 'epoch': 5.22}
{'loss': 0.1685, 'grad_norm': 0.003014058107510209, 'learning_rate': 9.81752188336331e-07, 'epoch': 5.51}
{'loss': 0.1677, 'grad_norm': 0.003691710764542222, 'learning_rate': 8.674088167621021e-07, 'epoch': 5.8}
{'loss': 0.1675, 'grad_norm': 0.01066081877797842, 'learning_rate': 7.530654451878729e-07, 'epoch': 6.09}
{'loss': 0.1682, 'grad_norm': 0.014575832523405552, 'learning_rate': 6.387220736136438e-07, 'epoch': 6.38}
{'loss': 0.1676, 'grad_norm': 0.005082960706204176, 'learning_rate': 5.243787020394147e-07, 'epoch': 6.67}
{'loss': 0.1673, 'grad_norm': 0.0024167955853044987, 'learning_rate': 4.1003533046518556e-07, 'epoch': 6.96}
{'loss': 0.1689, 'grad_norm': 0.01008246187120676, 'learning_rate': 2.956919588909565e-07, 'epoch': 7.25}
{'loss': 0.1673, 'grad_norm': 0.0040559531189501286, 'learning_rate': 1.8134858731672738e-07, 'epoch': 7.54}
{'loss': 0.1674, 'grad_norm': 0.004717786330729723, 'learning_rate': 6.700521574249826e-08, 'epoch': 7.83}
{'train_runtime': 703.9358, 'train_samples_per_second': 313.347, 'train_steps_per_second': 19.593, 'train_loss': 0.1678243250415386, 'epoch': 8.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13792/13792 [11:43<00:00, 19.59it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 3.47}
{'train_runtime': 48.5033, 'train_samples_per_second': 379.108, 'train_steps_per_second': 23.751, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1152/1152 [00:48<00:00, 23.75it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 0.87}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 2.61}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 3.48}
{'train_runtime': 85.1644, 'train_samples_per_second': 431.73, 'train_steps_per_second': 27.007, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2300/2300 [01:25<00:00, 27.01it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 0.58}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 1.16}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 2.32}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 2.9}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 3.48}
{'train_runtime': 131.5484, 'train_samples_per_second': 419.222, 'train_steps_per_second': 26.211, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3448/3448 [02:11<00:00, 26.21it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 0.44}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 0.87}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 1.31}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 2.18}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 2.61}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 3.05}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 3.92}
{'train_runtime': 181.1301, 'train_samples_per_second': 405.94, 'train_steps_per_second': 25.374, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4596/4596 [03:01<00:00, 25.37it/s] 
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 0.35}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 0.7}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 1.04}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 1.39}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 1.74}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 2.09}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 2.44}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 2.78}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 3.13}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 3.48}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.154047561503536e-06, 'epoch': 3.83}
{'train_runtime': 225.8089, 'train_samples_per_second': 407.017, 'train_steps_per_second': 25.455, 'train_loss': 0.0, 'epoch': 4.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5748/5748 [03:45<00:00, 25.46it/s] 
=== FT: fertig ===
[I 2025-08-12 17:29:05,156] A new study created in memory with name: meta_study
  0%|                                                                                                                                                                                                | 0/20 [00:00<?, ?it/s]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:29:11,850] Trial 0 finished with value: 0.28226022887594815 and parameters: {'d_token': 83, 'dropout': 0.29936624141382046, 'lr': 0.0004286865305228974}. Best is trial 0 with value: 0.28226022887594815.
  0%|                                                                                                                                                                                                | 0/20 [00:06<?, ?it/s][17:29:11] meta_study trial#0 value=0.28226022887594815 params={'d_token': 83, 'dropout': 0.29936624141382046, 'lr': 0.0004286865305228974}
[I 2025-08-12 17:29:18,275] Trial 1 finished with value: 0.251910948002335 and parameters: {'d_token': 46, 'dropout': 0.03243453662509205, 'lr': 0.00013165797186255447}. Best is trial 1 with value: 0.251910948002335.
Best trial: 0. Best value: 0.28226:   5%|███████▍                                                                                                                                            | 1/20 [00:13<02:07,  6.70s/it][17:29:18] meta_study trial#1 value=0.251910948002335 params={'d_token': 46, 'dropout': 0.03243453662509205, 'lr': 0.00013165797186255447}
[I 2025-08-12 17:29:25,035] Trial 2 finished with value: 0.31530344384975634 and parameters: {'d_token': 52, 'dropout': 0.026619068260772292, 'lr': 0.00047120952991558867}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  10%|██████████████▋                                                                                                                                    | 2/20 [00:19<01:57,  6.54s/it][17:29:25] meta_study trial#2 value=0.31530344384975634 params={'d_token': 52, 'dropout': 0.026619068260772292, 'lr': 0.00047120952991558867}
[I 2025-08-12 17:29:31,373] Trial 3 finished with value: 0.262267578156823 and parameters: {'d_token': 64, 'dropout': 0.09256885736619386, 'lr': 0.00021122761023924176}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  15%|██████████████████████                                                                                                                             | 3/20 [00:26<01:52,  6.64s/it][17:29:31] meta_study trial#3 value=0.262267578156823 params={'d_token': 64, 'dropout': 0.09256885736619386, 'lr': 0.00021122761023924176}
Best trial: 1. Best value: 0.251911:  20%|█████████████████████████████▍                                                                                                                     | 4/20 [00:26<01:44,  6.52s/it]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:29:35,960] Trial 4 finished with value: 0.28822567914595093 and parameters: {'d_token': 95, 'dropout': 0.09144708445959558, 'lr': 0.0007352650753192534}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  20%|█████████████████████████████▍                                                                                                                     | 4/20 [00:30<01:44,  6.52s/it][17:29:35] meta_study trial#4 value=0.28822567914595093 params={'d_token': 95, 'dropout': 0.09144708445959558, 'lr': 0.0007352650753192534}
Best trial: 1. Best value: 0.251911:  25%|████████████████████████████████████▊                                                                                                              | 5/20 [00:30<01:27,  5.82s/it]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:29:41,316] Trial 5 finished with value: 0.3073679715245786 and parameters: {'d_token': 33, 'dropout': 0.07620585658626701, 'lr': 0.0002127573221620517}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  25%|████████████████████████████████████▊                                                                                                              | 5/20 [00:36<01:27,  5.82s/it][17:29:41] meta_study trial#5 value=0.3073679715245786 params={'d_token': 33, 'dropout': 0.07620585658626701, 'lr': 0.0002127573221620517}
Best trial: 1. Best value: 0.251911:  30%|████████████████████████████████████████████                                                                                                       | 6/20 [00:36<01:19,  5.66s/it]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:29:47,803] Trial 6 finished with value: 0.28891603717453734 and parameters: {'d_token': 59, 'dropout': 0.20004360408146604, 'lr': 0.00012070785362671407}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  30%|████████████████████████████████████████████                                                                                                       | 6/20 [00:42<01:19,  5.66s/it][17:29:47] meta_study trial#6 value=0.28891603717453734 params={'d_token': 59, 'dropout': 0.20004360408146604, 'lr': 0.00012070785362671407}
Best trial: 1. Best value: 0.251911:  35%|███████████████████████████████████████████████████▍                                                                                               | 7/20 [00:42<01:17,  5.93s/it]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:29:54,360] Trial 7 finished with value: 0.29454437986136706 and parameters: {'d_token': 39, 'dropout': 0.13631131497640456, 'lr': 0.0001145082195479864}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  35%|███████████████████████████████████████████████████▍                                                                                               | 7/20 [00:49<01:17,  5.93s/it][17:29:54] meta_study trial#7 value=0.29454437986136706 params={'d_token': 39, 'dropout': 0.13631131497640456, 'lr': 0.0001145082195479864}
[I 2025-08-12 17:30:01,903] Trial 8 finished with value: 0.3011649384262646 and parameters: {'d_token': 122, 'dropout': 0.056686303781039546, 'lr': 0.0008793068890422394}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  40%|██████████████████████████████████████████████████████████▊                                                                                        | 8/20 [00:56<01:13,  6.13s/it][17:30:01] meta_study trial#8 value=0.3011649384262646 params={'d_token': 122, 'dropout': 0.056686303781039546, 'lr': 0.0008793068890422394}
[I 2025-08-12 17:30:09,921] Trial 9 finished with value: 0.2817443765672333 and parameters: {'d_token': 120, 'dropout': 0.2875850758792148, 'lr': 0.00029306724975372827}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  45%|██████████████████████████████████████████████████████████████████▏                                                                                | 9/20 [01:04<01:12,  6.57s/it][17:30:09] meta_study trial#9 value=0.2817443765672333 params={'d_token': 120, 'dropout': 0.2875850758792148, 'lr': 0.00029306724975372827}
[I 2025-08-12 17:30:16,648] Trial 10 finished with value: 0.2913856621910606 and parameters: {'d_token': 76, 'dropout': 0.2097716149288947, 'lr': 0.00015924411670030015}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  50%|█████████████████████████████████████████████████████████████████████████                                                                         | 10/20 [01:11<01:10,  7.02s/it][17:30:16] meta_study trial#10 value=0.2913856621910606 params={'d_token': 76, 'dropout': 0.2097716149288947, 'lr': 0.00015924411670030015}
Best trial: 1. Best value: 0.251911:  55%|████████████████████████████████████████████████████████████████████████████████▎                                                                 | 11/20 [01:11<01:02,  6.93s/it]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:30:23,251] Trial 11 finished with value: 0.26932610542437985 and parameters: {'d_token': 61, 'dropout': 0.009853781219514218, 'lr': 0.00020521282359959166}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  55%|████████████████████████████████████████████████████████████████████████████████▎                                                                 | 11/20 [01:18<01:02,  6.93s/it][17:30:23] meta_study trial#11 value=0.26932610542437985 params={'d_token': 61, 'dropout': 0.009853781219514218, 'lr': 0.00020521282359959166}
Best trial: 1. Best value: 0.251911:  60%|███████████████████████████████████████████████████████████████████████████████████████▌                                                          | 12/20 [01:18<00:54,  6.83s/it]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:30:30,076] Trial 12 finished with value: 0.2766727172139821 and parameters: {'d_token': 49, 'dropout': 0.12361501394497876, 'lr': 0.00018091461743541457}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  60%|███████████████████████████████████████████████████████████████████████████████████████▌                                                          | 12/20 [01:24<00:54,  6.83s/it][17:30:30] meta_study trial#12 value=0.2766727172139821 params={'d_token': 49, 'dropout': 0.12361501394497876, 'lr': 0.00018091461743541457}
Best trial: 1. Best value: 0.251911:  65%|██████████████████████████████████████████████████████████████████████████████████████████████▉                                                   | 13/20 [01:24<00:47,  6.83s/it]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:30:37,052] Trial 13 finished with value: 0.2790786942476599 and parameters: {'d_token': 71, 'dropout': 0.047822373982465305, 'lr': 0.00028843701937979443}. Best is trial 1 with value: 0.251910948002335.
Best trial: 1. Best value: 0.251911:  65%|██████████████████████████████████████████████████████████████████████████████████████████████▉                                                   | 13/20 [01:31<00:47,  6.83s/it][17:30:37] meta_study trial#13 value=0.2790786942476599 params={'d_token': 71, 'dropout': 0.047822373982465305, 'lr': 0.00028843701937979443}
Best trial: 1. Best value: 0.251911:  70%|██████████████████████████████████████████████████████████████████████████████████████████████████████▏                                           | 14/20 [01:31<00:41,  6.87s/it]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:30:44,145] Trial 14 finished with value: 0.2643057569934905 and parameters: {'d_token': 93, 'dropout': 0.10353908456375105, 'lr': 0.00014194850299038505}. Best is trial 1 with value: 0.251910948002335.     
Best trial: 1. Best value: 0.251911:  70%|██████████████████████████████████████████████████████████████████████████████████████████████████████▏                                           | 14/20 [01:38<00:41,  6.87s/it][17:30:44] meta_study trial#14 value=0.2643057569934905 params={'d_token': 93, 'dropout': 0.10353908456375105, 'lr': 0.00014194850299038505}
[I 2025-08-12 17:30:50,951] Trial 15 finished with value: 0.30813615686512796 and parameters: {'d_token': 48, 'dropout': 0.17191010071323432, 'lr': 0.00010843907697923605}. Best is trial 1 with value: 0.251910948002335.    
Best trial: 1. Best value: 0.251911:  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                    | 15/20 [01:45<00:34,  6.94s/it][17:30:50] meta_study trial#15 value=0.30813615686512796 params={'d_token': 48, 'dropout': 0.17191010071323432, 'lr': 0.00010843907697923605}
[I 2025-08-12 17:30:57,866] Trial 16 finished with value: 0.271132408771809 and parameters: {'d_token': 66, 'dropout': 0.007874853544874466, 'lr': 0.0002399611844965563}. Best is trial 1 with value: 0.251910948002335.      
Best trial: 1. Best value: 0.251911:  80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                             | 16/20 [01:52<00:27,  6.90s/it][17:30:57] meta_study trial#16 value=0.271132408771809 params={'d_token': 66, 'dropout': 0.007874853544874466, 'lr': 0.0002399611844965563}
Best trial: 1. Best value: 0.251911:  85%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                      | 17/20 [01:52<00:20,  6.90s/it]C:\trainers\ml-env\Lib\site-packages\torch\nn\modules\transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[I 2025-08-12 17:31:04,482] Trial 17 finished with value: 0.28384001057713343 and parameters: {'d_token': 41, 'dropout': 0.06070798130223664, 'lr': 0.00037849576440086814}. Best is trial 1 with value: 0.251910948002335.    
Best trial: 1. Best value: 0.251911:  85%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                      | 17/20 [01:59<00:20,  6.90s/it][17:31:04] meta_study trial#17 value=0.28384001057713343 params={'d_token': 41, 'dropout': 0.06070798130223664, 'lr': 0.00037849576440086814}
[I 2025-08-12 17:31:11,679] Trial 18 finished with value: 0.2620984557461991 and parameters: {'d_token': 88, 'dropout': 0.16449273043973664, 'lr': 0.00014559065313407915}. Best is trial 1 with value: 0.251910948002335.     
Best trial: 1. Best value: 0.251911:  90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍              | 18/20 [02:06<00:13,  6.82s/it][17:31:11] meta_study trial#18 value=0.2620984557461991 params={'d_token': 88, 'dropout': 0.16449273043973664, 'lr': 0.00014559065313407915}
[I 2025-08-12 17:31:19,831] Trial 19 finished with value: 0.25285068311071984 and parameters: {'d_token': 108, 'dropout': 0.17740618179780948, 'lr': 0.00013514189334160308}. Best is trial 1 with value: 0.251910948002335.   
Best trial: 1. Best value: 0.251911:  95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋       | 19/20 [02:14<00:06,  6.93s/it][17:31:19] meta_study trial#19 value=0.25285068311071984 params={'d_token': 108, 'dropout': 0.17740618179780948, 'lr': 0.00013514189334160308}
Best trial: 1. Best value: 0.251911: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [02:14<00:00,  6.73s/it]   
