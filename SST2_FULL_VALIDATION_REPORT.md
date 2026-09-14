# Full SST-2 Validation Set — Plaintext vs Encrypted (FHE) Accuracy Report

Generated: 2026-09-14 23:44:25 UTC
Progress: 439/872 sentences completed

Dataset: the real SST-2 (GLUE) validation split, 872 sentences, `notebooks/SST-2-val.parquet` (`sentence`, gold `label`: 0=negative, 1=positive). Plaintext circuit: the optimized, in-process `src/python/PlainCircuit.py` (`load_model()` once, `classify()` timed per sentence). Encrypted circuit: `./build/FHE-BERT-Tiny <sentence> --verbose`, one fresh subprocess per sentence (CKKS evaluation over the trained BERT-tiny/SST-2 weights, GPU-accelerated via FIDESlib). Both implement the same precomputed-LayerNorm approximation of BERT-tiny/SST-2. Unlike earlier reports in this repo, accuracy here is measured against the **actual SST-2 gold labels**, not a hand-assigned sanity-check label.

## Classification metrics vs SST-2 gold labels

| Circuit | N scored | Accuracy | Precision (pos) | Recall (pos) | F1 (pos) | Confusion matrix |
|---|---|---|---|---|---|---|
| Optimized plaintext | 439 | 0.7995 | 0.8087 | 0.8087 | 0.8087 | TP=186 FP=44 FN=44 TN=165 |
| Encrypted (FHE) | 293 | 0.8055 | 0.8387 | 0.7376 | 0.7849 | TP=104 FP=20 FN=37 TN=132 |

## Runtime & agreement

| Metric | Optimized plaintext | Encrypted (FHE) |
|---|---|---|
| Successful runs | 439/439 | 293/439 |
| Failed / timed out | 0 | 146 |
| Total internal retries | n/a | 0 |
| Mean time (s) | 0.0195 | 50.201 |
| Median time (s) | 0.0179 | 49.766 |
| Min / Max time (s) | 0.0115 / 0.4936 | 35.535 / 72.596 |

**Prediction agreement (optimized plaintext vs encrypted):** 293 pairs compared, 0.9283 agreement rate
**Mean absolute logit difference (avg of both logits):** 0.6569
**Max absolute logit difference:** 2.1331
**Encrypted / optimized-plaintext time ratio (slowdown):** 2574.2x

## Per-sentence results

| # | Gold | Sentence | Plain pred | Plain logits | Plain time (s) | Enc pred | Enc logits | Enc time (s) | Retries |
|---|---|---|---|---|---|---|---|---|---|
| 1 | positive | it 's a charming and often affecting journey . | positive | [-2.8203, 2.6318] | 0.4936 | positive | [-1.6606, 1.4896] | 44.71 | 0 |
| 2 | negative | unflinchingly bleak and desperate | negative | [1.0445, -0.6148] | 0.0194 | negative | [0.8968, -0.7968] | 42.97 | 0 |
| 3 | positive | allows us to hope that nolan is poised to embark a major ca… | positive | [-2.4844, 2.3244] | 0.0376 | positive | [-1.5471, 1.3899] | 51.03 | 0 |
| 4 | positive | the acting , costumes , music , cinematography and sound ar… | positive | [-1.4617, 1.5191] | 0.0179 | positive | [-0.1673, 0.0703] | 56.52 | 0 |
| 5 | negative | it 's slow -- very , very slow . | negative | [2.2975, -1.9950] | 0.0181 | negative | [0.9144, -0.9266] | 45.38 | 0 |
| 6 | positive | although laced with humor and a few fanciful touches , the … | positive | [-2.5077, 2.3682] | 0.0210 | positive | [-1.4919, 1.2856] | 61.24 | 0 |
| 7 | negative | a sometimes tedious film . | negative | [1.7480, -1.5298] | 0.0202 | negative | [1.2211, -0.9993] | 46.86 | 0 |
| 8 | negative | or doing last year 's taxes with your ex-wife . | negative | [1.3908, -1.2027] | 0.0118 | negative | [1.3663, -1.2046] | 53.94 | 0 |
| 9 | positive | you do n't have to know about music to appreciate the film … | positive | [-1.9701, 1.9297] | 0.0229 | positive | [-1.3355, 1.1828] | 49.34 | 0 |
| 10 | negative | in exactly 89 minutes , most of which passed as slowly as i… | negative | [2.1160, -1.7758] | 0.0216 | FAIL | n/a | 57.41 | 0 |
| 11 | positive | the mesmerizing performances of the leads keep the film gro… | positive | [-2.4866, 2.3672] | 0.0184 | FAIL | n/a | 58.64 | 0 |
| 12 | negative | it takes a strange kind of laziness to waste the talents of… | positive | [-0.5422, 0.6784] | 0.0196 | FAIL | n/a | 60.51 | 0 |
| 13 | negative | ... the film suffers from a lack of humor ( something neede… | negative | [2.1033, -1.7120] | 0.0184 | negative | [1.6734, -1.4404] | 65.95 | 0 |
| 14 | positive | we root for ( clara and paul ) , even like them , though pe… | negative | [0.4876, -0.2868] | 0.0185 | negative | [0.1846, -0.2629] | 56.83 | 0 |
| 15 | negative | even horror fans will most likely not find what they 're se… | negative | [1.0415, -0.6798] | 0.0139 | positive | [-0.1162, -0.0134] | 52.71 | 0 |
| 16 | positive | a gorgeous , high-spirited musical from india that exquisit… | positive | [-2.8600, 2.7075] | 0.0153 | FAIL | n/a | 64.21 | 0 |
| 17 | positive | the emotions are raw and will strike a nerve with anyone wh… | positive | [-0.4878, 0.5842] | 0.0180 | negative | [0.0103, -0.0813] | 49.22 | 0 |
| 18 | positive | audrey tatou has a knack for picking roles that magnify her… | positive | [-2.0319, 1.9702] | 0.0183 | FAIL | n/a | 55.12 | 0 |
| 19 | negative | ... the movie is just a plain old monster . | negative | [2.1559, -1.9078] | 0.0195 | negative | [0.5028, -0.7299] | 53.91 | 0 |
| 20 | negative | in its best moments , resembles a bad high school productio… | negative | [0.4496, -0.1990] | 0.0122 | FAIL | n/a | 64.26 | 0 |
| 21 | negative | pumpkin takes an admirable look at the hypocrisy of politic… | positive | [-0.0086, 0.1746] | 0.0182 | FAIL | n/a | 56.97 | 0 |
| 22 | negative | the iditarod lasts for days - this just felt like it did . | negative | [1.1953, -0.7763] | 0.0161 | negative | [0.7636, -0.5254] | 50.62 | 0 |
| 23 | negative | holden caulfield did it better . | positive | [-1.0207, 1.0503] | 0.0116 | positive | [-0.0339, -0.0082] | 55.03 | 0 |
| 24 | positive | a delectable and intriguing thriller filled with surprises … | positive | [-2.6966, 2.5452] | 0.0143 | positive | [-1.4682, 1.3243] | 52.28 | 0 |
| 25 | positive | seldom has a movie so closely matched the spirit of a man a… | positive | [-2.1123, 2.1123] | 0.0219 | positive | [-1.5167, 1.2505] | 44.27 | 0 |
| 26 | negative | nicks , seemingly uncertain what 's going to make people la… | negative | [1.6499, -1.3666] | 0.0120 | FAIL | n/a | 55.66 | 0 |
| 27 | negative | the action switches between past and present , but the mate… | negative | [0.9312, -0.6383] | 0.0200 | FAIL | n/a | 57.52 | 0 |
| 28 | positive | it 's an offbeat treat that pokes fun at the democratic exe… | positive | [-2.4063, 2.2442] | 0.0192 | positive | [-1.4016, 1.1295] | 53.23 | 0 |
| 29 | negative | it 's a cookie-cutter movie , a cut-and-paste job . | negative | [1.2145, -0.8818] | 0.0121 | negative | [0.8039, -0.8755] | 58.02 | 0 |
| 30 | negative | i had to look away - this was god awful . | negative | [1.7816, -1.3548] | 0.0160 | negative | [0.9351, -0.9541] | 43.49 | 0 |
| 31 | positive | thanks to scott 's charismatic roger and eisenberg 's sweet… | positive | [-2.5853, 2.4365] | 0.0209 | FAIL | n/a | 58.53 | 0 |
| 32 | negative | ... designed to provide a mix of smiles and tears , `` cros… | positive | [-0.2789, 0.3917] | 0.0122 | FAIL | n/a | 57.22 | 0 |
| 33 | positive | a gorgeous , witty , seductive movie . | positive | [-2.8750, 2.7239] | 0.0128 | FAIL | n/a | 54.83 | 0 |
| 34 | negative | if the movie succeeds in instilling a wary sense of ` there… | positive | [-0.8860, 0.9037] | 0.0178 | FAIL | n/a | 32.57 | 0 |
| 35 | negative | it does n't believe in itself , it has no sense of humor ..… | negative | [1.7595, -1.4447] | 0.0178 | negative | [0.4535, -0.5658] | 56.67 | 0 |
| 36 | negative | a sequence of ridiculous shoot - 'em - up scenes . | negative | [1.8421, -1.6181] | 0.0212 | negative | [1.6536, -1.4658] | 48.33 | 0 |
| 37 | positive | the weight of the piece , the unerring professionalism of t… | positive | [-2.0705, 2.0333] | 0.0121 | positive | [-1.0525, 0.8962] | 58.68 | 0 |
| 38 | negative | ( w ) hile long on amiable monkeys and worthy environmental… | positive | [-0.0777, 0.2436] | 0.0178 | FAIL | n/a | 57.63 | 0 |
| 39 | positive | as surreal as a dream and as detailed as a photograph , as … | positive | [-2.2163, 2.1394] | 0.0153 | positive | [-1.1888, 1.0741] | 52.64 | 0 |
| 40 | positive | escaping the studio , piccoli is warmly affecting and so is… | positive | [-0.9695, 1.1127] | 0.0176 | positive | [-0.7646, 0.7629] | 51.80 | 0 |
| 41 | positive | there 's ... tremendous energy from the cast , a sense of p… | positive | [-2.6953, 2.5288] | 0.0176 | positive | [-2.3854, 2.0945] | 58.25 | 0 |
| 42 | positive | this illuminating documentary transcends our preconceived v… | positive | [-2.7312, 2.5804] | 0.0182 | positive | [-2.0898, 1.8506] | 53.40 | 0 |
| 43 | positive | the subtle strength of `` elling '' is that it never loses … | negative | [1.0264, -0.6155] | 0.0174 | positive | [-0.3205, 0.2437] | 52.04 | 0 |
| 44 | positive | holm ... embodies the character with an effortlessly regal … | positive | [-2.7087, 2.5550] | 0.0174 | positive | [-0.5572, 0.4402] | 52.13 | 0 |
| 45 | negative | the title not only describes its main characters , but the … | negative | [0.8915, -0.6022] | 0.0209 | negative | [0.5269, -0.6753] | 51.21 | 0 |
| 46 | negative | it offers little beyond the momentary joys of pretty and we… | positive | [-0.0865, 0.2106] | 0.0178 | negative | [0.0401, -0.1234] | 52.17 | 0 |
| 47 | negative | a synthesis of cliches and absurdities that seems positivel… | negative | [0.5442, -0.1653] | 0.0182 | negative | [0.3360, -0.4055] | 50.70 | 0 |
| 48 | positive | a subtle and well-crafted ( for the most part ) chiller . | positive | [-2.6053, 2.4988] | 0.0224 | positive | [-1.4474, 1.2892] | 44.82 | 0 |
| 49 | positive | has a lot of the virtues of eastwood at his best . | positive | [-2.0111, 2.0144] | 0.0226 | positive | [-0.6237, 0.4503] | 45.84 | 0 |
| 50 | negative | it 's hampered by a lifetime-channel kind of plot and a lea… | negative | [1.4880, -1.1205] | 0.0226 | negative | [1.0959, -1.1002] | 72.60 | 0 |
| 51 | negative | it feels like an after-school special gussied up with some … | negative | [1.0791, -0.7948] | 0.0216 | FAIL | n/a | 58.23 | 0 |
| 52 | positive | for the most part , director anne-sophie birot 's first fea… | positive | [-2.4548, 2.2851] | 0.0218 | positive | [-1.4538, 1.2841] | 51.28 | 0 |
| 53 | positive | mr. tsai is a very original artist in his medium , and what… | negative | [0.4396, -0.1276] | 0.0181 | FAIL | n/a | 60.46 | 0 |
| 54 | positive | sade is an engaging look at the controversial eponymous and… | positive | [-2.1093, 2.0244] | 0.0181 | positive | [-1.7161, 1.5505] | 55.58 | 0 |
| 55 | negative | so devoid of any kind of intelligible story that it makes f… | negative | [0.6526, -0.1637] | 0.0188 | negative | [0.3829, -0.4481] | 66.04 | 0 |
| 56 | positive | a tender , heartfelt family drama . | positive | [-2.8312, 2.7200] | 0.0173 | FAIL | n/a | 53.30 | 0 |
| 57 | negative | ... a hollow joke told by a cinematic gymnast having too mu… | negative | [0.7959, -0.5050] | 0.0178 | FAIL | n/a | 61.81 | 0 |
| 58 | negative | the cold turkey would 've been a far better title . | negative | [1.8134, -1.3863] | 0.0175 | negative | [1.0813, -0.9650] | 45.97 | 0 |
| 59 | negative | manages to be both repulsively sadistic and mundane . | positive | [0.0431, 0.1456] | 0.0174 | negative | [0.3246, -0.4798] | 43.93 | 0 |
| 60 | negative | it 's just disappointingly superficial -- a movie that has … | negative | [0.2191, -0.0035] | 0.0204 | FAIL | n/a | 57.19 | 0 |
| 61 | positive | this is a story of two misfits who do n't stand a chance al… | negative | [1.1294, -0.7202] | 0.0175 | negative | [0.1047, -0.2149] | 53.16 | 0 |
| 62 | negative | schaeffer has to find some hook on which to hang his persis… | negative | [1.3839, -1.0423] | 0.0180 | FAIL | n/a | 62.59 | 0 |
| 63 | positive | the primitive force of this film seems to bubble up from th… | negative | [0.9685, -0.6451] | 0.0216 | negative | [0.5539, -0.6616] | 50.74 | 0 |
| 64 | positive | on this tricky topic , tadpole is very much a step in the r… | positive | [-2.2820, 2.1391] | 0.0170 | FAIL | n/a | 64.38 | 0 |
| 65 | negative | the script kicks in , and mr. hartley 's distended pace and… | negative | [1.1061, -0.7797] | 0.0157 | negative | [0.6942, -0.7573] | 57.11 | 0 |
| 66 | negative | you wonder why enough was n't just a music video rather tha… | negative | [1.0307, -0.6500] | 0.0166 | negative | [0.6158, -0.6386] | 48.37 | 0 |
| 67 | positive | if you 're hard up for raunchy college humor , this is your… | negative | [0.1760, -0.0813] | 0.0179 | negative | [-0.0340, -0.2373] | 50.21 | 0 |
| 68 | positive | a fast , funny , highly enjoyable movie . | positive | [-2.7911, 2.5976] | 0.0172 | FAIL | n/a | 50.14 | 0 |
| 69 | positive | good old-fashioned slash-and-hack is back ! | negative | [0.2949, -0.1157] | 0.0223 | negative | [0.5928, -0.6462] | 47.66 | 0 |
| 70 | negative | this one is definitely one to skip , even for horror movie … | negative | [0.3965, -0.0861] | 0.0222 | negative | [0.4912, -0.5445] | 46.27 | 0 |
| 71 | negative | for all its impressive craftsmanship , and despite an overb… | positive | [-1.1578, 1.1789] | 0.0175 | FAIL | n/a | 60.58 | 0 |
| 72 | positive | exquisitely nuanced in mood tics and dialogue , this chambe… | positive | [-2.7549, 2.5887] | 0.0162 | FAIL | n/a | 57.28 | 0 |
| 73 | positive | uses high comedy to evoke surprising poignance . | positive | [-2.6839, 2.5004] | 0.0209 | positive | [-1.9509, 1.7064] | 48.76 | 0 |
| 74 | positive | one of creepiest , scariest movies to come along in a long … | negative | [0.0998, -0.0085] | 0.0208 | negative | [0.4289, -0.4593] | 50.90 | 0 |
| 75 | negative | a string of rehashed sight gags based in insipid vulgarity . | negative | [1.6242, -1.2995] | 0.0203 | negative | [0.8446, -0.9333] | 45.20 | 0 |
| 76 | positive | among the year 's most intriguing explorations of alientati… | positive | [-2.0273, 2.0297] | 0.0207 | positive | [-0.7747, 0.6534] | 50.73 | 0 |
| 77 | negative | the movie fails to live up to the sum of its parts . | negative | [1.9012, -1.6455] | 0.0172 | FAIL | n/a | 57.38 | 0 |
| 78 | positive | the son 's room is a triumph of gentility that earns its mo… | positive | [-1.3669, 1.3204] | 0.0225 | positive | [-1.0705, 0.8737] | 54.82 | 0 |
| 79 | positive | there is nothing outstanding about this film , but it is go… | positive | [-1.1126, 1.1775] | 0.0185 | FAIL | n/a | 56.58 | 0 |
| 80 | negative | this is a train wreck of an action film -- a stupefying att… | negative | [1.3073, -0.9411] | 0.0191 | FAIL | n/a | 56.66 | 0 |
| 81 | positive | the draw ( for `` big bad love '' ) is a solid performance … | positive | [-2.2130, 2.1501] | 0.0171 | positive | [-2.2476, 2.0655] | 47.18 | 0 |
| 82 | negative | green might want to hang onto that ski mask , as robbery ma… | negative | [1.8610, -1.5844] | 0.0182 | negative | [1.2208, -1.1305] | 58.32 | 0 |
| 83 | negative | it 's one pussy-ass world when even killer-thrillers revolv… | negative | [0.3012, 0.0178] | 0.0218 | negative | [0.8479, -0.7808] | 50.61 | 0 |
| 84 | positive | though it 's become almost redundant to say so , major kudo… | positive | [-0.4276, 0.4354] | 0.0233 | positive | [-0.5012, 0.3192] | 60.59 | 0 |
| 85 | positive | the band 's courage in the face of official repression is i… | positive | [-2.2222, 2.1717] | 0.0216 | positive | [-1.2391, 1.1046] | 57.73 | 0 |
| 86 | positive | the movie achieves as great an impact by keeping these thou… | positive | [-1.9386, 1.9056] | 0.0178 | positive | [-1.1099, 0.9400] | 53.98 | 0 |
| 87 | negative | the film flat lines when it should peak and is more missed … | negative | [1.6107, -1.3208] | 0.0174 | negative | [0.7922, -0.8061] | 52.61 | 0 |
| 88 | positive | jaglom ... put ( s ) the audience in the privileged positio… | negative | [0.6478, -0.4760] | 0.0175 | FAIL | n/a | 63.96 | 0 |
| 89 | positive | fresnadillo 's dark and jolting images have a way of plying… | negative | [0.2666, -0.1207] | 0.0212 | FAIL | n/a | 56.39 | 0 |
| 90 | positive | we know the plot 's a little crazy , but it held my interes… | negative | [0.9388, -0.6272] | 0.0216 | negative | [0.6403, -0.8086] | 60.51 | 0 |
| 91 | positive | it 's a scattershot affair , but when it hits its mark it '… | positive | [-0.6983, 0.7951] | 0.0127 | positive | [-0.3159, 0.2117] | 62.82 | 0 |
| 92 | positive | hardly a masterpiece , but it introduces viewers to a good … | positive | [-2.0110, 2.0021] | 0.0179 | positive | [-1.5706, 1.3484] | 50.83 | 0 |
| 93 | negative | you wo n't like roger , but you will quickly recognize him . | positive | [-0.2848, 0.4137] | 0.0176 | negative | [0.1685, -0.1142] | 46.87 | 0 |
| 94 | positive | if steven soderbergh 's ` solaris ' is a failure it is a gl… | negative | [1.4575, -1.0587] | 0.0174 | FAIL | n/a | 27.83 | 0 |
| 95 | positive | byler reveals his characters in a way that intrigues and ev… | negative | [0.3021, 0.0177] | 0.0180 | negative | [0.5444, -0.6231] | 54.86 | 0 |
| 96 | negative | this riveting world war ii moral suspense story deals with … | positive | [-1.9666, 1.9397] | 0.0181 | positive | [-0.9067, 0.8517] | 54.60 | 0 |
| 97 | negative | it 's difficult to imagine the process that produced such a… | negative | [1.1889, -0.8496] | 0.0176 | FAIL | n/a | 59.44 | 0 |
| 98 | positive | no sophomore slump for director sam mendes , who segues fro… | positive | [-0.5802, 0.8000] | 0.0182 | positive | [-0.3840, 0.2662] | 56.77 | 0 |
| 99 | negative | on the whole , the movie lacks wit , feeling and believabil… | negative | [1.5058, -1.2133] | 0.0121 | FAIL | n/a | 59.52 | 0 |
| 100 | negative | why make a documentary about these marginal historical figu… | negative | [1.5862, -1.3326] | 0.0177 | negative | [1.3382, -1.1797] | 43.19 | 0 |
| 101 | positive | neither parker nor donovan is a typical romantic lead , but… | negative | [1.7938, -1.4856] | 0.0184 | negative | [1.0112, -1.0441] | 54.49 | 0 |
| 102 | negative | his last movie was poetically romantic and full of indelibl… | positive | [-1.0706, 1.2023] | 0.0176 | positive | [-0.4858, 0.4595] | 52.18 | 0 |
| 103 | positive | does paint some memorable images ... , but makhmalbaf keeps… | positive | [-1.4486, 1.4624] | 0.0179 | positive | [-0.5407, 0.4510] | 50.54 | 0 |
| 104 | positive | a gripping movie , played with performances that are all un… | positive | [-2.5153, 2.4424] | 0.0175 | positive | [-1.2194, 1.1192] | 51.39 | 0 |
| 105 | positive | it 's one of those baseball pictures where the hero is stoi… | positive | [-0.7800, 0.7667] | 0.0192 | FAIL | n/a | 57.48 | 0 |
| 106 | negative | combining quick-cut editing and a blaring heavy metal much … | negative | [1.6237, -1.2896] | 0.0137 | FAIL | n/a | 62.95 | 0 |
| 107 | positive | the movie 's relatively simple plot and uncomplicated moral… | positive | [-1.2208, 1.2545] | 0.0180 | negative | [-0.0731, -0.0820] | 56.72 | 0 |
| 108 | negative | what the director ca n't do is make either of val kilmer 's… | negative | [1.1058, -0.8497] | 0.0177 | negative | [0.1403, -0.2284] | 58.28 | 0 |
| 109 | negative | too often , the viewer is n't reacting to humor so much as … | negative | [1.3389, -0.9062] | 0.0158 | negative | [0.3904, -0.4568] | 50.19 | 0 |
| 110 | positive | it 's great escapist fun that recreates a place and time th… | positive | [-1.1348, 1.2391] | 0.0174 | positive | [-1.3276, 1.0733] | 50.97 | 0 |
| 111 | negative | scores no points for originality , wit , or intelligence . | negative | [1.1487, -0.9127] | 0.0179 | negative | [0.9049, -0.9237] | 44.06 | 0 |
| 112 | negative | there is n't nearly enough fun here , despite the presence … | positive | [-0.6040, 0.6365] | 0.0205 | positive | [-1.1850, 0.9614] | 49.09 | 0 |
| 113 | positive | hilariously inept and ridiculous . | positive | [-1.5376, 1.6367] | 0.0181 | positive | [-0.1459, 0.0831] | 47.90 | 0 |
| 114 | negative | this movie is maddening . | negative | [1.4698, -1.1836] | 0.0179 | negative | [0.7990, -0.7197] | 45.77 | 0 |
| 115 | positive | it haunts you , you ca n't forget it , you admire its conce… | positive | [-1.0012, 1.0929] | 0.0188 | FAIL | n/a | 54.53 | 0 |
| 116 | negative | sam mendes has become valedictorian at the school for soft … | positive | [-0.3166, 0.3832] | 0.0216 | negative | [0.1930, -0.2076] | 50.37 | 0 |
| 117 | positive | one of the smartest takes on singles culture i 've seen in … | positive | [-1.3755, 1.4063] | 0.0165 | positive | [-0.0452, -0.0207] | 44.56 | 0 |
| 118 | positive | moody , heartbreaking , and filmed in a natural , unforced … | positive | [-1.4803, 1.5274] | 0.0202 | positive | [-0.4232, 0.3662] | 50.64 | 0 |
| 119 | negative | every nanosecond of the the new guy reminds you that you co… | positive | [-1.4830, 1.5398] | 0.0179 | positive | [-0.0971, 0.1367] | 54.40 | 0 |
| 120 | negative | comes ... uncomfortably close to coasting in the treads of … | negative | [0.8107, -0.4955] | 0.0225 | negative | [0.7050, -0.8402] | 56.44 | 0 |
| 121 | positive | warm water under a red bridge is a quirky and poignant japa… | positive | [-2.5031, 2.3709] | 0.0177 | FAIL | n/a | 63.13 | 0 |
| 122 | negative | it seems to me the film is about the art of ripping people … | negative | [0.5504, -0.2942] | 0.0179 | negative | [0.5591, -0.5207] | 49.52 | 0 |
| 123 | positive | old-form moviemaking at its best . | positive | [-1.9332, 1.9827] | 0.0245 | positive | [-0.4202, 0.3400] | 39.42 | 0 |
| 124 | positive | turns potentially forgettable formula into something strang… | negative | [1.3377, -0.8702] | 0.0209 | negative | [0.7313, -0.5834] | 44.97 | 0 |
| 125 | positive | ( lawrence bounces ) all over the stage , dancing , running… | negative | [0.4076, -0.1225] | 0.0229 | FAIL | n/a | 53.33 | 0 |
| 126 | positive | a movie that reminds us of just how exciting and satisfying… | positive | [-2.3523, 2.2313] | 0.0245 | positive | [-1.5045, 1.2841] | 52.79 | 0 |
| 127 | negative | confirms the nagging suspicion that ethan hawke would be ev… | negative | [2.3013, -1.9713] | 0.0195 | negative | [1.5561, -1.3425] | 49.77 | 0 |
| 128 | negative | in the end , we are left with something like two ships pass… | positive | [-0.1043, 0.1807] | 0.0194 | FAIL | n/a | 55.21 | 0 |
| 129 | positive | montias ... pumps a lot of energy into his nicely nuanced n… | positive | [-1.0171, 1.0196] | 0.0220 | FAIL | n/a | 56.41 | 0 |
| 130 | positive | it provides the grand , intelligent entertainment of a supe… | positive | [-2.7895, 2.5663] | 0.0179 | FAIL | n/a | 55.31 | 0 |
| 131 | negative | suffers from the lack of a compelling or comprehensible nar… | negative | [2.3025, -2.1178] | 0.0180 | negative | [2.0217, -1.8447] | 54.46 | 0 |
| 132 | negative | in execution , this clever idea is far less funny than the … | negative | [0.9788, -0.7127] | 0.0146 | positive | [-0.1165, -0.0874] | 47.75 | 0 |
| 133 | positive | scooby dooby doo / and shaggy too / you both look and sound… | positive | [-0.1672, 0.1458] | 0.0187 | negative | [-0.0129, -0.1302] | 53.64 | 0 |
| 134 | negative | the tale of tok ( andy lau ) , a sleek sociopath on the tra… | positive | [0.0503, 0.1631] | 0.0187 | FAIL | n/a | 56.99 | 0 |
| 135 | negative | it all drags on so interminably it 's like watching a miser… | negative | [1.4524, -1.1339] | 0.0156 | negative | [0.6311, -0.5023] | 55.72 | 0 |
| 136 | negative | pumpkin means to be an outrageous dark satire on fraternity… | positive | [-1.8357, 1.7989] | 0.0220 | FAIL | n/a | 53.67 | 0 |
| 137 | negative | looks and feels like a project better suited for the small … | negative | [1.4347, -1.1350] | 0.0180 | negative | [0.7245, -0.8072] | 54.42 | 0 |
| 138 | negative | forced , familiar and thoroughly condescending . | negative | [2.0558, -1.6462] | 0.0175 | negative | [1.1495, -0.9226] | 46.48 | 0 |
| 139 | positive | that is a compliment to kuras and miller . | positive | [-2.7630, 2.6378] | 0.0172 | FAIL | n/a | 50.68 | 0 |
| 140 | negative | it 's not the ultimate depression-era gangster movie . | negative | [1.5467, -1.3139] | 0.0177 | negative | [1.2458, -1.0861] | 45.24 | 0 |
| 141 | negative | sacrifices the value of its wealth of archival foot-age wit… | positive | [0.0733, 0.1921] | 0.0128 | positive | [-0.5365, 0.3256] | 57.08 | 0 |
| 142 | negative | the character of zigzag is not sufficiently developed to su… | negative | [0.9659, -0.5212] | 0.0178 | negative | [0.0706, -0.1624] | 52.41 | 0 |
| 143 | positive | what better message than ` love thyself ' could young women… | negative | [0.2746, -0.0329] | 0.0177 | FAIL | n/a | 25.73 | 0 |
| 144 | positive | a solid film ... but more conscientious than it is truly st… | positive | [-2.7808, 2.6652] | 0.0187 | positive | [-2.6399, 2.4416] | 47.09 | 0 |
| 145 | negative | while ( hill ) has learned new tricks , the tricks alone ar… | negative | [1.8577, -1.6277] | 0.0139 | negative | [1.3631, -1.2974] | 55.36 | 0 |
| 146 | negative | the best that can be said about the work here of scottish d… | positive | [-1.6941, 1.5216] | 0.0178 | FAIL | n/a | 57.40 | 0 |
| 147 | positive | about a manga-like heroine who fights back at her abusers ,… | positive | [-0.7030, 0.7747] | 0.0176 | positive | [-0.6072, 0.4449] | 50.45 | 0 |
| 148 | negative | the talented and clever robert rodriguez perhaps put a litt… | positive | [-0.5160, 0.5055] | 0.0177 | positive | [-0.1559, 0.0071] | 51.24 | 0 |
| 149 | negative | feels too formulaic and too familiar to produce the transgr… | negative | [1.4179, -1.0625] | 0.0182 | negative | [0.5059, -0.5621] | 60.83 | 0 |
| 150 | positive | the volatile dynamics of female friendship is the subject o… | positive | [-2.1692, 2.0913] | 0.0179 | FAIL | n/a | 56.57 | 0 |
| 151 | positive | overall very good for what it 's trying to do . | negative | [0.2804, -0.1470] | 0.0221 | negative | [0.3697, -0.3850] | 41.24 | 0 |
| 152 | positive | a big , gorgeous , sprawling swashbuckler that delivers its… | positive | [-2.6686, 2.5028] | 0.0214 | FAIL | n/a | 62.76 | 0 |
| 153 | positive | a difficult , absorbing film that manages to convey more su… | negative | [0.2918, 0.0051] | 0.0186 | FAIL | n/a | 57.35 | 0 |
| 154 | negative | the heavy-handed film is almost laughable as a consequence . | negative | [1.8386, -1.5115] | 0.0235 | negative | [0.9301, -0.9995] | 56.19 | 0 |
| 155 | positive | a solid examination of the male midlife crisis . | positive | [-2.5862, 2.4989] | 0.0214 | positive | [-2.2456, 2.0231] | 43.85 | 0 |
| 156 | negative | a nightmare date with a half-formed wit done a great disser… | negative | [1.5886, -1.2783] | 0.0186 | FAIL | n/a | 53.63 | 0 |
| 157 | positive | manages to transcend the sex , drugs and show-tunes plot in… | positive | [-1.9345, 1.9026] | 0.0217 | positive | [-0.2358, 0.0910] | 47.74 | 0 |
| 158 | negative | it takes talent to make a lifeless movie about the most hei… | negative | [1.0487, -0.7462] | 0.0224 | negative | [0.7064, -0.8631] | 51.87 | 0 |
| 159 | negative | by getting myself wrapped up in the visuals and eccentricit… | positive | [-0.3690, 0.5493] | 0.0204 | FAIL | n/a | 57.71 | 0 |
| 160 | positive | like leon , it 's frustrating and still oddly likable . | positive | [-0.3488, 0.4342] | 0.0173 | positive | [-0.1960, 0.1299] | 42.41 | 0 |
| 161 | negative | uncommonly stylish but equally silly ... the picture fails … | negative | [0.8023, -0.4615] | 0.0142 | FAIL | n/a | 55.87 | 0 |
| 162 | negative | not exactly the bees knees | negative | [1.6517, -1.4039] | 0.0214 | negative | [0.6915, -0.7357] | 42.14 | 0 |
| 163 | negative | there seems to be no clear path as to where the story 's go… | negative | [1.5429, -1.1977] | 0.0134 | negative | [0.6946, -0.6139] | 59.95 | 0 |
| 164 | negative | slapstick buffoonery can tickle many a preschooler 's fancy… | negative | [0.9192, -0.6853] | 0.0187 | FAIL | n/a | 51.85 | 0 |
| 165 | positive | a woman 's pic directed with resonance by ilya chaiken . | positive | [-1.1664, 1.3453] | 0.0253 | negative | [0.0503, -0.1326] | 45.12 | 0 |
| 166 | negative | may reawaken discussion of the kennedy assassination but th… | negative | [1.0137, -0.7354] | 0.0152 | negative | [0.6404, -0.7823] | 48.07 | 0 |
| 167 | negative | characters still need to function according to some set of … | negative | [1.4222, -1.1161] | 0.0121 | FAIL | n/a | 53.25 | 0 |
| 168 | negative | the end result is a film that 's neither . | negative | [2.0916, -1.7855] | 0.0214 | negative | [1.1982, -0.9732] | 43.77 | 0 |
| 169 | positive | manages to be sweet and wickedly satisfying at the same tim… | positive | [-2.3393, 2.3159] | 0.0193 | positive | [-1.4665, 1.3073] | 45.95 | 0 |
| 170 | positive | leigh 's film is full of memorable performances from top to… | positive | [-0.6729, 0.6619] | 0.0221 | positive | [-0.5716, 0.4471] | 46.11 | 0 |
| 171 | positive | it 's also , clearly , great fun . | positive | [-2.6967, 2.5535] | 0.0203 | FAIL | n/a | 47.08 | 0 |
| 172 | negative | rarely has leukemia looked so shimmering and benign . | positive | [-0.8636, 1.1564] | 0.0242 | positive | [-0.3204, 0.3747] | 42.47 | 0 |
| 173 | positive | it seems like i have been waiting my whole life for this mo… | negative | [1.5200, -1.2185] | 0.0178 | negative | [0.9147, -0.8702] | 50.82 | 0 |
| 174 | negative | determined to be fun , and bouncy , with energetic musicals… | positive | [-1.9523, 1.8657] | 0.0189 | positive | [-1.7927, 1.5627] | 55.44 | 0 |
| 175 | positive | if you dig on david mamet 's mind tricks ... rent this movi… | positive | [-1.2836, 1.3148] | 0.0166 | positive | [-0.6016, 0.4117] | 48.18 | 0 |
| 176 | positive | bleakly funny , its characters all the more touching for re… | positive | [-0.3447, 0.4245] | 0.0191 | positive | [-0.2231, 0.1508] | 52.97 | 0 |
| 177 | negative | delivers the same old same old , tarted up with latin flava… | positive | [-1.0913, 1.1538] | 0.0225 | positive | [-0.2496, 0.2423] | 46.22 | 0 |
| 178 | negative | does n't offer much besides glib soullessness , raunchy lan… | negative | [0.8110, -0.4448] | 0.0215 | FAIL | n/a | 53.40 | 0 |
| 179 | negative | it made me want to wrench my eyes out of my head and toss t… | positive | [-0.2055, 0.1735] | 0.0176 | positive | [-0.0743, -0.0738] | 46.99 | 0 |
| 180 | positive | the film 's performances are thrilling . | positive | [-2.2389, 2.2170] | 0.0240 | positive | [-0.5400, 0.4871] | 43.40 | 0 |
| 181 | negative | unfortunately , it 's not silly fun unless you enjoy really… | negative | [1.3498, -0.9655] | 0.0205 | negative | [0.9473, -0.7595] | 49.87 | 0 |
| 182 | negative | it 's a bad thing when a movie has about as much substance … | negative | [1.9602, -1.6738] | 0.0154 | negative | [1.3209, -1.1931] | 62.11 | 0 |
| 183 | negative | i sympathize with the plight of these families , but the mo… | positive | [0.0365, 0.1019] | 0.0199 | FAIL | n/a | 56.24 | 0 |
| 184 | negative | the lower your expectations , the more you 'll enjoy it . | negative | [0.7353, -0.3499] | 0.0216 | negative | [0.1476, -0.1872] | 41.32 | 0 |
| 185 | negative | though perry and hurley make inspiring efforts to breathe l… | positive | [-0.2985, 0.3717] | 0.0221 | FAIL | n/a | 52.25 | 0 |
| 186 | positive | a must-see for the david mamet enthusiast and for anyone wh… | positive | [-2.5989, 2.4134] | 0.0176 | positive | [-1.1550, 0.9668] | 61.98 | 0 |
| 187 | positive | pacino is brilliant as the sleep-deprived dormer , his incr… | negative | [0.4257, -0.0497] | 0.0199 | positive | [-0.0385, -0.0026] | 49.35 | 0 |
| 188 | positive | ` de niro ... is a veritable source of sincere passion that… | positive | [-1.0997, 1.2146] | 0.0190 | FAIL | n/a | 26.19 | 0 |
| 189 | negative | a misogynistic piece of filth that attempts to pass itself … | negative | [1.3692, -0.9351] | 0.0197 | negative | [0.7689, -0.8233] | 51.02 | 0 |
| 190 | negative | its story may be a thousand years old , but why did it have… | negative | [1.5561, -1.2003] | 0.0215 | negative | [0.7484, -0.6024] | 52.29 | 0 |
| 191 | negative | try as i may , i ca n't think of a single good reason to se… | negative | [0.6060, -0.2019] | 0.0119 | FAIL | n/a | 25.53 | 0 |
| 192 | positive | the movie is beautiful to behold and engages one in a sense… | positive | [-2.6792, 2.5054] | 0.0181 | FAIL | n/a | 55.08 | 0 |
| 193 | positive | a celebration of quirkiness , eccentricity , and certain in… | negative | [0.4280, -0.1228] | 0.0176 | negative | [0.4233, -0.4921] | 56.76 | 0 |
| 194 | positive | morton uses her face and her body language to bring us morv… | positive | [-1.6790, 1.6500] | 0.0178 | FAIL | n/a | 64.44 | 0 |
| 195 | positive | instead of a hyperbolic beat-charged urban western , it 's … | negative | [0.2130, 0.1009] | 0.0175 | negative | [0.2947, -0.3722] | 48.98 | 0 |
| 196 | positive | my thoughts were focused on the characters . | negative | [1.4691, -0.9939] | 0.0172 | negative | [0.9973, -0.8716] | 38.53 | 0 |
| 197 | positive | so , too , is this comedy about mild culture clashing in to… | negative | [0.9816, -0.6813] | 0.0213 | FAIL | n/a | 56.41 | 0 |
| 198 | negative | for starters , the story is just too slim . | negative | [2.1086, -1.7989] | 0.0172 | negative | [1.3686, -1.1499] | 44.09 | 0 |
| 199 | positive | this is a winning ensemble comedy that shows canadians can … | positive | [-2.5578, 2.4159] | 0.0229 | FAIL | n/a | 56.35 | 0 |
| 200 | negative | at the very least , if you do n't know anything about derri… | negative | [0.3492, -0.1640] | 0.0213 | FAIL | n/a | 55.60 | 0 |
| 201 | positive | the format gets used best ... to capture the dizzying heigh… | positive | [-0.5887, 0.6565] | 0.0178 | FAIL | n/a | 55.22 | 0 |
| 202 | positive | inside the film 's conflict-powered plot there is a decent … | positive | [-0.3952, 0.4406] | 0.0120 | FAIL | n/a | 55.41 | 0 |
| 203 | negative | there ought to be a directing license , so that ed burns ca… | negative | [1.3466, -0.9103] | 0.0119 | negative | [1.0726, -0.9000] | 43.60 | 0 |
| 204 | negative | bad . | negative | [2.4829, -2.1841] | 0.0185 | negative | [0.8030, -0.7095] | 35.53 | 0 |
| 205 | positive | that dogged good will of the parents and ` vain ' jia 's de… | positive | [-0.8896, 1.0172] | 0.0144 | FAIL | n/a | 23.97 | 0 |
| 206 | positive | falls neatly into the category of good stupid fun . | negative | [1.5690, -1.3699] | 0.0188 | negative | [0.1856, -0.3676] | 44.06 | 0 |
| 207 | positive | an artful , intelligent film that stays within the confines… | positive | [-2.6824, 2.4798] | 0.0175 | positive | [-1.2198, 1.0468] | 46.89 | 0 |
| 208 | positive | smart , provocative and blisteringly funny . | positive | [-2.3149, 2.2307] | 0.0172 | FAIL | n/a | 47.07 | 0 |
| 209 | negative | and the lesson , in the end , is nothing new . | negative | [1.1037, -0.6783] | 0.0174 | negative | [0.7387, -0.6685] | 43.76 | 0 |
| 210 | negative | this is not the undisputed worst boxing movie ever , but it… | negative | [1.2015, -0.8792] | 0.0235 | negative | [0.6274, -0.7695] | 51.41 | 0 |
| 211 | positive | not only is undercover brother as funny , if not more so , … | positive | [-0.4369, 0.4343] | 0.0154 | FAIL | n/a | 56.02 | 0 |
| 212 | negative | to say this was done better in wilder 's some like it hot i… | negative | [1.1295, -0.7499] | 0.0205 | negative | [0.4443, -0.4849] | 50.08 | 0 |
| 213 | negative | the entire movie is about a boring , sad man being boring a… | negative | [2.3311, -2.0651] | 0.0175 | negative | [0.8527, -1.1035] | 45.57 | 0 |
| 214 | negative | this time mr. burns is trying something in the martin scors… | positive | [-0.1030, 0.2739] | 0.0176 | FAIL | n/a | 60.13 | 0 |
| 215 | negative | perceptive in its vision of nascent industrialized world po… | negative | [0.2846, 0.0518] | 0.0235 | FAIL | n/a | 59.53 | 0 |
| 216 | positive | the best revenge may just be living well because this film … | negative | [1.1103, -0.6861] | 0.0259 | FAIL | n/a | 55.31 | 0 |
| 217 | positive | the movie understands like few others how the depth and bre… | positive | [-2.4871, 2.3354] | 0.0266 | positive | [-1.4941, 1.3240] | 60.79 | 0 |
| 218 | negative | once ( kim ) begins to overplay the shock tactics and bait-… | positive | [0.0696, 0.1388] | 0.0197 | FAIL | n/a | 58.45 | 0 |
| 219 | negative | all that 's missing is the spontaneity , originality and de… | positive | [-0.2479, 0.5181] | 0.0178 | positive | [-0.2968, 0.2330] | 54.42 | 0 |
| 220 | positive | what the film lacks in general focus it makes up for in com… | positive | [-1.6719, 1.6237] | 0.0242 | FAIL | n/a | 56.58 | 0 |
| 221 | positive | the socio-histo-political treatise is told in earnest strid… | positive | [0.0541, 0.2295] | 0.0227 | FAIL | n/a | 53.18 | 0 |
| 222 | negative | my reaction in a word : disappointment . | negative | [2.3390, -2.0012] | 0.0154 | negative | [1.3329, -1.1558] | 41.84 | 0 |
| 223 | positive | a psychological thriller with a genuinely spooky premise an… | positive | [-2.4873, 2.3013] | 0.0138 | FAIL | n/a | 53.31 | 0 |
| 224 | positive | corny , schmaltzy and predictable , but still manages to be… | positive | [-0.8635, 0.9488] | 0.0176 | negative | [0.0388, -0.0960] | 46.81 | 0 |
| 225 | positive | nothing 's at stake , just a twisty double-cross you can sm… | positive | [-0.0161, 0.0262] | 0.0181 | FAIL | n/a | 53.14 | 0 |
| 226 | positive | far more imaginative and ambitious than the trivial , cash-… | positive | [-1.0757, 1.0650] | 0.0214 | positive | [-0.8670, 0.7000] | 52.02 | 0 |
| 227 | negative | of course , by more objective measurements it 's still quit… | negative | [1.5685, -1.2602] | 0.0210 | negative | [0.7938, -0.8010] | 49.24 | 0 |
| 228 | positive | as the two leads , lathan and diggs are charming and have c… | positive | [-2.2063, 2.1459] | 0.0179 | positive | [-1.3166, 1.1938] | 48.66 | 0 |
| 229 | positive | it provides an honest look at a community striving to ancho… | positive | [-2.6992, 2.5045] | 0.0183 | positive | [-1.4157, 1.1477] | 46.97 | 0 |
| 230 | negative | this movie seems to have been written using mad-libs . | negative | [1.1841, -0.8391] | 0.0209 | negative | [0.5592, -0.6068] | 44.32 | 0 |
| 231 | positive | reign of fire looks as if it was made without much thought … | positive | [-0.5407, 0.7618] | 0.0210 | positive | [-0.5603, 0.5503] | 51.87 | 0 |
| 232 | positive | martin and barbara are complex characters -- sometimes tend… | positive | [-2.4148, 2.3271] | 0.0218 | FAIL | n/a | 52.47 | 0 |
| 233 | negative | it 's not that kung pow is n't funny some of the time -- it… | negative | [1.5075, -1.2304] | 0.0259 | FAIL | n/a | 51.50 | 0 |
| 234 | negative | i 'd have to say the star and director are the big problems… | negative | [1.2778, -0.9689] | 0.0178 | negative | [1.1204, -0.9825] | 46.02 | 0 |
| 235 | positive | affleck and jackson are good sparring partners . | positive | [-1.8536, 1.7632] | 0.0223 | positive | [-0.3969, 0.2548] | 40.43 | 0 |
| 236 | positive | whether you like rap music or loathe it , you ca n't deny e… | positive | [-0.0608, 0.1781] | 0.0140 | FAIL | n/a | 52.14 | 0 |
| 237 | positive | not since japanese filmmaker akira kurosawa 's ran have the… | negative | [1.1495, -0.8345] | 0.0147 | FAIL | n/a | 56.72 | 0 |
| 238 | negative | a by-the-numbers effort that wo n't do much to enhance the … | negative | [1.6217, -1.3589] | 0.0176 | negative | [1.1053, -1.0351] | 51.17 | 0 |
| 239 | negative | an occasionally funny , but overall limp , fish-out-of-wate… | negative | [1.0735, -0.7446] | 0.0176 | negative | [0.4699, -0.7260] | 46.47 | 0 |
| 240 | positive | brilliantly explores the conflict between following one 's … | positive | [-2.6957, 2.5264] | 0.0178 | positive | [-1.9920, 1.7902] | 45.74 | 0 |
| 241 | positive | despite the 2-d animation , the wild thornberrys movie make… | positive | [-1.0104, 1.0937] | 0.0153 | positive | [-0.4792, 0.3580] | 45.14 | 0 |
| 242 | negative | it appears that something has been lost in the translation … | negative | [1.9091, -1.6403] | 0.0173 | negative | [1.1785, -1.0937] | 47.30 | 0 |
| 243 | negative | it all feels like a monty python sketch gone horribly wrong… | negative | [1.8422, -1.4542] | 0.0116 | negative | [1.2736, -1.1220] | 51.80 | 0 |
| 244 | positive | the film tunes into a grief that could lead a man across ce… | positive | [-0.2175, 0.4249] | 0.0173 | negative | [0.4472, -0.5437] | 45.20 | 0 |
| 245 | positive | dazzles with its fully-written characters , its determined … | positive | [-0.9784, 1.0392] | 0.0122 | FAIL | n/a | 53.66 | 0 |
| 246 | positive | it 's a work by an artist so in control of both his medium … | positive | [0.0460, 0.0650] | 0.0213 | negative | [0.1136, -0.2696] | 53.04 | 0 |
| 247 | positive | it 's the chemistry between the women and the droll scene-s… | positive | [-0.9361, 1.0167] | 0.0177 | FAIL | n/a | 55.42 | 0 |
| 248 | negative | stealing harvard is evidence that the farrelly bros. -- pet… | negative | [0.7213, -0.3845] | 0.0220 | FAIL | n/a | 51.27 | 0 |
| 249 | positive | a full world has been presented onscreen , not some series … | positive | [-0.2081, 0.2896] | 0.0216 | positive | [-0.1783, 0.0331] | 48.48 | 0 |
| 250 | positive | huston nails both the glad-handing and the choking sense of… | negative | [1.1968, -0.7442] | 0.0177 | negative | [0.9631, -0.9961] | 46.07 | 0 |
| 251 | positive | one of the more intelligent children 's movies to hit theat… | positive | [-1.3469, 1.3851] | 0.0220 | negative | [0.2618, -0.3251] | 48.10 | 0 |
| 252 | negative | the film tries too hard to be funny and tries too hard to b… | negative | [1.8101, -1.4432] | 0.0226 | negative | [1.5295, -1.2735] | 42.91 | 0 |
| 253 | positive | blanchett 's performance confirms her power once again . | positive | [-2.4487, 2.3849] | 0.0177 | positive | [-1.8334, 1.6977] | 47.95 | 0 |
| 254 | negative | if you believe any of this , i can make you a real deal on … | positive | [-0.8269, 0.8440] | 0.0192 | FAIL | n/a | 57.32 | 0 |
| 255 | negative | attempts by this ensemble film to impart a message are so h… | negative | [1.6453, -1.3600] | 0.0224 | negative | [0.9626, -1.0504] | 55.71 | 0 |
| 256 | negative | no one but a convict guilty of some truly heinous crime sho… | negative | [1.8034, -1.5336] | 0.0139 | negative | [1.4746, -1.2843] | 49.05 | 0 |
| 257 | negative | rarely has so much money delivered so little entertainment . | negative | [1.2305, -0.8505] | 0.0176 | negative | [0.8920, -0.7265] | 49.74 | 0 |
| 258 | negative | taylor appears to have blown his entire budget on soundtrac… | negative | [2.1623, -1.7986] | 0.0177 | negative | [1.9091, -1.6461] | 47.61 | 0 |
| 259 | negative | `` the time machine '' is a movie that has no interest in i… | negative | [1.6685, -1.3795] | 0.0129 | negative | [0.7120, -0.8314] | 43.94 | 0 |
| 260 | positive | a rarity among recent iranian films : it 's a comedy full o… | positive | [-2.4192, 2.2965] | 0.0251 | positive | [-1.3149, 1.2296] | 53.90 | 0 |
| 261 | negative | / but daphne , you 're too buff / fred thinks he 's tough /… | negative | [0.2993, -0.1142] | 0.0177 | negative | [0.5510, -0.6477] | 53.23 | 0 |
| 262 | positive | the very definition of the ` small ' movie , but it is a go… | positive | [-1.6062, 1.5066] | 0.0176 | FAIL | n/a | 25.47 | 0 |
| 263 | negative | it 's like every bad idea that 's ever gone into an after-s… | negative | [1.1199, -0.7338] | 0.0183 | FAIL | n/a | 54.79 | 0 |
| 264 | positive | chilling , well-acted , and finely directed : david jacobso… | positive | [-2.5423, 2.4055] | 0.0180 | positive | [-1.4099, 1.2717] | 47.55 | 0 |
| 265 | negative | it ca n't decide if it wants to be a mystery/thriller , a r… | negative | [1.0574, -0.6791] | 0.0176 | negative | [0.5550, -0.6710] | 49.16 | 0 |
| 266 | negative | paid in full is so stale , in fact , that its most vibrant … | positive | [-1.2874, 1.2864] | 0.0223 | positive | [-0.9048, 0.7461] | 59.74 | 0 |
| 267 | positive | a coda in every sense , the pinochet case splits time betwe… | negative | [0.7529, -0.4304] | 0.0235 | FAIL | n/a | 54.94 | 0 |
| 268 | negative | it 's played in the most straight-faced fashion , with litt… | positive | [-0.9768, 0.9994] | 0.0192 | positive | [-0.5145, 0.3716] | 51.17 | 0 |
| 269 | negative | a dumb movie with dumb characters doing dumb things and you… | negative | [1.5795, -1.2488] | 0.0217 | negative | [1.3369, -1.2103] | 50.33 | 0 |
| 270 | negative | with virtually no interesting elements for an audience to f… | negative | [0.2939, -0.1688] | 0.0239 | negative | [0.2075, -0.3598] | 54.61 | 0 |
| 271 | positive | dense with characters and contains some thrilling moments . | positive | [-0.9419, 1.1900] | 0.0175 | positive | [-0.7044, 0.5864] | 48.60 | 0 |
| 272 | positive | as unseemly as its title suggests . | negative | [1.7574, -1.5608] | 0.0171 | negative | [1.0271, -0.9135] | 43.25 | 0 |
| 273 | negative | it 's like watching a nightmare made flesh . | negative | [1.6825, -1.2593] | 0.0224 | negative | [0.5256, -0.6056] | 42.56 | 0 |
| 274 | positive | minority report is exactly what the title indicates , a rep… | negative | [1.6918, -1.4708] | 0.0243 | negative | [1.3479, -1.1956] | 50.91 | 0 |
| 275 | negative | it 's hard to like a film about a guy who is utterly unlike… | positive | [-0.7351, 0.8616] | 0.0188 | FAIL | n/a | 51.46 | 0 |
| 276 | positive | an entertaining , colorful , action-filled crime story with… | positive | [-2.8263, 2.6469] | 0.0179 | FAIL | n/a | 52.47 | 0 |
| 277 | positive | for this reason and this reason only -- the power of its ow… | positive | [-0.3023, 0.6210] | 0.0178 | FAIL | n/a | 53.70 | 0 |
| 278 | positive | it just may inspire a few younger moviegoers to read steven… | positive | [-1.6761, 1.6418] | 0.0177 | FAIL | n/a | 62.22 | 0 |
| 279 | negative | basically a static series of semi-improvised ( and semi-coh… | negative | [2.0999, -1.8165] | 0.0176 | negative | [0.4820, -0.6748] | 48.25 | 0 |
| 280 | positive | ... with `` the bourne identity '' we return to the more tr… | positive | [-1.8670, 1.7480] | 0.0175 | positive | [-0.5815, 0.4648] | 47.45 | 0 |
| 281 | positive | it 's so good that its relentless , polished wit can withst… | positive | [-0.6936, 0.7766] | 0.0135 | FAIL | n/a | 61.70 | 0 |
| 282 | negative | chokes on its own depiction of upper-crust decorum . | negative | [1.5181, -1.2582] | 0.0118 | negative | [0.7053, -0.7951] | 47.14 | 0 |
| 283 | positive | while there 's something intrinsically funny about sir anth… | negative | [0.1473, -0.0689] | 0.0121 | FAIL | n/a | 29.61 | 0 |
| 284 | positive | a rewarding work of art for only the most patient and chall… | positive | [-2.4337, 2.2824] | 0.0147 | positive | [-0.8677, 0.6937] | 49.01 | 0 |
| 285 | negative | directed in a paint-by-numbers manner . | negative | [1.5020, -1.2764] | 0.0165 | negative | [0.9516, -0.9059] | 48.52 | 0 |
| 286 | negative | k-19 exploits our substantial collective fear of nuclear ho… | negative | [1.0536, -0.7233] | 0.0226 | negative | [1.0523, -1.0382] | 45.16 | 0 |
| 287 | positive | at its best , queen is campy fun like the vincent price hor… | positive | [-2.2011, 2.1569] | 0.0200 | positive | [-1.0508, 0.9493] | 46.01 | 0 |
| 288 | positive | it 's a much more emotional journey than what shyamalan has… | positive | [-1.5622, 1.5560] | 0.0215 | FAIL | n/a | 55.39 | 0 |
| 289 | positive | the quality of the art combined with the humor and intellig… | positive | [-0.8037, 0.8742] | 0.0184 | FAIL | n/a | 55.21 | 0 |
| 290 | positive | cool ? | negative | [1.0064, -0.6658] | 0.0119 | negative | [0.6914, -0.6217] | 39.03 | 0 |
| 291 | positive | deliriously funny , fast and loose , accessible to the unin… | positive | [-1.1245, 1.1508] | 0.0173 | positive | [-0.5112, 0.3652] | 48.41 | 0 |
| 292 | negative | even with a green mohawk and a sheet of fire-red flame tatt… | negative | [0.9989, -0.6810] | 0.0141 | FAIL | n/a | 54.86 | 0 |
| 293 | negative | the story and the friendship proceeds in such a way that yo… | positive | [-2.1532, 2.1192] | 0.0188 | FAIL | n/a | 58.36 | 0 |
| 294 | positive | at a time when half the so-called real movies are little mo… | positive | [-1.4182, 1.4670] | 0.0172 | FAIL | n/a | 57.33 | 0 |
| 295 | positive | the old-world - meets-new mesh is incarnated in the movie '… | positive | [-1.4304, 1.5151] | 0.0204 | FAIL | n/a | 54.77 | 0 |
| 296 | positive | jones ... does offer a brutal form of charisma . | positive | [-0.7467, 0.7619] | 0.0216 | positive | [-0.3485, 0.1318] | 47.90 | 0 |
| 297 | negative | its well of thorn and vinegar ( and simple humanity ) has l… | positive | [-0.7387, 0.7731] | 0.0179 | FAIL | n/a | 56.65 | 0 |
| 298 | positive | travels a fascinating arc from hope and euphoria to reality… | positive | [-2.4639, 2.4035] | 0.0224 | positive | [-1.8341, 1.6931] | 46.91 | 0 |
| 299 | negative | serving sara does n't serve up a whole lot of laughs . | negative | [0.9518, -0.6553] | 0.0175 | negative | [0.4030, -0.5146] | 41.73 | 0 |
| 300 | positive | the sort of film that makes me miss hitchcock , but also fe… | positive | [-2.4223, 2.2880] | 0.0179 | positive | [-1.7684, 1.6058] | 59.23 | 0 |
| 301 | positive | fun , flip and terribly hip bit of cinematic entertainment . | positive | [-1.6468, 1.6698] | 0.0170 | positive | [-0.1317, 0.0330] | 49.95 | 0 |
| 302 | negative | the x potion gives the quickly named blossom , bubbles and … | positive | [-2.0133, 1.9649] | 0.0181 | FAIL | n/a | 54.92 | 0 |
| 303 | positive | the wild thornberrys movie is a jolly surprise . | positive | [-1.8990, 1.8983] | 0.0175 | positive | [-0.6036, 0.4269] | 41.67 | 0 |
| 304 | positive | entertains by providing good , lively company . | positive | [-2.8573, 2.6412] | 0.0172 | FAIL | n/a | 47.87 | 0 |
| 305 | positive | a densely constructed , highly referential film , and an au… | positive | [-1.3070, 1.2717] | 0.0180 | FAIL | n/a | 56.68 | 0 |
| 306 | negative | what was once original has been co-opted so frequently that… | negative | [1.0738, -0.7217] | 0.0211 | negative | [0.5179, -0.4774] | 48.21 | 0 |
| 307 | positive | the story and structure are well-honed . | positive | [0.0486, 0.2898] | 0.0171 | negative | [0.5126, -0.5206] | 47.22 | 0 |
| 308 | positive | macdowell , whose wifty southern charm has anchored lighter… | positive | [-2.6512, 2.4724] | 0.0179 | positive | [-1.9436, 1.7071] | 55.01 | 0 |
| 309 | positive | an intriguing cinematic omnibus and round-robin that occasi… | positive | [-1.9403, 1.8428] | 0.0118 | positive | [-1.0489, 0.8783] | 58.09 | 0 |
| 310 | positive | the second coming of harry potter is a film far superior to… | positive | [-0.7526, 0.7913] | 0.0196 | FAIL | n/a | 58.45 | 0 |
| 311 | positive | if you can stomach the rough content , it 's worth checking… | positive | [-0.6509, 0.6269] | 0.0175 | negative | [0.1556, -0.2118] | 47.92 | 0 |
| 312 | positive | a warm , funny , engaging film . | positive | [-2.8306, 2.6885] | 0.0174 | FAIL | n/a | 47.57 | 0 |
| 313 | negative | i 'll bet the video game is a lot more fun than the film . | positive | [-0.7665, 0.7919] | 0.0185 | negative | [-0.0352, -0.1140] | 44.47 | 0 |
| 314 | positive | the best film about baseball to hit theaters since field of… | positive | [-1.9252, 1.9722] | 0.0180 | positive | [-0.2695, 0.2055] | 45.43 | 0 |
| 315 | positive | it is great summer fun to watch arnold and his buddy gerald… | positive | [-2.0290, 1.9232] | 0.0218 | positive | [-0.3540, 0.1851] | 56.40 | 0 |
| 316 | negative | complete lack of originality , cleverness or even visible e… | negative | [1.8783, -1.5663] | 0.0217 | negative | [0.8578, -0.9842] | 43.82 | 0 |
| 317 | positive | awesome creatures , breathtaking scenery , and epic battle … | positive | [-2.7699, 2.6110] | 0.0213 | FAIL | n/a | 28.43 | 0 |
| 318 | positive | all-in-all , the film is an enjoyable and frankly told tale… | positive | [-1.0512, 1.0738] | 0.0174 | positive | [-0.6454, 0.4981] | 54.38 | 0 |
| 319 | negative | hit and miss as far as the comedy goes and a big ole ' miss… | negative | [0.4480, -0.1291] | 0.0269 | negative | [0.6756, -0.6908] | 50.41 | 0 |
| 320 | negative | too much of it feels unfocused and underdeveloped . | negative | [2.1936, -1.8557] | 0.0204 | negative | [1.1201, -1.0278] | 46.76 | 0 |
| 321 | positive | a deep and meaningful film . | positive | [-2.8188, 2.6869] | 0.0171 | positive | [-0.7709, 0.7037] | 38.55 | 0 |
| 322 | negative | but it could have been worse . | negative | [2.3972, -2.1250] | 0.0173 | negative | [0.9956, -0.8338] | 41.77 | 0 |
| 323 | negative | that 's pure pr hype . | negative | [1.9710, -1.6444] | 0.0176 | negative | [0.9552, -0.9137] | 42.15 | 0 |
| 324 | positive | a painfully funny ode to bad behavior . | negative | [1.9176, -1.5878] | 0.0204 | negative | [1.2805, -1.2252] | 41.62 | 0 |
| 325 | positive | you 'll gasp appalled and laugh outraged and possibly , wat… | positive | [-0.1733, 0.3394] | 0.0183 | FAIL | n/a | 60.73 | 0 |
| 326 | positive | liotta put on 30 pounds for the role , and has completely t… | positive | [-1.3241, 1.3529] | 0.0187 | positive | [-0.6083, 0.4938] | 49.07 | 0 |
| 327 | positive | a beguiling splash of pastel colors and prankish comedy fro… | positive | [-1.0508, 1.2065] | 0.0190 | positive | [-0.0625, -0.0115] | 52.41 | 0 |
| 328 | positive | it proves quite compelling as an intense , brooding charact… | positive | [-2.4242, 2.3184] | 0.0165 | positive | [-1.6618, 1.4958] | 51.16 | 0 |
| 329 | negative | an unwise amalgam of broadcast news and vibes . | negative | [0.2460, 0.0352] | 0.0258 | negative | [0.4984, -0.6386] | 45.41 | 0 |
| 330 | negative | utterly lacking in charm , wit and invention , roberto beni… | negative | [1.9047, -1.5653] | 0.0142 | negative | [1.3286, -1.2051] | 53.43 | 0 |
| 331 | negative | and that leaves a hole in the center of the salton sea . | negative | [1.4625, -1.2606] | 0.0188 | negative | [0.6477, -0.7619] | 44.91 | 0 |
| 332 | positive | the chateau cleverly probes the cross-cultural differences … | positive | [-2.0116, 1.9116] | 0.0211 | positive | [-0.6451, 0.6298] | 57.73 | 0 |
| 333 | positive | broomfield turns his distinctive ` blundering ' style into … | positive | [-1.7872, 1.6704] | 0.0171 | FAIL | n/a | 25.65 | 0 |
| 334 | positive | a pleasant enough romance with intellectual underpinnings ,… | positive | [-1.9275, 1.8882] | 0.0133 | positive | [-0.9970, 0.8765] | 51.01 | 0 |
| 335 | positive | what really makes it special is that it pulls us into its w… | positive | [-2.6087, 2.4375] | 0.0246 | FAIL | n/a | 57.80 | 0 |
| 336 | negative | with the exception of some fleetingly amusing improvisation… | positive | [-0.2237, 0.2862] | 0.0179 | FAIL | n/a | 58.21 | 0 |
| 337 | positive | having had the good sense to cast actors who are , generall… | positive | [-1.7207, 1.7141] | 0.0177 | FAIL | n/a | 58.01 | 0 |
| 338 | negative | ... a boring parade of talking heads and technical gibberis… | negative | [1.7513, -1.5134] | 0.0195 | negative | [1.1145, -1.2327] | 59.41 | 0 |
| 339 | negative | it 's of the quality of a lesser harrison ford movie - six … | negative | [1.1148, -0.7771] | 0.0181 | negative | [0.2999, -0.4264] | 49.31 | 0 |
| 340 | positive | if you enjoy more thoughtful comedies with interesting conf… | positive | [-2.6745, 2.5164] | 0.0176 | positive | [-1.0303, 0.8636] | 52.88 | 0 |
| 341 | negative | the most hopelessly monotonous film of the year , noteworth… | negative | [1.2606, -0.8912] | 0.0170 | FAIL | n/a | 56.90 | 0 |
| 342 | positive | it deserves to be seen by anyone with even a passing intere… | positive | [-1.8415, 1.8430] | 0.0176 | positive | [-1.4568, 1.2274] | 49.15 | 0 |
| 343 | negative | in an effort , i suspect , not to offend by appearing eithe… | negative | [2.1463, -1.8441] | 0.0176 | FAIL | n/a | 62.45 | 0 |
| 344 | negative | no way i can believe this load of junk . | negative | [2.4290, -2.2508] | 0.0197 | negative | [1.1907, -1.0272] | 40.30 | 0 |
| 345 | positive | there is a fabric of complex ideas here , and feelings that… | positive | [-0.7191, 0.8268] | 0.0176 | positive | [-0.1256, 0.0538] | 54.96 | 0 |
| 346 | positive | this tenth feature is a big deal , indeed -- at least the t… | positive | [-1.7613, 1.6929] | 0.0123 | FAIL | n/a | 53.03 | 0 |
| 347 | negative | not only unfunny , but downright repellent . | negative | [2.3804, -2.1028] | 0.0179 | negative | [1.6451, -1.4108] | 53.84 | 0 |
| 348 | negative | works hard to establish rounded characters , but then has n… | negative | [0.7127, -0.4170] | 0.0174 | FAIL | n/a | 58.93 | 0 |
| 349 | negative | just one bad idea after another . | negative | [2.3607, -2.0041] | 0.0172 | negative | [1.1822, -0.9897] | 41.27 | 0 |
| 350 | negative | ... turns so unforgivably trite in its last 10 minutes that… | negative | [1.1392, -0.6622] | 0.0245 | FAIL | n/a | 55.98 | 0 |
| 351 | negative | his comedy premises are often hackneyed or just plain crude… | negative | [1.3000, -0.8892] | 0.0208 | negative | [0.9086, -0.8996] | 58.90 | 0 |
| 352 | positive | ( næs ) directed the stage version of elling , and gets fin… | positive | [-1.5146, 1.5592] | 0.0175 | positive | [-0.9687, 0.8287] | 59.45 | 0 |
| 353 | positive | a swashbuckling tale of love , betrayal , revenge and above… | positive | [-2.4821, 2.3927] | 0.0176 | FAIL | n/a | 65.00 | 0 |
| 354 | positive | for movie lovers as well as opera lovers , tosca is a real … | positive | [-2.4707, 2.4005] | 0.0245 | FAIL | n/a | 57.02 | 0 |
| 355 | positive | the film is quiet , threatening and unforgettable . | negative | [0.3850, 0.0743] | 0.0218 | negative | [0.6134, -0.5500] | 46.57 | 0 |
| 356 | negative | there is no pleasure in watching a child suffer . | negative | [1.3943, -0.9388] | 0.0117 | negative | [0.6562, -0.6159] | 49.23 | 0 |
| 357 | negative | jason x is positively anti-darwinian : nine sequels and 400… | negative | [0.4478, -0.2291] | 0.0175 | FAIL | n/a | 54.67 | 0 |
| 358 | negative | stealing harvard aspires to comedic grand larceny but stand… | negative | [1.7024, -1.4471] | 0.0204 | negative | [1.3616, -1.2513] | 60.99 | 0 |
| 359 | positive | ( d ) oes n't bother being as cloying or preachy as equival… | negative | [1.1122, -0.7764] | 0.0196 | FAIL | n/a | 53.17 | 0 |
| 360 | positive | displaying about equal amounts of naiveté , passion and tal… | positive | [-2.0433, 2.0004] | 0.0212 | positive | [-1.6853, 1.4737] | 47.72 | 0 |
| 361 | positive | ` easily my choice for one of the year 's best films . ' | positive | [-2.5006, 2.4448] | 0.0164 | FAIL | n/a | 23.45 | 0 |
| 362 | negative | a very long movie , dull in stretches , with entirely too m… | negative | [1.6427, -1.4156] | 0.0175 | negative | [0.8380, -0.8634] | 54.03 | 0 |
| 363 | positive | as a first-time director , paxton has tapped something in h… | positive | [-2.5642, 2.3780] | 0.0178 | positive | [-2.1658, 1.8443] | 49.01 | 0 |
| 364 | negative | it 's a grab bag of genres that do n't add up to a whole lo… | negative | [1.5608, -1.2519] | 0.0174 | negative | [0.6818, -0.7376] | 46.27 | 0 |
| 365 | negative | instead of hiding pinocchio from critics , miramax should h… | negative | [1.8838, -1.5562] | 0.0136 | negative | [1.3270, -1.1388] | 46.67 | 0 |
| 366 | negative | portentous and pretentious , the weight of water is appropr… | negative | [1.6848, -1.3826] | 0.0228 | negative | [1.0838, -1.1258] | 57.78 | 0 |
| 367 | positive | altogether , this is successful as a film , while at the sa… | positive | [-2.5901, 2.4711] | 0.0249 | positive | [-1.6204, 1.4598] | 60.30 | 0 |
| 368 | positive | there has always been something likable about the marquis d… | positive | [-0.3690, 0.5663] | 0.0174 | negative | [-0.0014, -0.0836] | 46.88 | 0 |
| 369 | negative | the humor is forced and heavy-handed , and occasionally sim… | negative | [2.2838, -2.0161] | 0.0167 | negative | [1.5639, -1.3821] | 55.10 | 0 |
| 370 | positive | without ever becoming didactic , director carlos carrera ex… | positive | [-1.9245, 1.9028] | 0.0176 | FAIL | n/a | 62.12 | 0 |
| 371 | negative | partway through watching this saccharine , easter-egg-color… | negative | [1.1450, -0.6798] | 0.0186 | FAIL | n/a | 55.50 | 0 |
| 372 | positive | for the most part , it 's a work of incendiary genius , ste… | positive | [-1.1143, 1.0682] | 0.0140 | positive | [-0.9280, 0.7200] | 54.02 | 0 |
| 373 | positive | the special effects and many scenes of weightlessness look … | positive | [-1.3344, 1.3106] | 0.0121 | FAIL | n/a | 57.51 | 0 |
| 374 | negative | not since freddy got fingered has a major release been so p… | negative | [2.0923, -1.7477] | 0.0125 | negative | [1.5296, -1.3400] | 47.66 | 0 |
| 375 | negative | the movie is what happens when you blow up small potatoes t… | negative | [1.1096, -0.7604] | 0.0182 | negative | [0.4028, -0.5453] | 50.21 | 0 |
| 376 | positive | we have n't seen such hilarity since say it is n't so ! | negative | [1.3166, -0.9063] | 0.0205 | negative | [0.7845, -0.6725] | 47.87 | 0 |
| 377 | negative | to call the other side of heaven `` appalling '' would be t… | negative | [0.9046, -0.4929] | 0.0119 | negative | [0.4440, -0.4492] | 56.18 | 0 |
| 378 | negative | nothing is sacred in this gut-buster . | negative | [1.6443, -1.3846] | 0.0151 | negative | [1.0934, -0.8938] | 41.85 | 0 |
| 379 | negative | feels haphazard , as if the writers mistakenly thought they… | negative | [1.5946, -1.2526] | 0.0244 | FAIL | n/a | 53.37 | 0 |
| 380 | negative | tries to add some spice to its quirky sentiments but the ta… | negative | [1.1871, -0.7903] | 0.0226 | negative | [0.5305, -0.6304] | 57.39 | 0 |
| 381 | negative | at its worst , it implodes in a series of very bad special … | negative | [2.5277, -2.2351] | 0.0187 | negative | [2.0110, -1.7565] | 45.68 | 0 |
| 382 | positive | with tightly organized efficiency , numerous flashbacks and… | positive | [-2.3902, 2.2405] | 0.0250 | positive | [-1.4462, 1.2770] | 52.55 | 0 |
| 383 | negative | a great ensemble cast ca n't lift this heartfelt enterprise… | positive | [-2.5633, 2.4587] | 0.0232 | positive | [-2.2966, 2.1269] | 57.01 | 0 |
| 384 | positive | a warm but realistic meditation on friendship , family and … | positive | [-2.8696, 2.7107] | 0.0177 | FAIL | n/a | 56.59 | 0 |
| 385 | negative | at times , the suspense is palpable , but by the end there … | negative | [0.9135, -0.6022] | 0.0176 | FAIL | n/a | 53.86 | 0 |
| 386 | negative | while the resident evil games may have set new standards fo… | positive | [-1.3179, 1.3308] | 0.0185 | FAIL | n/a | 54.70 | 0 |
| 387 | positive | it 's a remarkably solid and subtly satirical tour de force… | positive | [-2.8469, 2.7116] | 0.0175 | positive | [-2.7481, 2.5381] | 42.14 | 0 |
| 388 | positive | director andrew niccol ... demonstrates a wry understanding… | positive | [-2.7725, 2.6513] | 0.0171 | positive | [-1.5765, 1.4011] | 48.88 | 0 |
| 389 | negative | when leguizamo finally plugged an irritating character late… | negative | [2.2910, -1.9957] | 0.0207 | negative | [1.6956, -1.4791] | 50.49 | 0 |
| 390 | positive | thekids will probably stay amused at the kaleidoscope of bi… | positive | [-1.4665, 1.4645] | 0.0196 | positive | [-1.2554, 1.0118] | 47.87 | 0 |
| 391 | negative | mattei is tiresomely grave and long-winded , as if circular… | negative | [2.4280, -2.1854] | 0.0161 | negative | [2.0743, -1.8690] | 56.35 | 0 |
| 392 | negative | ... plays like somebody spliced random moments of a chris r… | negative | [1.7319, -1.4002] | 0.0199 | FAIL | n/a | 56.93 | 0 |
| 393 | negative | an overemphatic , would-be wacky , ultimately tedious sex f… | negative | [1.7155, -1.3577] | 0.0174 | negative | [1.0229, -0.9325] | 45.24 | 0 |
| 394 | positive | it all adds up to good fun . | positive | [-2.8083, 2.6473] | 0.0172 | positive | [-1.3136, 1.0519] | 41.83 | 0 |
| 395 | positive | whether writer-director anne fontaine 's film is a ghost st… | negative | [0.6240, -0.3104] | 0.0177 | FAIL | n/a | 56.56 | 0 |
| 396 | negative | another in-your-face wallow in the lower depths made by peo… | negative | [0.3643, -0.1359] | 0.0173 | negative | [0.4500, -0.4886] | 55.35 | 0 |
| 397 | positive | a very well-made , funny and entertaining picture . | positive | [-2.7371, 2.5774] | 0.0167 | FAIL | n/a | 51.51 | 0 |
| 398 | positive | it 's worth seeing just on the basis of the wisdom , and at… | positive | [-2.7874, 2.6336] | 0.0186 | FAIL | n/a | 63.08 | 0 |
| 399 | positive | despite its title , punch-drunk love is never heavy-handed . | negative | [1.6701, -1.3049] | 0.0166 | FAIL | n/a | 53.74 | 0 |
| 400 | negative | if director michael dowse only superficially understands hi… | negative | [0.7333, -0.3574] | 0.0179 | negative | [0.8335, -0.9372] | 47.82 | 0 |
| 401 | positive | it 's refreshing to see a girl-power movie that does n't fe… | positive | [-1.0270, 1.1334] | 0.0116 | positive | [-1.5819, 1.4047] | 54.45 | 0 |
| 402 | positive | the film may appear naked in its narrative form ... but it … | positive | [-1.6829, 1.6520] | 0.0191 | positive | [-1.1959, 1.0285] | 53.91 | 0 |
| 403 | negative | however it may please those who love movies that blare with… | positive | [-1.2599, 1.3384] | 0.0173 | positive | [-0.3428, 0.2952] | 65.09 | 0 |
| 404 | negative | as vulgar as it is banal . | negative | [2.3494, -2.0938] | 0.0204 | negative | [1.4067, -1.2039] | 49.07 | 0 |
| 405 | positive | zhang ... has done an amazing job of getting realistic perf… | positive | [-2.3742, 2.2354] | 0.0170 | positive | [-2.0400, 1.7967] | 55.37 | 0 |
| 406 | negative | outer-space buffs might love this film , but others will fi… | positive | [-2.3346, 2.2797] | 0.0115 | positive | [-1.1324, 1.0312] | 44.54 | 0 |
| 407 | positive | maud and roland 's search for an unknowable past makes for … | positive | [-0.8546, 0.8373] | 0.0188 | FAIL | n/a | 53.25 | 0 |
| 408 | negative | more whiny downer than corruscating commentary . | negative | [0.7942, -0.5386] | 0.0128 | negative | [0.6171, -0.7462] | 43.22 | 0 |
| 409 | negative | there are simply too many ideas floating around -- part far… | negative | [1.2422, -0.7746] | 0.0170 | FAIL | n/a | 58.43 | 0 |
| 410 | negative | it 's another stale , kill-by-numbers flick , complete with… | negative | [2.1120, -1.8046] | 0.0191 | negative | [1.1306, -1.2718] | 52.87 | 0 |
| 411 | positive | what distinguishes time of favor from countless other thril… | positive | [-0.6789, 0.8297] | 0.0185 | positive | [-0.1486, 0.0952] | 57.76 | 0 |
| 412 | negative | i do n't mind having my heartstrings pulled , but do n't tr… | negative | [0.2014, -0.0468] | 0.0169 | positive | [-0.1861, 0.1211] | 56.65 | 0 |
| 413 | negative | the movie 's accumulated force still feels like an ugly kno… | negative | [1.8603, -1.5221] | 0.0203 | negative | [1.2209, -1.1834] | 56.61 | 0 |
| 414 | negative | at least one scene is so disgusting that viewers may be har… | negative | [1.7604, -1.2811] | 0.0193 | negative | [1.2409, -1.0590] | 47.33 | 0 |
| 415 | positive | it has charm to spare , and unlike many romantic comedies ,… | negative | [0.2500, 0.1007] | 0.0191 | negative | [0.3164, -0.4357] | 49.90 | 0 |
| 416 | positive | an operatic , sprawling picture that 's entertainingly acte… | positive | [-2.3449, 2.2220] | 0.0168 | FAIL | n/a | 58.30 | 0 |
| 417 | positive | a giggle a minute . | positive | [-1.7033, 1.8673] | 0.0167 | positive | [-0.1940, 0.1791] | 45.40 | 0 |
| 418 | positive | uses sharp humor and insight into human nature to examine c… | positive | [-2.7368, 2.5526] | 0.0145 | FAIL | n/a | 61.70 | 0 |
| 419 | positive | the continued good chemistry between carmen and juni is wha… | positive | [-1.7068, 1.7310] | 0.0171 | FAIL | n/a | 54.72 | 0 |
| 420 | negative | i 'm just too bored to care . | negative | [2.2144, -1.8629] | 0.0139 | negative | [1.6593, -1.4026] | 43.09 | 0 |
| 421 | negative | one of the more irritating cartoons you will see this , or … | negative | [1.2842, -0.8675] | 0.0164 | negative | [1.2026, -1.0225] | 51.53 | 0 |
| 422 | positive | it 's one heck of a character study -- not of hearst or dav… | negative | [0.3118, -0.1978] | 0.0182 | negative | [0.2390, -0.3525] | 51.14 | 0 |
| 423 | positive | it moves quickly , adroitly , and without fuss ; it does n'… | negative | [0.8417, -0.3875] | 0.0174 | FAIL | n/a | 51.39 | 0 |
| 424 | negative | i am sorry that i was unable to get the full brunt of the c… | negative | [0.9333, -0.5232] | 0.0171 | negative | [0.8494, -0.7143] | 45.46 | 0 |
| 425 | positive | a good piece of work more often than not . | positive | [-0.9368, 0.9129] | 0.0167 | positive | [-0.4046, 0.3249] | 43.04 | 0 |
| 426 | positive | while the ideas about techno-saturation are far from novel … | positive | [-2.5365, 2.4166] | 0.0167 | positive | [-1.0862, 0.9466] | 52.65 | 0 |
| 427 | positive | charles ' entertaining film chronicles seinfeld 's return t… | positive | [-0.0107, 0.1405] | 0.0178 | FAIL | n/a | 51.83 | 0 |
| 428 | positive | an exhilarating futuristic thriller-noir , minority report … | positive | [-2.6472, 2.4892] | 0.0168 | FAIL | n/a | 57.78 | 0 |
| 429 | positive | beautifully observed , miraculously unsentimental comedy-dr… | positive | [-1.8031, 1.9450] | 0.0177 | FAIL | n/a | 59.27 | 0 |
| 430 | negative | the film 's hackneyed message is not helped by the thin cha… | negative | [1.9032, -1.5946] | 0.0172 | negative | [1.1650, -1.1604] | 57.16 | 0 |
| 431 | positive | a breezy romantic comedy that has the punch of a good sitco… | positive | [-2.6327, 2.4183] | 0.0164 | positive | [-1.7095, 1.4684] | 69.95 | 0 |
| 432 | negative | should have been someone else - | negative | [1.7417, -1.3100] | 0.0163 | negative | [0.7126, -0.6360] | 44.54 | 0 |
| 433 | negative | coughs and sputters on its own postmodern conceit . | negative | [1.4737, -1.3276] | 0.0216 | negative | [1.0367, -0.9286] | 49.93 | 0 |
| 434 | positive | the lion king was a roaring success when it was released ei… | positive | [0.0443, 0.2509] | 0.0116 | FAIL | n/a | 61.76 | 0 |
| 435 | negative | almost gags on its own gore . | negative | [2.0059, -1.7845] | 0.0165 | negative | [1.1782, -1.0098] | 46.58 | 0 |
| 436 | positive | a marvel like none you 've seen . | negative | [1.0779, -0.6632] | 0.0117 | negative | [0.8433, -0.6983] | 40.97 | 0 |
| 437 | negative | trite , banal , cliched , mostly inoffensive . | negative | [2.4882, -2.2806] | 0.0164 | negative | [1.4475, -1.2643] | 45.07 | 0 |
| 438 | positive | immersing us in the endlessly inventive , fiercely competit… | positive | [-2.5110, 2.3971] | 0.0167 | FAIL | n/a | 53.10 | 0 |
| 439 | positive | the movie has an infectious exuberance that will engage any… | positive | [-1.1751, 1.1869] | 0.0213 | FAIL | n/a | 50.61 | 0 |
