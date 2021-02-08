import numpy as np
import matplotlib.pyplot as plt

precision_ce_death = [0.04586847, 0.04586847, 0.0461425 , 0.06041199, 0.07545766,
       0.09410013, 0.11661155, 0.14046628, 0.1666123 , 0.19063054,
       0.21460858, 0.2390453 , 0.26473868, 0.27871657, 0.30353278,
       0.31980136, 0.33404652, 0.35044419, 0.36021252, 0.37945369,
       0.39522872, 0.40387753, 0.41564071, 0.42590796, 0.43758593,
       0.45551012, 0.46707907, 0.47874868, 0.49063754, 0.50269142,
       0.51052113, 0.51828102, 0.52710264, 0.53475205, 0.54172214,
       0.54517609, 0.55539617, 0.55665526, 0.55620528, 0.55826857,
       0.56489337, 0.56687619, 0.57293048, 0.58333902, 0.58856331,
       0.59022123, 0.59628563, 0.5980247 , 0.60707099, 0.60874606,
       0.6125211 , 0.6155485 , 0.6169967 , 0.61946202, 0.62280142,
       0.62653612, 0.62959354, 0.63159238, 0.63375444, 0.63685155,
       0.63894623, 0.64571411, 0.64746802, 0.64967837, 0.6550835 ,
       0.65732855, 0.65933505, 0.66125296, 0.66284026, 0.66376672,
       0.66789858, 0.67142341, 0.67294872, 0.6755383 , 0.67816357,
       0.67816357, 0.67972133, 0.67933567, 0.67971311, 0.68547065,
       0.68967903, 0.68897845, 0.69117846, 0.69175466, 0.69418433,
       0.6983335 , 0.70027625, 0.70334871, 0.70518736, 0.70566544,
       0.71060829, 0.71123181, 0.71006477, 0.71070579, 0.71063245,
       0.71063245, 0.71129173, 0.71263873, 0.7185221 , 0.7217705 ,
       0.72153708, 0.72302626, 0.72376899, 0.72673552, 0.73153128,
       0.73243043, 0.7316555 , 0.73240712, 0.73240712, 0.73317038,
       0.73767287, 0.74348628, 0.74219267, 0.74224798, 0.74515494,
       0.7472077 , 0.74825445, 0.75049853, 0.75368035, 0.75345064,
       0.75522517, 0.75765438, 0.75618621, 0.75954051, 0.76042546,
       0.76134289, 0.76103217, 0.76123592, 0.76282279, 0.76552605,
       0.76479141, 0.76784225, 0.76719936, 0.76795435, 0.77551888,
       0.78536423, 0.78442458, 0.78843563, 0.79608117, 0.79428779,
       0.79292419, 0.79405661, 0.79237904, 0.79389348, 0.7937752 ,
       0.79582181, 0.79900418, 0.80186416, 0.80091814, 0.79999579,
       0.79870878, 0.80461928, 0.81235297, 0.82165435, 0.8261633 ,
       0.82423217, 0.83657972, 0.83473572, 0.82518192, 0.84085968,
       0.84283622, 0.83919494, 0.8475937 , 0.8631242 , 0.85877847,
       0.86394382, 0.88926185, 0.9057383 , 0.91098638, 0.91815156,
       0.91691835, 0.92068073, 0.93085213, 0.93409407, 0.93178244,
       0.9400303 , 0.93602399, 0.93274132, 0.92871259, 0.9452381 ,
       0.94095238, 0.93652174, 0.93126984, 0.9285213 , 0.91577965,
       0.92669683, 0.94      , 0.9       , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        ]


recall_ce_death = [1.        , 1.        , 1.        , 0.95294118, 0.93823529,
       0.90735294, 0.87058824, 0.86470588, 0.82647059, 0.79558824,
       0.74558824, 0.72058824, 0.70294118, 0.68823529, 0.67647059,
       0.67058824, 0.66176471, 0.65588235, 0.64558824, 0.61764706,
       0.60147059, 0.58382353, 0.57058824, 0.56029412, 0.55147059,
       0.54558824, 0.53970588, 0.53235294, 0.52941176, 0.52794118,
       0.52647059, 0.525     , 0.525     , 0.525     , 0.52058824,
       0.51617647, 0.51617647, 0.50882353, 0.50294118, 0.50147059,
       0.50147059, 0.50147059, 0.50147059, 0.50147059, 0.50147059,
       0.5       , 0.49852941, 0.49558824, 0.49411765, 0.49117647,
       0.48823529, 0.48823529, 0.48823529, 0.48676471, 0.48676471,
       0.48529412, 0.48235294, 0.48088235, 0.47941176, 0.47794118,
       0.47794118, 0.47647059, 0.47647059, 0.47647059, 0.47352941,
       0.47352941, 0.47205882, 0.47205882, 0.47205882, 0.47058824,
       0.46911765, 0.46764706, 0.46764706, 0.46617647, 0.46617647,
       0.46617647, 0.46470588, 0.46029412, 0.45882353, 0.45882353,
       0.45588235, 0.45441176, 0.45294118, 0.45294118, 0.45294118,
       0.45294118, 0.45147059, 0.45147059, 0.45      , 0.44852941,
       0.44852941, 0.44852941, 0.44411765, 0.44411765, 0.44264706,
       0.44264706, 0.44264706, 0.44264706, 0.44117647, 0.44117647,
       0.43823529, 0.43676471, 0.43529412, 0.43382353, 0.43235294,
       0.43088235, 0.42941176, 0.42941176, 0.42941176, 0.42941176,
       0.425     , 0.42205882, 0.41911765, 0.41764706, 0.41470588,
       0.41323529, 0.40588235, 0.40441176, 0.40441176, 0.40147059,
       0.4       , 0.39852941, 0.39264706, 0.39264706, 0.39264706,
       0.39117647, 0.38823529, 0.38676471, 0.38676471, 0.38382353,
       0.38235294, 0.38235294, 0.38088235, 0.37794118, 0.37647059,
       0.375     , 0.37058824, 0.36764706, 0.36617647, 0.36323529,
       0.35882353, 0.35735294, 0.35147059, 0.34558824, 0.34117647,
       0.33529412, 0.33235294, 0.32941176, 0.32352941, 0.31911765,
       0.31617647, 0.31470588, 0.31323529, 0.30441176, 0.3       ,
       0.28970588, 0.28529412, 0.28088235, 0.26911765, 0.25735294,
       0.24705882, 0.23529412, 0.22794118, 0.22352941, 0.21764706,
       0.21470588, 0.2       , 0.19558824, 0.18676471, 0.17794118,
       0.16617647, 0.15882353, 0.15      , 0.14411765, 0.13676471,
       0.13088235, 0.125     , 0.11764706, 0.11323529, 0.10588235,
       0.1       , 0.09117647, 0.08529412, 0.07794118, 0.06617647,
       0.05441176, 0.04705882, 0.02794118, 0.02058824, 0.01470588,
       0.00441176, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        ]

precition_cl_retain = [0.05059022, 0.05059022, 0.05059022, 0.05059022, 0.05059022,
       0.05575269, 0.07698048, 0.09860822, 0.12682813, 0.14603277,
       0.16111147, 0.17926014, 0.19649592, 0.21122631, 0.22513115,
       0.23802102, 0.25114737, 0.2628115 , 0.27974396, 0.29194346,
       0.30453726, 0.31656112, 0.32569257, 0.33541043, 0.343286  ,
       0.34910036, 0.35566331, 0.36224392, 0.3668088 , 0.37033619,
       0.37523777, 0.38102957, 0.38424256, 0.38944364, 0.39201762,
       0.39639778, 0.40099419, 0.40294359, 0.40732813, 0.411133  ,
       0.41328054, 0.41669927, 0.41924688, 0.42220488, 0.42654119,
       0.42877315, 0.43064411, 0.4331762 , 0.43386982, 0.43658247,
       0.43849326, 0.440522  , 0.44166175, 0.44430999, 0.4469696 ,
       0.44737258, 0.44800448, 0.4484324 , 0.45035303, 0.45109868,
       0.45137799, 0.45334198, 0.4551132 , 0.45627028, 0.45832352,
       0.45941597, 0.46042525, 0.46191615, 0.46434542, 0.46561539,
       0.46701137, 0.46714965, 0.46920607, 0.46966176, 0.47312593,
       0.47355131, 0.47670815, 0.47865029, 0.47847506, 0.48122096,
       0.48466689, 0.48852156, 0.49106549, 0.49321037, 0.49412244,
       0.49667274, 0.49663935, 0.49850433, 0.49960137, 0.50045628,
       0.50104382, 0.50110347, 0.50207349, 0.50197209, 0.50304163,
       0.50441484, 0.5074146 , 0.50841686, 0.50849548, 0.50935975,
       0.51237219, 0.51297644, 0.5128859 , 0.5149912 , 0.51583648,
       0.51843909, 0.52018221, 0.52157085, 0.5220438 , 0.52313169,
       0.52420688, 0.52529305, 0.52494937, 0.52495046, 0.52651227,
       0.53020409, 0.53089598, 0.53094197, 0.53511784, 0.53618947,
       0.54015049, 0.54186135, 0.54360385, 0.54676437, 0.55104248,
       0.55494105, 0.55781032, 0.55920011, 0.56334087, 0.565258  ,
       0.5659856 , 0.56786809, 0.5714784 , 0.57288137, 0.57525238,
       0.57607974, 0.57903623, 0.58145762, 0.58617879, 0.58950522,
       0.59154215, 0.59334024, 0.5931353 , 0.59852428, 0.60411522,
       0.61004255, 0.61288332, 0.61405339, 0.61568959, 0.61757344,
       0.62416433, 0.62744241, 0.62992175, 0.6322522 , 0.63351881,
       0.63881675, 0.64410195, 0.64438846, 0.64742568, 0.65098657,
       0.65238617, 0.65274488, 0.66106181, 0.66847799, 0.67010173,
       0.67170105, 0.68208641, 0.68808619, 0.69533136, 0.69853718,
       0.7062638 , 0.71394875, 0.72312421, 0.73414498, 0.73982304,
       0.75965653, 0.7573201 , 0.759597  , 0.76833485, 0.76765904,
       0.80640862, 0.81777003, 0.81391036, 0.8339308 , 0.89101307,
       0.90536381, 0.91043711, 0.92450142, 0.92411067, 0.92136223,
       0.92592593, 0.935     , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        ]

recall_cl_retain_death = [1.        , 1.        , 1.        , 1.        , 1.        ,
       0.988     , 0.95333333, 0.93466667, 0.90666667, 0.89733333,
       0.87866667, 0.85333333, 0.84266667, 0.83333333, 0.82533333,
       0.81733333, 0.79733333, 0.78266667, 0.776     , 0.76933333,
       0.75866667, 0.756     , 0.752     , 0.75066667, 0.74533333,
       0.744     , 0.74133333, 0.74      , 0.736     , 0.736     ,
       0.73333333, 0.73333333, 0.73066667, 0.73066667, 0.72666667,
       0.72666667, 0.72533333, 0.72133333, 0.72      , 0.71866667,
       0.71733333, 0.716     , 0.716     , 0.716     , 0.716     ,
       0.71466667, 0.71466667, 0.712     , 0.71066667, 0.70933333,
       0.708     , 0.70666667, 0.70533333, 0.70533333, 0.70533333,
       0.704     , 0.70266667, 0.69733333, 0.69733333, 0.696     ,
       0.696     , 0.69466667, 0.69333333, 0.69333333, 0.69333333,
       0.692     , 0.69066667, 0.69066667, 0.69066667, 0.68933333,
       0.68933333, 0.688     , 0.688     , 0.688     , 0.68666667,
       0.68533333, 0.684     , 0.684     , 0.68266667, 0.68266667,
       0.68266667, 0.68266667, 0.68266667, 0.68133333, 0.68      ,
       0.68      , 0.67866667, 0.67866667, 0.67866667, 0.67866667,
       0.67866667, 0.676     , 0.676     , 0.67466667, 0.672     ,
       0.672     , 0.672     , 0.672     , 0.67066667, 0.66933333,
       0.66933333, 0.66666667, 0.664     , 0.664     , 0.66266667,
       0.66266667, 0.66266667, 0.66266667, 0.66133333, 0.66133333,
       0.65866667, 0.65866667, 0.656     , 0.65333333, 0.652     ,
       0.652     , 0.65066667, 0.64933333, 0.64933333, 0.64666667,
       0.64266667, 0.64133333, 0.64      , 0.63866667, 0.63733333,
       0.63466667, 0.63333333, 0.63333333, 0.632     , 0.632     ,
       0.63066667, 0.62666667, 0.62666667, 0.62666667, 0.62533333,
       0.62533333, 0.624     , 0.62266667, 0.62133333, 0.62      ,
       0.61733333, 0.616     , 0.612     , 0.608     , 0.60533333,
       0.604     , 0.60266667, 0.60133333, 0.59866667, 0.59333333,
       0.59066667, 0.58933333, 0.58933333, 0.584     , 0.58      ,
       0.57733333, 0.57466667, 0.568     , 0.564     , 0.56      ,
       0.556     , 0.54666667, 0.54266667, 0.53733333, 0.53333333,
       0.524     , 0.51866667, 0.51066667, 0.504     , 0.488     ,
       0.476     , 0.46266667, 0.448     , 0.436     , 0.41466667,
       0.40133333, 0.38266667, 0.36      , 0.33733333, 0.30266667,
       0.28266667, 0.26      , 0.236     , 0.21333333, 0.19466667,
       0.17066667, 0.148     , 0.128     , 0.104     , 0.08      ,
       0.06      , 0.03466667, 0.00933333, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        ]

precision_ce_rnn_death = [0.04856661, 0.04856661, 0.04856661, 0.06388026, 0.09851238,
       0.13451837, 0.20547837, 0.23635951, 0.25490623, 0.26361312,
       0.28081658, 0.31519784, 0.34074671, 0.35298288, 0.36709253,
       0.38008691, 0.39091727, 0.40737028, 0.42325478, 0.44554145,
       0.46018696, 0.46998907, 0.47327419, 0.48291133, 0.49496988,
       0.50487531, 0.51354308, 0.52460769, 0.53331096, 0.54175742,
       0.54263203, 0.54897509, 0.55411689, 0.55822455, 0.56161333,
       0.5695447 , 0.57929092, 0.58280542, 0.59113259, 0.59795509,
       0.60029409, 0.60782787, 0.6101988 , 0.62029701, 0.6248844 ,
       0.62753393, 0.63030331, 0.63747313, 0.64310918, 0.6446271 ,
       0.64860996, 0.65432944, 0.65630406, 0.65974202, 0.66221219,
       0.66671962, 0.67234578, 0.67691046, 0.68387573, 0.69021681,
       0.69307512, 0.69569559, 0.69906644, 0.70664711, 0.71063566,
       0.71063566, 0.71785034, 0.72081084, 0.72322811, 0.72750362,
       0.72638532, 0.72520044, 0.72520044, 0.73000561, 0.73320442,
       0.73537037, 0.74015759, 0.74019601, 0.74212122, 0.74563578,
       0.74856221, 0.74906552, 0.75149548, 0.75264998, 0.75264998,
       0.75674054, 0.75840286, 0.76102544, 0.76541776, 0.76778485,
       0.77346999, 0.77282515, 0.77566459, 0.78072038, 0.78343785,
       0.78284548, 0.78592199, 0.78498512, 0.79341241, 0.80101664,
       0.80850534, 0.81244167, 0.81183561, 0.81319052, 0.81369518,
       0.81411028, 0.81573361, 0.81781162, 0.8209562 , 0.83643813,
       0.83312838, 0.83210872, 0.83328086, 0.83413304, 0.8407981 ,
       0.8389814 , 0.83982263, 0.85150338, 0.85754166, 0.85682121,
       0.85237971, 0.85730914, 0.8539276 , 0.85202283, 0.84322781,
       0.83674887, 0.8561685 , 0.85399197, 0.84310481, 0.82717949,
       0.8668796 , 0.85961538, 0.85761065, 0.86737968, 0.88064516,
       0.90142857, 0.89597701, 0.92857143, 0.93333333, 0.91428571,
       0.97142857, 0.98181818, 0.97777778, 0.97142857, 0.93333333,
       0.93333333, 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        ]

recall_ce_rnn_death_24 = [1.        , 1.        , 1.        , 0.96111111, 0.94583333,
       0.89444444, 0.81666667, 0.79861111, 0.79305556, 0.78055556,
       0.73888889, 0.72361111, 0.71388889, 0.70138889, 0.7       ,
       0.69444444, 0.68194444, 0.64305556, 0.62083333, 0.61666667,
       0.61111111, 0.60138889, 0.58333333, 0.57222222, 0.56527778,
       0.5625    , 0.55555556, 0.55      , 0.54722222, 0.54305556,
       0.53333333, 0.52916667, 0.52638889, 0.52222222, 0.52083333,
       0.51805556, 0.51805556, 0.5125    , 0.51111111, 0.50972222,
       0.50833333, 0.50555556, 0.50416667, 0.50138889, 0.5       ,
       0.49444444, 0.49305556, 0.49305556, 0.49305556, 0.48888889,
       0.48194444, 0.48055556, 0.47916667, 0.47638889, 0.47638889,
       0.475     , 0.47361111, 0.47222222, 0.46944444, 0.46805556,
       0.46805556, 0.46388889, 0.4625    , 0.4625    , 0.46111111,
       0.46111111, 0.45972222, 0.45972222, 0.45833333, 0.45694444,
       0.45416667, 0.45138889, 0.45138889, 0.45138889, 0.44722222,
       0.44444444, 0.44027778, 0.43333333, 0.43333333, 0.43333333,
       0.43194444, 0.42916667, 0.42638889, 0.425     , 0.425     ,
       0.425     , 0.42361111, 0.42083333, 0.41805556, 0.41388889,
       0.41111111, 0.40972222, 0.40694444, 0.40416667, 0.40138889,
       0.4       , 0.39861111, 0.39166667, 0.39166667, 0.38888889,
       0.38888889, 0.38611111, 0.38472222, 0.38194444, 0.37777778,
       0.37222222, 0.36388889, 0.35833333, 0.35277778, 0.34861111,
       0.34166667, 0.33888889, 0.33055556, 0.32638889, 0.32361111,
       0.31944444, 0.30833333, 0.30416667, 0.29444444, 0.28611111,
       0.275     , 0.25833333, 0.24444444, 0.22916667, 0.20833333,
       0.18611111, 0.17638889, 0.15138889, 0.13611111, 0.11944444,
       0.11388889, 0.10138889, 0.09166667, 0.08611111, 0.08194444,
       0.075     , 0.07083333, 0.06944444, 0.06388889, 0.05694444,
       0.04583333, 0.03888889, 0.02777778, 0.02083333, 0.0125    ,
       0.01111111, 0.00972222, 0.00833333, 0.00694444, 0.00555556,
       0.00416667, 0.00277778, 0.00138889, 0.00138889, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        ]

precision_cl_rnn_death=[0.04721754, 0.04721754, 0.0574815 , 0.07378008, 0.08981408,
       0.11690035, 0.1433672 , 0.16302377, 0.17827652, 0.19559601,
       0.21557679, 0.23423389, 0.24788451, 0.26051342, 0.26802064,
       0.28367262, 0.2992274 , 0.3115858 , 0.3230945 , 0.33134977,
       0.34092435, 0.34782496, 0.350588  , 0.36049095, 0.36371035,
       0.36654418, 0.37235315, 0.37566394, 0.38027764, 0.38587838,
       0.389838  , 0.39408713, 0.39853335, 0.40337958, 0.40927286,
       0.41329076, 0.41741628, 0.420263  , 0.4249204 , 0.42715382,
       0.43039869, 0.43064834, 0.43324489, 0.43728823, 0.43920678,
       0.44270517, 0.44441852, 0.4467843 , 0.45127235, 0.45574933,
       0.45925906, 0.4632044 , 0.46565771, 0.46772275, 0.46976437,
       0.47165094, 0.47586465, 0.48015325, 0.48378138, 0.48650849,
       0.48839846, 0.49049851, 0.49248641, 0.49614813, 0.49709559,
       0.50250626, 0.50601107, 0.5092633 , 0.51298179, 0.51502467,
       0.51645721, 0.51926116, 0.52395664, 0.52495784, 0.52721175,
       0.52822156, 0.53105136, 0.53427963, 0.53661043, 0.53803951,
       0.53914293, 0.54168886, 0.54496466, 0.54711762, 0.54742846,
       0.55189555, 0.55204243, 0.55341064, 0.55589752, 0.55645112,
       0.55989783, 0.56299002, 0.56472268, 0.56828478, 0.569243  ,
       0.5711124 , 0.57298532, 0.57495304, 0.57772068, 0.57978992,
       0.58269152, 0.5812673 , 0.58371508, 0.58394531, 0.58525204,
       0.58827874, 0.58900259, 0.5913728 , 0.59730111, 0.60058307,
       0.60217567, 0.60158421, 0.60614636, 0.61401911, 0.61699527,
       0.62163987, 0.62102291, 0.62429643, 0.62572888, 0.62671112,
       0.63002361, 0.63541859, 0.63874394, 0.64237846, 0.64585254,
       0.64847237, 0.65044157, 0.65159876, 0.65491881, 0.65749205,
       0.66165126, 0.66836305, 0.66980542, 0.67635821, 0.68047425,
       0.68291684, 0.68542659, 0.69405571, 0.69768499, 0.70410546,
       0.71118217, 0.71129966, 0.71605421, 0.72425601, 0.7283649 ,
       0.73778348, 0.7518312 , 0.75954719, 0.77003206, 0.78166676,
       0.77874509, 0.79763491, 0.80691193, 0.80256434, 0.81852919,
       0.81593208, 0.81595918, 0.83899411, 0.8412782 , 0.8400875 ,
       0.89664428, 0.92234606, 0.93762575, 0.94206566, 0.94928672,
       0.95779154, 0.96862745, 0.97174419, 0.98666667, 0.99047619,
       0.98      , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        ]

recall_cl_rnn_death_24 = [1.        , 1.        , 0.98285714, 0.97      , 0.95714286,
       0.92714286, 0.91285714, 0.91142857, 0.9       , 0.88285714,
       0.87428571, 0.86714286, 0.86714286, 0.86571429, 0.86      ,
       0.83714286, 0.82571429, 0.82      , 0.81857143, 0.81571429,
       0.81428571, 0.81428571, 0.80714286, 0.80714286, 0.80571429,
       0.80428571, 0.80428571, 0.80142857, 0.79857143, 0.79714286,
       0.79714286, 0.79714286, 0.79714286, 0.79571429, 0.79428571,
       0.79428571, 0.79428571, 0.79428571, 0.79428571, 0.79142857,
       0.79      , 0.78714286, 0.78571429, 0.78571429, 0.78571429,
       0.78428571, 0.78428571, 0.78285714, 0.78142857, 0.78      ,
       0.78      , 0.77857143, 0.77714286, 0.77571429, 0.77571429,
       0.77428571, 0.77428571, 0.77285714, 0.77142857, 0.77142857,
       0.76857143, 0.76714286, 0.76571429, 0.76571429, 0.76285714,
       0.76285714, 0.76285714, 0.76285714, 0.76285714, 0.75857143,
       0.75714286, 0.75428571, 0.75428571, 0.75285714, 0.75285714,
       0.75285714, 0.75142857, 0.75142857, 0.75142857, 0.74857143,
       0.74857143, 0.74571429, 0.74571429, 0.74428571, 0.74142857,
       0.74142857, 0.73857143, 0.73857143, 0.73714286, 0.73571429,
       0.73571429, 0.73428571, 0.73285714, 0.73142857, 0.73      ,
       0.73      , 0.73      , 0.73      , 0.72714286, 0.72571429,
       0.72428571, 0.71857143, 0.71714286, 0.71      , 0.70857143,
       0.70857143, 0.70714286, 0.70571429, 0.70571429, 0.70285714,
       0.70142857, 0.69714286, 0.69428571, 0.69285714, 0.69285714,
       0.69285714, 0.68714286, 0.68714286, 0.68428571, 0.67714286,
       0.67714286, 0.67428571, 0.67428571, 0.67285714, 0.67285714,
       0.67      , 0.66571429, 0.66428571, 0.65857143, 0.65285714,
       0.65142857, 0.64857143, 0.64428571, 0.64285714, 0.63714286,
       0.62285714, 0.61571429, 0.61142857, 0.60857143, 0.60285714,
       0.60142857, 0.59285714, 0.58571429, 0.58      , 0.56714286,
       0.55285714, 0.54428571, 0.52857143, 0.51571429, 0.49857143,
       0.47571429, 0.44571429, 0.42428571, 0.39      , 0.36      ,
       0.33285714, 0.30571429, 0.28714286, 0.26285714, 0.24571429,
       0.23      , 0.21714286, 0.20857143, 0.19857143, 0.18571429,
       0.16714286, 0.13857143, 0.11428571, 0.09285714, 0.05857143,
       0.02857143, 0.01428571, 0.00428571, 0.00285714, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        ]




plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Mortality Prediction", fontsize=14)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
x = [0.0, 1.0]
#plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

plt.plot(recall_ce_rnn_death_24, precision_ce_rnn_death, color='green', linestyle='dashed',linewidth=2, label='RNN+CE(AUC=0.823)')


plt.plot(recall_ce_death,precision_ce_death,color='blue',linestyle='dashed',linewidth=2,label='RETAIN+CE(AUC=0.837)')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(precision_cl_rnn_death,recall_cl_rnn_death_24,color='violet',linewidth=1.5,label='RNN+CL(AUC=0.901)')
plt.plot(recall_cl_retain_death, precition_cl_retain, color='red', linewidth=1.5, label='RETAIN+CL(AUC=0.887)')


plt.legend(loc='lower right')
plt.show()