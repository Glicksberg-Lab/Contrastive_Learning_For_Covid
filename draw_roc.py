import numpy as np
import matplotlib.pyplot as plt


tp_rates = [0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.00138889, 0.00138889,
       0.00277778, 0.00416667, 0.00555556, 0.00694444, 0.00833333,
       0.00972222, 0.01111111, 0.0125    , 0.02083333, 0.02777778,
       0.03888889, 0.04583333, 0.05694444, 0.06388889, 0.06944444,
       0.07083333, 0.075     , 0.08194444, 0.08611111, 0.09166667,
       0.10138889, 0.11388889, 0.11944444, 0.13611111, 0.15138889,
       0.17638889, 0.18611111, 0.20833333, 0.22916667, 0.24444444,
       0.25833333, 0.275     , 0.28611111, 0.29444444, 0.30416667,
       0.30833333, 0.31944444, 0.32361111, 0.32638889, 0.33055556,
       0.33888889, 0.34166667, 0.34861111, 0.35277778, 0.35833333,
       0.36388889, 0.37222222, 0.37777778, 0.38194444, 0.38472222,
       0.38611111, 0.38888889, 0.38888889, 0.39166667, 0.39166667,
       0.39861111, 0.4       , 0.40138889, 0.40416667, 0.40694444,
       0.40972222, 0.41111111, 0.41388889, 0.41805556, 0.42083333,
       0.42361111, 0.425     , 0.425     , 0.425     , 0.42638889,
       0.42916667, 0.43194444, 0.43333333, 0.43333333, 0.43333333,
       0.44027778, 0.44444444, 0.44722222, 0.45138889, 0.45138889,
       0.45138889, 0.45416667, 0.45694444, 0.45833333, 0.45972222,
       0.45972222, 0.46111111, 0.46111111, 0.4625    , 0.4625    ,
       0.46388889, 0.46805556, 0.46805556, 0.46944444, 0.47222222,
       0.47361111, 0.475     , 0.47638889, 0.47638889, 0.47916667,
       0.48055556, 0.48194444, 0.48888889, 0.49305556, 0.49305556,
       0.49305556, 0.49444444, 0.5       , 0.50138889, 0.50416667,
       0.50555556, 0.50833333, 0.50972222, 0.51111111, 0.5125    ,
       0.51805556, 0.51805556, 0.52083333, 0.52222222, 0.52638889,
       0.52916667, 0.53333333, 0.54305556, 0.54722222, 0.55      ,
       0.55555556, 0.5625    , 0.56527778, 0.57222222, 0.58333333,
       0.60138889, 0.61111111, 0.61666667, 0.62083333, 0.64305556,
       0.68194444, 0.69444444, 0.7       , 0.70138889, 0.71388889,
       0.72361111, 0.73888889, 0.78055556, 0.79305556, 0.79861111,
       0.81666667, 0.89444444, 0.94583333, 0.96111111, 1.        ,
       1.        , 1.        ]

fp_rates = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       7.08968451e-05, 7.08968451e-05, 7.08968451e-05, 7.08968451e-05,
       7.08968451e-05, 1.41793690e-04, 2.12690535e-04, 2.12690535e-04,
       2.83587380e-04, 3.54484225e-04, 3.54484225e-04, 4.25381071e-04,
       5.67174761e-04, 7.08968451e-04, 7.79865296e-04, 8.50762141e-04,
       9.92555831e-04, 1.13434952e-03, 1.20524637e-03, 1.41793690e-03,
       1.77242113e-03, 1.91421482e-03, 1.98511166e-03, 2.05600851e-03,
       2.12690535e-03, 2.33959589e-03, 2.33959589e-03, 2.41049273e-03,
       2.62318327e-03, 2.90677065e-03, 3.04856434e-03, 3.04856434e-03,
       3.26125487e-03, 3.33215172e-03, 3.47394541e-03, 3.47394541e-03,
       3.47394541e-03, 3.89932648e-03, 4.04112017e-03, 4.18291386e-03,
       4.32470755e-03, 4.39560440e-03, 4.46650124e-03, 4.53739809e-03,
       4.53739809e-03, 4.67919178e-03, 4.89188231e-03, 5.17546969e-03,
       5.45905707e-03, 5.52995392e-03, 5.67174761e-03, 5.67174761e-03,
       5.81354130e-03, 6.02623183e-03, 6.16802552e-03, 6.16802552e-03,
       6.38071606e-03, 6.52250975e-03, 6.73520028e-03, 6.87699397e-03,
       6.94789082e-03, 7.08968451e-03, 7.08968451e-03, 7.16058135e-03,
       7.30237504e-03, 7.37327189e-03, 7.51506558e-03, 7.65685927e-03,
       7.72775611e-03, 7.86954981e-03, 8.15313719e-03, 8.29493088e-03,
       8.50762141e-03, 8.72031195e-03, 8.72031195e-03, 8.72031195e-03,
       8.72031195e-03, 8.93300248e-03, 9.07479617e-03, 9.21658986e-03,
       9.57107409e-03, 9.57107409e-03, 9.78376462e-03, 1.01382488e-02,
       1.03509394e-02, 1.05636299e-02, 1.07054236e-02, 1.10599078e-02,
       1.14852889e-02, 1.17688763e-02, 1.21233605e-02, 1.24069479e-02,
       1.25487416e-02, 1.28323290e-02, 1.29741227e-02, 1.33286069e-02,
       1.37539879e-02, 1.39666785e-02, 1.43211627e-02, 1.47465438e-02,
       1.49592343e-02, 1.53137185e-02, 1.56682028e-02, 1.64480681e-02,
       1.66607586e-02, 1.72988302e-02, 1.75115207e-02, 1.80786955e-02,
       1.87876639e-02, 1.92839419e-02, 2.00638072e-02, 2.08436725e-02,
       2.11981567e-02, 2.17653314e-02, 2.23325062e-02, 2.31832683e-02,
       2.37504431e-02, 2.48847926e-02, 2.60900390e-02, 2.76497696e-02,
       2.91386033e-02, 3.06274371e-02, 3.28961361e-02, 3.53066289e-02,
       3.77880184e-02, 4.14746544e-02, 4.79262673e-02, 6.06168026e-02,
       8.65650479e-02, 2.03828430e-01, 2.29634881e-01, 2.32470755e-01,
       2.35377526e-01, 2.39560440e-01, 2.51825594e-01, 3.08897554e-01,
       4.27720666e-01, 4.31265509e-01, 4.36582772e-01, 4.50478554e-01,
       6.45586671e-01, 8.17511521e-01, 8.57497341e-01, 1.00000000e+00,
       1.00000000e+00, 1.00000000e+00]

tp_rates_CL = [0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.00933333,
       0.03466667, 0.06      , 0.08      , 0.104     , 0.128     ,
       0.148     , 0.17066667, 0.19466667, 0.21333333, 0.236     ,
       0.26      , 0.28266667, 0.30266667, 0.33733333, 0.36      ,
       0.38266667, 0.40133333, 0.41466667, 0.436     , 0.448     ,
       0.46266667, 0.476     , 0.488     , 0.504     , 0.51066667,
       0.51866667, 0.524     , 0.53333333, 0.53733333, 0.54266667,
       0.54666667, 0.556     , 0.56      , 0.564     , 0.568     ,
       0.57466667, 0.57733333, 0.58      , 0.584     , 0.58933333,
       0.58933333, 0.59066667, 0.59333333, 0.59866667, 0.60133333,
       0.60266667, 0.604     , 0.60533333, 0.608     , 0.612     ,
       0.616     , 0.61733333, 0.62      , 0.62133333, 0.62266667,
       0.624     , 0.62533333, 0.62533333, 0.62666667, 0.62666667,
       0.62666667, 0.63066667, 0.632     , 0.632     , 0.63333333,
       0.63333333, 0.63466667, 0.63733333, 0.63866667, 0.64      ,
       0.64133333, 0.64266667, 0.64666667, 0.64933333, 0.64933333,
       0.65066667, 0.652     , 0.652     , 0.65333333, 0.656     ,
       0.65866667, 0.65866667, 0.66133333, 0.66133333, 0.66266667,
       0.66266667, 0.66266667, 0.66266667, 0.664     , 0.664     ,
       0.66666667, 0.66933333, 0.66933333, 0.67066667, 0.672     ,
       0.672     , 0.672     , 0.672     , 0.67466667, 0.676     ,
       0.676     , 0.67866667, 0.67866667, 0.67866667, 0.67866667,
       0.67866667, 0.68      , 0.68      , 0.68133333, 0.68266667,
       0.68266667, 0.68266667, 0.68266667, 0.68266667, 0.684     ,
       0.684     , 0.68533333, 0.68666667, 0.688     , 0.688     ,
       0.688     , 0.68933333, 0.68933333, 0.69066667, 0.69066667,
       0.69066667, 0.692     , 0.69333333, 0.69333333, 0.69333333,
       0.69466667, 0.696     , 0.696     , 0.69733333, 0.69733333,
       0.70266667, 0.704     , 0.70533333, 0.70533333, 0.70533333,
       0.70666667, 0.708     , 0.70933333, 0.71066667, 0.712     ,
       0.71466667, 0.71466667, 0.716     , 0.716     , 0.716     ,
       0.716     , 0.71733333, 0.71866667, 0.72      , 0.72133333,
       0.72533333, 0.72666667, 0.72666667, 0.73066667, 0.73066667,
       0.73333333, 0.73333333, 0.736     , 0.736     , 0.74      ,
       0.74133333, 0.744     , 0.74533333, 0.75066667, 0.752     ,
       0.756     , 0.75866667, 0.76933333, 0.776     , 0.78266667,
       0.79733333, 0.81733333, 0.82533333, 0.83333333, 0.84266667,
       0.85333333, 0.87866667, 0.89733333, 0.90666667, 0.93466667,
       0.95333333, 0.988     , 1.        , 1.        , 1.        ,
       1.        , 1.        ]

fp_rates_cl = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 3.55239787e-04, 7.10479574e-04,
       9.94671403e-04, 1.20781528e-03, 1.42095915e-03, 1.84724689e-03,
       2.20248668e-03, 2.48667851e-03, 3.12611012e-03, 3.83658970e-03,
       4.33392540e-03, 4.97335702e-03, 5.75488455e-03, 6.18117229e-03,
       6.53641208e-03, 7.03374778e-03, 7.38898757e-03, 8.17051510e-03,
       8.88099467e-03, 9.66252220e-03, 1.03730018e-02, 1.09413854e-02,
       1.15808171e-02, 1.21492007e-02, 1.26465364e-02, 1.32149201e-02,
       1.38543517e-02, 1.42095915e-02, 1.44227353e-02, 1.49911190e-02,
       1.56305506e-02, 1.59147425e-02, 1.61278863e-02, 1.64831261e-02,
       1.68383659e-02, 1.70515098e-02, 1.75488455e-02, 1.80461812e-02,
       1.82593250e-02, 1.86145648e-02, 1.88277087e-02, 1.91119005e-02,
       1.97513321e-02, 2.01065719e-02, 2.03197158e-02, 2.04618117e-02,
       2.07460036e-02, 2.13143872e-02, 2.18827709e-02, 2.25222025e-02,
       2.26642984e-02, 2.28774423e-02, 2.31616341e-02, 2.35168739e-02,
       2.40142096e-02, 2.42984014e-02, 2.46536412e-02, 2.47246892e-02,
       2.50088810e-02, 2.51509769e-02, 2.55062167e-02, 2.58614565e-02,
       2.60035524e-02, 2.62166963e-02, 2.67140320e-02, 2.68561279e-02,
       2.72113677e-02, 2.77797513e-02, 2.83481350e-02, 2.87744227e-02,
       2.90586146e-02, 2.93428064e-02, 2.99822380e-02, 3.02664298e-02,
       3.07637655e-02, 3.08348135e-02, 3.09769094e-02, 3.14742451e-02,
       3.17584369e-02, 3.19005329e-02, 3.19715808e-02, 3.21136767e-02,
       3.23978686e-02, 3.25399645e-02, 3.26820604e-02, 3.28952043e-02,
       3.31083481e-02, 3.34635879e-02, 3.36767318e-02, 3.39609236e-02,
       3.41030195e-02, 3.43161634e-02, 3.47424512e-02, 3.49555950e-02,
       3.50266430e-02, 3.51687389e-02, 3.55950266e-02, 3.58081705e-02,
       3.60923623e-02, 3.61634103e-02, 3.63055062e-02, 3.64476021e-02,
       3.65186501e-02, 3.66607460e-02, 3.68738899e-02, 3.71580817e-02,
       3.72291297e-02, 3.75843694e-02, 3.77975133e-02, 3.82238011e-02,
       3.85790409e-02, 3.92184725e-02, 3.97868561e-02, 4.02131439e-02,
       4.02841918e-02, 4.05683837e-02, 4.11367673e-02, 4.12788632e-02,
       4.19182948e-02, 4.19893428e-02, 4.23445826e-02, 4.24866785e-02,
       4.26998224e-02, 4.29840142e-02, 4.34103020e-02, 4.36944938e-02,
       4.39786856e-02, 4.42628774e-02, 4.46181172e-02, 4.48312611e-02,
       4.52575488e-02, 4.57548845e-02, 4.58259325e-02, 4.61101243e-02,
       4.65364121e-02, 4.70337478e-02, 4.72468917e-02, 4.74600355e-02,
       4.79573712e-02, 4.85257549e-02, 4.88099467e-02, 4.93783304e-02,
       4.98756661e-02, 5.05150977e-02, 5.07992895e-02, 5.14387211e-02,
       5.18650089e-02, 5.25754885e-02, 5.34991119e-02, 5.42095915e-02,
       5.48490231e-02, 5.57726465e-02, 5.64120782e-02, 5.73357016e-02,
       5.84014210e-02, 5.93250444e-02, 6.07460036e-02, 6.18827709e-02,
       6.28774423e-02, 6.43694494e-02, 6.55772647e-02, 6.72824156e-02,
       6.90586146e-02, 7.03374778e-02, 7.22557726e-02, 7.48134991e-02,
       7.79396092e-02, 8.09236234e-02, 8.52575488e-02, 9.02309059e-02,
       9.61989343e-02, 1.06145648e-01, 1.19005329e-01, 1.37548845e-01,
       1.68170515e-01, 2.03410302e-01, 2.80781528e-01, 3.05719361e-01,
       3.28241563e-01, 3.58081705e-01, 3.97726465e-01, 4.81563055e-01,
       5.00959147e-01, 5.27673179e-01, 6.16483126e-01, 7.08490231e-01,
       9.13321492e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
       1.00000000e+00, 1.00000000e+00]

tp_rate_ce_retain=[0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.00441176, 0.01470588, 0.02058824, 0.02794118,
       0.04705882, 0.05441176, 0.06617647, 0.07794118, 0.08529412,
       0.09117647, 0.1       , 0.10588235, 0.11323529, 0.11764706,
       0.125     , 0.13088235, 0.13676471, 0.14411765, 0.15      ,
       0.15882353, 0.16617647, 0.17794118, 0.18676471, 0.19558824,
       0.2       , 0.21470588, 0.21764706, 0.22352941, 0.22794118,
       0.23529412, 0.24705882, 0.25735294, 0.26911765, 0.28088235,
       0.28529412, 0.28970588, 0.3       , 0.30441176, 0.31323529,
       0.31470588, 0.31617647, 0.31911765, 0.32352941, 0.32941176,
       0.33235294, 0.33529412, 0.34117647, 0.34558824, 0.35147059,
       0.35735294, 0.35882353, 0.36323529, 0.36617647, 0.36764706,
       0.37058824, 0.375     , 0.37647059, 0.37794118, 0.38088235,
       0.38235294, 0.38235294, 0.38382353, 0.38676471, 0.38676471,
       0.38823529, 0.39117647, 0.39264706, 0.39264706, 0.39264706,
       0.39852941, 0.4       , 0.40147059, 0.40441176, 0.40441176,
       0.40588235, 0.41323529, 0.41470588, 0.41764706, 0.41911765,
       0.42205882, 0.425     , 0.42941176, 0.42941176, 0.42941176,
       0.42941176, 0.43088235, 0.43235294, 0.43382353, 0.43529412,
       0.43676471, 0.43823529, 0.44117647, 0.44117647, 0.44264706,
       0.44264706, 0.44264706, 0.44264706, 0.44411765, 0.44411765,
       0.44852941, 0.44852941, 0.44852941, 0.45      , 0.45147059,
       0.45147059, 0.45294118, 0.45294118, 0.45294118, 0.45294118,
       0.45441176, 0.45588235, 0.45882353, 0.45882353, 0.46029412,
       0.46470588, 0.46617647, 0.46617647, 0.46617647, 0.46764706,
       0.46764706, 0.46911765, 0.47058824, 0.47205882, 0.47205882,
       0.47205882, 0.47352941, 0.47352941, 0.47647059, 0.47647059,
       0.47647059, 0.47794118, 0.47794118, 0.47941176, 0.48088235,
       0.48235294, 0.48529412, 0.48676471, 0.48676471, 0.48823529,
       0.48823529, 0.48823529, 0.49117647, 0.49411765, 0.49558824,
       0.49852941, 0.5       , 0.50147059, 0.50147059, 0.50147059,
       0.50147059, 0.50147059, 0.50147059, 0.50294118, 0.50882353,
       0.51617647, 0.51617647, 0.52058824, 0.525     , 0.525     ,
       0.525     , 0.52647059, 0.52794118, 0.52941176, 0.53235294,
       0.53970588, 0.54558824, 0.55147059, 0.56029412, 0.57058824,
       0.58382353, 0.60147059, 0.61764706, 0.64558824, 0.65588235,
       0.66176471, 0.67058824, 0.67647059, 0.68823529, 0.70294118,
       0.72058824, 0.74558824, 0.79558824, 0.82647059, 0.86470588,
       0.87058824, 0.90735294, 0.93823529, 0.95294118, 1.        ,
       1.        , 1.        ]

fp_rates_ce_retain=[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.41392718e-04, 2.12089077e-04,
       3.53481796e-04, 4.94874514e-04, 4.94874514e-04, 4.94874514e-04,
       4.94874514e-04, 4.94874514e-04, 4.94874514e-04, 5.65570873e-04,
       5.65570873e-04, 5.65570873e-04, 5.65570873e-04, 6.36267232e-04,
       6.36267232e-04, 7.06963591e-04, 9.19052669e-04, 9.89749028e-04,
       1.06044539e-03, 1.20183811e-03, 1.27253446e-03, 1.34323082e-03,
       1.62601626e-03, 1.69671262e-03, 1.69671262e-03, 1.90880170e-03,
       2.12089077e-03, 2.19158713e-03, 2.33297985e-03, 2.75715801e-03,
       2.82785437e-03, 2.82785437e-03, 3.18133616e-03, 3.25203252e-03,
       3.46412160e-03, 3.81760339e-03, 4.02969247e-03, 4.24178155e-03,
       4.24178155e-03, 4.31247791e-03, 4.45387063e-03, 4.59526334e-03,
       4.80735242e-03, 4.94874514e-03, 5.01944150e-03, 5.23153058e-03,
       5.30222694e-03, 5.37292329e-03, 5.44361965e-03, 5.44361965e-03,
       5.79710145e-03, 6.00919053e-03, 6.07988689e-03, 6.43336868e-03,
       6.71615412e-03, 6.85754684e-03, 6.85754684e-03, 6.92824320e-03,
       6.92824320e-03, 7.14033227e-03, 7.21102863e-03, 7.28172499e-03,
       7.35242135e-03, 7.49381407e-03, 7.56451043e-03, 7.63520679e-03,
       7.70590315e-03, 7.84729586e-03, 7.91799222e-03, 7.98868858e-03,
       8.05938494e-03, 8.13008130e-03, 8.27147402e-03, 8.48356310e-03,
       8.69565217e-03, 8.76634853e-03, 8.76634853e-03, 9.04913397e-03,
       9.26122305e-03, 9.33191941e-03, 9.33191941e-03, 9.40261577e-03,
       9.40261577e-03, 9.54400848e-03, 9.75609756e-03, 9.89749028e-03,
       9.96818664e-03, 1.00388830e-02, 1.01095794e-02, 1.03216684e-02,
       1.05337575e-02, 1.06751502e-02, 1.07458466e-02, 1.07458466e-02,
       1.08165429e-02, 1.08872393e-02, 1.09579357e-02, 1.10286320e-02,
       1.12407211e-02, 1.13821138e-02, 1.15235065e-02, 1.17355956e-02,
       1.18062920e-02, 1.20183811e-02, 1.20890774e-02, 1.21597738e-02,
       1.23011665e-02, 1.23011665e-02, 1.25839519e-02, 1.28667374e-02,
       1.29374337e-02, 1.30081301e-02, 1.32202192e-02, 1.32202192e-02,
       1.34323082e-02, 1.35737010e-02, 1.37150937e-02, 1.39978791e-02,
       1.42806645e-02, 1.44220573e-02, 1.44927536e-02, 1.47048427e-02,
       1.49169318e-02, 1.49876281e-02, 1.54825027e-02, 1.58359844e-02,
       1.60480735e-02, 1.64722517e-02, 1.67550371e-02, 1.70378226e-02,
       1.72499116e-02, 1.74620007e-02, 1.77447861e-02, 1.82396607e-02,
       1.87345352e-02, 1.90173206e-02, 1.91587133e-02, 1.95121951e-02,
       1.97949806e-02, 2.00777660e-02, 2.07847296e-02, 2.12089077e-02,
       2.18451750e-02, 2.23400495e-02, 2.31177094e-02, 2.40367621e-02,
       2.46730293e-02, 2.50265111e-02, 2.59455638e-02, 2.62283492e-02,
       2.66525274e-02, 2.75715801e-02, 2.87734182e-02, 2.99045599e-02,
       3.12477907e-02, 3.26617179e-02, 3.43584305e-02, 3.59137504e-02,
       3.73983740e-02, 3.93778720e-02, 4.24178155e-02, 4.58112407e-02,
       5.03358077e-02, 5.55673383e-02, 6.15058324e-02, 7.04842701e-02,
       8.29268293e-02, 9.96111700e-02, 1.24849770e-01, 1.93778720e-01,
       2.21420997e-01, 2.39377872e-01, 2.48639095e-01, 2.56415695e-01,
       2.68999647e-01, 2.86108165e-01, 3.15871333e-01, 3.67903853e-01,
       4.68292683e-01, 5.30081301e-01, 6.36125840e-01, 6.57122658e-01,
       7.30293390e-01, 8.27642276e-01, 8.54648286e-01, 9.93920113e-01,
       1.00000000e+00, 1.00000000e+00]



tp_rates_CL_RNN = [0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.00285714, 0.00428571,
       0.01428571, 0.02857143, 0.05857143, 0.09285714, 0.11428571,
       0.13857143, 0.16714286, 0.18571429, 0.19857143, 0.20857143,
       0.21714286, 0.23      , 0.24571429, 0.26285714, 0.28714286,
       0.30571429, 0.33285714, 0.36      , 0.39      , 0.42428571,
       0.44571429, 0.47571429, 0.49857143, 0.51571429, 0.52857143,
       0.54428571, 0.55285714, 0.56714286, 0.58      , 0.58571429,
       0.59285714, 0.60142857, 0.60285714, 0.60857143, 0.61142857,
       0.61571429, 0.62285714, 0.63714286, 0.64285714, 0.64428571,
       0.64857143, 0.65142857, 0.65285714, 0.65857143, 0.66428571,
       0.66571429, 0.67      , 0.67285714, 0.67285714, 0.67428571,
       0.67428571, 0.67714286, 0.67714286, 0.68428571, 0.68714286,
       0.68714286, 0.69285714, 0.69285714, 0.69285714, 0.69428571,
       0.69714286, 0.70142857, 0.70285714, 0.70571429, 0.70571429,
       0.70714286, 0.70857143, 0.70857143, 0.71      , 0.71714286,
       0.71857143, 0.72428571, 0.72571429, 0.72714286, 0.73      ,
       0.73      , 0.73      , 0.73      , 0.73142857, 0.73285714,
       0.73428571, 0.73571429, 0.73571429, 0.73714286, 0.73857143,
       0.73857143, 0.74142857, 0.74142857, 0.74428571, 0.74571429,
       0.74571429, 0.74857143, 0.74857143, 0.75142857, 0.75142857,
       0.75142857, 0.75285714, 0.75285714, 0.75285714, 0.75428571,
       0.75428571, 0.75714286, 0.75857143, 0.76285714, 0.76285714,
       0.76285714, 0.76285714, 0.76285714, 0.76571429, 0.76571429,
       0.76714286, 0.76857143, 0.77142857, 0.77142857, 0.77285714,
       0.77428571, 0.77428571, 0.77571429, 0.77571429, 0.77714286,
       0.77857143, 0.78      , 0.78      , 0.78142857, 0.78285714,
       0.78428571, 0.78428571, 0.78571429, 0.78571429, 0.78571429,
       0.78714286, 0.79      , 0.79142857, 0.79428571, 0.79428571,
       0.79428571, 0.79428571, 0.79428571, 0.79571429, 0.79714286,
       0.79714286, 0.79714286, 0.79714286, 0.79857143, 0.80142857,
       0.80428571, 0.80428571, 0.80571429, 0.80714286, 0.80714286,
       0.81428571, 0.81428571, 0.81571429, 0.81857143, 0.82      ,
       0.82571429, 0.83714286, 0.86      , 0.86571429, 0.86714286,
       0.86714286, 0.87428571, 0.88285714, 0.9       , 0.91142857,
       0.91285714, 0.92714286, 0.95714286, 0.97      , 0.98285714,
       1.        , 1.        ]

fp_rates_cl_RNN = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.07964602e-05,
       7.07964602e-05, 1.41592920e-04, 4.24778761e-04, 5.66371681e-04,
       9.20353982e-04, 1.27433628e-03, 1.55752212e-03, 1.76991150e-03,
       2.33628319e-03, 2.76106195e-03, 3.25663717e-03, 3.46902655e-03,
       3.82300885e-03, 4.46017699e-03, 4.81415929e-03, 5.02654867e-03,
       5.52212389e-03, 5.87610619e-03, 6.37168142e-03, 7.15044248e-03,
       7.36283186e-03, 8.00000000e-03, 8.49557522e-03, 9.06194690e-03,
       9.84070796e-03, 1.04778761e-02, 1.09026549e-02, 1.14690265e-02,
       1.18938053e-02, 1.21061947e-02, 1.25309735e-02, 1.30265487e-02,
       1.33097345e-02, 1.39469027e-02, 1.43008850e-02, 1.47964602e-02,
       1.52212389e-02, 1.57168142e-02, 1.59292035e-02, 1.64955752e-02,
       1.68495575e-02, 1.72035398e-02, 1.76283186e-02, 1.77699115e-02,
       1.80530973e-02, 1.83362832e-02, 1.86194690e-02, 1.89734513e-02,
       1.92566372e-02, 1.98230088e-02, 2.01061947e-02, 2.03893805e-02,
       2.06017699e-02, 2.08849558e-02, 2.10265487e-02, 2.14513274e-02,
       2.17345133e-02, 2.25132743e-02, 2.30088496e-02, 2.30796460e-02,
       2.32920354e-02, 2.37168142e-02, 2.43539823e-02, 2.46371681e-02,
       2.47787611e-02, 2.50619469e-02, 2.52743363e-02, 2.55575221e-02,
       2.59115044e-02, 2.59823009e-02, 2.63362832e-02, 2.66194690e-02,
       2.70442478e-02, 2.72566372e-02, 2.74690265e-02, 2.76814159e-02,
       2.78230088e-02, 2.82477876e-02, 2.85309735e-02, 2.89557522e-02,
       2.93805310e-02, 2.95221239e-02, 2.98761062e-02, 3.00884956e-02,
       3.02300885e-02, 3.07964602e-02, 3.09380531e-02, 3.12920354e-02,
       3.17168142e-02, 3.21415929e-02, 3.22831858e-02, 3.26371681e-02,
       3.29203540e-02, 3.33451327e-02, 3.37699115e-02, 3.39115044e-02,
       3.42654867e-02, 3.44778761e-02, 3.51150442e-02, 3.56106195e-02,
       3.58938053e-02, 3.63893805e-02, 3.69557522e-02, 3.75221239e-02,
       3.80176991e-02, 3.88672566e-02, 3.91504425e-02, 3.97168142e-02,
       4.01415929e-02, 4.05663717e-02, 4.10619469e-02, 4.15575221e-02,
       4.21946903e-02, 4.29734513e-02, 4.37522124e-02, 4.42477876e-02,
       4.46017699e-02, 4.51681416e-02, 4.56637168e-02, 4.65132743e-02,
       4.71504425e-02, 4.80000000e-02, 4.89911504e-02, 4.94867257e-02,
       4.98407080e-02, 5.06902655e-02, 5.11150442e-02, 5.19646018e-02,
       5.26017699e-02, 5.28141593e-02, 5.35929204e-02, 5.43008850e-02,
       5.53628319e-02, 5.60707965e-02, 5.70619469e-02, 5.80530973e-02,
       5.95398230e-02, 6.08141593e-02, 6.20176991e-02, 6.31504425e-02,
       6.42831858e-02, 6.58407080e-02, 6.75398230e-02, 6.88141593e-02,
       7.08672566e-02, 7.21415929e-02, 7.34867257e-02, 7.70265487e-02,
       7.89380531e-02, 8.21946903e-02, 8.77876106e-02, 9.39469027e-02,
       1.03362832e-01, 1.18796460e-01, 1.53840708e-01, 2.65911504e-01,
       2.74973451e-01, 2.82265487e-01, 2.91752212e-01, 3.11221239e-01,
       3.58725664e-01, 4.32212389e-01, 4.84814159e-01, 5.07964602e-01,
       5.75575221e-01, 7.36566372e-01, 8.38371681e-01, 8.82902655e-01,
       1.00000000e+00, 1.00000000e+00]
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Mortality Prediction", fontsize=14)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
x = [0.0, 1.0]
plt.plot(x, x, linestyle='dashed', color='red', linewidth=1, label='random')

plt.plot(fp_rates, tp_rates, color='green', linewidth=2, linestyle='dashed',label='RNN+CE(AUC=0.837)')


plt.plot(fp_rates_ce_retain,tp_rate_ce_retain,color='blue',linestyle='dashed',linewidth=2,label='RETAIN+CE(AUC=0.828)')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(fp_rates_cl_RNN,tp_rates_CL_RNN,color='violet',linewidth=1,label='RNN+CL(AUC=0.901)')
plt.plot(fp_rates_cl, tp_rates_CL, color='red', linewidth=1, label='RETAIN+CL(AUC=0.887)')


plt.legend(loc='lower right')
plt.show()