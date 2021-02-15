import numpy as np
import random
import math
import time
import pandas as pd
from scipy.stats import iqr
import json
from LSTM import LSTM_model
from Data_process import kg_process_data
from Dynamic_hgm_death_whole import dynamic_hgm
from MLP import MLP_model
from kg_constraction_whole_new import Kg_construct_ehr


def main(argv):
    kg = Kg_construct_ehr(argv)
    process_data.separate_train_test()
    dhgm = dynamic_hgm(kg, process_data, 4)

    print("now training 24h RETAIN with CL mortality")
    dhgm.cross_validation("cl_retain")

    df_prc = pd.DataFrame({"recall_ave_seq": dhgm.recall_ave_score, "precision_ave_seq": dhgm.precision_ave_score,
                           "std_precision": dhgm.std_precision})
    df_prc.to_csv("pr_curve_24_RETAIN_CL_mortality", index=False)

    df_roc = pd.DataFrame({"tp_ave_seq": dhgm.tp_ave_score, "fp_ave_seq": dhgm.fp_ave_score,
                           "std_tp": dhgm.std_tp})
    df_roc.to_csv("roc_curve_24_RETAIN_CL_mortality", index=False)

    dhgm.gen_heap_map_csv("heat_map_24h_retain_CL_mortality")

    dhgm.train_data = process_data.train_mortality
    dhgm.config_model_cl_retain()
    dhgm.train()
    dhgm.test_retain(dhgm.test_data_final)
    np.save("embedding_24h_retain_cl_mortality.npy", dhgm.test_patient)
    np.save("logit_24h_retain_cl_mortality.npy", dhgm.test_logit)

    dhgm.sess.close()



if __name__ == "__main__":
    main(sys.argv[1:])
