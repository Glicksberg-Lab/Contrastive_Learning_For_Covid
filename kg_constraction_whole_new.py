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


class Kg_construct_ehr():
    """
    construct knowledge graph out of EHR data
    """

    def __init__(self,args):
        """
        Read all input array
        """
        self.reg_ar = args[0]
        self.labtest_ar = args[1]
        self.vital_sign_ar = args[2]
        #self.lab_comb_ar = args[3]

    def create_kg_dic(self):
        self.dic_patient = {}
        self.dic_vital = {}
        self.dic_lab = {}
        self.dic_filter_patient = {}
        self.dic_lab_category = {}
        self.dic_demographic = {}
        self.dic_race = {}
        self.changed_death = []

        """
        create inital lab dictionary
        """
        index_lab = 0
        for i in range(index_name.shape[0]):
            name_category = self.lab_comb_keep[i][0]
            if name_category not in self.dic_lab:
                self.dic_lab[name_category] = {}
                self.dic_lab[name_category]['index'] = index_lab
                index_lab += 1

        """
        create initial vital sign dictionary
        """
        index_vital = 0
        for i in self.crucial_vital:
            self.dic_vital[i] = {}
            self.dic_vital[i]['index'] = index_vital
            index_vital += 1

        """
        get all patient with admit time
        """
        admit_time = np.where(self.reg_ar[:,1]==self.reg_ar[:,1])[0]
        self.admit = self.reg_ar[admit_time,:]
        covid_obv = np.where(self.admit[:,5]==self.admit[:,5])[0]
        self.covid_ar = self.admit[covid_obv,:]

        """
        filter out the first visit ID
        """
        for i in range(self.covid_ar.shape[0]):
            print("im here in filter visit ID")
            print(i)
            mrn_single = self.covid_ar[i,0]
            if mrn_single not in self.dic_patient.keys():
                self.dic_patient[mrn_single] = {}
                self.dic_patient[mrn_single]['prior_time_vital'] = {}
                self.dic_patient[mrn_single]['prior_time_lab'] = {}
            in_admit_time_single = self.covid_ar[i,1]

            self.in_admit_time = in_admit_time_single.split(' ')
            in_admit_date = [np.int(j) for j in self.in_admit_time[0].split('-')]
            in_admit_date_value = (in_admit_date[0] * 365.0 + in_admit_date[1] * 30 + in_admit_date[2]) * 24 * 60
            self.in_admit_time_ = [np.int(j) for j in self.in_admit_time[1].split(':')[0:-1]]
            in_admit_time_value = self.in_admit_time_[0] * 60.0 + self.in_admit_time_[1]
            total_in_admit_time_value = in_admit_date_value + in_admit_time_value
            self.dic_patient[mrn_single].setdefault('Admit_time_values', []).append(total_in_admit_time_value)
            """
            filter intubation
            """
            if self.covid_ar[i, 2] == self.covid_ar[i, 2]:
                self.dic_patient[mrn_single]['icu_label'] = 1
                in_time_single = self.covid_ar[i, 2]
                self.in_time = in_time_single.split(' ')
                in_date = [np.int(j) for j in self.in_time[0].split('-')]
                in_date_value = (in_date[0] * 365.0 + in_date[1] * 30 + in_date[2]) * 24 * 60
                self.in_time_ = [np.int(j) for j in self.in_time[1].split(':')[0:-1]]
                in_time_value = self.in_time_[0] * 60.0 + self.in_time_[1]
                total_in_time_value = in_date_value + in_time_value
                self.dic_patient[mrn_single]['in_icu_time'] = self.in_time
                self.dic_patient[mrn_single]['in_date'] = in_date
                self.dic_patient[mrn_single]['in_time'] = self.in_time_
                self.dic_patient[mrn_single]['total_in_icu_time_value'] = total_in_time_value
            else:
                self.dic_patient[mrn_single]['icu_label'] = 0
            """
            filter intubation
            """
            if self.covid_ar[i, 3] == self.covid_ar[i, 3]:
                self.dic_patient[mrn_single]['intubation_label'] = 1
                in_time_single = self.covid_ar[i, 3]
                self.in_time = in_time_single.split(' ')
                in_date = [np.int(i) for i in self.in_time[0].split('-')]
                in_date_value = (in_date[0] * 365.0 + in_date[1] * 30 + in_date[2]) * 24 * 60
                self.in_time_ = [np.int(i) for i in self.in_time[1].split(':')[0:-1]]
                in_time_value = self.in_time_[0] * 60.0 + self.in_time_[1]
                total_in_time_value = in_date_value + in_time_value
                self.dic_patient[mrn_single]['intubation_time'] = self.in_time
                self.dic_patient[mrn_single]['intubation_date'] = in_date
                self.dic_patient[mrn_single]['intubation_time'] = self.in_time_
                self.dic_patient[mrn_single]['total_intubation_time_value'] = total_in_time_value
            else:
                self.dic_patient[mrn_single]['intubation_label'] = 0

            """
            filter mortality
            """
            if self.covid_ar[i, 4] == self.covid_ar[i, 4]:
                death_flag = 1
                death_time_ = kg.covid_ar[i][4]
                self.dic_patient[mrn_single]['death_time'] = death_time_
                death_time = death_time_.split(' ')
                death_date = [np.int(l) for l in death_time[0].split('-')]
                death_date_value = (death_date[0] * 365.0 + death_date[1] * 30 + death_date[2]) * 24 * 60
                dead_time_ = [np.int(l) for l in death_time[1].split(':')[0:-1]]
                dead_time_value = dead_time_[0] * 60.0 + dead_time_[1]
                total_dead_time_value = death_date_value + dead_time_value
                self.dic_patient[mrn_single]['death_value'] = total_dead_time_value
            else:
                death_flag = 0
            self.dic_patient[mrn_single]['death_flag'] = death_flag


        """
        filter out labels
        """
        self.total_in_icu_time = []
        self.total_intubation_time = []
        self.total_death_time = []
        self.dic_death = {}
        self.dic_intubation = {}
        self.dic_in_icu = {}
        for i in self.dic_patient.keys():
            self.dic_patient[i]['Admit_time_values'] = np.sort(self.dic_patient[i]['Admit_time_values'])
            if self.dic_patient[i]['icu_label'] == 1:
                if len(self.dic_patient[i]['Admit_time_values'])>1:
                    if self.dic_patient[i]['total_in_icu_time_value']>self.dic_patient[i]['Admit_time_values'][1]:
                        self.dic_patient[i]['icu_label'] = 0
                        self.dic_patient[i]['filter_first_icu_visit'] = 1
            if self.dic_patient[i]['death_flag'] == 1:
                if len(self.dic_patient[i]['Admit_time_values'])>1:
                    if self.dic_patient[i]['death_value']>self.dic_patient[i]['Admit_time_values'][1]:
                        self.dic_patient[i]['death_flag'] = 0
                        self.dic_patient[i]['filter_first_death_visit'] = 1

            if self.dic_patient[i]['intubation_label'] == 1:
                if len(self.dic_patient[i]['Admit_time_values'])>1:
                    if self.dic_patient[i]['total_intubation_time_value']>self.dic_patient[i]['Admit_time_values'][1]:
                        self.dic_patient[i]['intubation_label'] = 0
                        self.dic_patient[i]['filter_first_intubation_visit'] = 1

        for i in self.dic_patient.keys():
            if self.dic_patient[i]['icu_label'] == 1:
                total_in_icu_time_value = self.dic_patient[i]['total_in_icu_time_value']
                total_in_admit_time_value = self.dic_patient[i]['Admit_time_values'][0]
                self.dic_patient[i]['in_icu_hour'] = np.int(
                    np.floor((total_in_icu_time_value - total_in_admit_time_value) / 60))
                self.total_in_icu_time.append(kg.dic_patient[i]['in_icu_hour'])
                self.dic_in_icu.setdefault(1, []).append(i)
            if self.dic_patient[i]['icu_label'] == 0:
                self.dic_in_icu.setdefault(0, []).append(i)

            if self.dic_patient[i]['death_flag'] == 1:
                total_death_value = self.dic_patient[i]['death_value']
                self.dic_patient[i]['death_hour'] = np.int(
                    np.floor((total_death_value - self.dic_patient[i]['Admit_time_values'][0]) / 60))
                self.total_death_time.append(self.dic_patient[i]['death_hour'])
                self.dic_death.setdefault(1, []).append(i)
            if self.dic_patient[i]['death_flag'] == 0:
                self.dic_death.setdefault(0, []).append(i)
            if self.dic_patient[i]['intubation_label'] == 1:
                total_intubation_time_value = self.dic_patient[i]['total_intubation_time_value']
                total_in_admit_time_value = self.dic_patient[i]['Admit_time_values'][0]
                self.dic_patient[i]['intubation_hour'] = np.int(
                    np.floor((total_intubation_time_value - total_in_admit_time_value) / 60))
                self.total_intubation_time.append(self.dic_patient[i]['intubation_hour'])
                self.dic_intubation.setdefault(1, []).append(i)
            if self.dic_patient[i]['intubation_label'] == 0:
                self.dic_intubation.setdefault(0, []).append(i)


        self.total_data_mortality = []
        self.un_correct_mortality = []
        self.total_data_intubation = []
        self.un_correct_intubation = []
        self.total_data_icu = []
        self.un_correct_icu = []

        for i in self.dic_patient.keys():
            if self.dic_patient[i]['death_flag'] == 0:
                self.total_data_mortality.append(i)
            if self.dic_patient[i]['death_flag'] == 1:
                if self.dic_patient[i]['death_hour'] > 0:
                    self.total_data_mortality.append(i)
                else:
                    self.un_correct_mortality.append(i)
            if self.dic_patient[i]['intubation_label'] == 0:
                self.total_data_intubation.append(i)
            if self.dic_patient[i]['intubation_label'] == 1:
                if self.dic_patient[i]['intubation_hour'] > 0:
                    self.total_data_intubation.append(i)
                else:
                    self.un_correct_intubation.append(i)
            if self.dic_patient[i]['icu_label'] == 0:
                self.total_data_icu.append(i)
            if self.dic_patient[i]['icu_label'] == 1:
                if self.dic_patient[i]['in_icu_hour'] > 0:
                    self.total_data_icu.append(i)
                else:
                    self.un_correct_icu.append(i)

        index = 0
        for i in self.dic_patient.keys():
            print(index)
            index += 1
            #in_icu_date = self.reg_ar
            self.single_patient_vital = np.where(self.vital_sign_ar[:, 0] == i)[0]
            in_time_value = self.dic_patient[i]['Admit_time_values'][0]
            self.single_patient_lab = np.where(self.labtest_ar[:, 0] == i)[0]
            total_value_lab = 0

            for k in self.single_patient_lab:
                obv_id = self.labtest_ar[k][1]
                patient_lab_mrn = self.labtest_ar[k][0]
                value = self.labtest_ar[k][2]
                self.check_data_lab = self.labtest_ar[k][3]
                date_year_value_lab = float(str(self.labtest_ar[k][3])[0:4]) * 365
                date_day_value_lab = float(str(self.check_data_lab)[4:6]) * 30 + float(str(self.check_data_lab)[6:8])
                date_value_lab = (date_year_value_lab + date_day_value_lab) * 24 * 60
                date_time_value_lab = float(str(self.check_data_lab)[8:10]) * 60 + float(
                    str(self.check_data_lab)[10:12])
                self.total_time_value_lab = date_value_lab + date_time_value_lab
                self.dic_patient[i].setdefault('lab_time_check', []).append(self.check_data_lab)
                if obv_id in self.dic_lab_category.keys():
                    category = self.dic_lab_category[obv_id]
                    self.prior_time = np.int(np.floor(np.float((self.total_time_value_lab - in_time_value) / 60)))
                    if self.prior_time < 0:
                        continue
                    try:
                        value = float(value)
                    except:
                        continue
                    if not value == value:
                        continue
                    if i not in self.dic_lab[category]:
                        # self.dic_lab[category]['patient_values'][i]={}
                        self.dic_lab[category].setdefault('lab_value_patient', []).append(value)
                    else:
                        self.dic_lab[category].setdefault('lab_value_patient', []).append(value)
                    if self.prior_time not in self.dic_patient[i]['prior_time_lab']:
                        self.dic_patient[i]['prior_time_lab'][self.prior_time] = {}
                        self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(category, []).append(value)
                    else:
                        self.dic_patient[i]['prior_time_lab'][self.prior_time].setdefault(category, []).append(value)


            for j in self.single_patient_vital:
                obv_id = self.vital_sign_ar[j][1]
                if obv_id in self.crucial_vital:
                    self.check_data = self.vital_sign_ar[j][3]
                    self.dic_patient[i].setdefault('time_capture', []).append(self.check_data)
                    date_year_value = float(str(self.vital_sign_ar[j][3])[0:4]) * 365
                    date_day_value = float(str(self.check_data)[4:6]) * 30 + float(str(self.check_data)[6:8])
                    date_value = (date_year_value + date_day_value) * 24 * 60
                    date_time_value = float(str(self.check_data)[8:10]) * 60 + float(str(self.check_data)[10:12])
                    total_time_value = date_value + date_time_value
                    self.prior_time = np.int(np.floor(np.float((total_time_value - in_time_value) / 60)))
                    if self.prior_time < 0:
                        continue
                    self.check_value = self.vital_sign_ar[j][2]
                    self.check_obv = obv_id
                    self.check_ar = self.vital_sign_ar[j]
                    if self.check_value == '""':
                        continue
                    value = float(self.vital_sign_ar[j][2])
                    if np.isnan(value):
                        continue
                    if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                        self.dic_patient[i]['prior_time_vital'][self.prior_time] = {}
                        self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                            value)
                    else:
                        self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id, []).append(
                            value)
                    self.dic_vital[obv_id].setdefault('value', []).append(value)

    def gen_demo_csv(self, name_to_store):
        pick_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 19,
                    20, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 41, 43,
                    45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 66,
                    67, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 83]
        pick_num = np.array(pick_num)
        self.feature_mean = []
        feature = list(np.array(list(self.dic_vital.keys()) + list(self.dic_lab.keys()))[pick_num])
        self.feature_copy = []
        self.feature_iqr = []
        feature_csv = feature #+ feature + feature + feature
        for i in feature:
            if i in self.dic_vital.keys():
                #self.feature_mean.append(self.dic_vital[i]['mean_value'])
                #mean = self.dic_vital[i]['mean_value']
                self.feature_copy.append(i)
                #std = np.float(self.dic_vital[i]['std'])
                values = [np.float(i) for i in self.dic_vital[i]['value'] if np.float(i)<mean+std]
                percent_75 = np.percentile(values,75)
                values_correction = [i for i in values if i < percent_75]
                mean_value = np.mean(values_correction)
                self.feature_mean.append(mean_value)
                irq_value = iqr(values_correction)
                self.feature_iqr.append(irq_value)
            if i in self.dic_lab.keys():
                #self.feature_mean.append(self.dic_lab[i]['mean_value'])
                #mean = self.dic_lab[i]['mean_value']
                self.feature_copy.append(i)
                #std = np.float(self.dic_labl[i]['std'])
                values = [np.float(i) for i in self.dic_lab[i]['lab_value_patient'] if np.float(i)<mean+std]
                percent_75 = np.percentile(values, 75)
                values_correction = [i for i in values if i < percent_75]
                mean_value = np.mean(values_correction)
                self.feature_mean.append(mean_value)
                irq_value = iqr(values_correction)
                self.feature_iqr.append(irq_value)

        #time_seq = list(np.ones(63)) + list(2 * np.ones(63)) + list(3 * np.ones(63)) + list(4 * np.ones(63))
        #time_step1 = self.ave_data_scores_total[0, :][pick_num]
        #time_step2 = self.ave_data_scores_total[1, :][pick_num]
        #time_step3 = self.ave_data_scores_total[2, :][pick_num]
        #time_step4 = self.ave_data_scores_total[3, :][pick_num]
        #variable_scores = list(time_step1) + list(time_step2) + list(time_step3) + list(time_step4)
        df = pd.DataFrame(
            {"Demographic Features": self.feature_copy, "mean_value": self.feature_mean,"irq":self.feature_iqr})
        df.to_csv(name_to_store, index=False)

    def gen_race_csv(self,name_to_store):
        self.dic_race_sec = {}
        self.dic_gen_sec = {}
        self.age_list = []
        for i in self.total_data_mortality:
            race = self.dic_demographic[i]['race']
            gender = self.dic_demographic[i]['gender']
            if race not in self.dic_race_sec.keys():
                self.dic_race_sec[race] = {}
                self.dic_race_sec[race] = 1
            else:
                self.dic_race_sec[race] += 1

            if gender not in self.dic_gen_sec.keys():
                self.dic_gen_sec[gender] = {}
                self.dic_gen_sec[gender] = 1
            else:
                self.dic_gen_sec[gender] += 1
            age = self.dic_demographic[i]['Age']
            self.age_list.append(age)
