#one hot encoding
data_current_students_final = pd.get_dummies(data_current_students_final, columns=["leadsource_code"])

#normalizing data (used for feature selection but ultimately did not use these features)
data_current_students_final["miles_norm"] = (data_current_students_final["IMP_ZIP_MILES_FROM_CAMPUS"]-data_current_students_final["IMP_ZIP_MILES_FROM_CAMPUS"].min()) / (data_current_students_final["IMP_ZIP_MILES_FROM_CAMPUS"].max()-data_current_students_final["IMP_ZIP_MILES_FROM_CAMPUS"].min())
data_current_students_final["gpa_norm"] = (data_current_students_final["gpa_points"]-data_current_students_final["gpa_points"].min()) / (data_current_students_final["gpa_points"].max()-data_current_students_final["gpa_points"].min())
data_current_students_final["income_rank_norm"] = (data_current_students_final["est_household_income_rank"]-data_current_students_final["est_household_income_rank"].min()) / (data_current_students_final["est_household_income_rank"].max()-data_current_students_final["est_household_income_rank"].min())
data_current_students_final["zip_rank_norm"] = (data_current_students_final["IMP_ZIP_MEDIAN_INCOME_2"]-data_current_students_final["IMP_ZIP_MEDIAN_INCOME_2"].min()) / (data_current_students_final["IMP_ZIP_MEDIAN_INCOME_2"].max()-data_current_students_final["IMP_ZIP_MEDIAN_INCOME_2"].min())
