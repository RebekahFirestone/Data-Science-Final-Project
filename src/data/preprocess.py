#filtering for only completed academic years
mask = data["academic_year"].isin([2020,2021,2022,2023,2024,2025])
data_current_students = data[mask]
data_current_students.isna().sum()

#dropping nulls from gender from campus (only 4)
data_current_students_final=data_current_students.dropna(inplace=False, axis=0)

#filtering for accepted students only
data_current_students_final = data_current_students_final[data_current_students_final["ACCEPTED"]==1]
data_current_students_final.shape

#renaming columns
data_current_students_final.rename(columns = {"IMP_EST_HOUSEHOLD_INCOME_RANK":"est_household_income_rank"},inplace=True)
data_current_students_final.rename(columns = {"IMP_LEADSOURCE_CODE":"leadsource_code"},inplace=True)
data_current_students_final.rename(columns = {"IMP_MINORITY":"minority"},inplace=True)

#normalizing data (used for feature selection but ultimately dropped)
data_current_students_final["miles_norm"] = (data_current_students_final["IMP_ZIP_MILES_FROM_CAMPUS"]-data_current_students_final["IMP_ZIP_MILES_FROM_CAMPUS"].min()) / (data_current_students_final["IMP_ZIP_MILES_FROM_CAMPUS"].max()-data_current_students_final["IMP_ZIP_MILES_FROM_CAMPUS"].min())
data_current_students_final["gpa_norm"] = (data_current_students_final["gpa_points"]-data_current_students_final["gpa_points"].min()) / (data_current_students_final["gpa_points"].max()-data_current_students_final["gpa_points"].min())
data_current_students_final["income_rank_norm"] = (data_current_students_final["est_household_income_rank"]-data_current_students_final["est_household_income_rank"].min()) / (data_current_students_final["est_household_income_rank"].max()-data_current_students_final["est_household_income_rank"].min())
data_current_students_final["zip_rank_norm"] = (data_current_students_final["IMP_ZIP_MEDIAN_INCOME_2"]-data_current_students_final["IMP_ZIP_MEDIAN_INCOME_2"].min()) / (data_current_students_final["IMP_ZIP_MEDIAN_INCOME_2"].max()-data_current_students_final["IMP_ZIP_MEDIAN_INCOME_2"].min())
