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
