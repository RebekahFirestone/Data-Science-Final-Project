import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay 

#featuere selection visualizations

sns.countplot(data = data_current_students_final, x="ACTIVE_PAID", palette=["#522D72"])

sns.countplot(data=data_current_students_final, x='legacy', hue='ACTIVE_PAID')
plt.show()

df_legacy = data_current_students_final.groupby('legacy')['ACTIVE_PAID'].value_counts(normalize=True).unstack() * 100
df_legacy.plot(kind='bar', stacked=True, figsize=(5, 3), color = ["#522D72","#B2B4BE"])

sns.scatterplot(data=data_current_students_final, x='gpa_points', y='IMP_ZIP_MILES_FROM_CAMPUS', hue='ACTIVE_PAID')
plt.show()

sns.scatterplot(data=data_current_students_final, y='est_household_income_rank', x='IMP_ZIP_MILES_FROM_CAMPUS', hue='ACTIVE_PAID')
plt.show()

sns.countplot(data=data_current_students_final, x='campus_visit_date_exists', hue='ACTIVE_PAID')
plt.show()

df_cv = data_current_students_final.groupby('campus_visit_date_exists')['ACTIVE_PAID'].value_counts(normalize=True).unstack() * 100
df_cv.plot(kind='bar', stacked=True, figsize=(5, 3), color = ["#522D72","#B2B4BE"])

sns.countplot(data=data_current_students_final, x='gpa_norm', hue='ACTIVE_PAID')
plt.show()

df_gpa = data_current_students_final.groupby('gpa_norm')['ACTIVE_PAID'].value_counts(normalize=True).unstack() * 100
df_gpa.plot(kind='bar', stacked=True, figsize=(5, 3))

sns.countplot(data=data_current_students_final, x='minority', hue='ACTIVE_PAID')
plt.show()

df_minority = data_current_students_final.groupby('minority')['ACTIVE_PAID'].value_counts(normalize=True).unstack() * 100
df_minority.plot(kind='bar', stacked=True, figsize=(5, 3), color = ["#522D72","#B2B4BE"])

sns.countplot(data=data_current_students_final, x='est_household_income_rank', hue='ACTIVE_PAID')
plt.show()

df_hir = data_current_students_final.groupby('est_household_income_rank')['ACTIVE_PAID'].value_counts(normalize=True).unstack() * 100
df_hir.plot(kind='bar', stacked=True, figsize=(5, 3))

sns.countplot(data=data_current_students_final, x='gender', hue='ACTIVE_PAID')
plt.show()

df_hir = data_current_students_final.groupby('gender')['ACTIVE_PAID'].value_counts(normalize=True).unstack() * 100
df_hir.plot(kind='bar', stacked=True, figsize=(5, 3), color = ["#522D72","#B2B4BE"])


df = data_current_students_final.groupby(["leadsource_code"]).size().reset_index()
remove_list = ["IAP","IEM","IMP","INC","INF","INV","IOE","IRS","IYC"]
dff = data_current_students_final[~data_current_students_final["leadsource_code"].isin(remove_list)]
sns.countplot(data=dff, x='leadsource_code', hue='ACTIVE_PAID')
plt.show()
df_hir = dff.groupby('leadsource_code')['ACTIVE_PAID'].value_counts(normalize=True).unstack() * 100
df_hir.plot(kind='bar', stacked=True, figsize=(8, 3), color = ["#522D72","#B2B4BE"])

#confusion matrices
var = sns.color_palette("blend:#FFFFFF,#522D72",as_cmap=True)
cm_nb=confusion_matrix(y_validate, y_pred_v)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap=var, cbar=True)
plt.title('Confusion Matrix - NB')
plt.xlabel('Predicted')
plt.ylabel('Actual')

var = sns.color_palette("blend:#FFFFFF,#522D72",as_cmap=True)
cm_knn=confusion_matrix(y_validate, knn_y_predict_v)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap=var, cbar=True)
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')

var = sns.color_palette("blend:#FFFFFF,#522D72",as_cmap=True)
cm_tree=confusion_matrix(y_validate, tree_y_predict_v)
sns.heatmap(cm_tree, annot=True, fmt='d', cmap = var, cbar=True)
plt.title('Confusion Matrix - Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual') 

#ROC curve stuff
roc_curve1 = RocCurveDisplay.from_predictions(y_validate, tree_predict_a, name="Tree", color="#00AFCE")
roc_curve2 = RocCurveDisplay.from_predictions(y_validate, knn_predict_a, ax=roc_curve1.ax_, name="KNN", color="#522D72")
roc_curve3 = RocCurveDisplay.from_predictions(y_validate, nb_predict_a, ax=roc_curve1.ax_, name="NB", color="#E14F3D")

plt.show()
