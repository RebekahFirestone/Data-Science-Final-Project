from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

#nb

# Make predictions
y_pred_v = model.predict(X_validate)

nb_predict_a = model.predict_proba(X_validate)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_validate, y_pred_v)
precision = precision_score(y_validate, y_pred_v, pos_label=1)
recall = recall_score(y_validate, y_pred_v, pos_label=1)
f1_nb = f1_score(y_validate, y_pred_v)


print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_nb}')
tn, fp, fn, tp = cm_nb.ravel()
specificity = tn / (tn + fp)
print(specificity)

#knn

knn_y_predict_v = knn_fit.predict(X_validate)

knn_predict_a = knn_spec.predict_proba(X_validate)[:, 1]

accuracy = accuracy_score(y_validate, knn_y_predict_v)
precision = precision_score(y_validate, knn_y_predict_v, pos_label=1)
recall = recall_score(y_validate, knn_y_predict_v, pos_label=1)
f1_knn = f1_score(y_validate, knn_y_predict_v)


print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_knn}')
tn, fp, fn, tp = cm_knn.ravel()
specificity = tn / (tn + fp)
print(specificity)

#tree

# Evaluate on test set
tree_y_predict_v = dt_clf.predict(X_validate)

tree_predict_a = dt_clf.predict_proba(X_validate)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_validate, tree_y_predict_v)
precision = precision_score(y_validate, tree_y_predict_v, pos_label=1)
recall = recall_score(y_validate, tree_y_predict_v, pos_label=1)
f1_tree = f1_score(y_validate, tree_y_predict_v)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_tree}')
tn, fp, fn, tp = cm_tree.ravel()
specificity = tn / (tn + fp)
print(specificity)
