from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import set_config
from sklearn.tree import DecisionTreeClassifier
import random

random.seed(123)
rando = random.randint(0,100)

# Prepare data for scikit-learn
X = data_current_students_final.drop(columns=["gpa_norm","miles_norm","IMP_ZIP_MILES_FROM_CAMPUS","zip_rank_norm","gpa_norm","income_rank_norm","minority","ACCEPTED","ACTIVE_PAID","academic_year", "gpa_points", "IMP_ZIP_MEDIAN_INCOME_2","IMP_SCHOOL_TYPE","gender"])
y = data_current_students_final["ACTIVE_PAID"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rando, stratify = y)

# Split train into training and validation
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=.25, random_state=rando, stratify = y_train)

# Initialize and train the model
model = BernoulliNB()
model.fit(X_train, y_train)

##knn
knn_spec = KNeighborsClassifier(n_neighbors=10)

knn_fit = knn_spec.fit(X_train, y_train)
print(knn_fit.classes_)

#tree
# Create decision tree classifier
dt_clf = DecisionTreeClassifier()

# Train the model
dt_clf.fit(X_train, y_train)
