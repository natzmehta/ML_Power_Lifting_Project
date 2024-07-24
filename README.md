# ML_Power_Lifting_Project

In recent years, the intersection of sports analytics and machine learning has opened new avenues for deriving insights and predictions from large datasets. The sport of powerlifting, with its rich history and diverse range of athletes, provides a unique opportunity to leverage machine learning techniques for predictive analytics.

Powerlifting is a strength sport that consists of three attempts at maximal weight on three lifts: squat, bench press, and deadlift. Olympic weightlifting involves the athlete attempting a maximal weight single-lift effort of a barbell loaded with weight plates. Powerlifting evolved from a sport known as "odd lifts," which followed the same three-attempt format but used a wider variety of events, akin to strongman competition. Eventually, odd lifts became standardized to the current three.

In competition, lifts may be performed equipped or un-equipped (typically referred to as 'classic' or 'raw' lifting in the IPF specifically). Equipment in this context refers to a supportive bench shirt or squat/deadlift suit or briefs. In some federations, knee wraps are permitted in the equipped but not un-equipped division; in others, they may be used in both equipped and un-equipped lifting. Weight belts, knee sleeves, wrist wraps, and special footwear may also be used but are not considered when distinguishing equipped from un-equipped lifting.

The motivation or problem definition for this project, therefore, can be summarized as the following:
- Performance analysis and athlete coaching: Evaluate an athlete's performance potential based on their stats. Identify areas of improvement, set realistic goals, and tailor training programs for individuals to enhance their overall performance.
- Equipment Optimization: Analyze which types of equipment contribute most significantly to better performance.
- Competition prediction and strategy: Help athletes devise strategies, such as selecting attempts or understanding their chances against competitors with similar or different profiles.

**Data Loading & Cleaning:**

The Powerlifting dataset contains details about competitions (meets) and participating athletes. The dataset was obtained from Kaggle in CSV format; the link for reference is https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database/data. It consists of 1048576 records and 37 attributes. The dataset includes categorical features such as sex, event, equipment, age class, etc. and numerical features like age, body weight, weight class, weights lifted in each of the three attempts in each powerlifting event - squat, deadlift and bench press as well as the total or maximum weight an athlete could lift in kilograms. Further, the “place” column indicates the ranking or position of the athlete in the competition. Information about the competition, such as the event date, location, and the name of the meet, is also available.


![image](https://github.com/user-attachments/assets/bdcb0fc2-a158-44ab-9430-058aefc7ab3e)

From the heatmap, we can clearly identify columns with the most null values, and therefore we drop ‘Squat4Kg', 'Bench4Kg', and 'Deadlift4Kg’ as a part of data cleaning. Further on this raw data, we extract the month and year values from the ‘Date’ column. Since straps and wraps are essentially the same equipment, we combine them. The 'strapswraps' function alters the input: if it matches 'Straps', it replaces it with 'Wraps'; otherwise, it returns the input unchanged.
Further, we try to remove duplicates, but there doesn't seem to be a good column in the openpowerlifting.csv dataset to eliminate duplicate data. For example, the most unique column, ‘Name’, has nearly 200 instances of 'Sverre Paulsen'. And that is not to say that we are talking about the same single person, though we probably are. Therefore, we utilize drop_duplicates to remove duplicate records from the data frame 'openpl', ensuring that unique instances remain within the dataset.
At this point, we also start to analyze the possibility of grouping the data by column values to process the data further. The 'divisions' column typically signifies the categories or classes within powerlifting competitions, such as age groups, weight classes, or experience levels. Grouping by 'divisions' allows insights into performance trends across these categories.

However, the data contains 4246 unique divisions. Given the extensive range of unique divisions within the dataset, it might be more effective to organize the data by alternative criteria rather than relying solely on the 'Divisions' column for grouping.

![image](https://github.com/user-attachments/assets/677ef43d-3654-4182-8c77-404e5b3e85d3)

The visualization indicates a notable surge in the popularity of weightlifting as a sport over the years. However, a closer examination of the participant counts by year highlights incomplete data for 2019, prompting us to omit all records from that year. Notably, 2010 and 2014 stand out with significant increases in participant numbers, a trend we aim to scrutinize further in our exploratory data analysis section.

**Data Processing**

After identifying missing values in the dataset and dealing with duplicate records, we closely understand the data types for each column and decide on strategies to address the missing data. An integral column like ‘BodyweightKg’ had many missing values. Consequently, we impute the missing body weight with the participating weight class. While direct body weight measurements might be missing, weight class categorizations often closely correspond to specific weight ranges. Imputing missing values with the weight class retains some pertinent information regarding the participants' weights.

Outliers become apparent upon visual examination using boxplots and histograms for the columns 'Age', 'BodyweightKg', and 'TotalKg'. Observations above the upper whisker in the boxplot indicate potential outliers within the 'Age' column. This suspicion is further supported by right-skewed histograms for both the 'BodyweightKg' and 'TotalKg' columns. The distribution's elongation towards higher values notably indicates the presence of outliers.

![image](https://github.com/user-attachments/assets/e19b402e-db18-4ce9-8e1d-23f024a24478)

![image](https://github.com/user-attachments/assets/5a206a97-4fc5-4bd7-91ac-965166609045)

![image](https://github.com/user-attachments/assets/55a76e5c-822d-4e7b-a5d9-9208c0cd456f)

The function 'remove_outliers' calculates quartiles and the Interquartile Range (IQR) for specified columns, establishing upper and lower limits based on 1.5 times the IQR from the median. It filters data points falling outside these limits for outlier removal. We also observe negative lifts in some columns of the squat, deadlift, or bench press events. We convert these values to absolute and recalculate each event's best of three attempts columns.
In the final leg of pre-processing the data, we perform one hot encoding to convert categorical data in the columns ‘Sex’, ‘Equipment’ and ‘Age Category’ into the numeric format. This is done so that the machine learning models, especially those based on mathematical equations, can process them effectively. One-hot encoding converts categorical variables into a binary format, creating new binary columns (often called dummy variables). Each unique category in the original column becomes a new column. For each observation, only one of these columns will be 1 (indicating the presence of that category), while others will be 0.

Finally, we use MinMaxScaler from sklearn to scale specific columns ('BodyweightKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg'). Scaling ensures these numerical features are within a consistent range (0 to 1), aiding in model convergence and reducing the impact of differing feature magnitudes on algorithms.

**EDA**

We plot a bar plot showcasing the distribution of male and female athletes in the 'openpl' dataset. It visualizes the count of male and female lifters, indicating the predominance of male participants. The percentage calculation reveals approximately 75 per cent of the dataset comprises male lifters, while the 'Sex' column's value counts display the raw count for each gender category.

![image](https://github.com/user-attachments/assets/79597743-8954-4716-ab9c-d78dafdb396f)

![image](https://github.com/user-attachments/assets/ec7628f6-b452-4642-96aa-ab868b534b2f)

From the plot we can see that most athletes prefer single ply followed by raw which means they chose to perform the lift unequipped.

![image](https://github.com/user-attachments/assets/fa84a520-ca2a-4f5e-8f12-0195c4c1a006)

Each bar represents the frequency of body weights within specific ranges (bins=50) for both genders, showcasing the spread and overlap of body weight distributions. From the histogram we can conclude that male lifters are heavier than female lifters. There is an obvious anatomical explanation to this which often results in the average male lifters being able to lift heavier in kilograms compared to a female athlete.

In our data preprocessing, we encountered challenges with the 'Division' column due to its extensive unique values. Realizing the importance of a better grouping approach, we addressed missing 'Age' values by estimating them from 'AgeClass'. By categorizing ages into ranges using the 'age_calc' function, such as '5-10', '10-20', we created a new 'ageCategory' column. This categorization simplifies data segmentation, aiding analysis by grouping athletes based on age ranges providing clearer insights.

We plot the distribution of athletes that secured first place overall by age (as seen in fig. 4.4). Most winning athletes belonged to the age group of 20 -30 for both males and females.

![image](https://github.com/user-attachments/assets/b98193b2-c2e4-4c42-bf97-0bdd4c66b1ed)

In our final analysis, we assess absolute strength concerning body weight and equipment types to discern the effectiveness of specific equipment in powerlifting. By examining the relationship between equipment used and absolute strength, we aim to determine if using equipment provides a competitive advantage in lifts. This exploration aids in devising tailored training programs for athletes, leveraging insights from their historical preferences and performance to optimize future training strategies.

![image](https://github.com/user-attachments/assets/39bbe878-32a2-4194-a044-f571b61664ce)

**Models and Performance Evaluation:**

As mentioned in our problem definition, we want to estimate the maximum weight ie. ‘TotalKg’ a powerlifting athlete can lift based on dependent variables like age, sex, best bench lift weight, best dead lift weight and type of equipment used. Therefore, this is a regression problem i.e. predicting a continuous numerical value based on input features. We will model the relationship between independent variables (features) and a dependent variable (target) to make predictions or understand the underlying pattern in the data.

A. Linear Regression: The model is used to predict the TotalKg lifted in powerlifting. It is tuned via GridSearchCV. The potential hyperparameters considered are 'fit_intercept' and 'normalize'. GridSearchCV explores different combinations within the specified parameter grid, evaluating them using 5-fold cross-validation to find the best model based on 'neg_mean_squared_error'. Evaluation is based on several metrics including mean absolute error, mean squared error, and R-squared to assess the model's performance. Best parameters: {'fit_intercept': True, 'normalize': False}

Performance Metrics:
• Mean Absolute Error (MAE): 0.0275
• Mean Squared Error (MSE): 0.0023
• Root Mean Squared Error (RMSE): 0.0476
• R-squared (R2): 0.9456

B. Random Forest Regressor: The model’s algorithm works by constructing multiple decision trees during training and outputs the average prediction (regression) of individual trees for regression tasks. Each tree is trained on a random subset of the data and a random subset of features, reducing overfitting, and improving generalization compared to a single decision tree.

Performance Metrics:
• Mean Absolute Error (MAE): 0.0284
• Mean Squared Error (MSE): 0.0022
• Root Mean Squared Error (RMSE): 0.0473
• R-squared (R2): 0.9461

C.
XG Boost Regressor: The algorithm of this boosting ensemble method employed works by iteratively training weak learners (decision trees) and optimizing subsequent trees to correct errors made by preceding models. It focuses on minimizing the overall prediction error using gradient descent, using a weighted sum of multiple decision trees to generate the final prediction. GridSearchCV is applied to explore the best combination of hyperparameters such as 'n_estimators', 'learning_rate', 'max_depth', 'subsample', and 'colsample_bytree' through cross-validation, aiming to minimize 'neg_mean_squared_error'. The process identifies the best hyperparameters and the resulting best model based on the specified metric, using it to predict 'TotalKg' on the test set. Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 1.0} 

Performance Metrics:
• Mean Absolute Error (MAE): 0.0268
• Mean Squared Error (MSE): 0.0021
• Root Mean Squared Error (RMSE): 0.0455
• R-squared (R2): 0.9502

D. KNN Regressor: K Neighbors Regressor predicts the target variable by averaging the 'k' nearest neighbors' target values. The model identifies the 'k' nearest data points based on specified distance metrics (Manhattan or Euclidean distance) and computes the average or weighted average of their target values for predictions. For hyperparameter tuning, GridSearchCV is employed to explore the best combinations of 'n_neighbors' (number of neighbors), 'weights' (uniform or distance-based weighting), and 'p' (distance metric). The GridSearchCV process identifies the best hyperparameters based on 'neg_mean_squared_error' using cross-validation. The resulting best model is used to predict 'TotalKg' on the test set, and the code outputs the best parameters and the best model chosen.
Best Parameters: {'n_neighbors': 10, 'p': 2, 'weights': 'distance'}
Best Model: KNeighborsRegressor (n_neighbors=10, weights='distance') 

Performance Metrics:
• Mean Absolute Error (MAE): 0.0273
• Mean Squared Error (MSE): 0.0022
• Root Mean Squared Error (RMSE): 0.0473
• R-squared (R2): 0.9462

E. Adaboost Regressor: This regression model builds a strong model by sequentially adding weak models to correct errors made by previous models. AdaBoost iteratively learns from the mistakes of preceding models by assigning higher weights to misclassified data, thereby focusing on more challenging instances. GridSearchCV explores optimal combinations of 'n_estimators' (number of estimators) and 'learning_rate' (contribution of each model to the ensemble) to minimize 'neg_mean_squared_error' through cross-validation. For Best Model Identification, the GridSearchCV process identifies the best hyperparameters, and the resulting best model is used to predict 'TotalKg' on the test set. The code displays the best parameters and the best model chosen based on the specified metric.
Best Parameters: {'learning_rate': 0.1, 'n_estimators': 50}
Best Model: AdaBoostRegressor(learning_rate=0.1)

Performance Metrics:
• Mean Absolute Error (MAE): 0.0408
• Mean Squared Error (MSE): 0.0034
• Root Mean Squared Error (RMSE): 0.0583
• R-squared (R2): 0.9184

**Conclusion:**

![image](https://github.com/user-attachments/assets/061518c7-b7af-40a4-af90-487988fff99e)

• Across all models, XG Boost Regressor consistently demonstrates lower error metrics (MAE, MSE, and RMSE)

• XG Boost Regressor has the highest R-squared values indicating better predictive accuracy and goodness of fit

• Different equipment types significantly influenced lift performance as well as maximum weight lifted by an athlete. Athletes that used single ply excelled while athletes that used wraps in their lifts tended to perform lower.

• Athletes in their early career stages i.e. 20s or 30s tended to achieve higher rankings and better performance due to peak physical condition and training.

• Features like body weight, equipment type and age showed notable importance in predicting TotalKg indicating strong influence on an athlete powerlifting performance

