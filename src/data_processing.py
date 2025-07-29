train_df = pd.read_csv('/content/train (1) (2).csv')
test_df = pd.read_csv('/content/test (1) (2).csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Missing values in train:")
print(train_df.isnull().sum())

property_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train_df_clean = train_df.dropna(subset=property_columns, how='all').copy()
print(f"Clean train shape after removing all-NaN rows: {train_df_clean.shape}")

knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
train_df_clean[property_columns] = knn_imputer.fit_transform(train_df_clean[property_columns])

print(f"Missing values after KNN imputation:")
print(train_df_clean[property_columns].isnull().sum())

iterative_imputer = IterativeImputer(random_state=42, max_iter=10)
train_df_clean[property_columns] = iterative_imputer.fit_transform(train_df_clean[property_columns])

print(f"Missing values after Iterative imputation:")
print(train_df_clean[property_columns].isnull().sum())
