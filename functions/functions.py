import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def missing_data(data):
    """
    Calculates the total number and percentage of missing values in each column of the given DataFrame.
    
    Parameters:
    data (DataFrame): Input DataFrame containing the data.

    Returns:
    DataFrame: A DataFrame showing the total number and percentage of missing values for each column,
               sorted in descending order based on the percentage of missing values.
    """
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def get_column_description(col_desc, col):
    """
    Retrieves the description of the specified column(s) from the given column description DataFrame.

    Parameters:
    col_desc (DataFrame): DataFrame containing column descriptions with 'Row' and 'Description' columns.
    col (str or list): Name of the column(s) for which descriptions are to be retrieved.

    Returns:
    str: A concatenated string containing the descriptions of the specified column(s).
    """
    if type(col) != list:
        col = [col]
    all_cols_desc = ""
    for c in col:
        all_cols_desc += f"{c}: {col_desc[col_desc.Row == c].Description.values[0]}\n"
    return all_cols_desc

def create_dtypes_na_df(df):
    dtypes = df.dtypes
    dtypes_df = pd.DataFrame({"column": dtypes.index, "dtype": dtypes.values})

    na = df.isna().sum() / len(df)
    na_df = pd.DataFrame({"column": na.index, "prop_na": na.values})

    dtypes_na_df = dtypes_df.merge(na_df, on="column")

    dtypes_na_df["n_levels"] = [df[col].nunique() if dtypes_na_df[dtypes_na_df.column == col].dtype.values[0] == object else -1 for col in dtypes_na_df.column]

    return dtypes_na_df

def plot_cat_features(df, features):
    num_features = len(features)
    fig, axes = plt.subplots(nrows=num_features, ncols=2, figsize=(12, 6 * num_features))
    for i, feature in enumerate(features):
        plot_cat_feature_distribution(df, feature, axes[i, 0])
        plot_cat_feature_target_re(df, feature, axes[i, 1])
    plt.tight_layout()
    plt.show()

def plot_cat_feature_distribution(df, feature, ax=None):
    """
    Plot the distribution of a specified feature.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - feature (str): The name of the feature to plot.
    - ax (Axes, optional): The Axes object to plot on. If not provided, a new figure and Axes will be created.

    Returns:
    - None
    """
    if ax is None:
        if df[feature].nunique() > 8:
            if df[feature].nunique() > 30:
                figsize = (10, 14)
                horizontal = True
            else:
                figsize = (10, 8)
                horizontal = True
        else:
            figsize = (10, 6)
            horizontal = False
        fig, ax = plt.subplots(figsize=figsize)
    else:
        horizontal = False
    if horizontal:
        sns.countplot(y=feature, data=df, ax=ax, order=df[feature].value_counts().index)
        ax.set_xlabel('Count')
        ax.set_ylabel(f'{feature}')
    else:
        sns.countplot(x=feature, data=df, ax=ax, order=df[feature].value_counts().index)
        ax.set_ylabel('Count')
        ax.set_xlabel('')
    ax.set_title(f'Distribution of {feature}')

    if df[feature].nunique() > 3 and not horizontal:
        ax.tick_params(axis='x', rotation=90)

    total_count = df[feature].count()
    for p in ax.patches:
        width = p.get_width() if horizontal else p.get_height()
        percentage = width / total_count * 100
        if horizontal:
            ax.text(width + 2000, p.get_y() + p.get_height() / 2., f'{percentage:.2f}%', va="center")
        else:
            ax.text(p.get_x() + p.get_width() / 2., width + 2000, f'{percentage:.2f}%', ha="center", va="bottom", fontsize=10, color='black')

    if ax is None:
        plt.show()

def plot_cat_feature_target_re(df, feature, ax=None):
    """
    Plot the percentage of TARGET by a specified feature.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - feature (str): The name of the feature to plot.
    - ax (Axes, optional): The Axes object to plot on. If not provided, a new figure and Axes will be created.

    Returns:
    - None
    """
    if ax is None:
        if df[feature].nunique() > 8:
            if df[feature].nunique() > 8:
                figsize = (10, 20)
            else:
                figsize = (10, 16)
        else:
            figsize = (10, 4)
        fig, ax = plt.subplots(figsize=figsize)
    count_df = df.groupby([feature, 'TARGET']).size().reset_index(name='count')
    total_df = df.groupby(feature).size().reset_index(name='total')
    merged_df = pd.merge(count_df, total_df, on=feature)
    merged_df['percentage'] = (merged_df['count'] / merged_df['total']) * 100
    merged_df['TARGET'] = merged_df['TARGET'].astype(str)
    merged_df_sorted = merged_df.sort_values(by='percentage', ascending=False)

    horizontal = False
    if df[feature].nunique() > 8:
        horizontal = True

    if horizontal:
        sns.barplot(y=feature, x='percentage', hue='TARGET', data=merged_df_sorted, ax=ax)
        ax.set_xlabel('Percentage (%)')
        ax.set_ylabel(f'{feature}')
    else:
        sns.barplot(x=feature, y='percentage', hue='TARGET', data=merged_df, ax=ax,
                    order=df[feature].value_counts().index)
        ax.set_ylabel('Percentage (%)')

    ax.set_title(f'Percentage of TARGET by {feature}')

    if df[feature].nunique() > 3:
        if horizontal:
            ax.tick_params(axis='y', rotation=0)
            ax.legend(title='TARGET', loc='upper center', bbox_to_anchor=(0.85, -0.1), ncol=2)
        else:
            ax.tick_params(axis='x', rotation=90)
            ax.legend(title='TARGET', loc='upper center', bbox_to_anchor=(0.85, -0.45), ncol=2)
    else: 
        ax.legend(title='TARGET', loc='upper center', bbox_to_anchor=(0.85, -0.15), ncol=2)
    for p in ax.patches:
        width = p.get_width() if horizontal else p.get_height()
        if width > 0:
            if horizontal:
                ax.text(width + 1, p.get_y() + p.get_height() / 2., f'{width:.2f}%', va="center")
            else:
                ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 1, f'{width:.2f}%', ha="center")

    if ax is None:
        plt.show()

def chi_squared_test_for_features(df, features):
    """
    Perform chi-squared test for each variable in the list of features.

    Parameters:
    - df: DataFrame containing the data.
    - features: List of categorical variables to test.

    Returns:
    None (prints the test results for each feature).
    """
    for var in features:
        contingency_table = pd.crosstab(df[var], df['TARGET'])
        
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        alpha = 0.05

        print(f"Chi-squared test results for {var}:")
        print(f"Chi-squared statistic: {chi2_stat}")
        print(f"P-value: {p_value}")
        
        if p_value < alpha:
            print("Conclusion: There is a significant association between", var, "and TARGET.")
        else:
            print("Conclusion: There is no significant association between", var, "and TARGET.")
        print()

def plot_distribution_comparison_logx(dataframe, features, nrow=2):
    '''
    Plot distribution comparison for numerical features with log-scaled x-axis.

    Parameters:
    - dataframe (DataFrame): The DataFrame containing the data.
    - features (list): List of numerical features to plot.
    - nrow (int, optional): Number of rows for subplot layout. Defaults to 2.

    Returns:
    - None
    '''

    loan_num_features = [
    "AMT_CREDIT",
    "AMT_ANNUITY"
]
    i = 0
    t1 = dataframe.loc[dataframe['TARGET'] != 0, loan_num_features].copy()
    t0 = dataframe.loc[dataframe['TARGET'] == 0, loan_num_features].copy()

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow, 2, figsize=(12, 6*nrow))
    for feature in features:
        i += 1
        plt.subplot(nrow, 2, i)
        sns.kdeplot(t1[feature], bw_method=1, label="TARGET = 1")
        sns.kdeplot(t0[feature], bw_method=1, label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        plt.legend()
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xscale('log')
    plt.show()

def plot_boxplot_logx(data, feature):
    """
    Plot a boxplot of the specified numerical feature.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - feature (str): The name of the numerical feature to plot.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data[feature])
    plt.xlabel(feature)
    plt.title(f'Boxplot of {feature} (Log Scale)')
    plt.xscale('log')
    plt.show()

def classify_outliers(df, column, threshold=1.5):
    """
    Classify outliers in a DataFrame column based on the interquartile range (IQR) method.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column (str): The name of the column to classify outliers for.
    - threshold (float, optional): The threshold for determining outliers. Defaults to 1.5.

    Returns:
    - DataFrame: DataFrame containing only the rows classified as outliers based on the specified column.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
def plot_target_percentages(df_outliers, df_non_outliers, title):
    """
    Plot the percentage of default and non-default loans for outliers and non-outliers.

    Parameters:
    - df_outliers (DataFrame): DataFrame containing outlier loan data.
    - df_non_outliers (DataFrame): DataFrame containing non-outlier loan data.
    - title (str): Title of the plot.

    Returns:
    - None
    """
    outliers_default_percentage = (df_outliers['TARGET'].sum() / len(df_outliers)) * 100
    outliers_non_default_percentage = 100 - outliers_default_percentage

    non_outliers_default_percentage = (df_non_outliers['TARGET'].sum() / len(df_non_outliers)) * 100
    non_outliers_non_default_percentage = 100 - non_outliers_default_percentage

    labels = ['Outliers', 'Non-Outliers']
    default_percentages = [outliers_default_percentage, non_outliers_default_percentage]
    non_default_percentages = [outliers_non_default_percentage, non_outliers_non_default_percentage]

    x = range(len(labels))
    width = 0.4

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, non_default_percentages, width, bottom=default_percentages, label='Non-default')
    rects2 = ax.bar(x, default_percentages, width, label='Default')

    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()

def plot_AMT_comparison(dataframe, main_col, target_col, amount_col):
    """
    Plot density comparison based on target values using line graphs with KDE plots.

    Parameters:
    - dataframe (DataFrame): The DataFrame containing the data.
    - main_col (str): Column representing main feature.
    - target_col (str): Column representing target values.
    - amount_col (str): Column representing the amount.

    Returns:
    - None
    """
    dataframe = dataframe[dataframe[main_col] != 'XNA']
    if dataframe[main_col].nunique() > 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
    for cat in dataframe[main_col].unique():
        subset = dataframe[dataframe[main_col] == cat]
        sns.kdeplot(subset[subset[target_col] == 1][amount_col], bw_method=0.5, ax=axes[0], label=f'{cat} (Default)')
        sns.kdeplot(subset[subset[target_col] == 0][amount_col], bw_method=0.5, ax=axes[1], label=f'{cat} (Non-Default)')

    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    axes[0].set_xlabel(amount_col)
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Density Plot of Defaults by {amount_col}')
    axes[1].set_xlabel(amount_col)
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Density Plot of Non-Defaults by {amount_col}')

    if dataframe[main_col].nunique() > 2:
        axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=1)
        axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=1)
    else:
        axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    
    plt.tight_layout()
    plt.show()

def plot_density_comparison(dataframe, main_feature, feature_to_compare):
    """
    Plot KDE plot showing differences in feature_to_compare based on main_feature.

    Parameters:
    - dataframe (DataFrame): The DataFrame containing the data.
    - main_feature (str): The main feature used for comparison (e.g., gender).
    - feature_to_compare (str): The feature to compare across different main_feature groups.

    Returns:
    - None
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    dataframe = dataframe[dataframe[main_feature] != 'XNA']

    for category in dataframe[main_feature].unique():
        sns.kdeplot(data=dataframe[dataframe[main_feature] == category][feature_to_compare],
                    label=f'{category}', bw_method=0.5)

    plt.xlabel(feature_to_compare)
    plt.ylabel('Density')
    plt.title(f'{main_feature} Comparison for {feature_to_compare}')
    plt.legend()
    plt.xscale('log')

    plt.show()


def calculate_feature_stats(df, feature_col):
    """
    Calculate the percentage of default and non-default loans for each unique value of a feature,
    alongside the average income total and average credit amount.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - feature_col (str): Column representing the feature for which statistics are calculated.

    Returns:
    - DataFrame: DataFrame containing the percentage of default and non-default loans,
      average income total, and average credit amount for each unique value of the feature.
    """
    if feature_col in df.index.names:
        df.reset_index(inplace=True)

    feature_target_percentages = df.groupby(feature_col)['TARGET'].value_counts(normalize=True).unstack(level=-1).fillna(0) * 100
    
    feature_target_percentages.columns = ['Non-Default', 'Default']

    avg_values = df.groupby(feature_col)[['AMT_INCOME_TOTAL', 'AMT_CREDIT']].mean()

    feature_stats_df = feature_target_percentages.merge(avg_values, left_index=True, right_index=True).reset_index()

    feature_stats_df = feature_stats_df.sort_values(by='Default', ascending=False)

    return feature_stats_df

def format_with_commas(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, int):
        return '{:,.0f}'.format(x)
    elif isinstance(x, float):
        return '{:,.2f}'.format(x)
    else:
        return x


def plot_AMT_comparison_num(dataframe, target_col, num_col):
    """
    Plot density comparison based on target values using line graph with KDE plots for a numerical feature.

    Parameters:
    - dataframe (DataFrame): The DataFrame containing the data.
    - target_col (str): Column representing target values.
    - num_col (str): Column representing the numerical feature.

    Returns:
    - None
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    unique_targets = sorted(dataframe[target_col].unique())

    for target_value in unique_targets:
        sns.kdeplot(data=dataframe[dataframe[target_col] == target_value][num_col], bw_method=0.5, label=target_value)

    plt.xlabel(num_col)
    plt.ylabel('Density')
    plt.title(f'Density Plot of {target_col} by {num_col}')
    
    plt.legend()

    plt.show()
