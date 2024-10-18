import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import category_encoders as ce
import sweetviz as sv

# Load the dataset
# Assuming 'df' is your loaded dataframe
# df = pd.read_csv('your_dataset.csv')  # Load your dataset accordingly

# Step 1: Analyze the dataset using SweetViz
# Generate the SweetViz report
report = sv.analyze(df)
report.show_html("sweetviz_report.html")  # This will generate a detailed report

# You can inspect the SweetViz report for more details.
# SweetViz will show you the distribution, missing values, and data types.

# Step 2: Based on analysis, we preprocess the data.

# Split categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Step 3: Handling Missing Values based on distribution analysis

# SweetViz will give you distributions of columns.
# If the distribution is normal-like (bell curve), use mean.
# If skewed or has outliers, use median. For categorical, use mode.

# 3.1 Numerical Features
# Analyze distributions using SweetViz. If the column is skewed or has outliers, use median; otherwise, use mean.

def get_imputation_strategy(column):
    # Based on SweetViz, determine whether to use 'mean' or 'median'
    if df[column].skew() > 1 or df[column].skew() < -1:  # Detecting skewness for choosing median
        return 'median'
    else:
        return 'mean'

# Apply the strategy to all numerical columns
for col in numerical_cols:
    strategy = get_imputation_strategy(col)
    imputer = SimpleImputer(strategy=strategy)
    df[col] = imputer.fit_transform(df[[col]])

# 3.2 Categorical Features
# Use mode (most frequent) for categorical columns
mode_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = mode_imputer.fit_transform(df[categorical_cols])

# Step 4: Encoding Categorical Features
# If categorical features have many categories, use Target Encoding; otherwise, use Label Encoding

for col in categorical_cols:
    if df[col].nunique() > 50:  # Arbitrary threshold for deciding when to use Target Encoding
        target_encoder = ce.TargetEncoder(cols=[col])
        df[col] = target_encoder.fit_transform(df[col], df['resp_tag'])  # Assuming 'resp_tag' is the target variable
    else:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

# Step 5: Scaling Numerical Features

# Use StandardScaler to normalize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Now your data is preprocessed and ready for modeling.


//     CRS Report Management: Up until August, I was responsible for managing 11 (CRS) reports. During this time, I developed a solid understanding of the reporting processes, including technologies such as SAS code, process management, and execution. I became proficient enough to provide knowledge transfer (KT) to Abhijit.
// Code Version Control : My team and I encountered significant challenges with code management, which led to delays and conflicts during code merging and the development of new features. I conducted extensive research to identify alternative solutions available within the organization and tested several options. Ultimately, I developed a solution using Git and OneDrive, which is currently undergoing testing to ensure it effectively addresses our code management issues.
// API Testing:  With the development of V2, which introduced a client-server architecture, new features (KEYS) were implemented in the backend, requiring thorough testing. Initially, testing was done primarily through the frontend and Python packages, but as the complexity increased, a dedicated API testing tool became essential. After researching available tools within the organization and facing limitations with CMPs, I explored several alternatives. I selected ThunderClient after conducting thorough research and testing because it integrates seamlessly as a Visual Studio Code extension, making it the best fit for our workflow. Unlike other tools, which we couldn't install on our system, ThunderClient was ideal for our needs. It has greatly simplified the process of testing new API keys, resolving issues with key names and structures, and proving to be an effective solution to the challenges we encountered.

// DATA for Convo BI : Collaborated with Asmanjas on preparing CRS data for Convo BI. The tasks involved end-to-end data handling, starting with data extraction using SAS, followed by transferring the dataset to the edge node. Additionally, I optimized the process by compressing the data into ZIP files and segmenting it into manageable chunks for efficient processing and transfer.

// Extra Activity : volunteered in the event organised on the floor for USBP, and participated in Modelling hackathon.



    //////////
    CRS Report Management: Up until August, I was responsible for managing 11 (CRS) reports. During this time, I developed a solid understanding of the reporting processes, including technologies such as SAS code, process management, and execution. I became proficient enough to provide knowledge transfer (KT) to Abhijit.
Code Version Control : My team and I encountered significant challenges with code management, resulting in delays and conflicts during code merging and new feature development. I took the initiative to conduct extensive research into alternative solutions available within the organization and tested several options. Ultimately, I implemented a solution using Git and OneDrive, which is currently undergoing testing to ensure it effectively resolves our code management issues. By leveraging open-source tools, we were able to achieve this solution without incurring additional costs, saving resources for the organization while enhancing our development process.
API Testing:  With the development of V2, which introduced a client-server architecture, new features (KEYS) were implemented in the backend, requiring thorough testing. Initially, testing was conducted primarily through the frontend and Python packages, but as the complexity increased, a dedicated API testing tool became necessary. After researching available tools within the organization and encountering limitations with CMPs, I explored several open-source alternatives. I selected ThunderClient after conducting thorough research and testing, as it integrates seamlessly as a Visual Studio Code extension and was the best fit for our workflow. Unlike other tools, which we couldn't install on our system, ThunderClient offered a cost-effective, open-source solution, saving the organization resources while simplifying the process of testing new API keys. It has effectively resolved issues with key names and structures, addressing the challenges we faced..

DATA for Convo BI : Collaborated with Asmanjas on preparing CRS data for Convo BI. The tasks involved end-to-end data handling, starting with data extraction using SAS, followed by transferring the dataset to the edge node. Additionally, I optimized the process by compressing the data into ZIP files and segmenting it into manageable chunks for efficient processing and transfer. 

///////
    Extra Activity : volunteered in the event organised on the floor for USBP, and participated in Modelling hackathon.

Taking to New people who are working on same niche: Mostly Tech people ,how they work on POC’s do they get a cloud to host the POC’s to they do it on system, specially Gen AI , how they are building the products like, they hve open source hugging face models or call the   




    import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import category_encoders as ce

# Exclude columns that should not be preprocessed
exclude_cols = ['cust_no', 'HH_NO', 'resp_tag']  # Add other unique identifier columns as needed

# Split the columns into categorical and numerical
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(exclude_cols)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols)

# Step 1: Handling Missing Values
# Handle missing values for numerical and categorical columns separately

# 1.1 Numerical Features
# If skewed, use median; otherwise, use mean
def get_imputation_strategy(column):
    if df[column].skew() > 1 or df[column].skew() < -1:
        return 'median'
    else:
        return 'mean'

# Apply imputation dynamically for each numerical column
for col in numerical_cols:
    strategy = get_imputation_strategy(col)
    imputer = SimpleImputer(strategy=strategy)
    df[col] = imputer.fit_transform(df[[col]])

# 1.2 Categorical Features
# Use mode (most frequent) to fill missing values in categorical columns
mode_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = mode_imputer.fit_transform(df[categorical_cols])

# Step 2: Encoding Categorical Features
# Label Encoding for low-cardinality columns, Target Encoding for high-cardinality

for col in categorical_cols:
    if df[col].nunique() > 50:  # High cardinality
        target_encoder = ce.TargetEncoder(cols=[col])
        df[col] = target_encoder.fit_transform(df[col], df['resp_tag'])
    else:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

# Step 3: Scaling Numerical Features
# Standardize the numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Now the dataset is preprocessed, excluding 'cust_no', 'HH_NO', and 'resp_tag'

