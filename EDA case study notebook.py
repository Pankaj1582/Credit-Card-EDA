#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max.rows',130)
pd.set_option('display.max.columns',130)
pd.set_option('float_format', '{:.2f}'.format)


# In[27]:


df = pd.read_csv("C:\\Users\\hp\\Downloads\\application_data.csv")
df1 = pd.read_csv("C:\\Users\\hp\\Downloads\\previous_application.csv")


# In[5]:


df.head()


# In[8]:


###check data structure
df.info(verbose = True,null_counts = True)


# In[9]:


##We don't see any columns with nullable values


# In[10]:


df.shape


# In[11]:


df.describe()


# In[14]:


# Analyzing catagorical variables


# In[15]:


df.select_dtypes(include = "object").columns


# In[16]:


# Checking number of categorical variables
len(df.select_dtypes(include = "object").columns)


# In[17]:


#Dealing with incorrect data type


# In[18]:


df.dtypes


# In[19]:


df.head()


# In[20]:


# Looking at the data and its corresponding data types, we can conclude that no data type changes are required.


# In[21]:


#### missing values


# In[22]:


df.isnull().values.any()


# In[23]:


df.isnull().values.sum()


# In[24]:


df.columns[df.isnull().any()]


# In[72]:


len(df.columns[df.isnull().any()])


# In[73]:


## there are 67 columns having one or more null values


# In[74]:


pd.set_option("display.max_rows", None, "display.max_columns", None) #TO DISPLAY ALL COLUMNS
(df.isnull().sum()/len(df.index))*100


# In[75]:


###Remove columns with > 50% missing data


# In[76]:


columnsToDelete = ["OWN_CAR_AGE", "EXT_SOURCE_1", "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE"]


# In[77]:


df.drop(columnsToDelete, axis = 1, inplace = True)


# In[78]:


df.shape


# In[79]:


#Now to verify that our targeted columns are removed
(df.isnull().sum()/len(df.index))*100


# In[80]:


## Dealing with missing values


# In[81]:


###COLUMN: [AMT_GOODS_PRICE]


# In[82]:


df["AMT_GOODS_PRICE"].describe()


# In[83]:


plt.hist(df["AMT_GOODS_PRICE"])
plt.show()


# In[84]:


##There is a clear difference between the min and max values, so we will use the median to impute missing values since the mean will skew the data


# In[85]:


df["AMT_GOODS_PRICE"].isnull().sum()


# In[86]:


df["AMT_GOODS_PRICE"].median()


# In[87]:


df["AMT_GOODS_PRICE"] = df["AMT_GOODS_PRICE"].fillna(df["AMT_GOODS_PRICE"].median())


# In[88]:


## Now verify that all the null values are replaced
df["AMT_GOODS_PRICE"].isnull().sum()


# In[89]:


###COLUMN: [NAME_TYPE_SUITE]


# In[90]:


df["NAME_TYPE_SUITE"].isnull().sum()


# In[91]:


df["NAME_TYPE_SUITE"].isnull()


# In[92]:


df["NAME_TYPE_SUITE"].describe()


# In[93]:


df["NAME_TYPE_SUITE"].value_counts()


# In[94]:


df[df["NAME_TYPE_SUITE"].isnull()]


# In[95]:


df["NAME_TYPE_SUITE"] = df["NAME_TYPE_SUITE"].fillna("Unaccompanied")


# In[96]:


pd.set_option("display.max_rows", 100, "display.max_columns", 100)
df.isnull().sum()


# In[97]:


##COLUMN: [AMT_ANNUITY]


# In[98]:


df[df["AMT_ANNUITY"].isnull()]


# In[99]:


plt.hist(df["AMT_ANNUITY"])
plt.show()


# In[100]:


## Assinging the median to null values
df["AMT_ANNUITY"].median()


# In[101]:


df["AMT_ANNUITY"] = df["AMT_ANNUITY"].fillna(0)


# In[103]:


df.AMT_ANNUITY.isnull().sum()


# In[104]:


###COLUMN: [OCCUPATION_TYPE]


# In[105]:


df["OCCUPATION_TYPE"].describe()


# In[106]:


df["OCCUPATION_TYPE"].head()


# In[107]:


df["OCCUPATION_TYPE"].value_counts()


# In[108]:


df["OCCUPATION_TYPE"].notna()


# In[109]:


df["OCCUPATION_TYPE"].value_counts()


# In[110]:


## now we remove the missing values
df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"][~df["OCCUPATION_TYPE"].isnull()]


# In[111]:


df.dropna(subset=['OCCUPATION_TYPE'], inplace = True)


# In[112]:


df["OCCUPATION_TYPE"].isnull().sum()


# In[113]:


df.isnull().sum()


# In[114]:


###COLUMN: [CODE_GENDER]


# In[115]:


df["CODE_GENDER"].describe()


# In[116]:


df["CODE_GENDER"].value_counts()


# In[117]:


df["CODE_GENDER"].mode()


# In[119]:


# Add XNA values to the schema, since XNA values are negligible, adding them to the schema will not affect the analysis
df["CODE_GENDER"].replace("XNA", "F", inplace = True)


# In[120]:


### Now we will drop unwanted columns


# In[121]:


df.drop([ "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "FLAG_EMAIL", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "DAYS_LAST_PHONE_CHANGE","FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21", "EXT_SOURCE_2", "EXT_SOURCE_3", "YEARS_BEGINEXPLUATATION_AVG", "FLOORSMAX_AVG", "YEARS_BEGINEXPLUATATION_MODE", "FLOORSMAX_MODE", "YEARS_BEGINEXPLUATATION_MEDI", "FLOORSMAX_MEDI", "TOTALAREA_MODE", "EMERGENCYSTATE_MODE"], axis=1, inplace = True)


# In[122]:


df.isnull().sum()


# In[123]:


(df.isnull().sum()/len(df.index))*100


# In[124]:


###COLUMN: [DEF_60_CNT_SOCIAL_CIRCLE]


# In[125]:


df["DEF_60_CNT_SOCIAL_CIRCLE"].describe()


# In[126]:


df.boxplot("DEF_60_CNT_SOCIAL_CIRCLE")
plt.show()


# In[127]:


### Now we will use median to replace missing values
df["DEF_60_CNT_SOCIAL_CIRCLE"] = df["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(df["DEF_60_CNT_SOCIAL_CIRCLE"].median())


# In[128]:


df.boxplot("DEF_60_CNT_SOCIAL_CIRCLE")
plt.show()


# In[129]:


### COLUMN: [AMT_REQ_CREDIT_BUREAU_HOUR]


# In[130]:


df["AMT_REQ_CREDIT_BUREAU_HOUR"].describe()


# In[134]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_HOUR")
plt.show()


# In[135]:


#filling missing values with mean as no outliers are seen

df["AMT_REQ_CREDIT_BUREAU_HOUR"] = df["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(df["AMT_REQ_CREDIT_BUREAU_HOUR"].mean())


# In[136]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_HOUR")
plt.show()


# In[137]:


### COLUMN: [AMT_REQ_CREDIT_BUREAU_DAY]


# In[138]:


df["AMT_REQ_CREDIT_BUREAU_DAY"].describe()


# In[139]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_DAY")
plt.show()


# In[140]:


## Again we will use mean to handle missing value


# In[141]:


df["AMT_REQ_CREDIT_BUREAU_DAY"] = df["AMT_REQ_CREDIT_BUREAU_DAY"].fillna(df["AMT_REQ_CREDIT_BUREAU_DAY"].mean())


# In[142]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_DAY")
plt.show()


# In[143]:


### COLUMN: [AMT_REQ_CREDIT_BUREAU_WEEK]


# In[144]:


df["AMT_REQ_CREDIT_BUREAU_WEEK"].describe()


# In[145]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_WEEK")
plt.show()


# In[146]:


df["AMT_REQ_CREDIT_BUREAU_WEEK"] = df["AMT_REQ_CREDIT_BUREAU_WEEK"].fillna(df["AMT_REQ_CREDIT_BUREAU_WEEK"].mean())


# In[147]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_WEEK")
plt.show()


# In[148]:


### COLUMN: [AMT_REQ_CREDIT_BUREAU_MON]


# In[149]:


df["AMT_REQ_CREDIT_BUREAU_MON"].describe()


# In[150]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_MON")
plt.show()


# In[151]:


df["AMT_REQ_CREDIT_BUREAU_MON"] = df["AMT_REQ_CREDIT_BUREAU_MON"].fillna(df["AMT_REQ_CREDIT_BUREAU_MON"].mean())


# In[152]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_MON")
plt.show()


# In[153]:


### COLUMN: [AMT_REQ_CREDIT_BUREAU_QRT]


# In[154]:


df["AMT_REQ_CREDIT_BUREAU_QRT"].describe()


# In[155]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_QRT")
plt.show()


# In[156]:


### Outliers are detected but no removal is required according to the problem statement


# In[157]:


### To deal with missing values
df["AMT_REQ_CREDIT_BUREAU_QRT"] = df["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(df["AMT_REQ_CREDIT_BUREAU_QRT"].median())


# In[158]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_QRT")
plt.show()


# In[159]:


### COLUMN: [AMT_REQ_CREDIT_BUREAU_YEAR]


# In[160]:


df["AMT_REQ_CREDIT_BUREAU_YEAR"].describe()


# In[161]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_YEAR")
plt.show()


# In[162]:


df["AMT_REQ_CREDIT_BUREAU_YEAR"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(df["AMT_REQ_CREDIT_BUREAU_YEAR"].mean())


# In[163]:


df.boxplot("AMT_REQ_CREDIT_BUREAU_YEAR")
plt.show()


# In[166]:


df["AMT_REQ_CREDIT_BUREAU_YEAR"].describe()


# In[167]:


df["AMT_REQ_CREDIT_BUREAU_YEAR"].isnull().sum()


# In[169]:


df.isnull().sum()


# In[170]:


### COLUMN: [OBS_30_CNT_SOCIAL_CIRCLE]


# In[171]:


df.OBS_30_CNT_SOCIAL_CIRCLE.describe()


# In[172]:


df.OBS_30_CNT_SOCIAL_CIRCLE = df["OBS_30_CNT_SOCIAL_CIRCLE"].fillna(df.OBS_30_CNT_SOCIAL_CIRCLE.mean())


# In[174]:


plt.boxplot(df.OBS_30_CNT_SOCIAL_CIRCLE)
plt.show()


# In[175]:


### COLUMN: [DEF_30_CNT_SOCIAL_CIRCLE]


# In[176]:


df["DEF_30_CNT_SOCIAL_CIRCLE"].describe()


# In[177]:


df.boxplot("DEF_30_CNT_SOCIAL_CIRCLE")
plt.show()


# In[178]:


df["DEF_30_CNT_SOCIAL_CIRCLE"] = df["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(df["DEF_30_CNT_SOCIAL_CIRCLE"].mean())


# In[180]:


df.boxplot("DEF_30_CNT_SOCIAL_CIRCLE")
plt.show()


# In[181]:


### COLUMN: [OBS_60_CNT_SOCIAL_CIRCLE]


# In[182]:


df["OBS_60_CNT_SOCIAL_CIRCLE"].describe()


# In[183]:


df.boxplot("OBS_60_CNT_SOCIAL_CIRCLE")
plt.show()


# In[184]:


df["OBS_60_CNT_SOCIAL_CIRCLE"] = df["OBS_60_CNT_SOCIAL_CIRCLE"].fillna(df["OBS_60_CNT_SOCIAL_CIRCLE"].median())


# In[185]:


df.boxplot("OBS_60_CNT_SOCIAL_CIRCLE")
plt.show()


# In[186]:


df.isnull().sum()


# In[187]:


df.info(verbose = True, null_counts = True)


# In[188]:


## Now we deal with previous application data set


# In[28]:


df1.shape


# In[29]:


df1.head()


# In[30]:


df1.info()


# In[31]:


df1.isnull().sum()


# In[32]:


# percentage of missing values
(df1.isnull().sum()/len(df1.index))*100


# In[33]:


#####Data cleaning


# In[35]:


#Deleting columns with missing values > 50%

df1=df1.drop(["AMT_DOWN_PAYMENT", "RATE_INTEREST_PRIMARY", "RATE_DOWN_PAYMENT", "RATE_INTEREST_PRIVILEGED"], axis=1)


# In[38]:


(df1.isnull().sum()/len(df1.index))*100


# In[39]:


#Treating missing values


# In[40]:


#COLUMN: [AMT_ANNUITY]


# In[41]:


#Total no off null values
df1["AMT_ANNUITY"].isnull().sum()


# In[42]:


pd.options.display.float_format = "{:.2f}".format #to display in decimal form
df1["AMT_ANNUITY"].describe()


# In[43]:


##Histogram plotting of this data


# In[44]:


plt.hist(df1["AMT_ANNUITY"])
plt.show()


# In[45]:


#Now it is clearly visible that most values are between min and 50%
#so we use median to impute missing values
df1["AMT_ANNUITY"].median(skipna = True)


# In[46]:


df1["AMT_ANNUITY"] = df1["AMT_ANNUITY"].fillna(df1["AMT_ANNUITY"].median(skipna = True))


# In[47]:


pd.options.display.float_format = "{:.2f}".format #to display in decimal form
df1["AMT_ANNUITY"].describe()


# In[48]:


plt.hist(df1["AMT_ANNUITY"])
plt.show()


# In[49]:


#now the graph seems to be more informative


# In[50]:


#Now we check for outliers in AMT_ANNUITY 


# In[53]:


df1.boxplot(["AMT_ANNUITY"], figsize=[4,4])
plt.show()


# In[54]:


#Data points appear to be evenly distributed
#No need to remove/handle outliers (also seen from the distribution, values are mostly continuous)


# In[55]:


#Check that there are no null values in AMT_ANNUITY after imputation
df1.isnull().sum()


# In[56]:


###COLUMN: [CNT_PAYMENT]


# In[62]:


df1["CNT_PAYMENT"].describe()


# In[63]:


df1["CNT_PAYMENT"].value_counts()


# In[64]:


df1["CNT_PAYMENT"].isnull().sum()


# In[65]:


plt.hist(df1["CNT_PAYMENT"])
plt.show()


# In[66]:


##Max of the data is between 0 and 20


# In[67]:


## so in this case mean is supposed to be most appropriate to assign values to missing attributes


# In[68]:


df1["CNT_PAYMENT"].mean()


# In[69]:


df1["CNT_PAYMENT"] = df1["CNT_PAYMENT"].fillna(df1["CNT_PAYMENT"].mean())


# In[70]:


##After assigning
plt.hist(df1["CNT_PAYMENT"])
plt.show()


# In[71]:


##now to verify their is no missing values
df1["CNT_PAYMENT"].isnull().sum()


# In[189]:


##COLUMN: [DAYS_TERMINATION]


# In[190]:


df1["DAYS_TERMINATION"].describe()


# In[191]:


plt.hist(df1["DAYS_TERMINATION"])
plt.show()


# In[192]:


df1["DAYS_TERMINATION"].median()


# In[193]:


df1["DAYS_TERMINATION"] = df1["DAYS_TERMINATION"].fillna(df1["DAYS_TERMINATION"].median())


# In[194]:


plt.hist(df1["DAYS_TERMINATION"])
plt.show()


# In[195]:


df1.isnull().sum()


# In[196]:


### COLUMN: [AMT_GOODS_PRICE]


# In[197]:


df1["AMT_GOODS_PRICE"].describe()


# In[198]:


plt.hist(df1["AMT_GOODS_PRICE"])
plt.show()


# In[199]:


df1["AMT_GOODS_PRICE"] = df1["AMT_GOODS_PRICE"].fillna(df1["AMT_GOODS_PRICE"].isnull().sum())


# In[200]:


df1["AMT_GOODS_PRICE"].isnull().sum()


# In[201]:


df1.boxplot(["AMT_GOODS_PRICE"], figsize = [8, 10])
plt.show()


# In[202]:


### COLUMN: [DAYS_FIRST_DRAWING]


# In[203]:


df1["DAYS_FIRST_DRAWING"].describe()


# In[204]:


plt.hist(df1["DAYS_FIRST_DRAWING"])
plt.show()


# In[205]:


df1["DAYS_FIRST_DRAWING"].median()


# In[206]:


### we will use medium to assign values to missing values 
df1["DAYS_FIRST_DRAWING"] = df1["DAYS_FIRST_DRAWING"].fillna(df1["DAYS_FIRST_DRAWING"].median())


# In[207]:


plt.hist(df1["DAYS_FIRST_DRAWING"])
plt.show()


# In[208]:


### COLUMN: [DAYS_FIRST_DUE]


# In[209]:


df1['DAYS_FIRST_DUE'].describe()


# In[210]:


plt.hist(df1["DAYS_FIRST_DUE"])
plt.show()


# In[211]:


df1['DAYS_FIRST_DUE'] = df1['DAYS_FIRST_DUE'].fillna(df1['DAYS_FIRST_DUE'].median())


# In[212]:


plt.hist(df1["DAYS_FIRST_DUE"])
plt.show()


# In[213]:


df1['DAYS_FIRST_DUE'].isnull().sum()


# In[214]:


### COLUMN: [DAYS_LAST_DUE_1ST_VERSION]


# In[215]:


df1["DAYS_LAST_DUE_1ST_VERSION"].describe()


# In[216]:


plt.hist(df1["DAYS_LAST_DUE_1ST_VERSION"])
plt.show()


# In[217]:


#Impute missing values the median
df1["DAYS_LAST_DUE_1ST_VERSION"] = df1["DAYS_LAST_DUE_1ST_VERSION"].fillna(df1["DAYS_LAST_DUE_1ST_VERSION"].median())


# In[218]:


plt.hist(df1["DAYS_LAST_DUE_1ST_VERSION"])
plt.show()


# In[219]:


### COLUMN: [DAYS_LAST_DUE]


# In[220]:


df1["DAYS_LAST_DUE"].describe()


# In[221]:


plt.hist(df1["DAYS_LAST_DUE"])
plt.show()


# In[222]:


df1["DAYS_LAST_DUE"] = df1["DAYS_LAST_DUE"].fillna(df1["DAYS_LAST_DUE"].median())


# In[223]:


plt.hist(df1["DAYS_LAST_DUE"])
plt.show()


# In[224]:


### COLUMN[NFLAG_INSURED_ON_APPROVAL]


# In[225]:


df1["NFLAG_INSURED_ON_APPROVAL"].describe()


# In[226]:


plt.hist(df1["NFLAG_INSURED_ON_APPROVAL"])
plt.show()


# In[227]:


### In this case median is 0 so is better for us to replace the values with 0 as compared to 1 as this may lead to false decission and affect our final analysis 


# In[228]:


df1["NFLAG_INSURED_ON_APPROVAL"] = df1["NFLAG_INSURED_ON_APPROVAL"].fillna(df1["NFLAG_INSURED_ON_APPROVAL"].median())


# In[229]:


plt.hist(df1["NFLAG_INSURED_ON_APPROVAL"])
plt.show()


# In[230]:


### COLUMN: [Product_Combination]


# In[231]:


df1["PRODUCT_COMBINATION"].describe()


# In[232]:


df1["PRODUCT_COMBINATION"].isnull().sum()


# In[233]:


df1["PRODUCT_COMBINATION"].value_counts()


# In[234]:


df1["PRODUCT_COMBINATION"] = df1["PRODUCT_COMBINATION"].fillna("Cash")


# In[235]:


df1["PRODUCT_COMBINATION"].value_counts()


# In[236]:


df1.isnull().sum()


# In[237]:


### Now we have done with the data cleaning 


# In[238]:


### We will analyze the data now


# In[239]:


### some columns of our data in df have negative values which have no meaning. so we will use abs function to make these values positive.


# In[240]:


df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
df['DAYS_ID_PUBLISH'] = abs(df['DAYS_ID_PUBLISH'])
df['DAYS_ID_PUBLISH'] = abs(df['DAYS_ID_PUBLISH'])


# In[241]:


df.head()


# In[242]:


bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

df['AMT_INCOME_RANGE'] = pd.cut(df['AMT_INCOME_TOTAL'], bins=bins, labels=slot)


# In[243]:


df["AMT_INCOME_RANGE"].head()


# In[244]:


bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
slots = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000 and above']

df['AMT_CREDIT_RANGE'] = pd.cut(df['AMT_CREDIT'], bins=bins, labels=slots)


# In[246]:


df["AMT_CREDIT_RANGE"].head()


# In[247]:


#Split the dataset into two datasets with target=1 (customers with payment issues) and target=0 (all other cases)
#We will use these two datasets for a small number of comparisons


# In[248]:


target0 = df[df["TARGET"]==0]
target1 = df[df["TARGET"]==1]


# In[249]:


print(target0.shape)
print(target1.shape)


# In[250]:


## income range plotting


# In[259]:


total = len(df["TARGET"])
explode = [0, 0.05]

def my_fmt(x):
    return '{:.2f}%\n({:.0f})'.format(x, total*x/100) #to print both the percentage and value together

plt.figure(figsize = [6, 6])
plt.title("Imbalance between target0 and target1")
df["TARGET"].value_counts().plot.pie(autopct = my_fmt, colors = ["teal", "gold"], explode = explode)

plt.show()


# In[260]:


### 8.78% of customers are customers with payment problems. 91.21% of customers fall into the "all other cases" category


# In[261]:


def functionPlot(df, col, title, xtitle, ytitle, hue = None):
    
    sns.set_style("white")
    sns.set_context("notebook")    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation = 45)
    plt.yscale('log')
    plt.title(title)

    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue, palette='deep') 
    ax.set(xlabel = xtitle, ylabel = ytitle)    
    plt.show()


# In[262]:


functionPlot(target0, col='NAME_CONTRACT_TYPE', title='Distribution of contract type (Segregated based on gender)', hue='CODE_GENDER', xtitle = "Type of Contract", ytitle = "Count")


# In[263]:


### Cash loans obviously have more customers per loan than revolving loans. In both cases, there were more female customers than male customers.


# In[264]:


functionPlot(target1, col='AMT_INCOME_RANGE', title='Distribution of Income Range (Customers with payment difficulties)', hue='CODE_GENDER', xtitle = "Income Range", ytitle = "Count")


# In[265]:


### 100,000 to 200,000 income, the highest credit limit. There are far fewer than income brackets of 400,000 and above. On average, there are more male customers with less credit.


# In[266]:


functionPlot(target1, col='NAME_CONTRACT_TYPE', title='Distribution of Contract Type (Customers with payment difficulties)', hue='NAME_EDUCATION_TYPE', xtitle = "Type of contract", ytitle = "Count")


# In[267]:


### As we have seen, cash loans are overwhelmingly preferred by customers of all educational levels.
### People who only have diplomas don't like revolving loans at all.


# In[271]:


plt.figure(figsize = [30, 7])

plt.suptitle("Distribution of clients with difficulties and all other cases")

plt.subplot(1,2,1)
ax = df["TARGET"].value_counts().plot(kind = "barh", colormap = "plasma")

plt.subplot(1,2,2)
df["TARGET"].value_counts().plot.pie(autopct = "%.2f%%", startangle = 40, colors = ["blue", "pink"])


for i,j in enumerate(df["TARGET"].value_counts().values):
    ax.text(.7, i, j, weight = "bold")

plt.show()


# In[272]:


### 8.79% (18,547) of the total number of customers (192,573) are struggling to repay their loans.


# In[273]:


#####Concatenating applicationData and previousApplication


# In[274]:


data = pd.merge(df, df1, on = 'SK_ID_CURR', how = 'inner')
data.sort_values(by = ['SK_ID_CURR','SK_ID_PREV'], ascending = [True, True], inplace = True)


# In[275]:


data.head()


# In[276]:


## Analyzing gender column in concatenating data set


# In[279]:


fig = plt.figure(figsize=(13,6))

plt.subplot(1,2,1)
data["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%", colors = ["orange", "green"])
plt.title("Gender distribution")
plt.show()


# In[280]:


### In the app data file we see 61% female and 39% male, but now in the combined dataset we see: - Female: 62% Male: 38%


# In[282]:


## Client income type distribution


# In[286]:


plt.figure(figsize = [20, 8])

plt.subplot(1,2,2)
plt.title("Distribution of client income type",  weight = "bold")
ax = sns.countplot(y = data["NAME_INCOME_TYPE"], palette = "deep", order = data["NAME_INCOME_TYPE"].value_counts().index[:4])
ax.set(xlabel = "Count", ylabel = "Income Type")

plt.subplot(2,2,1)
plt.title("Distribution of client income  type by target (repayment status)",  weight = "bold")
ax = sns.countplot(y = data["NAME_INCOME_TYPE"],  hue = data["TARGET"], palette="deep", order = data["NAME_INCOME_TYPE"].value_counts().index[:4])
ax.set(xlabel = "Count", ylabel = "")

plt.show()


# In[287]:


###Most customers work based on their refund status in both cases.
###Conversely, the fewest customers are retirees (retired customers)


# In[288]:


### Breakdown of type of education by loan repayment status


# In[290]:


plt.figure(figsize = [20, 10])

plt.subplot(1,2,2)
plt.title("Distribution of Family status for Defaulters (Target1)", weight = "bold")
data[data["TARGET"]==1]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%2.0f%%")

plt.subplot(1,2,1)
plt.title("Distribution of Family status for Repayers (Target0)",  weight = "bold")
data[data["TARGET"]==0]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%2.0f%%")



plt.show()


# In[291]:


## There was a -4% difference among married customers who had difficulty paying.
## Marital Status in Both Repayment Situations Divided Almost Equally Marital Status (Family Members Living with Client)


# In[292]:


#### TOP10 Correlation variables


# In[293]:


repayerData = data[data['TARGET'] == 0]
defaulterData = data[data['TARGET'] == 1]


# In[294]:


### Most correlated columns


# In[295]:


repayerData.corr().unstack().sort_values(ascending = False).drop_duplicates()


# In[296]:


### Top 10 correlated columns are as follows


# In[297]:


###SK_ID_CURR                SK_ID_CURR                  
###OBS_60_CNT_SOCIAL_CIRCLE  OBS_30_CNT_SOCIAL_CIRCLE    
###AMT_CREDIT_x              AMT_GOODS_PRICE_x            
###AMT_CREDIT_y              AMT_APPLICATION              
###DAYS_TERMINATION          DAYS_LAST_DUE                
                                                        
###DAYS_BIRTH                DAYS_REGISTRATION           
###CNT_PAYMENT               DAYS_LAST_DUE_1ST_VERSION   
###DAYS_FIRST_DRAWING        DAYS_TERMINATION            
                          #DAYS_LAST_DUE_1ST_VERSION   
###SK_ID_CURR                TARGET                      


# In[298]:


### now we will form a data set using these corelated variables
top10_CorrTarget0 = repayerData[["OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "AMT_APPLICATION", "DAYS_TERMINATION", "DAYS_LAST_DUE", "CNT_FAM_MEMBERS", "CNT_CHILDREN", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "AMT_GOODS_PRICE_y", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY", "AMT_CREDIT_y", "AMT_ANNUITY_y"]].copy()


# In[299]:


top10_CorrTarget0.shape


# In[300]:


### Now we can use heatmap to visualize these 10 corelated variable


# In[305]:


corr_target0 = top10_CorrTarget0.corr()

plt.figure(figsize = [20, 20])
sns.heatmap(data = corr_target0, cmap="magma", annot=True)

plt.show()


# In[306]:


###There is a strong correlation between AMT_GOODS_PRICE and AMT_APPLICATION, i.e. the higher the credit previously requested by the customer, the more it is proportional to the price of the product previously requested by the customer.
###AMT_ANNUITY and AMT_APPLICATION also have a high correlation, meaning that the higher the loan annuity issued, the higher the price of the product the customer previously requested.
###If the customer's contact address does not match the business address, chances are the customer's permanent address does not match the business address either.
###The first result of the previous request is strongly correlated with the expected end of the previous request
###CNT_CHILDREN and CNT_FAM_MEMBERS are strongly correlated, which means that customers with children will also have family members.

