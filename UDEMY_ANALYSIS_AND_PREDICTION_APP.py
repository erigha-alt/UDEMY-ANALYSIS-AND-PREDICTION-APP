#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 05:28:08 2023

@author: user
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import warnings
import base64
#warnings.PyplotGlobalUseWarning
warnings.filterwarnings('ignore')

# Streamlit app
st.set_page_config(
    page_title = "Udemy Analysis and Prediction",
    page_icon = "📉",
    layout = "wide"
    )


 # Streamlit functions
 
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-repeat: repeat;
        background-attachment: scroll;
   }}
    </style>
    """,
        unsafe_allow_html = True
    )
    
background = add_bg_from_local("/Users/user/Downloads/magicpattern-grid-pattern-1690805793107.png")


# STEP 1: Store pages inside session_state
# store['page'] = 0
if 'page' not in st.session_state:
    st.session_state['page'] = 0

def next_page():
    st.session_state['page'] += 1

def previous_page():
    st.session_state['page'] -= 1

# Load the dataset
udemy_dataset = pd.read_csv("/Users/user/Downloads/udemy_courses 2.csv")
udemy = udemy_dataset.drop(['course_id', 'url'], axis = 1)
# Initialize the label encoder
# encode = LabelEncoder()
# udemy['level'] = encode.fit_transform(udemy['level'])
# udemy['subject'] = encode.fit_transform(udemy['subject'])
# udemy['course_title'] = encode.fit_transform(udemy['course_title'])
# udemy['is_paid'] = encode.fit_transform(udemy['is_paid'])



if st.session_state['page'] == 0:
    st.title("UDEMY DATASET")
    st.divider()
    st.dataframe(udemy_dataset, use_container_width = True, height = 725)
    
 # Creating columns for navigating to next page and previous page
    col1, col2 = st.columns([10, 1])
    with col1:
        pass

    with col2:
        st.button("Next page", on_click = next_page)
    

elif st.session_state['page'] == 1:
    st.title("Exploratory Data Analysis")
    st.divider()
    
    head =udemy_dataset.head()
    tail = udemy_dataset.tail()
    descriptive_stats = udemy_dataset.describe()
    # Dropping unnecessary columns
    udemy = udemy_dataset.drop(['course_id', 'url'], axis=1)

    correlation_matrix = udemy.corr()
   
    check_null = udemy.isnull().sum()
    check_null = pd.Series(check_null, name = "Null_Value_Count")
    total_null = udemy.isnull().sum().sum()
    distinct_count = udemy.nunique()
    distinct_count = pd.Series(distinct_count, name = "Unique_Value_Count")
    
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Data Head", "Data Tail", "Data Descriptive Statistics", "Dropping columns", "Data Correlation Matrix", "Null Value Count", "Unique Count"])
    
    with tab1:
        st.subheader("Data Head")
        st.write("Finding the head of our dataset means we look at the first 5 values of our dataset. This is used in exploratory data analysis as a way to share insight on large datasets, what is happening at the extreme end of our data. In pandas, this is accomplished using the code below. ")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    head =udemy_dataset.head()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to find the head of the dataset.")
        st.dataframe(head, use_container_width = True, height = 225)
        
    
        
    with tab2:
        st.subheader("Data Tail")
        st.write("Finding the tail of our dataset means we look at the bottom 5 values of our dataset. This is used in exploratory data analysis as a way to share insight on large datasets, what is happening at the extreme end of our data. In pandas, this is accomplished using the code below. ")
        # Using Column Design
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    tail = udemy_dataset.tail()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to find the tail of the dataset.")
        st.dataframe(tail, use_container_width =True, height = 225)

    with tab3:    
       st.subheader("Data Descriptive Statistics")
       st.write("printing out the discriptive statistics in each colum in our dataset which is the (mean, min, max, count, std ...). It helps us to describe the features and understand the basic characteristics of our data. it also helps us to identify outliers")
       col1, col2 = st.columns([1, 2])
       with col1:
           # Using Expander Design
           with st.expander("Code"):
               st.code(
                   """
                   import pandas as pd
                   
                   dataset = pd.read_csv("file_directory")
                   descriptive_stats = udemy.describe()
                   """)
       with col2:
           st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to check for the descriptive statistics of the data.")
       st.dataframe(descriptive_stats, height = 325)
       

    with tab4:    
        st.subheader("Dropping columns")
        st.write("dropping irrelivant columns in our dataset.In pandas, this is accomplished using the code below.")
        col1, col2 = st.columns([1, 2])
        with col1:
           # Using Expander Design
            with st.expander("Code"):
                st.code(
                   """
                   import pandas as pd
                   
                   dataset = pd.read_csv("file_directory")
                   udemy = udemy_dataset.drop(['course_id', 'url'], axis=1)
                   """)
        with col2:
           st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library in EDA before you check for the drop column in the dataset.")
        st.dataframe(udemy, use_container_width = True, height = 750)
       
          
    with tab5:    
        st.subheader("Data Correlation Matrix")
        st.write("Checking for the correlation between each columns(features) in our dataset,which means checking the relationship between each columns and finding if our columns are highly correlated (positively or negatively) or not.In pandas, this is accomplished using the code below.")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    corr = udemy.corr()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to find the correlation matrix of the dataset.")
        st.dataframe(correlation_matrix, use_container_width = True, height = 250)
        
   
         
            
    with tab6:
        st.subheader("Null Value Count (Columns)")
        st.write("checking if there are any null values in our dataset by checking every columns in our dataset if there are any missing values. in pandas this is accomplish by using the code below")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    check_null = udemy.isnull().sum()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to check if there are any missing values in the dataset.")
        st.dataframe(check_null, width = 300, height = 400)
    
    with tab7:    
        st.subheader("Unique Values (Columns)")
        st.write("checking for the unique values of each column in the dataset")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    distinct_count = cudemy.nunique()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to check for the nunique values in the dataset.")
        st.dataframe(distinct_count, width = 250, height = 400)
        
    
    
    st.divider()
    
    
    # Creating columns for navigating to next page and previous page
    col3, col4 = st.columns([10, 1])
    with col3:
        st.button("Previous Page", on_click = previous_page)

    with col4:
        st.button("Next page", on_click = next_page)
    

elif st.session_state['page'] == 2:
    st.title("Data Visualization.")
    st.write("Data Visualization is an important stept in EDA. Data visualization is the graphical representation of data and information using charts, graphs, and other visual elements to help people understand patterns, trends, and insights within the data")
    udemy = udemy_dataset.drop(['course_id', 'url'], axis = 1)
    # Display the pie chart for count of paid and free courses
    st.write("courses payment distribution:")
    is_paid = udemy_dataset['is_paid'].value_counts()
    label = ['true', 'false']
    plt.figure(figsize = (2,2))
    plt.pie(is_paid.values, labels=label, autopct='%0.2f%%', shadow=True, explode=[0, 0.2], startangle=180)
    plt.title('Count of paid and free')
    st.pyplot()
    # Display the bar chart for number of courses posted per year
    st.write("Number of courses posted per year:")
    udemy['published_timestamp'] = pd.to_datetime(udemy['published_timestamp'])
    udemy['published_year'] = udemy['published_timestamp'].dt.year
    year_counts = udemy['published_year'].value_counts()
    year_counts = year_counts.sort_index()
    plt.figure(figsize=(8, 6))
    plt.bar(year_counts.index, year_counts.values)
    plt.xlabel('Year')
    plt.ylabel('Number of Courses')
    plt.title('Number of Courses Posted per Year')
    plt.grid(False)
    st.pyplot()

    # Display the bar chart for number of courses per subject
    plt.figure(figsize=(8, 6))
    st.write("Number of courses per subject:")
    sub = udemy['subject'].value_counts()
    plt.xlabel('Subject')
    plt.ylabel('Number of Courses')
    plt.title('Number of Courses Per Subjects')
    plt.bar(sub.index, sub.values, width=0.4)
    plt.xticks(fontsize=8)
    st.pyplot()


    # Display the bar chart for number of courses per level
    st.write("Number of courses per level:")
    level = udemy['level'].value_counts()
    plt.xlabel('Levels')
    plt.ylabel('Number of Courses')
    plt.title('Number of Courses Per Levels')
    plt.bar(level.index, level.values, width=0.4)
    plt.xticks(fontsize=8)
    st.pyplot()

    # Display the bar chart for number of lectures in paid and free courses
    st.write("Number of lectures in paid and free courses:")
    num_lec = udemy.groupby('is_paid')['num_lectures'].sum().sort_values(ascending=False)
    label = ['true', 'false']
    fig, ax = plt.subplots(figsize=(2, 3))
    ax.bar(label, num_lec.values, width=0.4)
    plt.ylabel('Number of Lectures')
    plt.xlabel('Type of Course')
    st.pyplot()

    # Display the distribution of the is_paid variable for each level
    st.write("Distribution of is_paid variable for each level:")
    levels = udemy['level'].unique()

    for level in levels:
        is_paid = udemy[udemy['level'] == level].groupby('is_paid').size()
        num_slices = len(is_paid)
        explode = [0.1] + [0] * (num_slices - 1)  # Separate the first slice from the center
        plt.pie(is_paid, labels=is_paid.index, autopct='%0.1f%%', explode=explode, startangle=180)
        #plt.legend(['true', 'false'], title='is_paid')
        plt.title(f'Distribution of is_paid for {level} Level')
        st.pyplot()

    # Display the bar chart for Top 20 Most Popular Courses by Number of Subscribers: Paid vs Free
    st.write("Top 10 Most Popular Courses by Number of Subscribers:")
    #Top 20 Most Popular Courses by Number of Subscribers: Paid vs Free
    top_20_subscribers = udemy.nlargest(20,'num_subscribers')
    plt.figure(figsize=(8, 6))
    # Plotting paid courses
    plt.barh(top_20_subscribers[top_20_subscribers['is_paid'] == True]['course_title'],
             top_20_subscribers[top_20_subscribers['is_paid'] == True]['num_subscribers'], color='orange', label='Paid')
    # Plotting free courses
    plt.barh(top_20_subscribers[top_20_subscribers['is_paid'] == False]['course_title'],
             top_20_subscribers[top_20_subscribers['is_paid'] == False]['num_subscribers'], color='blue', label='Free')
    plt.xlabel('Number of Subscribers')
    plt.ylabel('Course Title')
    plt.title('Top 20 Most Popular Courses by Number of Subscribers: Paid vs Free')
    plt.legend()
    plt.gca().invert_yaxis()  # Invert the y-axis to have the course with the highest subscribers at the top
    plt.tight_layout()
    plt.grid(False)
    st.pyplot()

    # Display the bar chart for Top 20 Courses with Highest Number of Reviews: Paid vs Free
    st.write("Course with the Highest Number of Reviews:")
    top_20_reviews = udemy.nlargest(20, 'num_reviews')
    plt.figure(figsize=(10, 6))
    # Plotting paid courses
    plt.barh(top_20_reviews[top_20_reviews['is_paid'] == True]['course_title'],
             top_20_reviews[top_20_reviews['is_paid'] == True]['num_reviews'], color='orange', label='Paid')
    # Plotting free courses
    plt.barh(top_20_reviews[top_20_reviews['is_paid'] == False]['course_title'],
             top_20_reviews[top_20_reviews['is_paid'] == False]['num_reviews'], color='blue', label='Free')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Course Title')
    plt.title('Top 20 Courses with Highest Number of Reviews: Paid vs Free')
    plt.legend()
    plt.gca().invert_yaxis()  # Invert the y-axis to have the course with the highest reviews at the top
    plt.tight_layout()
    st.pyplot()

    # Display the pie chart for count of subjects
    st.write("Count of subjects:")
    subject = udemy['subject'].value_counts()
    plt.pie(subject.values, labels=subject.index, autopct='%0.2f%%', shadow=True)
    plt.title('Count of subjects')
    st.pyplot()

    st.write("Correlation Heatmap:")
    udemy.content_duration = udemy['content_duration'].apply(lambda x: int(np.floor(x)))
    
    st.write("Average Price of Paid and Free Courses by Level:")
    # Group data by level and calculate average price for each level and payment status
    level_price = udemy.groupby(['level', 'is_paid'])['price'].mean().reset_index()
    # Pivot the data for plotting
    pivot_data = level_price.pivot(index='level', columns='is_paid', values='price')
    # Plotting the data
    pivot_data.plot(kind='bar', stacked=True, color= ['blue', 'orange'])
    plt.xlabel('Level')
    plt.ylabel('Average Price')
    plt.title('Average Price of Paid and Free Courses by Level')
    plt.xticks(rotation=45)
    plt.legend(title='is_paid', labels=['Free', 'Paid'])
    plt.tight_layout()
    st.pyplot()
    
    
    def correlation_heatmap(df):
        _, ax = plt.subplots(figsize=(8, 10))
        colormap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(udemy.corr(), annot=True, cmap=colormap)

    correlation_heatmap(udemy)
    st.pyplot()
    
    
    # Creating columns for navigating to next page and previous page
    col5, col6 = st.columns([10, 1])
    with col5:
        st.button("Previous Page", on_click = previous_page)

    with col6:
        st.button("Next page", on_click = next_page)


elif st.session_state['page'] == 3:
    introduction_text = '''
    INTRODUCTION TO UDEMY:
        
    Udemy has emerged as a leading global online learning platform, providing a diverse array of courses on various subjects. With millions of students and thousands of instructors, Udemy offers a vast range of courses, from technical skills like programming and data analysis to soft skills like leadership and communication. This platform empowers individuals around the world to acquire new knowledge, enhance existing skills, and pursue personal and professional growth.

    ABOUT THE APP:
        
    Predicting whether a Udemy course is paid or free can have valuable implications for both learners and content creators. This prediction can provide insights into the dynamics of the online education marketplace and help learners make informed decisions. By visualizing trends in course pricing and distribution, learners can confidently select courses aligned with their budget and learning objectives. Additionally, content creators can tailor their offerings based on the popularity of paid and free courses, optimizing their revenue strategy. This visual exploration enhances our understanding of Udemy's educational ecosystem. Here's how the prediction's usefulness can be highlighted:
    
    Learner Decision-Making: For learners exploring courses, knowing whether a course is paid or free can significantly influence their decision to enroll. Predicting the courses payment status can assist learners in selecting courses that align with their budget and learning objectives.
    
    Content Creators and Instructors: Content creators, instructors, and course publishers can benefit from understanding the factors that contribute to a course being paid or free. This prediction could guide instructors in determining the pricing strategy for their courses and help them tailor their offerings to match market demand.

    Platform Strategy: Udemy platform administrators can gain insights into the popularity and distribution of paid and free courses. This information can influence strategic decisions related to course promotion, platform revenue, and user engagement.

    Market Analysis: Predictive models that classify courses as paid or free can be used for market analysis and research. By analyzing the trends and patterns of paid and free courses, researchers can identify shifts in educational preferences and industry demands.

    In summary, predicting whether a Udemy course is paid or free is more than just a technical exercise; it holds practical implications for both learners and course creators. The insights gained from such predictions can shape learning experiences, instructor strategies, and platform decisions, ultimately contributing to the growth and impact of online education.
    
    VISUALIVATION
    
    visualizations provide valuable insights into various aspects of Udemy's course offerings, such as their distribution, popularity, price trends, and correlations between attributes.
    Each aspect of the visualization is talkin about:
        
        
    Count of Paid and Free Courses :
        
    This chart visually displays the distribution of paid and free courses on Udemy. The orange slice represents paid courses, while the blue slice represents free courses. The percentages indicate the proportion of each type of course in the dataset.

    Number of Courses Posted per Year:
    This chart illustrates the number of courses posted on Udemy for each year. The x-axis represents the years, and the y-axis represents the count of courses. The chart provides insights into how the number of course offerings has evolved over time.

    Number of Courses Per Subject:
        
    This chart showcases the count of courses available in different subjects. Each subject category is plotted along the x-axis, and the corresponding count of courses is shown on the y-axis. This visualization gives an overview of the diversity of subjects covered on the platform.

    Number of Courses Per Level:
        
    This chart depicts the distribution of courses across different skill levels. The x-axis represents the skill levels (e.g., Beginner, Intermediate), while the y-axis represents the count of courses. The chart offers insights into the level of expertise catered to by Udemy courses.

    Number of Lectures in Paid and Free Courses:
        
    This chart compares the total number of lectures in paid and free courses. The x-axis denotes the course type (paid or free), and the y-axis shows the cumulative number of lectures. The chart highlights any differences in content volume between paid and free courses.

    Distribution of is_paid Variable for Each Level:
        
    A series of charts illustrates the distribution of paid and free courses for each skill level category. Each pie chart represents a different level, with the sizes of the "Paid" and "Free" segments indicating the relative prevalence of those course types within the level.

    Top 20 Most Popular Courses by Number of Subscribers:
        
    This chart ranks the top 20 courses based on the number of subscribers they have. Courses are categorized by payment status (paid or free) and color-coded with orange for paid and blue for free. The y-axis displays the course titles, and the x-axis shows the number of subscribers.

    Top 20 Courses with Highest Number of Reviews:
        
    Similar to the previous chart, this chart ranks the top 20 courses by the number of reviews. Courses are grouped by payment status (paid or free) and color-coded. The y-axis represents the course titles, and the x-axis indicates the number of reviews.

    Count of Subjects :
        
    This chart presents the distribution of courses among different subjects. Each slice corresponds to a subject, with the slice size reflecting the proportion of courses in that subject category.

    Average Price of Paid and Free Courses by Level :
        
    This chart showcases the average price of courses across different skill levels. Each skill level is plotted along the x-axis, and the y-axis represents the average price. The bars are stacked to differentiate between paid (orange) and free (blue) courses.

    Correlation Heatmap:
        
    The correlation heatmap visually represents the correlation between different numerical variables in the dataset. Darker colors indicate stronger positive or negative correlations, helping to identify relationships between features.
    
    '''

    st.write(introduction_text)
    # Creating columns for navigating to next page and previous page
    col3, col4 = st.columns([10, 1])
    with col3:
        st.button("Previous Page", on_click = previous_page)

    with col4:
        st.button("Next page", on_click = next_page)
    # st.button("Next page", on_click=next_page)
    # st.button("Previous Page", on_click=previous_page)


elif st.session_state['page'] == 4:
    st.title("UDEMY COURSE PREDICTION: PAID OR FREE")
    st.write("This app helps to  predicts if a udemy course is paid or free .")
     
    udemy_dataset = pd.read_csv("/Users/user/Downloads/udemy_courses 2.csv")

    #printing the length of the data
    print("number of datapoint:",len(udemy_dataset))

    #Data cleaning
    #print the head of the csv
    print(udemy_dataset.head)

    #print the tail of the csv
    print(udemy_dataset.tail)

    #print the info of the csv 
    print(udemy_dataset.info)

    #print the discription of the csv
    print(udemy_dataset.describe)

    # droping unnecessary columns
    udemy = udemy_dataset.drop(['course_id', 'url'], axis = 1)
    # cheching the null values in the dataset
    print(udemy.isnull())

    #printing the null values in each row
    print(udemy.isnull().sum())

    #printing the total null values in the csv
    print(udemy.isnull().sum().sum())

    #printing out the unique values in each column
    print(udemy.nunique())

    #printing out duplicated row if there is any
    print(udemy.duplicated())

    # Initialize the label encoder
    encode = LabelEncoder()
    udemy['level'] = encode.fit_transform(udemy['level'])
    udemy['subject'] = encode.fit_transform(udemy['subject'])
    udemy['course_title'] = encode.fit_transform(udemy['course_title'])


    udemy.content_duration=udemy['content_duration'].apply(lambda x: int(np.floor(x)))

    #showing the correlations of each of the columns with price
    def correlation_heatmap(df):
        _,ax=plt.subplots(figsize=(8,10))
        colormap=sns.diverging_palette(220,10,as_cmap=True)
        sns.heatmap(udemy.corr(),annot=True,cmap=colormap)

    correlation_heatmap(udemy)
    # Drop irrelavant columns
    udemy.drop(["published_timestamp", "course_title", "content_duration", "price"],axis=1,inplace=True)

    x = udemy[["num_lectures", "subject", "level", "num_subscribers", "num_reviews"]]
    y = udemy["is_paid"]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Initialize and train multiple classification models

    model = LogisticRegression()
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)


    st.write("Enter course details to predict whether the course is free or paid:")

    # User inputs
    num_lectures = st.number_input("Number of Lectures:")
    subject = st.selectbox("subject:", ["web development", "Business_Finance", "Musical Instrument", "Graphic Design"])
    level = st.selectbox("Course Level:", ["All Levels", "Beginner Level", "Intermediate Level", "Expert Level"])
    num_subscribers = st.number_input("Number of Subscribers:")
    num_reviews = st.number_input("Number of Reviews:")

    level_mapping = {"All Levels": 0, "Beginner Level": 1, "Intermediate Level": 2, "Expert Level": 3}
    level_code = level_mapping[level]

    subject_mapping = {"web development": 0, "Business_Finance": 1, "Musical Instrument": 2, "Graphic Design": 3}
    subject_code = subject_mapping[subject]

    # Prepare input data for prediction
    if st.button('Predict'):
        input_data = pd.DataFrame({

            "num_lectures": [num_lectures],
            "subject": [subject_code],
            "level": [level_code],
            "num_subscribers": [num_subscribers],
            "num_reviews": [num_reviews]
        
    
    })

        prediction = model.predict(input_data)
        prediction_text = "Free" if prediction[0] == 0 else "Paid" 
        st.write(f"Based on the input data, the predicted course pricing is: {prediction_text}")
        
             # Calculate precision, recall, and F1-score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
        st.write("Model Evaluation Metrics:")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")

    st.button("Previous Page", on_click = previous_page)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    