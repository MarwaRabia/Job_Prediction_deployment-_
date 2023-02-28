
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from flask import Flask,render_template,request
import plotly.express as px
from JobPrediction import JobPrediction

MLFLOW_TRACKING_URI = 'models/mlruns'
MLFLOW_RUN_ID = "c161fa669278438ab3f99def61973436"
CLUSTERS_YAML_PATH = "data/features_skills_clusters_description.yaml"
DATA_PATH = 'data/02_cleaned_df.pkl'

ROLE_COLS = ['DevType']
TECH_COLS = ['LanguageHaveWorkedWith',
             'DatabaseHaveWorkedWith',
             'WebframeHaveWorkedWith',
             'MiscTechHaveWorkedWith',
             'ToolsTechHaveWorkedWith']

TECH_NAMES = {'LanguageHaveWorkedWith': "Languages",
              'DatabaseHaveWorkedWith': "Databases",
              'WebframeHaveWorkedWith': "Web Frameworks",
              'MiscTechHaveWorkedWith': "Other Tech",
              'ToolsTechHaveWorkedWith': "Tools"}

jobs=['Academic researcher',
 'Data or business analyst',
 'Data scientist or machine learning specialist',
 'Database administrator',
 'DevOps specialist',
 'Developer, QA or test',
 'Developer, back-end',
 'Developer, desktop or enterprise applications',
 'Developer, embedded applications or devices',
 'Developer, front-end',
 'Developer, full-stack',
 'Developer, game or graphics',
 'Developer, mobile',
 'Engineer, data',
 'Scientist',
 'System administrator']


## Intialize


def one_hot_encode(df: pd.DataFrame, columns):
    """One-hot-encode columns with multiple answers"""
    df = df.copy()

    if not isinstance(columns, list):
        raise ValueError('arg: column has to be a list')

    encoded_dfs = {}
    for column in columns:
        binarizer = MultiLabelBinarizer()
        encoded_df = pd.DataFrame(binarizer.fit_transform(df[column]),
                                  columns=binarizer.classes_,
                                  index=df[column].index)
        encoded_dfs[column] = encoded_df

    # Merge 1-hot encoded dfs and return
    encoded_dfs = pd.concat(encoded_dfs, axis=1)
    return encoded_dfs


df = pd.read_pickle(DATA_PATH)
skills_groups = {col: one_hot_encode(df, [col]).columns.get_level_values(1).tolist()
                 for col in TECH_COLS}





app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST','GET'])
def test():
    job_model = JobPrediction(mlflow_uri=MLFLOW_TRACKING_URI,
                          run_id=MLFLOW_RUN_ID,
                          clusters_yaml_path=CLUSTERS_YAML_PATH)
    sample_skills=[]
    
    dropdown1_values = request.form.getlist('dropdown1')
    dropdown2_values = request.form.getlist('dropdown2')
    dropdown3_values = request.form.getlist('dropdown3')
    dropdown4_values = request.form.getlist('dropdown4')
    dropdown5_values = request.form.getlist('dropdown5')
    
    # print(dropdown6_values)
    
    sample_skills=dropdown1_values+dropdown2_values+dropdown3_values+dropdown4_values+dropdown5_values
    
    dropdown1_options = skills_groups['LanguageHaveWorkedWith']
    dropdown2_options = skills_groups['DatabaseHaveWorkedWith']
    dropdown3_options = skills_groups['WebframeHaveWorkedWith']
    dropdown4_options = skills_groups['MiscTechHaveWorkedWith']
    dropdown5_options = skills_groups['ToolsTechHaveWorkedWith']
    dropdown6_options = jobs
    
    
    
    if request.method == 'POST':
       
        if "job" in request.form:
            # print("Download")
            # print(request.form)
            
            predictions = job_model.predict_jobs_probabilities(sample_skills)
            fig = px.bar(predictions.sort_values(),
                    orientation='h',
                    width=1500, height=500)
            fig \
            .update_xaxes(title='',
                        visible=True,
                        tickformat=',.0%',
                        range=[-0.03,1],
                        showticklabels=False) \
            .update_yaxes(title='Job',
                        visible=True) \
            .update_layout(showlegend=False,
                        font=dict(size=16),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)')
            fig.update_traces(marker_color='#30b7f6')
            
            fig.write_image('static/predict.png',scale=3)

            return render_template('test.html', dropdown1_options=dropdown1_options, dropdown2_options=dropdown2_options, dropdown3_options=dropdown3_options,
                            dropdown4_options=dropdown4_options,dropdown5_options=dropdown5_options,dropdown6_options=dropdown6_options,pred_val_job="../static/predict.png")
        
        
        
        elif "skill" in request.form:
            
            dropdown6_values = request.form.getlist('dropdown6')
            print(sample_skills,dropdown6_values)
            predictions = job_model.recommend_new_skills(sample_skills,dropdown6_values[0],.3)
            fig = px.bar(predictions.sort_values(),
                    orientation='h',
                    width=1200, height=600)
            fig \
            .update_xaxes(title='',
                        visible=True,
                        tickformat=',.0%',
                        range=[-0.03,1],
                        showticklabels=False) \
            .update_yaxes(title='Skills',
                        visible=True) \
            .update_layout(showlegend=False,
                        font=dict(size=16),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)')
            fig.update_traces(marker_color='#30b7f6')
            
            fig.write_image('static/predictskill.png',scale=3)

            return render_template('test.html', dropdown1_options=dropdown1_options, dropdown2_options=dropdown2_options, dropdown3_options=dropdown3_options,
                            dropdown4_options=dropdown4_options,dropdown5_options=dropdown5_options,dropdown6_options=dropdown6_options,pred_val_skill="../static/predictskill.png")
        else:
            return render_template('test.html', dropdown1_options=dropdown1_options, dropdown2_options=dropdown2_options, dropdown3_options=dropdown3_options,
                           dropdown4_options=dropdown4_options,dropdown5_options=dropdown5_options,dropdown6_options=dropdown6_options)
        
    else:
        return render_template('test.html', dropdown1_options=dropdown1_options, dropdown2_options=dropdown2_options, dropdown3_options=dropdown3_options,
                           dropdown4_options=dropdown4_options,dropdown5_options=dropdown5_options,dropdown6_options=dropdown6_options)
        


@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == '__main__':
    app.run(debug=True)