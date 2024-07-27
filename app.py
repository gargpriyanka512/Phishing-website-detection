import gradio as gr

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pathlib
import matplotlib.lines as mlines

global_pr_qm=[]
global_da_ph=[]
answer=''
white_circle = mlines.Line2D([], [], color='white', markeredgecolor='black', marker='o', markersize=10, label='New Point', linestyle='None')
red_circle = mlines.Line2D([], [], color='red', marker='o', markersize=10, label='Legitimate', linestyle='None')
green_circle = mlines.Line2D([], [], color='green', marker='o', markersize=10, label='Phishing', linestyle='None')

def visualisations():
    global global_pr_qm
    with open('dataframe.pkl', 'rb') as file:
        df = pickle.load(file)
        plt.figure(figsize=(8, 5))

        red_points = df[df['is_legitimate'] == 0]
        blue_points = df[df['is_legitimate'] == 1]

        plt.scatter(red_points['nb_qm'], red_points['page_rank'], color='red' )
        plt.scatter(blue_points['nb_qm'], blue_points['page_rank'], color='green')
        if len(global_pr_qm)!=0:
            # if answer.split(',')[0]=='Legitimate':
            #     ccccc='green'
            # else:
            #     ccccc='red'

            ccccc=[]
            for i in answer[:-1].split(','):
                if i.startswith('Le'):
                    ccccc.append('green')
                else:
                    ccccc.append('red')

            x_coords, y_coords = zip(*global_pr_qm)
            plt.scatter(x_coords, y_coords,color=ccccc , s=100, edgecolor='black', zorder=5)

            # plt.scatter(global_pr_qm[0][0],global_pr_qm[0][1],color=ccccc , s=100, edgecolor='black', label='New Point', zorder=5)
        plt.title('scatter plot')
        plt.xlabel('Number of Question marks')
        plt.ylabel('page_rank')
        # plt.legend(title="Status")
        plt.legend(title="Status",handles=[white_circle, red_circle, green_circle])
        
        return plt.gcf() 
    
def visualisations2():
    global global_da_ph
    with open('dataframe.pkl', 'rb') as file:
        df = pickle.load(file)
        plt.figure(figsize=(8, 5))

        red_points = df[df['is_legitimate'] == 0]
        blue_points = df[df['is_legitimate'] == 1]

        plt.scatter(red_points['domain_age'], red_points['phish_hints'], color='red')
        plt.scatter(blue_points['domain_age'], blue_points['phish_hints'], color='green')

        if len(global_da_ph)!=0:
            
            ccccc=[]
            for i in answer[:-1].split(','):
                if i.startswith('Le'):
                    ccccc.append('green')
                else:
                    ccccc.append('red')

            # if answer.split(',')[0]=='Legitimate':
            #     ccccc='green'
            # else:
            #     ccccc='red'
            x_coords, y_coords = zip(*global_da_ph)
            plt.scatter(x_coords, y_coords,color=ccccc , s=100, edgecolor='black', zorder=5)

        plt.title('scatter plot')
        plt.xlabel('domain_age')
        plt.ylabel('phish_hints')
        # plt.legend(title="Status")
        plt.legend(title="Status",handles=[white_circle, red_circle, green_circle])
        return plt.gcf() 


def greet(aa):
    df_for_test=pd.DataFrame(columns=['length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at',
       'nb_qm', 'nb_and', 'nb_eq', 'nb_slash', 'nb_colon', 'nb_semicolumn',
       'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token',
       'ratio_digits_url', 'ratio_digits_host', 'tld_in_path',
       'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains',
       'prefix_suffix', 'shortening_service', 'nb_external_redirection',
       'length_words_raw', 'shortest_word_host', 'shortest_word_path',
       'longest_words_raw', 'longest_word_host', 'longest_word_path',
       'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints',
       'domain_in_brand', 'brand_in_subdomain', 'brand_in_path',
       'suspecious_tld', 'statistical_report', 'nb_hyperlinks',
       'ratio_inthyperlinks', 'ratio_exthyperlinks', 'nb_extcss',
       'ratio_extredirection', 'external_favicon', 'links_in_tags',
       'ratio_intmedia', 'ratio_extmedia', 'popup_window', 'safe_anchor',
       'empty_title', 'domain_in_title', 'domain_with_copyright',
       'whois_registered_domain', 'domain_registration_length', 'domain_age',
       'web_traffic', 'dns_record', 'google_index', 'page_rank'])
    df_for_test.loc[0]=eval(aa)

    with open('minMaxScalerForTestingData.pkl','rb') as f:
        scaler_objects=pickle.load(f)

    for i in list(scaler_objects.keys()):
        if i[0] in df_for_test.columns:
            df_for_test[i[0]]=scaler_objects[i].transform(df_for_test[i[0]].values.reshape(-1,1))

    with open('all_models.pkl', 'rb') as file:
        all_models = pickle.load(file)

    def predict_all_models(data):
        if all_models['SVM'].predict([data])[0] ==1:
            return 'Legitimate'
        return 'Phishing'    
    return predict_all_models(df_for_test.iloc[0].tolist()) 

def combined_interface(url=None):
    if url is None:
        greeting = "Please enter a feature vector to analyze."  
    else:
        greeting = greet(url)  
    return visualisations(), greeting

def load_on_start():
    global global_pr_qm,global_da_ph,answer
    answer=''
    global_pr_qm=[]
    global_da_ph=[]
    return visualisations2(),visualisations(), "Please enter a feature vector to analyze."

def upload_file(files):
    global global_da_ph,global_pr_qm
    global answer
    try:
        a=pd.read_csv(files)
        # print(a.columns)
    
        for i in range(len(a)):
            global_da_ph.append((a.loc[i,'domain_age'],a.loc[i,'phish_hints']))
            global_pr_qm.append((a.loc[i,'nb_qm'],a.loc[i,'page_rank']))
            
            answer=answer+greet(str(a.loc[i].values.tolist()))+','
    except Exception as e:
        print('Error:', e)
        return "Failed to process the file."
        

    print(files)
    # visualisations2()
    return answer[:-1],visualisations2(),visualisations()
    

with gr.Blocks() as demo:
    # plot_output2 = None
    with gr.Row():
        with gr.Column():
            # file_output = gr.File()
            text_output2 = gr.Textbox(label='Status of Website')
            upload_button=gr.UploadButton(label='upload csv',file_count="single")
            # plot_output2= gr.Plot()
            
            gr.DownloadButton("Download Input Template", value=pathlib.Path('FeatureVector.csv'))
            # url_input = gr.Textbox(label="Feature Vector")
            # submit_btn = gr.Button("Submit")

    with gr.Row():
        text_output = gr.Textbox(label='Status of Website',visible=False)

    gr.Markdown('# Visualisations')
    with gr.Row():
        with gr.Column():
            plot_output = gr.Plot()
        with gr.Column():
            gr.Markdown("## We can clearly understand that as the number of question marks are increasing (indicating more queries in URL), there are a lot of phishing sites indicating that phishing websites generally have more number of query parameters in a URL when compared to a legitimate one.") 
    with gr.Row():
        with gr.Column():
            plot_output2= gr.Plot()
            upload_button.upload(upload_file, upload_button,[text_output2,plot_output2,plot_output])
        with gr.Column():
            gr.Markdown("## We can see that as the age of the domain is increasing there are relatively less phishing site which indicates that Older domains might have fewer phishing hints due to established legitimacy over time. We can also interpret this as websites which are just created have more phish_hints potentially being a phishing website.") 
    # submit_btn.click(
    #     fn=lambda url: (visualisations2(),visualisations(), greet(url)),
    #     inputs=url_input,
    #     outputs=[plot_output, plot_output2,text_output]
    # )


    demo.load(load_on_start, inputs=None, outputs=[plot_output2,plot_output, text_output])


demo.launch()

# import pandas as pd

# df_for_test=pd.DataFrame(columns=['length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at',
#        'nb_qm', 'nb_and', 'nb_eq', 'nb_slash', 'nb_colon', 'nb_semicolumn',
#        'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token',
#        'ratio_digits_url', 'ratio_digits_host', 'tld_in_path',
#        'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains',
#        'prefix_suffix', 'shortening_service', 'nb_external_redirection',
#        'length_words_raw', 'shortest_word_host', 'shortest_word_path',
#        'longest_words_raw', 'longest_word_host', 'longest_word_path',
#        'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints',
#        'domain_in_brand', 'brand_in_subdomain', 'brand_in_path',
#        'suspecious_tld', 'statistical_report', 'nb_hyperlinks',
#        'ratio_inthyperlinks', 'ratio_exthyperlinks', 'nb_extcss',
#        'ratio_extredirection', 'external_favicon', 'links_in_tags',
#        'ratio_intmedia', 'ratio_extmedia', 'popup_window', 'safe_anchor',
#        'empty_title', 'domain_in_title', 'domain_with_copyright',
#        'whois_registered_domain', 'domain_registration_length', 'domain_age',
#        'web_traffic', 'dns_record', 'google_index', 'page_rank'])

# df_for_test.to_csv('./FeatureVector.csv')

# df=pd.read_csv('FeatureVector.csv')
# df.loc[0]=[68.0, 40.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0,2.0, 0.0, 0.0, 1.0, 0.029411765, 0.05, 0.0, 0.0, 0.0, 3.0, 0.0,0.0, 0.0, 7.0, 5.0, 3.0, 17.0, 17.0, 6.0, 6.857142857, 10.33333333,4.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0,100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 1.0, 0.0, 0.0, 2381.0, 0, 0.0,0.0, 1.0, 2.0]
# df.loc[1]=[55.0, 15.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 0.0, 1.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,6.0, 3.0, 4.0, 11.0, 7.0, 11.0, 6.333333333, 5.0, 7.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 102.0, 0.470588235, 0.529411765, 0.0,0.537037037, 0.0, 76.47058824, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 224.0, 8175.0, 8725.0, 0.0, 0.0, 6.0,]
# df.to_csv('test.csv',index=False)
