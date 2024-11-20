import json
import os

import pandas as pd
from dash import Dash, html, dcc, dash_table
import plotly.express as px

if __name__ == '__main__':
    path_folder = './results/test_res'
    res_complets = []
    partie_train = False
    for file in os.listdir(path_folder):
        if file.startswith('zout_ranllm'):
            res_act = {'info_run': {}, 'train': [], 'eval': {}}
            filename = os.path.join(path_folder, file)
            with open(filename) as f:
                run_id = file.split('.')[0].split('_')[-1]
                res_act['info_run']['run_id'] = run_id
                partie_eval = False
                for line in f:
                    # Chargement des données des arguments
                    if line.startswith('Namespace'):
                        infos = line.strip().split('(')[1][:-1]
                        for arg in infos.split(','):
                            arg_left, arg_right = arg.split('=')
                            res_act['info_run'][arg_left.strip()] = arg_right.replace('"',"").replace("'","").strip()
                            
                    # Chargement des données de train
                    elif line.startswith("{'loss'"):
                        partie_train = True
                        line = line.strip()[1:-1].replace('"',"").replace("'","").strip()
                        dict_res_act = {}
                        for metric in line.split(','):
                            arg_left, arg_right = metric.split(':')
                            dict_res_act[arg_left.strip()] = float(arg_right.strip())
                        res_act['train'].append(dict_res_act)

                    # Chargement des données d'entrainement
                    elif line.startswith("AVG Jaccard"):
                        partie_eval = True
                        res_act['eval']['AvgJSim'] = float(line.strip().split(':')[1])
                    elif partie_eval:
                        if line.startswith("Length of dataset"): res_act['eval']['length_dataset'] = int(line.strip().split(':')[1])
                        elif line.startswith("NB Jaccard Similarity"): res_act['eval']['NBJSimNo0'] = int(line.strip().split(':')[1])
                        elif line.startswith("{'recall@1'"):
                            line = line.strip()[1:-1].replace('"',"").replace("'","").strip()
                            for metric in line.split(','):
                                arg_left, arg_right = metric.split(':')
                                res_act['eval'][arg_left.strip()] = float(arg_right.strip())

            res_complets.append(res_act)
    #print(res_complets)


    app = Dash()
    app.layout = [html.H1('Résultats des expés')]
    
    sorted_data_eval = {'model_name': [], 'learning_rate': [], 'epoch':[], 'recall@1': [], 'AvgJSim': [], '%JSimNo0':[]}
    for exp in res_complets:
        if partie_train:
            data_dict = {'loss': [], 'grad_norm': [], 'learning_rate': [], 'epoch': []}
            for data in exp['train']:
                data_dict['loss'].append(data['loss'])
                data_dict['grad_norm'].append(data['grad_norm'])
                data_dict['learning_rate'].append(data['learning_rate'])
                data_dict['epoch'].append(data['epoch'])
            df_train = pd.DataFrame.from_dict(data_dict)
        
        data_eval = {'name': [], 'value': []}
        for key in exp['eval']:
            data_eval['name'].append(key)
            data_eval['value'].append(exp['eval'][key])
        df_eval = pd.DataFrame.from_dict(data_eval)

        sorted_data_eval['model_name'].append(exp['info_run']['model_name'])
        sorted_data_eval['learning_rate'].append(exp['info_run']['learning_rate'])
        sorted_data_eval['epoch'].append(exp['info_run']['test_cpt'])
        sorted_data_eval['AvgJSim'].append(exp['eval']['AvgJSim'])
        sorted_data_eval['%JSimNo0'].append(exp['eval']['NBJSimNo0']/exp['eval']['length_dataset'])
        sorted_data_eval['recall@1'].append(exp['eval']['recall@1'])
        print(sorted_data_eval)
            
        app.layout.append(html.Details([
            html.Summary(f"Expé : {exp['info_run']['run_id']} // Modèle : {exp['info_run']['model_name']} // Dataset : {exp['info_run']['dataset']}"),
            html.H3("Details de l'expérience"),
            html.Ul([html.Li(f"{key} : {exp['info_run'][key]}") for key in exp['info_run']]),

            html.H3("Résultats de l'évaluation"),
            dash_table.DataTable(df_eval.to_dict('records'), [{"name": i, "id": i} for i in df_eval.columns])
        ]))
        
        if partie_train:
            app.layout.append(html.Details([
                html.H3("Métriques d'entrainement"),
                dcc.Graph(figure=px.line(df_train, x='epoch', y=['loss', 'grad_norm', 'learning_rate']))
            ]))
        app.layout.append(html.Br())
    
    df_sorted_eval = pd.DataFrame.from_dict(sorted_data_eval)
    df_sorted_eval = df_sorted_eval.sort_values(by=['recall@1', '%JSimNo0', 'AvgJSim'], ascending=False)
    app.layout.append(html.H3("Résultats des expés triés"))
    app.layout.append(dash_table.DataTable(df_sorted_eval.to_dict('records'), [{"name": i, "id": i} for i in df_sorted_eval.columns]))

    app.run(debug=True, port=8352)
                    
