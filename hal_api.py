import requests
import json
import os
import PyPDF2
from io import StringIO, BytesIO
import sys

def get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list, start):
    base_uri = 'https://api.archives-ouvertes.fr/search/?'
    uri = base_uri + "&".join(["q="+"+".join(mots_cles),
                              "wt="+format_retour,
                              "rows="+max_rows,
                              "sort="+tri,
                              "fl="+",".join(field_list)
                              ])
    for filter in filters :
        uri += "&fq="+filter
    
    uri += "&start="+str(start)
    print(uri)
    r = requests.get(uri)
    return json.loads(r.text)

def prompt():
    val = input().lower()
    try:
        ret = val=="y" if val in ("y","n") else ValueError(f"Invalid truth value: {val}")
    except ValueError:
        sys.stdout.write("Please answer with y/n")
        return prompt()
    return ret

if __name__ == '__main__':
    out_folder = './hal_data'
    
    uri_test = 'https://api.archives-ouvertes.fr/search/?q=cyber+juridique&wt=xml&rows=10000&sort=submittedDate_tdate%20desc&fq=language_s:fr&fl=docid,label_s,keyword_s,uri_s,title_s,text_fulltext,text,fulltext_t,fileMain_s'
    base_uri = 'https://api.archives-ouvertes.fr/search/?'
    mots_cles = ["*"]
    format_retour = "json"
    max_rows = "10000"
    pages = 1
    tri = "submittedDate_tdate%20desc"
    filters = ["language_s:fr","docType_s:ART","NOT%20level0_domain_s:shs","submittedDateY_i:2024","publicationDateY_i:2024"]
    field_list = ["docid","title_s","abstract_s","docType_s","level0_domain_s","submittedDate_tdate","fileMain_s"]
    start = 0
    res = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list, start)
    for i in range(1,pages):
        for doc in get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list, start):
            res["response"]["docs"].append(doc)        
    
    print(len(res["response"]["docs"]))
    with_docs = [i for i in res["response"]["docs"] if "fileMain_s" in i and "title_s" in i and "abstract_s" in i and "No abstract available" not in i["abstract_s"]]
    print(len(with_docs))

    res = []
    out_file = './hal/hal_extract.json'
    with open(out_file, mode='w', encoding='utf8') as file_write:
        for i, doc in enumerate(with_docs):
            doc_uri = doc["fileMain_s"]
            # r = requests.get(doc_uri)
            # print(doc_uri)
            # print(doc)
            # print(doc["title_s"])
            # print("Garder le document ?", i)
            # save_file = prompt()
            # if save_file and not("<html" in r.text):
            #     try:
            #         memoryFile = BytesIO(r.content)
            #         pdfFile = PyPDF2.PdfReader(memoryFile)
            #         pdf_text = ""
            #         for page in pdfFile.pages:
            #             pdf_text += page.extract_text() + "\n"
            #         doc["full_text"] = pdf_text
            #         #json.dump(doc, file_write, indent=2)
            #         res.append(doc)
            #     except PyPDF2.errors.PdfReadError:
            #         print(doc_uri)
            #         print("err pypdf2")
            ndoc = {"docid": doc["docid"], "title": doc["title_s"][0], "abstract": doc["abstract_s"][0]}
            res.append(ndoc)
        json.dump(res, file_write, indent=2, ensure_ascii=False)
                
    
