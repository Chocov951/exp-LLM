import requests
import json
import os
import PyPDF2
import fitz
from io import StringIO, BytesIO
import sys

def get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list):
    base_uri = 'https://api.archives-ouvertes.fr/search/?'
    uri = base_uri + "&".join(["q="+"+".join(mots_cles),
                              "wt="+format_retour,
                              "rows="+max_rows,
                              "sort="+tri,
                              "fq="+filters,
                              "fl="+",".join(field_list)
                              ])
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
    mots_cles = ["cyber*", "juridique"]
    format_retour = "json"
    max_rows = "10000"
    tri = "submittedDate_tdate%20desc"
    filters = "language_s:fr"
    field_list = ["docid","label_s","keyword_s","uri_s","title_s","text_fulltext","text","fulltext_t","fileMain_s"]
    res = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list)
    
    mots_cles = ["cyber*", "legal*"]
    new_docs = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list)["response"]["docs"]
    new_docs = [i for i in new_docs if i["docid"] not in [j["docid"] for j in res["response"]["docs"]]]
    res["response"]["docs"] += new_docs

    mots_cles = ["cyber*", "droit*"]
    new_docs = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list)["response"]["docs"]
    new_docs = [i for i in new_docs if i["docid"] not in [j["docid"] for j in res["response"]["docs"]]]
    res["response"]["docs"] += new_docs
    
    mots_cles = ["cyber*"]
    filters = "language_s:fr&fq=primaryDomain_s:shs.droit"
    new_docs = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list)["response"]["docs"]
    new_docs = [i for i in new_docs if i["docid"] not in [j["docid"] for j in res["response"]["docs"]]]
    res["response"]["docs"] += new_docs

    mots_cles = ["informatique*"]
    new_docs = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list)["response"]["docs"]
    new_docs = [i for i in new_docs if i["docid"] not in [j["docid"] for j in res["response"]["docs"]]]
    res["response"]["docs"] += new_docs

    mots_cles = ["RGPD*"]
    new_docs = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list)["response"]["docs"]
    new_docs = [i for i in new_docs if i["docid"] not in [j["docid"] for j in res["response"]["docs"]]]
    res["response"]["docs"] += new_docs

    mots_cles = ["gouvernance*"]
    new_docs = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list)["response"]["docs"]
    new_docs = [i for i in new_docs if i["docid"] not in [j["docid"] for j in res["response"]["docs"]]]
    res["response"]["docs"] += new_docs

    mots_cles = ["compliance*"]
    new_docs = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list)["response"]["docs"]
    new_docs = [i for i in new_docs if i["docid"] not in [j["docid"] for j in res["response"]["docs"]]]
    res["response"]["docs"] += new_docs

    mots_cles = ["données+personnelles*"]
    new_docs = get_api_res_from_param(mots_cles, format_retour, max_rows, tri, filters, field_list)["response"]["docs"]
    new_docs = [i for i in new_docs if i["docid"] not in [j["docid"] for j in res["response"]["docs"]]]
    res["response"]["docs"] += new_docs
    
    print(len(res["response"]["docs"]))
    with_docs = [i for i in res["response"]["docs"] if "fileMain_s" in i]
    print(len(with_docs))

    # on vire les thèses :
    with_docs = [i for i in with_docs if not(i["fileMain_s"].startswith("https://theses"))]
    print(len(with_docs))
    print(with_docs)

    res = []
    out_file = './hal/hal_extract.json'
    with open(out_file, mode='w', encoding='utf8') as file_write:
        for i, doc in enumerate(with_docs):
            new_folder = os.path.join(out_folder, doc["docid"])
            #os.makedirs(new_folder)
            json_file_path = os.path.join(new_folder, "info.json")
            #with open(json_file_path, 'w') as fp:
            #    json.dump(doc, fp)
            doc_uri = doc["fileMain_s"]
            r = requests.get(doc_uri)
            print(doc_uri)
            print(doc)
            print(doc["title_s"])
            print("Garder le document ?", i)
            save_file = prompt()
            if save_file and not("<html" in r.text):
                try:
                    memoryFile = BytesIO(r.content)
                    """pdfFile = PyPDF2.PdfReader(memoryFile)
                    pdf_text = ""
                    for page in pdfFile.pages:
                        pdf_text += page.extract_text() + "\n"
                    print(pdf_text)"""
                    pdf_file_fitz = fitz.open("pdf", memoryFile)
                    pdf_text = ""
                    for page in pdf_file_fitz:
                        pdf_text += page.get_text()
                    doc["full_text"] = pdf_text
                    #json.dump(doc, file_write, indent=2)
                    res.append(doc)
                except PyPDF2.errors.PdfReadError:
                    print(doc_uri)
                    print("err pypdf2")
                    #print(r.content)
                except fitz.fitz.FileDataError:
                    print(doc_uri)
                    print("err fitz")
        json.dump(res, file_write, indent=2, ensure_ascii=False)
                
    
