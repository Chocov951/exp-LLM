import json
from langdetect import detect, DetectorFactory

# Fixer le seed pour des résultats reproductibles
DetectorFactory.seed = 0

# Charger les fichiers JSON
with open('hal/hal_extract_24.json', 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

with open('hal/hal_extract_21-23.json', 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# Parcourir et traiter les documents de data1
for doc in data1:
    abstract = doc.get('abstract', '')
    if abstract:
        try:
            language = detect(abstract)
            if language == 'en':
                doc['en'] = True
        except:
            print(f"Error detecting language for doc {doc.get('docid')}")

# Parcourir et traiter les documents de data2
for doc in data2:
    abstract = doc.get('abstract', '')
    if abstract:
        try:
            language = detect(abstract)
            if language == 'en':
                doc['en'] = True
        except:
            print(f"Error detecting language for doc {doc.get('docid')}")

# Enregistrer les modifications dans les fichiers JSON d'origine
with open('hal/hal_extract_24.json', 'w', encoding='utf-8') as f1:
    json.dump(data1, f1, ensure_ascii=False, indent=4)
with open('hal/hal_extract_21-23.json', 'w', encoding='utf-8') as f2:
    json.dump(data2, f2, ensure_ascii=False, indent=4)