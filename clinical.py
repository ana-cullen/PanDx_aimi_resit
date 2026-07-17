import csv
import json

csvfile = open('clinical_information.csv', 'r')
jsonfile = open('patient_sex_map.json', 'w')

fieldnames = ("PANORAMA_patient_id","PANORAMA_study_id","anonymized_study_date","patient_age","patient_sex","scanner","label","level","annotation_tier","lesion_gt_gold,external_source")
reader = csv.DictReader( csvfile, fieldnames)
int_map = { "F": 1, "M": 2}
dict = {}
for row in reader:
    if row["PANORAMA_study_id"] == "PANORAMA_study_id":
        continue
    if row["patient_sex"] == "":
        dict.update({ row["PANORAMA_study_id"]: 0})
    else:
        dict.update({ row["PANORAMA_study_id"]: int_map[row["patient_sex"]]})

out = json.dumps(dict)
jsonfile.write(out)

