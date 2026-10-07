from pathlib import Path
import csv, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
PROCESSED=ROOT/'data'/'processed'
DEST=Path(__file__).resolve().parent/'data'
DEST.mkdir(exist_ok=True)
def records(path):
    p=Path(path)
    if not p.exists(): return []
    with p.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
def load_json(path):
    p=Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
def numeric_rows(rows):
    result=[]
    for row in rows:
        item={}
        for k,v in row.items():
            if v is None or v=='': item[k]=None; continue
            try: item[k]=float(v) if '.' in v or 'e' in v.lower() else int(v)
            except (ValueError,AttributeError): item[k]=v
        result.append(item)
    return result
def dump(name,obj):
    (DEST/name).write_text(json.dumps(obj,separators=(',',':')),encoding='utf-8')
for name,path in {
 'queue.json':OUT/'investigation_queue.csv',
 'features.json':PROCESSED/'provider_features.csv',
 'importance.json':OUT/'global_importance.csv',
 'metrics.json':OUT/'final_test_results.csv',
 'ranking.json':OUT/'topk_ranking.csv'
}.items(): dump(name,numeric_rows(records(path)))
selection=load_json(OUT/'selection.json'); threshold=load_json(OUT/'threshold_choice.json'); audit=load_json(OUT/'audit_raw.json')
dump('meta.json',{'selection':selection,'threshold':threshold,'audit':audit})
print('Dashboard data snapshots generated:', ', '.join(p.name for p in DEST.glob('*.json')))
