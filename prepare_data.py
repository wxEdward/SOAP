import argparse, json, pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--in_csv', required=True)
parser.add_argument('--dialogue_col', default='dialogue')
parser.add_argument('--soap_col', default='soap')
parser.add_argument('--out_jsonl', required=True)
args = parser.parse_args()

df = pd.read_csv(args.in_csv)
with open(args.out_jsonl, 'w', encoding='utf-8') as f:
    for i, row in df.iterrows():
        rec = {'id': str(i), 'dialogue': row[args.dialogue_col]}
        if args.soap_col in df.columns:
            rec['soap'] = row[args.soap_col]
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
print('Wrote', args.out_jsonl)
