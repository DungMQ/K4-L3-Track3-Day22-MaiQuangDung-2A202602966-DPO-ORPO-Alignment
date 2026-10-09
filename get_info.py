import json

nb = json.load(open('colab/Lab22_DPO_T4.ipynb', 'r', encoding='utf-8'))
for c in nb['cells']:
  if c['cell_type'] == 'code':
    for o in c.get('outputs', []):
      t = ''.join(o.get('text', []))
      if 'dài hơn' in t: 
          print('NB2 length:', t.strip().split('\n')[-1])
      if '"sanity_accuracy":' in t: 
          print('JSON:\n', t)
