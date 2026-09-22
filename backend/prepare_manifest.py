import json
from pathlib import Path
from collections import Counter

backend = Path(__file__).resolve().parent

SPECIALIST_NAMES = {
    'SINGLE_IMAGE_VQA',
    'GROUNDING_CAPTIONING',
    'CHANGE_ANALYSIS',
    'OPTICAL_SAR_ANALYSIS',
}

manifests_info = [
    {
        'file': backend / 'router_manifest_cdvqa.jsonl',
        'type': 'cdvqa',
        'fix_spec': lambda specs: specs,
        'fix_img': lambda imgs: [p.replace('/kaggle/working/', '').replace('\\', '/') for p in imgs]
    },
    {
        'file': backend / 'router_manifest_dev2.jsonl',
        'type': 'vrsbench',
        'fix_spec': lambda specs: specs,
        'fix_img': lambda imgs: [('satquery_vrsbench_images/' + Path(p).name).replace('\\', '/') for p in imgs]
    },
    {
        'file': backend / 'router_manifest.jsonl',
        'type': 'satquery',
        'fix_spec': lambda specs: specs,
        'fix_img': lambda imgs: [('satquery_cached_images/' + Path(p).name).replace('\\', '/') for p in imgs]
    },
    {
        'file': backend / 'rsvqa_router_manifest (1).jsonl',
        'type': 'rsvqa',
        'fix_spec': lambda specs: ['SINGLE_IMAGE_VQA' if s == 'OPTICAL_VQA' else s for s in specs],
        'fix_img': lambda imgs: [('rsvqa_cached_images/' + Path(p).name).replace('\\', '/') for p in imgs]
    }
]

combined = []
seen_ids = {}

for m in manifests_info:
    count = 0
    m_type = m['type']
    with open(m['file'], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            
            raw_id = str(row.get('id', f"{m_type}_{count}"))
            if raw_id in seen_ids:
                seen_ids[raw_id] += 1
                sample_id = f"{raw_id}_{seen_ids[raw_id]}"
            else:
                seen_ids[raw_id] = 1
                sample_id = raw_id
                
            query = str(row['query'])
            images = m['fix_img'](row['images'])
            modalities = [str(x) for x in row['modalities']]
            specialists = m['fix_spec'](row['required_specialists'])
            use_general = bool(row.get('use_general', False))
            
            # Check image existence
            for p in images:
                full_p = backend / p
                if not full_p.exists():
                    raise FileNotFoundError(f"{sample_id}: Image not found: {full_p}")
            
            # Check specialist names
            unknown = set(specialists) - SPECIALIST_NAMES
            if unknown:
                raise ValueError(f"{sample_id}: Unknown specialists: {unknown}")
                
            entry = {
                'id': sample_id,
                'query': query,
                'images': images,
                'modalities': modalities,
                'required_specialists': specialists,
                'use_general': use_general
            }
            combined.append(entry)
            count += 1
    print(f"Processed {count} from {m['file'].name}")

print(f"\nTotal combined samples: {len(combined)}")
print(f"Unique sample IDs: {len(set(x['id'] for x in combined))}")
print("Specialists distribution:", Counter(tuple(x['required_specialists']) for x in combined))

# Write combined manifest
out_path = backend / 'router_manifest_combined.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for item in combined:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Successfully wrote: {out_path} ({out_path.stat().st_size} bytes)")
