#!/usr/bin/env python3
import os
import json
import argparse
import time
import mrcfile
from statistics import mean

def get_pixel_size(mrc_path):
    try:
        with mrcfile.open(mrc_path) as mrc:
            return round(float(mrc.voxel_size['x']), 3)
    except:
        return None

def get_max_image_size(mrc_path):
    try:
        with mrcfile.open(mrc_path) as mrc:
            h = mrc.header
            return int(max(h.nx, h.ny, h.nz))
    except:
        return None

def normalise_prefix(name):
    return name.lower().replace('-', '').replace('_', '')

def extract_emdb_info(data):
    out = {}
    # Administrative dates
    dates = data.get('admin', {}).get('key_dates', {})
    out['deposition_date']  = dates.get('deposition')
    out['header_release']   = dates.get('header_release')
    out['map_release_date'] = dates.get('map_release')

    # Resolution
    sd_list = data.get('structure_determination_list', {}) \
                 .get('structure_determination', [])
    if sd_list:
        ip_list = sd_list[0].get('image_processing', [])
        if ip_list:
            final = ip_list[0].get('final_reconstruction', {})
            res   = final.get('resolution', {})
            out['resolution_value']  = res.get('valueOf_')
            out['resolution_type']   = res.get('res_type')
            out['resolution_method'] = final.get('resolution_method')

    # Sample & macromolecules
    sample = data.get('sample', {})
    out['sample_name'] = sample.get('name', {}).get('valueOf_')
    mac_list = sample.get('macromolecule_list', {}).get('macromolecule', [])
    organisms = set(); macromolecules = []
    for mol in mac_list:
        nm = mol.get('name', {}).get('valueOf_')
        if nm: macromolecules.append(nm)
        org = mol.get('natural_source', {}) \
                 .get('organism', {}) \
                 .get('valueOf_')
        if org: organisms.add(org)
    out['macromolecule_names'] = macromolecules or None
    out['organisms']            = list(organisms) or None

    # Fitted models
    pdb_refs = data.get('crossreferences', {}) \
                   .get('pdb_list', {}) \
                   .get('pdb_reference', [])
    out['fitted_models'] = [p.get('pdb_id') for p in pdb_refs] or None

    # Map parameters
    mp = data.get('map', {})
    contours = mp.get('contour_list', {}).get('contour', [])
    out['recommended_contour_level'] = contours[0].get('level') if contours else None
    dims = mp.get('dimensions', {})
    out['grid_dimensions'] = (
        dims.get('col'), dims.get('row'), dims.get('sec')
    ) if dims else None
    pix = mp.get('pixel_spacing', {})
    out['voxel_size'] = (
        pix.get('x', {}).get('valueOf_'),
        pix.get('y', {}).get('valueOf_'),
        pix.get('z', {}).get('valueOf_')
    )
    stats = mp.get('statistics', {})
    out['statistics'] = {
        'min':  stats.get('minimum'),
        'max':  stats.get('maximum'),
        'mean': stats.get('average'),
        'std':  stats.get('std'),
    } if stats else None

    # Microscopy + magnification
    if sd_list:
        mic_list = sd_list[0] \
            .get('microscopy_list', {}) \
            .get('microscopy', [])
        if mic_list:
            mic = mic_list[0]
            out['imaging_mode']             = mic.get('imaging_mode')
            out['microscopy_instance_type'] = mic.get('instance_type')
            out['microscope']               = mic.get('microscope')
            cal = mic.get('calibrated_magnification')
            nom = mic.get('nominal_magnification')
            out['magnification']            = cal if cal is not None else nom
            out['electron_source']          = mic.get('electron_source')
            out['illumination_mode']        = mic.get('illumination_mode')
            film = mic.get('film_or_detector_model', {})
            out['film_or_detector_model']   = film.get('valueOf_')

    return out

def build_entry_from_folder(folder, suffixes):
    entry = {'directory': folder}
    norm = normalise_prefix(os.path.basename(folder))

    # Required suffix files
    for suf in suffixes:
        pat = normalise_prefix(f"{norm}_{suf}.mrc")
        match = next((f for f in os.listdir(folder)
                      if normalise_prefix(f).startswith(pat)), None)
        if not match:
            return None
        entry[f"{suf}_file"] = os.path.join(folder, match)

    # Map file
    pat = normalise_prefix(f"{norm}.mrc")
    match = next((f for f in os.listdir(folder)
                  if normalise_prefix(f).startswith(pat)), None)
    if not match:
        return None
    mp = os.path.join(folder, match)
    px = get_pixel_size(mp)
    ms = get_max_image_size(mp)
    if px is None or ms is None:
        return None
    entry['map_file']   = mp
    entry['pixel_size'] = px
    entry['max_size']   = ms

    # Sidecar info JSON
    emd = os.path.basename(folder)
    info_path = os.path.join(folder, f"{emd}_info.json")
    if os.path.exists(info_path):
        with open(info_path) as jf:
            info = json.load(jf)
        entry.update(extract_emdb_info(info))
    else:
        print(f"  · warning: no info JSON for {emd}")

    return entry

def scan_all_maps(input_dir, suffixes):
    results = []
    for root, dirs, _ in os.walk(input_dir):
        for d in dirs:
            f = os.path.join(root, d)
            ent = build_entry_from_folder(f, suffixes)
            if ent:
                results.append(ent)
    return results

def load_subset(list_file, subset):
    with open(list_file) as f:
        data = json.load(f)
    if subset == 'train':
        return data.get('train', [])
    if subset == 'test':
        return data.get('test', [])
    return data.get('train', []) + data.get('test', [])

def write_summary(entries, summary_path):
    cols = [
        'directory','pixel_size','max_size','deposition_date','resolution_value',
        'imaging_mode','microscopy_instance_type','microscope',
        'magnification','electron_source','illumination_mode',
        'film_or_detector_model'
    ]
    with open(summary_path, 'w') as w:
        w.write('\t'.join(cols) + '\n')
        for e in entries:
            year = ''
            dd = e.get('deposition_date')
            if dd: year = dd.split('-')[0]
            row = []
            for c in cols:
                if c == 'deposition_date':
                    row.append(year)
                else:
                    v = e.get(c)
                    row.append(str(v) if v is not None else '')
            w.write('\t'.join(row) + '\n')

def compute_stats(entries):
    # Numeric fields
    nums = {}
    for field in ['pixel_size','max_size','resolution_value','magnification']:
        vals = []
        for e in entries:
            v = e.get(field)
            try:
                vals.append(float(v))
            except:
                pass
        if vals:
            nums[field] = {
                'mean': mean(vals),
                'min': min(vals),
                'max': max(vals)
            }
    # Categorical fields
    cats = {}
    for field in [
        'deposition_date','imaging_mode','microscopy_instance_type',
        'microscope','electron_source','illumination_mode',
        'film_or_detector_model'
    ]:
        counts = {}
        for e in entries:
            v = e.get(field)
            if field == 'deposition_date' and v:
                v = v.split('-')[0]
            if v:
                counts[v] = counts.get(v, 0) + 1
        if counts:
            cats[field] = counts
    return {'numeric': nums, 'categorical': cats}

def write_stats(train_entries, test_entries, stats_path):
    stats = {
        'train': compute_stats(train_entries),
        'test':  compute_stats(test_entries)
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Build, enrich EMDB entries; optional summary & stats."
    )
    p.add_argument('--d',      help="Base dir of EMD-*/ folders")
    p.add_argument('--suffix', nargs='+',
                   help="Required file suffixes, e.g. locres angelo1_20A_mask")
    p.add_argument('--list',   help="JSON of {'train':[...],'test':[...]} dirs")
    p.add_argument('--subset', choices=['train','test','both'],
                   default='both', help="Subset for main output")
    p.add_argument('--summary', help="Path for tab-delimited summary.txt")
    p.add_argument('--stats',   help="Path for JSON train/test stats")
    p.add_argument('--o',      required=True, help="Output JSON file")
    args = p.parse_args()

    # Load main entries based on --list/--subset or --d scan
    if args.list:
        dirs = load_subset(args.list, args.subset)
        entries = [e for e in (build_entry_from_folder(d, args.suffix or []) for d in dirs) if e]
    else:
        entries = scan_all_maps(args.d, args.suffix or [])

    # Write enriched JSON
    with open(args.o, 'w') as out:
        json.dump(entries, out, indent=4)
    print(f"Written {len(entries)} entries to {args.o}")

    # Summary
    if args.summary:
        write_summary(entries, args.summary)
        print(f"Summary written to {args.summary}")

    # Stats (requires --list)
    if args.stats and args.list:
        # build train & test separately
        all_data = json.load(open(args.list))
        train_dirs = all_data.get('train', [])
        test_dirs  = all_data.get('test', [])
        train_entries = [e for e in (build_entry_from_folder(d, args.suffix or []) for d in train_dirs) if e]
        test_entries  = [e for e in (build_entry_from_folder(d, args.suffix or []) for d in test_dirs) if e]
        write_stats(train_entries, test_entries, args.stats)
        print(f"Stats written to {args.stats}")
    elif args.stats:
        print("Warning: --stats only supported when --list is provided; skipping stats.")

