import os
import re
import csv
import time
import json
import requests
import datetime
from collections import defaultdict
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

# === KONFIGURACE CEST ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../structures"))
PDB_DIR = os.path.join(BASE_DIR, "all_pdbs")
os.makedirs(PDB_DIR, exist_ok=True)

LOG_FILE = os.path.join(BASE_DIR, "pipeline_50k_no_org_filter.log")
# Pokud používáte název dataset_metadata.tsv, ponechte takto:
METADATA_FILE = os.path.join(BASE_DIR, "dataset_metadata.tsv")
DATASET_CACHE_FILE = os.path.join(BASE_DIR, "master_dataset_cache.json")

TARGET_PER_CLASS = {
    'B12': 2500,
    'acetyl-CoA': 6500,
    'FAD': 10000,
    'NAD': 14000,
    'ATP': 17000
}

MAX_PER_EC = 40
MAX_PER_ORG_EC = 1
MAX_UNASSIGNED_PER_ORG = 1
MIN_LENGTH = 60
MAX_LENGTH = 1400
NUM_WORKERS = 16

HEADERS = {
    "User-Agent": "EnzymeCofactorResearch/6.0 (structural screening; contact: user@domain.cz)"
}

def write_log(msg, print_to_console=True):
    if print_to_console:
        print(msg)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if msg.startswith("=") or msg.strip() == "":
            f.write(f"{msg}\n")
        else:
            f.write(f"[{timestamp}] {msg}\n")

SWISSPROT_QUERIES = {
    'B12': (
        'reviewed:true AND ('
        'keyword:KW-0171 OR '
        'cc_cofactor_chebi:"CHEBI:176843" OR '
        'cc_cofactor_chebi:"CHEBI:18409" OR '
        'cc_cofactor_chebi:"CHEBI:25281"'
        ')'
    ),
    'acetyl-CoA': (
        'reviewed:true AND ('
        'keyword:KW-0008 OR '
        'cc_cofactor_chebi:"CHEBI:15351" OR '
        'ft_binding:"acetyl-CoA"'
        ')'
    ),
    'FAD': (
        'reviewed:true AND ('
        'keyword:KW-0274 OR '
        'cc_cofactor_chebi:"CHEBI:57618" OR '
        'ft_binding:FAD'
        ')'
    ),
    'NAD': (
        'reviewed:true AND ('
        'keyword:KW-0524 OR '
        'cc_cofactor_chebi:"CHEBI:57540" OR '
        'cc_cofactor_chebi:"CHEBI:57945" OR '
        'ft_binding:NAD'
        ')'
    ),
    'ATP': (
        'reviewed:true AND ('
        'keyword:KW-0067 OR '
        'cc_cofactor_chebi:"CHEBI:30616" OR '
        'ft_binding:ATP'
        ')'
    )
}

TREMBL_QUERIES = {
    'B12': (
        'reviewed:false AND fragment:false AND ('
        'keyword:KW-0171 OR '
        'xref:interpro-IPR001214 OR '
        'xref:interpro-IPR006158 OR '
        'xref:interpro-IPR003711'
        ')'
    ),
    'acetyl-CoA': (
        'reviewed:false AND fragment:false AND ('
        'keyword:KW-0008 OR '
        'xref:interpro-IPR000182 OR '
        'xref:interpro-IPR016181 OR '
        'ec:2.3.1.*'
        ')'
    ),
    'FAD': (
        'reviewed:false AND fragment:false AND ('
        'keyword:KW-0274 OR '
        'xref:interpro-IPR001327 OR '
        'xref:interpro-IPR003952'
        ')'
    ),
    'NAD': (
        'reviewed:false AND fragment:false AND ('
        'keyword:KW-0524 OR '
        'xref:interpro-IPR001709 OR '
        'xref:interpro-IPR027417'
        ')'
    ),
    'ATP': (
        'reviewed:false AND fragment:false AND ('
        'keyword:KW-0067 OR '
        'ec:2.7.1.* OR ec:2.7.11.*'
        ')'
    )
}

def robust_request(url, max_retries=5, backoff_factor=2, timeout=60):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff_factor ** attempt)
            else:
                return resp
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            time.sleep(backoff_factor ** attempt)
    return None

def stream_swissprot(query):
    fields = "accession,organism_id,length,ec"
    url = f"https://rest.uniprot.org/uniprotkb/stream?query={quote(query)}&format=tsv&fields={fields}"
    resp = robust_request(url, timeout=120)
    if not resp or resp.status_code != 200:
        return []
        
    lines = resp.text.strip().splitlines()
    if not lines:
        return []
        
    reader = csv.reader(lines, delimiter='\t')
    headers = [h.strip().lower() for h in next(reader)]
    
    acc_idx = next((i for i, h in enumerate(headers) if "entry" in h or "accession" in h), 0)
    org_idx = next((i for i, h in enumerate(headers) if "organism" in h), 1)
    len_idx = next((i for i, h in enumerate(headers) if "length" in h), 2)
    ec_idx = next((i for i, h in enumerate(headers) if "ec" in h), 3)
    
    parsed = []
    for row in reader:
        if not row or len(row) <= max(acc_idx, len_idx):
            continue
        length = int(row[len_idx]) if row[len_idx].isdigit() else 0
        if not (MIN_LENGTH <= length <= MAX_LENGTH):
            continue
        ec_val = row[ec_idx].strip() if len(row) > ec_idx else ""
        parsed.append({
            "accession": row[acc_idx].strip(),
            "taxon_id": row[org_idx].strip() if len(row) > org_idx else "unknown",
            "length": length,
            "ec": ec_val if ec_val else "unassigned",
            "source": "Swiss-Prot"
        })
    return parsed

def fetch_trembl_paged(query, needed_count, org_ec_counts, ec_counts, unassigned_org_counts):
    fields = "accession,organism_id,length,ec"
    url = f"https://rest.uniprot.org/uniprotkb/search?query={quote(query)}&fields={fields}&format=tsv&size=500"
    added_entries = []
    
    while url and len(added_entries) < needed_count:
        resp = robust_request(url, timeout=45)
        if not resp or resp.status_code != 200:
            break
            
        lines = resp.text.strip().splitlines()
        if len(lines) <= 1:
            break
            
        reader = csv.reader(lines, delimiter='\t')
        headers = [h.strip().lower() for h in next(reader)]
        
        acc_idx = next((i for i, h in enumerate(headers) if "entry" in h or "accession" in h), 0)
        org_idx = next((i for i, h in enumerate(headers) if "organism" in h), 1)
        len_idx = next((i for i, h in enumerate(headers) if "length" in h), 2)
        ec_idx = next((i for i, h in enumerate(headers) if "ec" in h), 3)
        
        for row in reader:
            if not row or len(row) <= max(acc_idx, len_idx):
                continue
            length = int(row[len_idx]) if row[len_idx].isdigit() else 0
            if not (MIN_LENGTH <= length <= MAX_LENGTH):
                continue
                
            acc = row[acc_idx].strip()
            org = row[org_idx].strip() if len(row) > org_idx else "unknown"
            ec_val = row[ec_idx].strip() if len(row) > ec_idx else ""
            primary_ec = ec_val.split(";")[0].strip() if ec_val else "unassigned"
            
            if primary_ec != "unassigned":
                if ec_counts[primary_ec] >= MAX_PER_EC:
                    continue
                if org != "unknown" and org_ec_counts[(org, primary_ec)] >= MAX_PER_ORG_EC:
                    continue
            else:
                if org != "unknown" and unassigned_org_counts[org] >= MAX_UNASSIGNED_PER_ORG:
                    continue
                    
            if primary_ec != "unassigned":
                ec_counts[primary_ec] += 1
                if org != "unknown":
                    org_ec_counts[(org, primary_ec)] += 1
            else:
                if org != "unknown":
                    unassigned_org_counts[org] += 1
                    
            added_entries.append({
                "accession": acc,
                "taxon_id": org,
                "length": length,
                "ec": ec_val if ec_val else "unassigned",
                "source": "TrEMBL"
            })
            
            if len(added_entries) >= needed_count:
                break
                
        link_header = resp.headers.get("Link") or resp.headers.get("link")
        if not link_header:
            break
        match = re.search(r'<([^>]+)>;\s*rel=["\']?next["\']?', link_header, re.IGNORECASE)
        url = match.group(1) if match else None
        time.sleep(0.1)
        
    return added_entries

def collect_cofactor_dataset(cofactor, target_count):
    write_log(f"\n--- Sběr dat: {cofactor} (Cílová kvóta: {target_count}) ---")
    sp_data = stream_swissprot(SWISSPROT_QUERIES[cofactor])
    write_log(f"  [Swiss-Prot] Staženo {len(sp_data)} surových záznamů.")
    
    selected = {}
    ec_counts = defaultdict(int)
    org_ec_counts = defaultdict(int)
    unassigned_org_counts = defaultdict(int)
    
    for item in sp_data:
        if len(selected) >= target_count:
            break
        org = item["taxon_id"]
        primary_ec = item["ec"].split(";")[0].strip() if item["ec"] != "unassigned" else "unassigned"
        
        if primary_ec != "unassigned":
            if ec_counts[primary_ec] >= MAX_PER_EC:
                continue
            if org != "unknown" and org_ec_counts[(org, primary_ec)] >= MAX_PER_ORG_EC:
                continue
        else:
            if org != "unknown" and unassigned_org_counts[org] >= MAX_UNASSIGNED_PER_ORG:
                continue
                
        if primary_ec != "unassigned":
            ec_counts[primary_ec] += 1
            if org != "unknown":
                org_ec_counts[(org, primary_ec)] += 1
        else:
            if org != "unknown":
                unassigned_org_counts[org] += 1
                
        selected[item["accession"]] = item
        
    write_log(f"  [Swiss-Prot] Vybráno po funkčním pre-filteru: {len(selected)} enzymů.")
    
    if len(selected) < target_count:
        needed = target_count - len(selected)
        write_log(f"  [TrEMBL] Doplňuji {needed} zástupců z unreviewed databáze...")
        trembl_data = fetch_trembl_paged(
            TREMBL_QUERIES[cofactor], 
            needed, 
            org_ec_counts, 
            ec_counts, 
            unassigned_org_counts
        )
        for item in trembl_data:
            selected[item["accession"]] = item
        write_log(f"  [TrEMBL] Doplněno: +{len(trembl_data)} enzymů.")
        
    write_log(f"  🏁 Celkem pro {cofactor}: {len(selected)} enzymů.")
    return selected

def download_alphafold_pdb(acc, out_dir):
    """Stáhne PDB z AlphaFold DB pouze v případě, že ještě není na disku."""
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
    resp = robust_request(api_url, timeout=15)
    
    if not resp or resp.status_code != 200:
        return acc, [], "NOT_FOUND" if (resp and resp.status_code == 404) else "ERROR"
        
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        return acc, [], "ERROR"
        
    saved_pdbs = []
    for idx, fragment in enumerate(payload, start=1):
        pdb_url = fragment.get("pdbUrl")
        if not pdb_url:
            continue
            
        frag_resp = robust_request(pdb_url, timeout=25)
        if not frag_resp or frag_resp.status_code != 200:
            continue
            
        filename = f"{acc}.pdb" if len(payload) == 1 else f"{acc}_F{idx}.pdb"
        full_path = os.path.join(out_dir, filename)
        
        with open(full_path, "wb") as f:
            f.write(frag_resp.content)
        saved_pdbs.append(filename)
        
    return acc, saved_pdbs, "DOWNLOADED" if saved_pdbs else "FAILED"

def write_tsv_entry(f_out, acc, pdbs, data):
    cofactors_str = ";".join(sorted(list(data["cofactors"])))
    if pdbs:
        for pdb_file in pdbs:
            f_out.write(f"{acc}\t{pdb_file}\t{cofactors_str}\t{data['ec']}\t{data['length']}\t{data['source']}\n")
    else:
        f_out.write(f"{acc}\tNONE\t{cofactors_str}\t{data['ec']}\t{data['length']}\t{data['source']}\n")
    f_out.flush()

# === HLAVNÍ BĚH ===
if __name__ == "__main__":
    write_log("=" * 65)
    write_log("START: STAHY S OCHRANOU PROTI PŘERUŠENÍ A DOPLNĚNÍM METADAT")
    write_log("=" * 65)

    # 1. KROK: Načtení nebo vytvoření master_datasetu
    master_dataset = {}
    if os.path.exists(DATASET_CACHE_FILE):
        write_log(f"Načítám uložený seznam proteinů z cache: {DATASET_CACHE_FILE}")
        with open(DATASET_CACHE_FILE, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            for acc, entry in cached_data.items():
                entry["cofactors"] = set(entry["cofactors"])
                master_dataset[acc] = entry
        write_log(f"Úspěšně načteno {len(master_dataset)} proteinů z JSON cache.")
    else:
        write_log("Cache nenalezena, spouštím dotazy na UniProt...")
        for cof, target_cnt in TARGET_PER_CLASS.items():
            subset = collect_cofactor_dataset(cof, target_cnt)
            for acc, entry in subset.items():
                if acc not in master_dataset:
                    master_dataset[acc] = entry
                    master_dataset[acc]["cofactors"] = {cof}
                else:
                    master_dataset[acc]["cofactors"].add(cof)
                    
        # Uložení do JSONu pro příští běhy
        write_log(f"Ukládám sestavený dataset do cache: {DATASET_CACHE_FILE}")
        serializable = {k: {**v, "cofactors": list(v["cofactors"])} for k, v in master_dataset.items()}
        with open(DATASET_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

    # 2. KROK: Analýza existujících souborů na disku a existujícího TSV
    write_log(f"\n📂 Indexuji lokální soubory v: {PDB_DIR}")
    disk_pdbs = defaultdict(list)
    for fname in os.listdir(PDB_DIR):
        if fname.endswith(".pdb"):
            fpath = os.path.join(PDB_DIR, fname)
            # Ignorovat poškozené nebo nulové soubory
            if os.path.getsize(fpath) > 0:
                acc_part = fname.split(".")[0].split("_")[0]
                disk_pdbs[acc_part].append(fname)
    write_log(f"  • Nalezeno platných PDB na disku pro {len(disk_pdbs)} proteinů.")

    logged_accs = set()
    tsv_exists = os.path.exists(METADATA_FILE)
    if tsv_exists:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            headers = next(reader, None)
            for row in reader:
                if row and len(row) > 0:
                    logged_accs.add(row[0].strip())
        write_log(f"  • V TSV souboru {os.path.basename(METADATA_FILE)} již existuje {len(logged_accs)} záznamů.")

    # Otevření TSV pro append režim
    meta_fp = open(METADATA_FILE, "a", encoding="utf-8")
    if not tsv_exists or os.path.getsize(METADATA_FILE) == 0:
        meta_fp.write("uniprot_id\tpdb_file\tcofactors\tec\tlength\tsource\n")
        meta_fp.flush()

    # 3. KROK: Záchrana dat z disku, která chybí v TSV
    missing_from_tsv_but_on_disk = 0
    to_download = []

    for acc, data in master_dataset.items():
        if acc in logged_accs:
            continue  # Již kompletně zapsáno v TSV
            
        if acc in disk_pdbs:
            # Máme PDB na disku, ale chybělo v TSV -> rovnou zapíšeme bez dotazování sítě
            write_tsv_entry(meta_fp, acc, disk_pdbs[acc], data)
            logged_accs.add(acc)
            missing_from_tsv_but_on_disk += 1
        else:
            # Chybí na disku i v TSV -> nutno stáhnout
            to_download.append(acc)

    if missing_from_tsv_but_on_disk > 0:
        write_log(f"  ⚡ Zpětně doplněno do TSV bez stahování: {missing_from_tsv_but_on_disk} proteinů z disku.")

    write_log(f"\nZbývá reálně stáhnout z AlphaFoldu: {len(to_download)} / {len(master_dataset)} proteinů.")

    # 4. KROK: Paralelní stahování zbývajících položek s průběžným zápisem
    if to_download:
        write_log(f"🚀 Spouštím paralelní stahování ({NUM_WORKERS} workerů)...")
        stats = defaultdict(int)
        done_counter = 0
        total_needed = len(to_download)

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_acc = {
                executor.submit(download_alphafold_pdb, acc, PDB_DIR): acc 
                for acc in to_download
            }
            
            for future in as_completed(future_to_acc):
                acc = future_to_acc[future]
                done_counter += 1
                
                try:
                    _, pdbs, status = future.result()
                except Exception:
                    pdbs, status = [], "ERROR"

                stats[status] += 1
                data = master_dataset[acc]
                
                # Okamžitý zápis na disk po každém dokončeném stažení
                write_tsv_entry(meta_fp, acc, pdbs, data)

                if done_counter % 250 == 0 or done_counter == total_needed:
                    write_log(f"  [{done_counter:5d}/{total_needed:5d}] Staženo: {stats['DOWNLOADED']:5d} | 404/Chyba: {stats['NOT_FOUND'] + stats['FAILED'] + stats['ERROR']:4d}")

    meta_fp.close()
    write_log("\n" + "=" * 65)
    write_log("HOTOVO - Všechna data a metadata jsou kompletní a synchronizovaná.")
    write_log("=" * 65)