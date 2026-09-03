import requests
import os 
from urllib.parse import quote
import time
import json
import re
import datetime
from collections import defaultdict
from Bio.PDB import PDBParser, PDBIO

# Cesta k ukládání struktur
structures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../structures"))
os.makedirs(structures_dir, exist_ok=True)
log_file = os.path.join(structures_dir, "run_log.txt")

def write_log(msg, print_to_console=True, write_to_file=True):
    """Vypíše zprávu do konzole a zároveň ji uloží do logu s časovým razítkem."""
    if print_to_console:
        print(msg)
    if write_to_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            if msg.startswith("=") or msg.strip() == "":
                f.write(f"{msg}\n")
            else:
                f.write(f"[{timestamp}] {msg}\n")

write_log("="*60)
write_log("SPUŠTĚNÍ PIPELINE PRO STAŽENÍ DIVERZIFIKOVANÝCH STRUKTUR")
write_log("="*60)
write_log(f"Cílová složka: {structures_dir}")

# Robustní vyhledávací dotazy pro jednotlivé kofaktory v UniProtKB
QUERIES = {
    'NAD': '(ft_binding:NAD OR keyword:KW-0524 OR "NAD" AND (cc_cofactor:* OR ft_binding:*))',
    'ATP': '(ft_binding:ATP OR keyword:KW-0067 OR "ATP" AND (cc_cofactor:* OR ft_binding:*))',
    'acetyl-CoA': '(ft_binding:"acetyl-CoA" OR keyword:KW-0008 OR "acetyl-CoA" AND (cc_cofactor:* OR ft_binding:*))',
    'B12': '(chebi:176843 OR keyword:KW-0171 OR "cobalamin" AND (cc_cofactor:* OR ft_binding:*))',
    'FAD': '(keyword:KW-0274 OR ft_binding:FAD OR "FAD" AND (cc_cofactor:* OR ft_binding:*))'
}

def get_next_url_from_link_header(link_header):
    """Extrahuje URL další stránky (cursor) z HTTP hlavičky 'Link' UniProt REST API."""
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel=["\']?next["\']?', link_header, re.IGNORECASE)
    return match.group(1) if match else None

def fetch_diverse_uniprots(
    cofactor, 
    base_query, 
    target_count=2500, 
    max_per_family=35,    # Max 35 proteinů se stejnou Pfam rodinou
    max_per_organism=3,   # Max 3 proteiny ze stejného organismu
    page_size=500
):
    """
    Stáhne UniProt ID s důrazem na maximální strukturní a taxonomickou diverzitu:
    1. Využívá korektní cursorovou paginaci UniProtKB REST API (Link header).
    2. Prioritizuje manuálně kurátované (Swiss-Prot / reviewed:true) proteiny.
    3. Doplňuje z TrEMBL, pokud Swiss-Prot nestačí do cílového počtu.
    4. Kontroluje Pfam rodiny a taxonId organismu.
    """
    cache_file = os.path.join(structures_dir, f"uniprot_ids_{cofactor}.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_ids = json.load(f)
        if len(cached_ids) >= target_count:
            write_log(f"  ⚡ [CACHE] Načteno {len(cached_ids)} UniProt ID z lokálního souboru pro {cofactor}.")
            return cached_ids[:target_count]
        else:
            write_log(f"  ℹ️ [CACHE] V cache je {len(cached_ids)} ID, požadováno {target_count}, stahuji nový set...")

    write_log(f"  🌐 [API] Stahuji diverzifikované proteiny pro {cofactor} (cíl: {target_count})...")
    
    selected_ids = []
    family_counts = defaultdict(int)
    organism_counts = defaultdict(int)
    seen_ids = set()
    
    # 2 fáze: Nejprve kurátovaný Swiss-Prot, poté TrEMBL
    phases = [
        ("Swiss-Prot (reviewed)", f"{base_query} AND (reviewed:true)"),
        ("TrEMBL (unreviewed fallback)", f"{base_query} AND (reviewed:false)")
    ]
    
    fields = "accession,reviewed,organism_id,protein_name,xref_pfam"
    
    for phase_name, query in phases:
        if len(selected_ids) >= target_count:
            break
            
        write_log(f"    ↳ Fáze: {phase_name}")
        encoded_query = quote(query)
        current_url = f"https://rest.uniprot.org/uniprotkb/search?query={encoded_query}&format=json&fields={fields}&size={page_size}"
        page_num = 1
        
        while current_url and len(selected_ids) < target_count:
            try:
                response = requests.get(current_url, timeout=30)
                if response.status_code != 200:
                    write_log(f"    ❌ Chyba UniProt API: HTTP {response.status_code}")
                    break
                
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                    
                added_in_page = 0
                
                for item in results:
                    acc = item.get("primaryAccession")
                    if not acc or acc in seen_ids:
                        continue
                        
                    # 1. Taxonomický filtr
                    org_id = item.get("organism", {}).get("taxonId")
                    if org_id and organism_counts[org_id] >= max_per_organism:
                        continue
                        
                    # 2. Doménový / Pfam rodinný filtr
                    cross_refs = item.get("uniProtKBCrossReferences", [])
                    fam_ids = [x["id"] for x in cross_refs if x.get("database") == "Pfam"]
                    
                    if fam_ids:
                        # Pokud jsou VŠECHNY jeho Pfam rodiny již zaplněné na max_per_family, přeskočíme
                        if all(family_counts[fid] >= max_per_family for fid in fam_ids):
                            continue
                        for fid in fam_ids:
                            family_counts[fid] += 1
                            
                    # Akceptujeme protein
                    seen_ids.add(acc)
                    if org_id:
                        organism_counts[org_id] += 1
                    selected_ids.append(acc)
                    added_in_page += 1
                    
                    if len(selected_ids) >= target_count:
                        break
                        
                print(f"      [Strana {page_num:3d}] +{added_in_page:3d} unikátních | Celkem vybráno: {len(selected_ids):4d}/{target_count}")
                
                # Získání URL další stránky pomocí cursoru v Link headeru
                link_header = response.headers.get("Link") or response.headers.get("link")
                current_url = get_next_url_from_link_header(link_header)
                page_num += 1
                time.sleep(0.2)
                
            except Exception as e:
                write_log(f"    ❌ Výjimka při komunikaci s UniProt: {e}")
                time.sleep(2)
                # Při chybě spojení zkusíme znovu nebo ukončíme
                break
                
    # Uložení do cache
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(selected_ids, f, indent=2)
    write_log(f"  💾 Uloženo {len(selected_ids)} unikátních proteinů do cache: {cache_file}")
    
    return selected_ids

def download_alphafold_structures(cofactor, uniprot_ids, max_downloads=None):
    """Stáhne AlphaFold struktury, kontroluje existenci na disku."""
    if max_downloads is None:
        max_downloads = len(uniprot_ids)
    
    cofactor_dir = os.path.join(structures_dir, cofactor)
    os.makedirs(cofactor_dir, exist_ok=True)
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    for i, uniprot_id in enumerate(uniprot_ids[:max_downloads]):
        existing_files = [f for f in os.listdir(cofactor_dir) if uniprot_id in f and f.endswith('.pdb')]
        if existing_files:
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{max_downloads}] ⚡ Kontrola lokální cache ({skipped} již existuje)...")
            skipped += 1
            continue
            
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        
        try:
            response = requests.get(api_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for fragment in data:
                    pdb_url = fragment.get("pdbUrl")
                    if pdb_url:
                        pdb_response = requests.get(pdb_url, timeout=20)
                        if pdb_response.status_code == 200:
                            filename = os.path.join(cofactor_dir, pdb_url.split("/")[-1])
                            with open(filename, "wb") as f:
                                f.write(pdb_response.content)
                            downloaded += 1
                        else:
                            failed += 1
            elif response.status_code == 404:
                failed += 1
            else:
                failed += 1
        except Exception as e:
            write_log(f"    ❌ Chyba stahování AF u {uniprot_id}: {e}")
            failed += 1
            
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{max_downloads}] Nově staženo: {downloaded}, V cache: {skipped}, Selhalo: {failed}")
        
        time.sleep(0.2)
    
    write_log(f"  📊 {cofactor} AF DB report: {downloaded} nově staženo, {skipped} v cache, {failed} selhalo/nenalezeno.")
    return downloaded, failed, skipped

def merge_fragments(cofactor_dir, uniprot_id):
    """
    Standardizuje názvy stažených AlphaFold souborů:
    1. Pokud má protein 1 soubor (99 % případů v AF DB): přejmenuje ho na {uniprot_id}_MERGED.pdb pro konzistenci.
    2. Pokud má protein více fragmentů (AF-F1, AF-F2 u obřích proteinů > 2700 aa):
       Každý fragment byl predikován AlphaFoldem v samostatném lokálním souřadném systému s překryvem.
       Ponecháme je jako samostatné domény {uniprot_id}_F1_MERGED.pdb, {uniprot_id}_F2_MERGED.pdb,
       což zabrání chybnému míchání atomů a zaručí správnou detekci kapes v P2Ranku.
    """
    pdb_files = sorted([f for f in os.listdir(cofactor_dir) if uniprot_id in f and f.endswith('.pdb') and not f.endswith('_MERGED.pdb')])
    
    if not pdb_files:
        return 0
        
    if len(pdb_files) == 1:
        merged_filename = os.path.join(cofactor_dir, f"{uniprot_id}_MERGED.pdb")
        if not os.path.exists(merged_filename):
            os.rename(os.path.join(cofactor_dir, pdb_files[0]), merged_filename)
        return 1
    
    # Více fragmentů (AlphaFold F1, F2, F3...)
    processed = 0
    for i, pdb_file in enumerate(pdb_files, start=1):
        # Pokusíme se extrahovat číslo fragmentu z původního názvu (např. AF-P12345-F2-model_v4.pdb)
        frag_match = re.search(r'-F(\d+)-', pdb_file)
        frag_num = frag_match.group(1) if frag_match else str(i)
        
        frag_filename = os.path.join(cofactor_dir, f"{uniprot_id}_F{frag_num}_MERGED.pdb")
        if not os.path.exists(frag_filename):
            os.rename(os.path.join(cofactor_dir, pdb_file), frag_filename)
            processed += 1
            
    return processed

# === HLAVNÍ SPUŠTĚNÍ ===

TARGET_PROTEINS_PER_CLASS = 2500  # Požadovaný počet unikátních struktur na třídu (lze upravit)
MAX_PER_FAMILY = 35              # Max 35 proteinů ze stejné Pfam domény
MAX_PER_ORGANISM = 3             # Max 3 proteiny ze stejného druhu

total_downloaded = 0
total_failed = 0
total_skipped = 0

for cofactor, query in QUERIES.items():
    write_log(f"\n{'='*50}")
    write_log(f"Kofaktor: {cofactor}")
    write_log(f"{'='*50}")
    
    # 1. KROK: Získání diverzifikovaných UniProt IDs
    uniprots = fetch_diverse_uniprots(
        cofactor=cofactor,
        base_query=query,
        target_count=TARGET_PROTEINS_PER_CLASS,
        max_per_family=MAX_PER_FAMILY,
        max_per_organism=MAX_PER_ORGANISM
    )
    
    if uniprots:
        write_log(f"\n=== Stažení AlphaFold struktur pro {cofactor} ===")
        # 2. KROK: Stahování z AlphaFold DB
        downloaded, failed, skipped = download_alphafold_structures(cofactor, uniprots, max_downloads=TARGET_PROTEINS_PER_CLASS)
        
        # 3. KROK: Sjednocení a úklid
        cofactor_dir = os.path.join(structures_dir, cofactor)
        merged_count = 0
        for uid in uniprots:
            merged_count += merge_fragments(cofactor_dir, uid)
            
        write_log(f"  🧩 Sjednoceno {merged_count} struktur pro {cofactor}.")
        
        total_downloaded += downloaded
        total_failed += failed
        total_skipped += skipped

write_log(f"\n\n{'='*50}")
write_log("CELKOVÉ SHRNUTÍ DIVERZIFIKOVANÉHO STAHOVÁNÍ")
write_log(f"{'='*50}")
write_log(f"✅ Nově staženo z AF DB: {total_downloaded}")
write_log(f"⚡ Přeskočeno (již na disku): {total_skipped}")
write_log(f"❌ Selhalo / Nenalezeno v AF DB: {total_failed}")
write_log("Konec skriptu.\n")