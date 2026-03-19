import pandas as pd
import numpy as np

def clean_mamajek():
    # Read the file line by line
    with open('data/raw/EEM_dwarf_UBVIJHK_colors_Teff.txt', 'r') as f:
        lines = f.readlines()
        
    # Extract data lines
    data_lines = [line for line in lines if not line.startswith('#') and len(line.strip()) > 0]
    
    # Process lines
    processed_data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 30:
            try:
                # Find indices by searching from start/end
                spt = parts[0]
                teff = parts[1]
                
                # Column counting from the header:
                # 0: SpT, 1: Teff, 2: logT, 3: BCv, 4: logL, 5: Mbol, 6: R_Rsun, 7: Mv, 8: B-V, 9: Bt-Vt, 10: G-V, 11: Bp-Rp, 12: G-Rp, 13: M_G
                # Msun is the second to last column, parts[-2]
                
                bp_rp = parts[11]
                mg = parts[13]
                msun = parts[-2]
                
                # Check for valid numeric data
                if bp_rp not in ['...', '.....'] and mg not in ['...', '.....'] and msun not in ['...', '.....']:
                    processed_data.append({
                        'SpT': spt,
                        'Teff': float(teff),
                        'Bp_Rp': float(bp_rp),
                        'M_G': float(mg),
                        'M_sun': float(msun)
                    })
            except ValueError:
                pass
                
    df = pd.DataFrame(processed_data)
    df.to_csv('data/processed/mamajek_clean.csv', index=False)
    print(f"Parsed {len(df)} rows from Mamajek table.")
    
clean_mamajek()
