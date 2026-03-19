from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def clean_mamajek(input_path, output_path):
    with open(input_path, 'r') as f:
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Parsed {len(df)} rows from Mamajek table.")


if __name__ == "__main__":
    clean_mamajek(
        PROJECT_ROOT / "data" / "raw" / "EEM_dwarf_UBVIJHK_colors_Teff.txt",
        PROJECT_ROOT / "data" / "processed" / "mamajek_clean.csv",
    )
