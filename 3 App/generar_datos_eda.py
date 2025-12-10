import pandas as pd
import os

# Carpeta donde están los CSV generados a partir de los .dat
BASE_PATH = "../data/TEP_csv/"

# d00 → normal, d01 → fallo 1 ... d21 → fallo 21
FALLOS = list(range(22))  # 0–21

dfs = []

for fallo in FALLOS:
    # Los nombres pueden ser d00.csv, d01.csv ... d21.csv
    file_name = f"d{fallo:02d}.csv"
    file_path = os.path.join(BASE_PATH, file_name)

    if not os.path.exists(file_path):
        print(f"⚠️ No existe: {file_path}, se omite.")
        continue

    print(f"📄 Cargando {file_name} ...")

    df = pd.read_csv(file_path)

    # Añadimos las columnas de fallo
    df["fault"] = fallo

    if fallo == 0:
        df["fault_type"] = "Normal"
    else:
        df["fault_type"] = f"Fault_{fallo}"

    dfs.append(df)

# Unimos todos en un único dataset
df_total = pd.concat(dfs, ignore_index=True)

# Guardamos el archivo final
output_path = os.path.join(BASE_PATH, "datos_eda.csv")
df_total.to_csv(output_path, index=False)

print(f"\n✅ Archivo generado correctamente:")
print(f"   → {output_path}")
print(f"   Total de filas: {len(df_total)}")