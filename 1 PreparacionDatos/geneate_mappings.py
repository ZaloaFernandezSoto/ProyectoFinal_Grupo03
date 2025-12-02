import pandas as pd
import numpy as np
import os
from pathlib import Path

# Diccionario de mapeo de variables
VARIABLE_NAMES = {
    # Variables medidas (XMEAS) - Columnas 0-40
    0: 'A_Feed_stream1',
    1: 'D_Feed_stream2',
    2: 'E_Feed_stream3',
    3: 'AC_Feed_stream4',
    4: 'Recycle_Flow_stream8',
    5: 'Reactor_Feed_Rate_stream6',
    6: 'Reactor_Pressure',
    7: 'Reactor_Level',
    8: 'Reactor_Temperature',
    9: 'Purge_Rate_stream9',
    10: 'Product_Sep_Temp',
    11: 'Product_Sep_Level',
    12: 'Prod_Sep_Pressure',
    13: 'Prod_Sep_Underflow_stream10',
    14: 'Stripper_Level',
    15: 'Stripper_Pressure',
    16: 'Stripper_Underflow_stream11',
    17: 'Stripper_Temperature',
    18: 'Stripper_Steam_Flow',
    19: 'Compressor_Work',
    20: 'Reactor_CW_Outlet_Temp',
    21: 'Separator_CW_Outlet_Temp',
    22: 'Reactor_Feed_CompA',
    23: 'Reactor_Feed_CompB',
    24: 'Reactor_Feed_CompC',
    25: 'Reactor_Feed_CompD',
    26: 'Reactor_Feed_CompE',
    27: 'Reactor_Feed_CompF',
    28: 'Purge_Gas_CompA',
    29: 'Purge_Gas_CompB',
    30: 'Purge_Gas_CompC',
    31: 'Purge_Gas_CompD',
    32: 'Purge_Gas_CompE',
    33: 'Purge_Gas_CompF',
    34: 'Purge_Gas_CompG',
    35: 'Purge_Gas_CompH',
    36: 'Product_CompD',
    37: 'Product_CompE',
    38: 'Product_CompF',
    39: 'Product_CompG',
    40: 'Product_CompH',
    
    # Variables manipuladas (XMV) - Columnas 41-51
    41: 'D_Feed_Flow',              # XMV(1) - Flow, no valve
    42: 'E_Feed_Flow',              # XMV(2) - Flow, no valve
    43: 'A_Feed_Flow',              # XMV(3) - Flow, no valve
    44: 'AC_Feed_Flow',             # XMV(4) - Flow, no valve
    45: 'Compressor_Recycle_Valve', # XMV(5) - Valve
    46: 'Purge_Valve',              # XMV(6) - Valve
    47: 'Separator_Pot_Liquid_Flow',# XMV(7) - Flow (pot liquid)
    48: 'Stripper_Product_Flow',    # XMV(8) - Flow
    49: 'Stripper_Steam_Valve',     # XMV(9) - Valve
    50: 'Reactor_CW_Flow',          # XMV(10) - Cooling water flow
    51: 'Condenser_CW_Flow'         # XMV(11) - Cooling water flow
}

# Unidades de medida
VARIABLE_UNITS = {
    0: 'kscmh', 1: 'kg/hr', 2: 'kg/hr', 3: 'kscmh', 4: 'kscmh',
    5: 'kscmh', 6: 'kPa_gauge', 7: '%', 8: 'Deg_C', 9: 'kscmh',
    10: 'Deg_C', 11: '%', 12: 'kPa_gauge', 13: 'm3/hr', 14: '%',
    15: 'kPa_gauge', 16: 'm3/hr', 17: 'Deg_C', 18: 'kg/hr', 19: 'kW',
    20: 'Deg_C', 21: 'Deg_C',
    22: 'Mole_%', 23: 'Mole_%', 24: 'Mole_%', 25: 'Mole_%', 26: 'Mole_%',
    27: 'Mole_%', 28: 'Mole_%', 29: 'Mole_%', 30: 'Mole_%', 31: 'Mole_%',
    32: 'Mole_%', 33: 'Mole_%', 34: 'Mole_%', 35: 'Mole_%', 36: 'Mole_%',
    37: 'Mole_%', 38: 'Mole_%', 39: 'Mole_%', 40: 'Mole_%',
    41: '%', 42: '%', 43: '%', 44: '%', 45: '%', 46: '%',
    47: '%', 48: '%', 49: '%', 50: '%', 51: '%'
}

def convert_dat_to_csv(input_path, output_path):
    """
    Convierte archivos .dat del TEP a CSV con nombres de columnas descriptivos
    
    Args:
        input_path: Ruta al archivo .dat
        output_path: Ruta de salida para el CSV
    """
    # Leer archivo .dat (separado por espacios)
    data = pd.read_csv(input_path, sep=r'\s+', header=None)
    
    print(f"  Archivo: {Path(input_path).name}")
    print(f"  Dimensiones: {data.shape}")
    
    # Verificar que tenga 52 columnas
    if data.shape[1] != 52:
        print(f"  ⚠ ADVERTENCIA: Se esperaban 52 columnas, encontradas {data.shape[1]}")
        # Tomar solo las primeras 52 columnas si hay más
        if data.shape[1] > 52:
            data = data.iloc[:, :52]
            print(f"  → Usando solo las primeras 52 columnas")
        else:
            print(f"  ⚠ No se puede procesar este archivo, tiene menos de 52 columnas")
            return None
    
    # Renombrar columnas
    data.columns = [VARIABLE_NAMES[i] for i in range(52)]
    
    # Extraer información del nombre del archivo
    filename = Path(input_path).stem
    
    # Determinar si es normal (d00) o con fallo
    if 'd00' in filename:
        data['fault'] = 0
        data['fault_type'] = 'Normal'
    elif '_te' in filename:
        # Archivo de testing
        fault_num = int(filename.replace('d', '').replace('_te', ''))
        data['fault'] = fault_num
        data['fault_type'] = f'Fault_{fault_num}'
    else:
        # Archivo de training
        fault_num = int(filename.replace('d', ''))
        data['fault'] = fault_num
        data['fault_type'] = f'Fault_{fault_num}'
    
    # Guardar como CSV
    data.to_csv(output_path, index=False)
    print(f"  ✓ Convertido -> {output_path.name}\n")
    
    return data

def convert_all_files(data_dir, output_dir):
    """
    Convierte todos los archivos .dat en un directorio
    
    Args:
        data_dir: Directorio con archivos .dat
        output_dir: Directorio de salida para CSV
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los archivos .dat
    dat_files = list(data_dir.glob('*.dat'))
    
    if not dat_files:
        print(f"⚠ No se encontraron archivos .dat en {data_dir}")
        return
    
    print(f"Encontrados {len(dat_files)} archivos .dat")
    print("-" * 60)
    
    success_count = 0
    error_count = 0
    
    for dat_file in sorted(dat_files):
        output_file = output_dir / f"{dat_file.stem}.csv"
        try:
            result = convert_dat_to_csv(dat_file, output_file)
            if result is not None:
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"  ✗ Error procesando {dat_file.name}: {e}\n")
            error_count += 1
    
    print("-" * 60)
    print(f"✓ Conversión completada")
    print(f"  Exitosos: {success_count}")
    print(f"  Errores: {error_count}")
    print(f"  Archivos guardados en: {output_dir}")



def create_variable_info_csv(output_path='variable_info.csv'):
    """
    Crea un CSV con información sobre todas las variables
    """
    info_data = []
    for idx, name in VARIABLE_NAMES.items():
        var_type = 'Measured (XMEAS)' if idx < 41 else 'Manipulated (XMV)'
        unit = VARIABLE_UNITS.get(idx, '')
        info_data.append({
            'Column_Index': idx,
            'Variable_Name': name,
            'Type': var_type,
            'Unit': unit
        })
    
    df_info = pd.DataFrame(info_data)
    df_info.to_csv(output_path, index=False)
    print(f"✓ Información de variables guardada en: {output_path}")

if __name__ == "__main__":
    # Rutas
    DATA_DIR = Path(__file__).parent.parent / 'data' / 'TEP_data'
    OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'TEP_csv'
    
    print("="*60)
    print("CONVERSIÓN DE DATOS TENNESSEE EASTMAN PROCESS")
    print("="*60)
    
    # Convertir todos los archivos
    convert_all_files(DATA_DIR, OUTPUT_DIR)
    
    # Crear archivo con información de variables
    create_variable_info_csv(OUTPUT_DIR / 'variable_info.csv')
    

    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)