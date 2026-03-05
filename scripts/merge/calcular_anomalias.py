import os
import xarray as xr
import pandas as pd
import numpy as np
from calendar import monthrange

# =============================================================================
# --- 1. CONFIGURAÇÕES DE DIRETÓRIOS (Caminhos Relativos GitHub) ---
# =============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DIR_OBSERVADO = os.path.join(BASE_DIR, "data", "merge", "observado")
DIR_CLIMATOLOGIA = os.path.join(BASE_DIR, "data", "merge", "climatologia")
DIR_SAIDA_BASE = os.path.join(BASE_DIR, "data", "merge", "anomalia")

# Nomes esperados dos arquivos
ARQ_OBS = "MERGE_GPM_COMPLETO_2026.nc"
ARQ_CLIM = "climatologia_366_dias_1998_2024.nc"

VAR_NOME_PREF = 'prec' 

# =============================================================================
# --- 2. MOTOR MATEMÁTICO ---
# =============================================================================

def criar_pasta(ano, mes):
    path = os.path.join(DIR_SAIDA_BASE, str(ano), f"{mes:02d}")
    os.makedirs(path, exist_ok=True)
    return path

def padronizar_coords(ds):
    renames = {}
    for var in ds.coords:
        if var.lower() in ['latitude', 'lat']: renames[var] = 'lat'
        if var.lower() in ['longitude', 'lon']: renames[var] = 'lon'
    if renames: ds = ds.rename(renames)
    return ds

def calcular_anomalia_diaria_base(ds_obs, ds_clim):
    """
    Função central: Alinha as grades, cruza os dias do ano e subtrai Obs - Clim.
    Retorna um DataArray com a anomalia diária de todo o período.
    """
    ds_obs = padronizar_coords(ds_obs)
    ds_clim = padronizar_coords(ds_clim)

    var_obs = VAR_NOME_PREF if VAR_NOME_PREF in ds_obs else list(ds_obs.data_vars)[0]
    var_clim = VAR_NOME_PREF if VAR_NOME_PREF in ds_clim else list(ds_clim.data_vars)[0]

    vals_lat = ds_obs['lat'].values
    vals_lon = ds_obs['lon'].values
    clean_lat = xr.DataArray(vals_lat, dims='lat', coords={'lat': vals_lat})
    clean_lon = xr.DataArray(vals_lon, dims='lon', coords={'lon': vals_lon})
    
    ds_clim_regrid = ds_clim[var_clim].interp(lat=clean_lat, lon=clean_lon, method='linear')

    dias_do_ano_obs = ds_obs['time'].dt.dayofyear
    clim_alinhada = ds_clim_regrid.sel(time=ds_clim_regrid['time'].dt.dayofyear.isin(dias_do_ano_obs))
    
    if len(ds_obs.time) != len(clim_alinhada.time):
        min_len = min(len(ds_obs.time), len(clim_alinhada.time))
        ds_obs = ds_obs.isel(time=slice(0, min_len))
        clim_alinhada = clim_alinhada.isel(time=slice(0, min_len))

    da_anomalia = ds_obs[var_obs].copy()
    da_anomalia.values = ds_obs[var_obs].values - clim_alinhada.values
    
    return da_anomalia

# =============================================================================
# --- 3. ORQUESTRADOR ---
# =============================================================================

def main():
    # --- ESCOLHA O MODO AQUI ---
    MODO = "historico"  
    
    print(f"=== CÁLCULO DE ANOMALIA | MODO: {MODO.upper()} ===")
    
    caminho_obs = os.path.join(DIR_OBSERVADO, ARQ_OBS)
    caminho_clim = os.path.join(DIR_CLIMATOLOGIA, ARQ_CLIM)

    if not os.path.exists(caminho_obs):
        print(f"ERRO: Arquivo não encontrado: {caminho_obs}")
        return

    print("1. Carregando arquivos e calculando matemática base (Obs - Clim)...")
    ds_obs = xr.open_dataset(caminho_obs)
    ds_clim = xr.open_dataset(caminho_clim)
    
    # Faz o processamento pesado uma única vez na memória
    da_anomalia_total = calcular_anomalia_diaria_base(ds_obs, ds_clim)

    # ---------------------------------------------------------
    # MODO 1: CONTÍNUO (Foca apenas no final da série de dados)
    # ---------------------------------------------------------
    if MODO == "continuo":
        print("2. Gerando recortes contínuos (foco na cauda dos dados)...")
        ultima_data = pd.to_datetime(da_anomalia_total.time.values[-1])
        ano_atual, mes_atual, dia_atual = ultima_data.year, ultima_data.month, ultima_data.day
        pasta_destino = criar_pasta(ano_atual, mes_atual)
        
        # === CORREÇÃO: Puxa 15 dias corridos exatos cruzando meses ===
        data_ini_15 = ultima_data - pd.Timedelta(days=14)
        slice_15dias = da_anomalia_total.sel(time=slice(data_ini_15, ultima_data))
        
        anom_15d_rolling = slice_15dias.sum(dim='time')
        nome_arq_15 = f"Anomalia_Ultimos15Dias_{ano_atual}{mes_atual:02d}_dia_{data_ini_15.day:02d}_a_{ultima_data.day:02d}.nc"
        anom_15d_rolling.to_netcdf(os.path.join(pasta_destino, nome_arq_15))
        print(f"   [SALVO] {nome_arq_15}")

        # Fechamento do mês
        _, ultimo_dia_calendario = monthrange(ano_atual, mes_atual)
        if dia_atual == ultimo_dia_calendario:
            slice_mes = da_anomalia_total.sel(time=f"{ano_atual}-{mes_atual:02d}")
            slice_mes.sum(dim='time').to_netcdf(os.path.join(pasta_destino, f"Anomalia_Mensal_{ano_atual}{mes_atual:02d}.nc"))
            print(f"   [SALVO MENSAL] Anomalia_Mensal_{ano_atual}{mes_atual:02d}.nc")

    # ---------------------------------------------------------
    # MODO 2: HISTÓRICO (Itera por tudo: Dia Atual, Mensal, Acumulado)
    # ---------------------------------------------------------
    elif MODO == "historico":
        print("2. Iterando sobre o histórico completo...")
        anos_disponiveis = np.unique(da_anomalia_total['time'].dt.year)
        
        for ano in anos_disponiveis:
            da_ano = da_anomalia_total.sel(time=str(ano))
            meses_no_ano = np.unique(da_ano['time'].dt.month)
            
            for mes in meses_no_ano:
                print(f"\n   -> Processando: {mes:02d}/{ano}...")
                da_mes = da_ano.sel(time=f"{ano}-{mes:02d}")
                pasta_saida = criar_pasta(ano, mes)
                
                dias_disponiveis = da_mes['time'].dt.day.values
                if len(dias_disponiveis) == 0: continue
                ultimo_dia = dias_disponiveis[-1]

                # A. Acumulado do Mês (1 ao dia disponível)
                anom_acumulada_mes = da_mes.sum(dim='time')
                nome_acumulado = f"Anomalia_Acumulada_{ano}{mes:02d}_ate_dia_{ultimo_dia:02d}.nc"
                anom_acumulada_mes.to_netcdf(os.path.join(pasta_saida, nome_acumulado))
                print(f"      [SALVO] Acumulado Parcial: {nome_acumulado}")

                # B. Últimos 15 Dias Corridos (Cruza os meses!)
                # === CORREÇÃO AQUI ===
                data_fim_historico = pd.to_datetime(f"{ano}-{mes:02d}-{ultimo_dia:02d}")
                data_ini_historico = data_fim_historico - pd.Timedelta(days=14)
                
                # Lê do arquivo GLOBAL (da_anomalia_total) para voltar ao mês anterior se precisar
                da_ultimos_15 = da_anomalia_total.sel(time=slice(data_ini_historico, data_fim_historico))
                anom_15d = da_ultimos_15.sum(dim='time')
                
                nome_arq_15 = f"Anomalia_Ultimos15Dias_{ano}{mes:02d}_dia_{data_ini_historico.day:02d}_a_{data_fim_historico.day:02d}.nc"
                anom_15d.to_netcdf(os.path.join(pasta_saida, nome_arq_15))
                print(f"      [SALVO] 15 Dias: {nome_arq_15}")

                # C. Mês Fechado
                _, ultimo_dia_calendario = monthrange(ano, mes)
                if ultimo_dia == ultimo_dia_calendario:
                    nome_mensal = f"Anomalia_Mensal_{ano}{mes:02d}.nc"
                    anom_acumulada_mes.to_netcdf(os.path.join(pasta_saida, nome_mensal))
                    print(f"      [SALVO MÊS FECHADO] {nome_mensal}")

    print("\nProcesso Finalizado.")

if __name__ == "__main__":
    main()