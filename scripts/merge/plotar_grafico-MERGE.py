import os
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
from datetime import datetime, timedelta
from shapely.geometry import mapping

# --- IMPORTAÇÃO ANTI-SOBREPOSIÇÃO ---
try:
    from adjustText import adjust_text
except ImportError:
    print("ERRO: Instale 'adjustText' rodando 'pip install adjustText'")
    exit()

# =============================================================================
# --- 1. CONFIGURAÇÕES DE DIRETÓRIOS (Caminhos Relativos GitHub) ---
# =============================================================================

# Descobre a pasta atual do script (scripts/merge) e sobe duas pastas para a raiz
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Diretórios de Entrada e Saída
DIR_ANOMALIA      = os.path.join(BASE_DIR, "data", "merge", "anomalia")
DIR_SAIDA_FIGURAS = os.path.join(BASE_DIR, "outputs", "merge", "figuras")

# Caminhos dos Shapefiles e DEM (Usando os novos nomes de pastas que você definiu)
SHP_BACIAS       = os.path.join(BASE_DIR, "data", "shapes", "shp_bacias", "BACIA_AMAZONICA_.shp")
SHP_RIOS         = os.path.join(BASE_DIR, "data", "shapes", "shp_rios", "geometrias-corrigidas-Massa-Agua.shp")
SHP_SUB_SOLIMOES = os.path.join(BASE_DIR, "data", "shapes", "shp_sub_solimoes", "sub_solimoes.shp")
CAMINHO_DEM      = os.path.join(BASE_DIR, "data", "shapes", "mde", "bacia.asc")

# =============================================================================

SUB_BACIAS = {
    "48": {"nome": "Rio Negro",    "x": 0.03, "y": 0.03, "ha": "left", "va": "bottom"},
    "46": {"nome": "Rio Madeira",  "x": 0.03, "y": 0.97, "ha": "left", "va": "top"},
    "49": {"nome": "Rio Solimoes", "x": 0.97, "y": 0.97, "ha": "right", "va": "top"}
}

DADOS_CIDADES = [
    {"nome": "Tabatinga",     "lat": -4.2347,  "lon": -69.9447},
    {"nome": "Coari",         "lat": -4.0856,  "lon": -63.0833},
    {"nome": "Manacapuru",    "lat": -3.3106,  "lon": -60.6094},
    {"nome": "Beruri",        "lat": -3.8989,  "lon": -61.3742},
    {"nome": "Lábrea",        "lat": -7.2581,  "lon": -64.7975},
    {"nome": "Rio Branco",    "lat": -9.9758,  "lon": -67.8000},
    {"nome": "Barcelos",      "lat": -0.9658,  "lon": -62.9311},
    {"nome": "Moura",         "lat": -1.4567,  "lon": -61.6347},
    {"nome": "Manaus",        "lat": -3.1383,  "lon": -60.0272},
    {"nome": "Porto Velho",   "lat": -8.7483,  "lon": -63.9169},
    {"nome": "Humaitá",       "lat": -7.5028,  "lon": -63.0183},
    {"nome": "Manicoré",      "lat": -5.8167,  "lon": -61.3019},
    {"nome": "Ji-Paraná",     "lat": -10.8736, "lon": -61.9356},
    {"nome": "Itacoatiara",   "lat": -3.1539,  "lon": -58.4114},
    {"nome": "Óbidos",        "lat": -1.9192,  "lon": -55.5131},
    {"nome": "Eirunepé",                 "lat": -6.684444,  "lon": -69.8811},
    {"nome": "Ipixuna",                  "lat": -7.050833,  "lon": -71.68417},
    {"nome": "Cruzeiro do Sul",             "lat": -7.610556,  "lon": -72.68111},
    {"nome": "Manoel Urbano",               "lat": -8.884167,  "lon": -69.26806},
    {"nome": "Boca do Acre",                "lat": -8.735556,  "lon": -67.4000},
    {"nome": "Palmeiras do Javari",         "lat": -5.133333,  "lon": -72.8000},
    {"nome": "Fonte Boa (Solimões)",        "lat": -2.491389,  "lon": -66.06167},
    {"nome": "Tefé",                        "lat": -3.375833,  "lon": -64.65472},
    {"nome": "São Gabriel da Cachoeira", "lat": -0.136111,  "lon": -67.08472},
    {"nome": "Boa Vista",                   "lat":  2.826111,  "lon": -60.65805},
    {"nome": "Caracaraí",                   "lat":  1.821389,  "lon": -61.12361},
    {"nome": "Príncipe da Beira",           "lat": -12.42667,  "lon": -64.42528},
    {"nome": "Abunã",                       "lat": -9.7053,    "lon": -65.3667},
    {"nome": "Guajará-Mirim",               "lat": -10.7925,   "lon": -65.3478},
]

# --- 2. FUNÇÕES ---

def encontrar_arquivos_nc(ano, mes):
    """
    Retorna os arquivos mais recentes para processamento com melhor critério de desempate.
    """
    pasta = os.path.join(DIR_ANOMALIA, str(ano), f"{mes:02d}")
    if not os.path.exists(pasta): return []
    todos_arquivos = glob.glob(os.path.join(pasta, "*.nc"))
    
    arquivos_finais = []

    # 1. Filtra "Ultimos 15 Dias"
    ultimos15 = [f for f in todos_arquivos if "Ultimos15Dias" in f]
    
    if ultimos15:
        try:
            def chave_ordenacao(x):
                match = re.search(r"dia_(\d+)_a_(\d+)", x)
                if match:
                    dia_ini = int(match.group(1))
                    dia_fim = int(match.group(2))
                    return (dia_fim, dia_ini)
                return (0, 0)

            ultimos15.sort(key=chave_ordenacao, reverse=True)
            
            maior_arq = ultimos15[0]
            
            match = re.search(r"dia_(\d+)_a_(\d+)", os.path.basename(maior_arq))
            print(f"   [INFO] Arquivo selecionado: Dia {match.group(1)} a {match.group(2)}")
            
            arquivos_finais.append(maior_arq)
        except Exception as e: 
            print(f"   [AVISO] Erro ao ordenar arquivos: {e}")
            arquivos_finais.append(ultimos15[-1])

    # 2. Filtra Mensal
    mensais = [f for f in todos_arquivos if "Mensal" in f]
    if mensais: arquivos_finais.extend(mensais)

    return arquivos_finais

def calcular_stats_xarray(da):
    if da.isnull().all(): return None
    return float(da.mean()), float(da.min()), float(da.max())

def preparar_coordenadas(ds):
    x_name, y_name = None, None
    for c in ds.coords:
        if c.lower() in ['longitude', 'lon', 'x']: x_name = c
        if c.lower() in ['latitude', 'lat', 'y']: y_name = c
    if not x_name or not y_name: return ds
    if ds[x_name].max() > 180:
        ds = ds.assign_coords({x_name: (((ds[x_name] + 180) % 360) - 180)})
        ds = ds.sortby([x_name, y_name])
    renames = {}
    if x_name != 'x': renames[x_name] = 'x'
    if y_name != 'y': renames[y_name] = 'y'
    if renames: ds = ds.rename(renames)
    return ds

def plotar_nc(nc_path, gdf_recorte, titulo_base, nome_arquivo_saida, 
              pos_config=None, cor_texto='black', gdf_cidades=None, gdf_rios=None, dem_path=None):
    try:
        ds = xr.open_dataset(nc_path)
        ds = preparar_coordenadas(ds)
        var_nome = list(ds.data_vars)[0]
        da = ds[var_nome].squeeze()
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        da = da.rio.write_crs("epsg:4326")

        if gdf_recorte.crs != "epsg:4326":
            gdf_recorte = gdf_recorte.to_crs("epsg:4326")
            
        try:
            geometrias = gdf_recorte.geometry.values 
            da_clipped = da.rio.clip(geometrias, gdf_recorte.crs, drop=True)
        except Exception as e:
            print(f"   [AVISO] Erro no recorte: {e}")
            return

        stats = calcular_stats_xarray(da_clipped)
        if not stats: return
        media, _, _ = stats
        seta, cor_stats = ("↑", "blue") if media >= 0 else ("↓", "red")
        texto_stats = f"Média: {media:.1f} mm {seta}"

        # Suavização
        try:
            fator = 4
            xmin, xmax = da_clipped.x.min().values, da_clipped.x.max().values
            ymin, ymax = da_clipped.y.min().values, da_clipped.y.max().values
            nx, ny = int(len(da_clipped.x) * fator), int(len(da_clipped.y) * fator)
            new_x, new_y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
            da_filled = da_clipped.fillna(0)
            da_plot = da_filled.interp(x=new_x, y=new_y, method='cubic')
        except:
            da_plot = da_clipped

        # Relevo (DEM)
        hillshade_arr = None
        hillshade_extent = None
        if dem_path and os.path.exists(dem_path):
            try:
                rds = rioxarray.open_rasterio(dem_path)
                if rds.rio.crs is None: rds.rio.write_crs("epsg:4326", inplace=True)
                if rds.rio.crs != da.rio.crs: rds = rds.rio.reproject("epsg:4326")
                
                rds_clipped = rds.rio.clip(geometrias, gdf_recorte.crs, drop=True)
                dem_data = rds_clipped.isel(band=0).values.astype('float32')
                dem_data = np.where(dem_data < -500, np.nan, dem_data)
                
                ls = LightSource(azdeg=315, altdeg=45)
                hillshade_arr = ls.hillshade(dem_data, vert_exag=2000)
                b = rds_clipped.rio.bounds()
                hillshade_extent = [b[0], b[2], b[1], b[3]]
            except Exception as e:
                pass 

        # PLOT
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if hillshade_arr is not None:
            ax.imshow(hillshade_arr, cmap='gray', extent=hillshade_extent, 
                      origin='upper', alpha=1.0, zorder=1)
        
        # Barra de Cores (Horizontal em Baixo)
        vmin_user, vmax_user = -160, 160
        passo_user = 20
        ticks_bar = np.arange(vmin_user, vmax_user + 1, passo_user)

        im = da_plot.plot(ax=ax, cmap=plt.cm.BrBG, 
                          vmin=vmin_user, vmax=vmax_user,
                          add_colorbar=False, alpha=0.9, zorder=2)
        
        gdf_recorte.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8, zorder=4)
        try:
            boundary = gpd.GeoSeries(gdf_recorte.union_all(), crs=gdf_recorte.crs)
            boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=4)
        except: pass

        if gdf_rios is not None:
            if gdf_rios.crs != gdf_recorte.crs: gdf_rios = gdf_rios.to_crs(gdf_recorte.crs)
            gpd.clip(gdf_rios, gdf_recorte).plot(ax=ax, color='blue', linewidth=0.8, zorder=3)

        if gdf_cidades is not None:
            cidades_clip = gpd.clip(gdf_cidades.to_crs(gdf_recorte.crs), gdf_recorte)
            if not cidades_clip.empty:
                cidades_clip.plot(ax=ax, color='red', markersize=35, edgecolor='white', linewidth=0.8, zorder=5)
                textos = []
                for _, row in cidades_clip.iterrows():
                    t = ax.text(row.geometry.x, row.geometry.y, row['nome'], 
                                fontsize=9, fontweight='bold', zorder=6,
                                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
                    textos.append(t)
                if textos: adjust_text(textos, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, zorder=4))

        # --- BARRA DE CORES HORIZONTAL ---
        cbar = plt.colorbar(im, ax=ax, 
                            orientation='horizontal', 
                            shrink=0.75, 
                            pad=0.05, 
                            ticks=ticks_bar,
                            aspect=40)
        cbar.set_label('Anomalia Acumulada (mm)', fontsize=12)
        cbar.ax.tick_params(labelsize=9)
        # ---------------------------------
        
        ax.set_title(titulo_base, fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')
        
        if pos_config:
            ax.text(pos_config['x'], pos_config['y'], texto_stats,
                    transform=ax.transAxes, fontsize=11, fontweight='bold', color=cor_stats,
                    verticalalignment=pos_config['va'], horizontalalignment=pos_config['ha'],
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85), zorder=7)

        os.makedirs(os.path.dirname(nome_arquivo_saida), exist_ok=True)
        plt.savefig(nome_arquivo_saida, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> Figura salva: {os.path.basename(nome_arquivo_saida)}")
        ds.close()

    except Exception as e:
        print(f"   [ERRO PLOT] {e}")
        import traceback
        traceback.print_exc()

# --- 3. EXECUÇÃO ---
agora = datetime.now()
ano, mes = agora.year, agora.month 

print(f"=== PLOTAGEM ANOMALIA - ({mes:02d}/{ano}) ===")

print("Carregando Vetores...")
gdf_bacias = gpd.read_file(SHP_BACIAS)
gdf_bacias['wts_cd_p_2'] = gdf_bacias['wts_cd_p_2'].astype(str).str.strip()
if gdf_bacias.crs != "epsg:4326": gdf_bacias = gdf_bacias.to_crs("epsg:4326")

gdf_rios = gpd.read_file(SHP_RIOS)
if gdf_rios.crs is None: gdf_rios.set_crs("epsg:4326", inplace=True)
elif gdf_rios.crs != "epsg:4326": gdf_rios = gdf_rios.to_crs("epsg:4326")

df_cidades = pd.DataFrame(DADOS_CIDADES)
gdf_cidades = gpd.GeoDataFrame(df_cidades, geometry=gpd.points_from_xy(df_cidades.lon, df_cidades.lat), crs="EPSG:4326")

arquivos_nc = encontrar_arquivos_nc(ano, mes)

if not arquivos_nc:
    print(f"Nenhum arquivo NC encontrado na pasta: {os.path.join(DIR_ANOMALIA, str(ano), f'{mes:02d}')}")
    print("DICA: Rode o script de CÁLCULO primeiro!")
    exit()

for nc_file in arquivos_nc:
    nome_arquivo = os.path.basename(nc_file)
    print(f"\nProcessando: {nome_arquivo}")
    
    # === LÓGICA DE TÍTULO DINÂMICO ===
    titulo_tipo = "Anomalia de Chuva"
    
    if "Ultimos15Dias" in nome_arquivo:
        match_dia = re.search(r"dia_(\d+)_a_(\d+)", nome_arquivo)
        if match_dia:
            dia_ini = int(match_dia.group(1))
            dia_fim = int(match_dia.group(2))
            
            # Se dia inicial > dia final (Ex: 18 > 01), é mês anterior
            mes_ini = mes - 1 if dia_ini > dia_fim else mes
            if mes_ini == 0: mes_ini = 12
            
            titulo_tipo = f"Anomalia Últimos 15 Dias ({dia_ini:02d}/{mes_ini:02d} até {dia_fim:02d}/{mes:02d})"
        else:
            titulo_tipo = f"Anomalia Últimos 15 Dias ({mes:02d}/{ano})"
    
    elif "Mensal" in nome_arquivo:
        titulo_tipo = f"Anomalia Mensal ({mes:02d}/{ano})"

    titulo_final = f"Bacia Amazônica \n {titulo_tipo}"

    # GERAL
    config_geral = {"x": 0.97, "y": 0.97, "ha": "right", "va": "top"}
    path_saida = os.path.join(DIR_SAIDA_FIGURAS, str(ano), f"{mes:02d}", f"Geral_{nome_arquivo.replace('.nc', '.png')}")
    plotar_nc(nc_file, gdf_bacias, titulo_final, path_saida, 
              pos_config=config_geral, gdf_cidades=gdf_cidades, gdf_rios=gdf_rios, dem_path=CAMINHO_DEM)

    # SUB-BACIAS
    for codigo, info in SUB_BACIAS.items():
        nome_rio = info["nome"]
        titulo_rio = f"{nome_rio} \n {titulo_tipo}"
        
        if nome_rio == "Rio Solimoes" and os.path.exists(SHP_SUB_SOLIMOES):
             sub_gdf = gpd.read_file(SHP_SUB_SOLIMOES)
             if sub_gdf.crs != "epsg:4326": sub_gdf = sub_gdf.to_crs("epsg:4326")
        else:
            sub_gdf = gdf_bacias[gdf_bacias['wts_cd_p_2'] == codigo]
        
        if not sub_gdf.empty:
            path_rio = os.path.join(DIR_SAIDA_FIGURAS, str(ano), f"{mes:02d}", f"{nome_rio}_{nome_arquivo.replace('.nc', '.png')}")
            plotar_nc(nc_file, sub_gdf, titulo_rio, path_rio, 
                      pos_config=info, gdf_cidades=gdf_cidades, gdf_rios=gdf_rios, dem_path=CAMINHO_DEM)

print("\nProcesso Finalizado.")