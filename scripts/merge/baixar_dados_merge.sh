#!/bin/bash

# =============================================================================
# --- 1. CONFIGURAÇÕES DE DIRETÓRIOS (Caminhos Relativos GitHub) ---
# =============================================================================

# Descobre a pasta exata onde este script está salvo (scripts/merge)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sobe duas pastas para chegar na raiz do repositório
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Define onde os dados serão salvos (NÃO vão para o GitHub, ficam no /data/)
DIR_BASE="${BASE_DIR}/data/merge/observado"

URL_BASE="https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY"

# Cores para o terminal
VERDE='\033[0;32m'
AZUL='\033[0;34m'
AMARELO='\033[1;33m'
VERMELHO='\033[0;31m'
SEM_COR='\033[0m'

# =============================================================================
# --- 2. ORQUESTRADOR DE MODOS E DATAS ---
# =============================================================================

# ESCOLHA O MODO DE DOWNLOAD AQUI:
# "atualizacao" -> Atualiza do dia 1º de Janeiro do ano atual até HOJE (ignora os já baixados).
# "historico"   -> Baixa um período específico (ex: um mês antigo que você perdeu).
MODO="atualizacao"

if [ "$MODO" == "atualizacao" ]; then
    ANO_ATUAL=$(date +%Y)
    DATA_INICIO="${ANO_ATUAL}-01-01"
    DATA_FIM=$(date +%Y-%m-%d) # Hoje
    ARQUIVO_FINAL="$DIR_BASE/MERGE_GPM_COMPLETO_${ANO_ATUAL}.nc"

elif [ "$MODO" == "historico" ]; then
    # Se quiser baixar um mês específico do passado, mude aqui:
    DATA_INICIO="2025-01-01"
    DATA_FIM="2025-01-31"
    ANO_HIST=$(date -d "$DATA_INICIO" +%Y)
    ARQUIVO_FINAL="$DIR_BASE/MERGE_GPM_COMPLETO_${ANO_HIST}.nc"
else
    echo -e "${VERMELHO}ERRO: MODO inválido.${SEM_COR}"
    exit 1
fi

echo -e "${AZUL}==========================================================${SEM_COR}"
echo -e "   ATUALIZADOR DE DADOS MERGE (GRIB2 -> NC -> ÚNICO)"
echo -e "   MODO: ${AMARELO}${MODO^^}${SEM_COR} | Período: ${DATA_INICIO} a ${DATA_FIM}"
echo -e "${AZUL}==========================================================${SEM_COR}"

# Verifica CDO
if ! command -v cdo &> /dev/null; then
    echo -e "${VERMELHO}ERRO: O 'cdo' é obrigatório. Instale: sudo apt install cdo${SEM_COR}"
    exit 1
fi

# =============================================================================
# --- ETAPA 1: DOWNLOAD E CONVERSÃO DIÁRIA ---
# =============================================================================

DATA_ATUAL="$DATA_INICIO"

while [ "$DATA_ATUAL" != $(date -I -d "$DATA_FIM + 1 day") ]; do
    
    ANO=$(date -d "$DATA_ATUAL" +%Y)
    MES=$(date -d "$DATA_ATUAL" +%m)
    DATA_STR=$(date -d "$DATA_ATUAL" +%Y%m%d)

    PASTA_DIA="$DIR_BASE/$ANO/$MES"
    mkdir -p "$PASTA_DIA"

    ARQ_GRIB="MERGE_CPTEC_${DATA_STR}.grib2"
    ARQ_NC_DIA="MERGE_CPTEC_${DATA_STR}.nc"
    
    PATH_GRIB="$PASTA_DIA/$ARQ_GRIB"
    PATH_NC_DIA="$PASTA_DIA/$ARQ_NC_DIA"

    # 1.1 Baixa GRIB se não existir
    if [ ! -f "$PATH_GRIB" ]; then
        echo -n -e "[$DATA_ATUAL] Baixando... "
        URL="$URL_BASE/$ANO/$MES/$ARQ_GRIB"
        wget -q --no-check-certificate -P "$PASTA_DIA" "$URL"
        
        if [ $? -ne 0 ]; then
            echo -e "${VERMELHO}Indisponível.${SEM_COR}"
            if [ -f "$PATH_GRIB" ]; then rm "$PATH_GRIB"; fi
            DATA_ATUAL=$(date -I -d "$DATA_ATUAL + 1 day")
            continue 
        else
            echo -n -e "${VERDE}OK. ${SEM_COR}"
        fi
    fi

    # 1.2 Converte GRIB para NC Diário (se ainda não existir)
    if [ ! -f "$PATH_NC_DIA" ] && [ -f "$PATH_GRIB" ]; then
        echo -n -e "${AMARELO}Convertendo para NC... ${SEM_COR}"
        cdo -s -f nc copy "$PATH_GRIB" "$PATH_NC_DIA"
    fi
    
    # Pula linha visual no log se houve ação
    if [ -f "$PATH_GRIB" ]; then echo ""; fi

    DATA_ATUAL=$(date -I -d "$DATA_ATUAL + 1 day")
done

echo -e "\n----------------------------------------------------------"
echo -e "ETAPA 2: UNIFICANDO TUDO EM UM ÚNICO ARQUIVO..."
echo -e "----------------------------------------------------------"

# =============================================================================
# --- ETAPA 2: MERGETIME (A Mágica) ---
# =============================================================================

# Encontra todos os arquivos .nc diários daquele ano, ordena e junta.
ANO_ALVO=$(date -d "$DATA_INICIO" +%Y)
LISTA_ARQUIVOS=$(find "$DIR_BASE/$ANO_ALVO" -name "MERGE_CPTEC_*.nc" | sort)

if [ -z "$LISTA_ARQUIVOS" ]; then
    echo -e "${VERMELHO}Nenhum arquivo NC encontrado para juntar no ano ${ANO_ALVO}.${SEM_COR}"
else
    echo -e "Gerando arquivo consolidado: ${AMARELO}$ARQUIVO_FINAL${SEM_COR}"
    echo "Isso pode levar alguns segundos dependendo do tamanho..."
    
    # O comando mergetime faz a união
    cdo -s -O mergetime $LISTA_ARQUIVOS "$ARQUIVO_FINAL"
    
    if [ $? -eq 0 ]; then
        echo -e "${VERDE}✅ SUCESSO! Arquivo único atualizado.${SEM_COR}"
        ls -lh "$ARQUIVO_FINAL"
    else
        echo -e "${VERMELHO}❌ ERRO ao unir arquivos.${SEM_COR}"
    fi
fi