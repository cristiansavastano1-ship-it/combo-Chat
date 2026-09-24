
# COMBO — Audit Model V12 CLEAN
# Derivato dalla V12.1 CPU OPTIMIZED.
# Mantiene la logica V12.1 e rimuove l'esecuzione/codice delle vecchie V non necessari.
# Obiettivo: ridurre CPU/RAM su Streamlit senza modificare il calcolo V12.

import streamlit as st

import pandas as pd

import numpy as np

import requests

import io

import time

import os

import pickle

from datetime import datetime, date

from scipy.stats import poisson

from sklearn.isotonic import IsotonicRegression

AUDIT_SCORE_MAX = 12

CAMPIONATI_DOMESTICI = {
    "Italia - Serie A": {"id_fd": "I1"},
    "Inghilterra - Premier League": {"id_fd": "E0"},
    "Spagna - La Liga": {"id_fd": "SP1"},
    "Germania - Bundesliga": {"id_fd": "D1"},
    "Francia - Ligue 1": {"id_fd": "F1"},
}

HEADERS_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

K_SHRINKAGE = 10

def normalizza_nome_squadra(nome):
    nome = str(nome)
    for suffisso in [" FC", " CF", " AFC", " AC", " SC", " CFC"]:
        nome = nome.replace(suffisso, "")
    return nome.strip().lower()

def nomi_corrispondono(a, b):
    na, nb = normalizza_nome_squadra(a), normalizza_nome_squadra(b)
    return na == nb or na in nb or nb in na

def codici_stagione(oggi=None):
    oggi = oggi or date.today()
    anno_inizio_corrente = oggi.year if oggi.month >= 7 else oggi.year - 1
    anno_inizio_precedente = anno_inizio_corrente - 1
    fmt = lambda a: f"{a % 100:02d}{(a + 1) % 100:02d}"
    return fmt(anno_inizio_corrente), fmt(anno_inizio_precedente)

def tau_dixon_coles(gc, gt, lc, lt, rho):
    if gc == 0 and gt == 0: return 1 - (lc * lt * rho)
    elif gc == 0 and gt == 1: return 1 + (lc * rho)
    elif gc == 1 and gt == 0: return 1 + (lt * rho)
    elif gc == 1 and gt == 1: return 1 - rho
    return 1.0

def media_ewma(serie, span):
    serie = serie.dropna()
    if len(serie) == 0: return None
    return serie.ewm(span=span, min_periods=1).mean().iloc[-1]

def media_pesata_decadimento(df, colonna, data_riferimento, emivita):
    if colonna not in df.columns: return None
    sub = df[[colonna, 'Date_parsed']].dropna()
    if len(sub) == 0: return None
    giorni = (data_riferimento - sub['Date_parsed']).dt.days.clip(lower=0)
    pesi = 0.5 ** (giorni / emivita)
    tot = pesi.sum()
    return sub[colonna].mean() if tot <= 0 else (sub[colonna] * pesi).sum() / tot

def scarica_csv_robusto(url, tentativi=3, attesa_secondi=2):
    """FIX #11 — BOM (Byte Order Mark): fixtures.csv inizia con un carattere
    invisibile Unicode che, se non gestito, si attacca al nome della prima
    colonna ("Div" diventa "\\ufeffDiv"), facendo fallire in silenzio ogni
    controllo tipo "if 'Div' in colonne" — nessuna fixture futura veniva mai
    trovata, su nessun campionato, per questo motivo esatto. 'utf-8-sig' lo
    rimuove in fase di decodifica; non fa danno se il file non ha il BOM."""
    varianti_url = [url]
    if "://www." in url:
        varianti_url.append(url.replace("://www.", "://"))
    elif "://" in url:
        varianti_url.append(url.replace("://", "://www."))

    ultimo_errore = None
    for tentativo in range(tentativi):
        for url_prova in varianti_url:
            try:
                resp = requests.get(url_prova, headers=HEADERS_BROWSER, timeout=15)
                resp.raise_for_status()
                testo = resp.content.decode('utf-8-sig', errors='replace')
                df = pd.read_csv(io.StringIO(testo))
                df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]  # rete di sicurezza extra
                return df, None
            except Exception as e:
                ultimo_errore = str(e)
        if tentativo < tentativi - 1:
            time.sleep(attesa_secondi)
    return None, ultimo_errore

def carica_dati_campionato(id_fd):
    """FIX B — aggiunto il tag 'Stagione' (precedente/corrente), necessario
    per poter validare il value bet SOLO sui risultati reali più recenti,
    non su quelli già usati per calibrare le medie di lega."""
    codice_corrente, codice_precedente = codici_stagione()
    frames = []
    for codice, label in [(codice_precedente, 'precedente'), (codice_corrente, 'corrente')]:
        url = f"https://football-data.co.uk/mmz4281/{codice}/{id_fd}.csv"
        df, _ = scarica_csv_robusto(url)
        if df is not None:
            df.columns = df.columns.str.strip()
            df['Stagione'] = label
            frames.append(df)
    if frames:
        dati = pd.concat(frames, ignore_index=True, sort=False)
        dati['Date_parsed'] = pd.to_datetime(dati['Date'], errors='coerce', dayfirst=True)
        return dati.dropna(subset=['Date_parsed']).sort_values('Date_parsed').reset_index(drop=True)
    return None

def estrai_partite_squadra_intelligente(squadra, df_coppa, df_globale):
    """Matching morbido (FIX #4): prima prova nel dataset della competizione
    stessa, poi nel dataset globale domestico come fallback."""
    if df_coppa is not None and not df_coppa.empty:
        maschera = df_coppa['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra)) | \
                   df_coppa['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
        f_coppa = df_coppa[maschera]
        if len(f_coppa) > 0:
            return f_coppa

    if df_globale is not None and not df_globale.empty:
        maschera = df_globale['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra)) | \
                   df_globale['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
        f_glob = df_globale[maschera]
        if len(f_glob) > 0:
            return f_glob

    return pd.DataFrame()

def serie_squadra(df, squadra, tipo):
    """tipo: 'gol_fatti', 'gol_subiti', 'tiri_fatti', 'corner_fatti'.
    Ritorna una Series con i valori attribuiti correttamente alla squadra,
    indipendentemente dal fatto che giocasse in casa o in trasferta."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    colonne_casa = {'gol_fatti': 'FTHG', 'gol_subiti': 'FTAG', 'tiri_fatti': 'HST', 'corner_fatti': 'HC'}
    colonne_trasf = {'gol_fatti': 'FTAG', 'gol_subiti': 'FTHG', 'tiri_fatti': 'AST', 'corner_fatti': 'AC'}
    col_c, col_t = colonne_casa[tipo], colonne_trasf[tipo]

    if col_c not in df.columns or col_t not in df.columns:
        return pd.Series(dtype=float)

    gioca_in_casa = df['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
    valori = df[col_c].where(gioca_in_casa, df[col_t])
    return valori.dropna()

def calcola_modello_completo(giocate_coppa, squadra_casa, squadra_trasferta, rho, ewma_span,
                              emivita, df_globale, data_riferimento=None):
    giocate_validi = giocate_coppa.dropna(subset=['FTHG', 'FTAG']) if giocate_coppa is not None else pd.DataFrame()
    if data_riferimento is None:
        data_riferimento = giocate_validi['Date_parsed'].max() if not giocate_validi.empty else pd.Timestamp(date.today())

    # Baseline di lega/competizione. Se il campione della competizione è
    # troppo piccolo (tipico per le coppe a inizio stagione), lo arricchiamo
    # con il dataset domestico globale per una stima più stabile.
    base_per_media = giocate_validi
    if len(giocate_validi) < 20 and df_globale is not None and not df_globale.empty:
        base_per_media = pd.concat([giocate_validi, df_globale.dropna(subset=['FTHG', 'FTAG'])], ignore_index=True, sort=False)

    m_gol_casa = media_pesata_decadimento(base_per_media, 'FTHG', data_riferimento, emivita) or 1.65
    m_gol_trasf = media_pesata_decadimento(base_per_media, 'FTAG', data_riferimento, emivita) or 1.25
    m_tiri_casa_lega = media_pesata_decadimento(base_per_media, 'HST', data_riferimento, emivita) or 4.8
    m_tiri_trasf_lega = media_pesata_decadimento(base_per_media, 'AST', data_riferimento, emivita) or 4.1
    m_corner_casa_lega = media_pesata_decadimento(base_per_media, 'HC', data_riferimento, emivita) or 5.4
    m_corner_trasf_lega = media_pesata_decadimento(base_per_media, 'AC', data_riferimento, emivita) or 4.6

    forma_casa = estrai_partite_squadra_intelligente(squadra_casa, giocate_coppa, df_globale)
    forma_trasf = estrai_partite_squadra_intelligente(squadra_trasferta, giocate_coppa, df_globale)
    n_casa, n_trasf = len(forma_casa), len(forma_trasf)

    # FIX CRITICO #12 — i valori vengono attribuiti alla squadra giusta
    # guardando, partita per partita, se giocava in casa o in trasferta.
    gf_casa_rec = media_ewma(serie_squadra(forma_casa, squadra_casa, 'gol_fatti'), ewma_span) if n_casa else None
    gs_casa_rec = media_ewma(serie_squadra(forma_casa, squadra_casa, 'gol_subiti'), ewma_span) if n_casa else None
    gf_trasf_rec = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'gol_fatti'), ewma_span) if n_trasf else None
    gs_trasf_rec = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'gol_subiti'), ewma_span) if n_trasf else None

    tiri_casa = media_ewma(serie_squadra(forma_casa, squadra_casa, 'tiri_fatti'), ewma_span) if n_casa else None
    corner_casa = media_ewma(serie_squadra(forma_casa, squadra_casa, 'corner_fatti'), ewma_span) if n_casa else None
    tiri_trasf = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'tiri_fatti'), ewma_span) if n_trasf else None
    corner_trasf = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'corner_fatti'), ewma_span) if n_trasf else None

    # Shrinkage: peso -> 0 quando n_casa/n_trasf sono pochi o zero, quindi la
    # stima converge verso il rapporto neutro 1.0 (= "come la media") invece
    # di un profilo inventato. Peso -> 1 quando il campione è ampio, quindi ci
    # si fida del dato specifico della squadra.
    peso_casa = n_casa / (n_casa + K_SHRINKAGE)
    peso_trasf = n_trasf / (n_trasf + K_SHRINKAGE)

    # FIX CRITICO #13 — riferimento corretto per i rapporti attacco/difesa.
    # I dati di una squadra mescolano partite in casa e in trasferta, quindi
    # vanno confrontati con la media di lega COMPLESSIVA (casa+trasferta), non
    # con quella specifica di casa o di trasferta. Prima si confrontava il dato
    # misto della squadra di casa con la media "solo casa" (più alta) e quello
    # della squadra ospite con la media "solo trasferta" (più bassa): risultato,
    # le squadre di casa risultavano sistematicamente sottovalutate e le ospiti
    # sopravvalutate — per questo il modello prevedeva più vittorie in trasferta
    # che in casa, il contrario di come funziona davvero il calcio.
    # Il vantaggio del fattore campo resta comunque applicato, più sotto, dal
    # fatto che lambda_casa usa m_gol_casa e lambda_trasferta usa m_gol_trasf.
    m_gol_complessiva = (m_gol_casa + m_gol_trasf) / 2.0

    rapp_attacco_casa = (gf_casa_rec / max(0.1, m_gol_complessiva)) if gf_casa_rec is not None else 1.0
    rapp_difesa_casa = (gs_casa_rec / max(0.1, m_gol_complessiva)) if gs_casa_rec is not None else 1.0
    rapp_attacco_trasf = (gf_trasf_rec / max(0.1, m_gol_complessiva)) if gf_trasf_rec is not None else 1.0
    rapp_difesa_trasf = (gs_trasf_rec / max(0.1, m_gol_complessiva)) if gs_trasf_rec is not None else 1.0

    attacco_casa = peso_casa * rapp_attacco_casa + (1 - peso_casa) * 1.0
    difesa_casa = peso_casa * rapp_difesa_casa + (1 - peso_casa) * 1.0
    attacco_trasf = peso_trasf * rapp_attacco_trasf + (1 - peso_trasf) * 1.0
    difesa_trasf = peso_trasf * rapp_difesa_trasf + (1 - peso_trasf) * 1.0

    # Stessa logica del FIX #13 anche per tiri e angoli: i dati della squadra
    # sono misti casa+trasferta, quindi il riferimento verso cui convergono
    # (shrinkage) dev'essere la media complessiva, non quella specifica.
    m_tiri_complessiva = (m_tiri_casa_lega + m_tiri_trasf_lega) / 2.0
    m_corner_complessiva = (m_corner_casa_lega + m_corner_trasf_lega) / 2.0

    tiri_casa_finale = peso_casa * tiri_casa + (1 - peso_casa) * m_tiri_complessiva if tiri_casa is not None else m_tiri_casa_lega
    corner_casa_finale = peso_casa * corner_casa + (1 - peso_casa) * m_corner_complessiva if corner_casa is not None else m_corner_casa_lega
    tiri_trasf_finale = peso_trasf * tiri_trasf + (1 - peso_trasf) * m_tiri_complessiva if tiri_trasf is not None else m_tiri_trasf_lega
    corner_trasf_finale = peso_trasf * corner_trasf + (1 - peso_trasf) * m_corner_complessiva if corner_trasf is not None else m_corner_trasf_lega

    lam_c = max(0.2, attacco_casa * difesa_trasf * m_gol_casa)
    lam_t = max(0.2, attacco_trasf * difesa_casa * m_gol_trasf)

    prob_1, prob_x, prob_2 = 0.0, 0.0, 0.0
    prob_goal, prob_nogoal = 0.0, 0.0
    limiti_under = [1.5, 2.5, 3.5]
    prob_under = {l: 0.0 for l in limiti_under}
    multigol_casa = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    multigol_trasf = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    combo_stats = {"1 + Goal": 0.0, "1 + Over 2.5": 0.0, "X + Under 2.5": 0.0, "2 + Goal": 0.0}
    griglia_risultati = []  # FIX #5 — griglia completa per il motore combo libero

    tot_p = 0.0
    for gc in range(AUDIT_SCORE_MAX + 1):
        for gt in range(AUDIT_SCORE_MAX + 1):
            p = poisson.pmf(gc, lam_c) * poisson.pmf(gt, lam_t) * tau_dixon_coles(gc, gt, lam_c, lam_t, rho) * 100
            tot_p += p
            segno = 'X' if gc == gt else ('1' if gc > gt else '2')
            griglia_risultati.append({"gc": gc, "gt": gt, "p": p, "segno": segno})

            if segno == '1': prob_1 += p
            elif segno == 'X': prob_x += p
            else: prob_2 += p

            if gc > 0 and gt > 0: prob_goal += p
            else: prob_nogoal += p

            for l in limiti_under:
                if gc + gt < l: prob_under[l] += p

            for mg_key, (mi_c, ma_c) in [("0-1", (0,1)), ("0-2", (0,2)), ("1-2", (1,2)), ("1-3", (1,3)), ("2-3", (2,3)), ("2-4", (2,4))]:
                if mi_c <= gc <= ma_c: multigol_casa[mg_key] += p
                if mi_c <= gt <= ma_c: multigol_trasf[mg_key] += p

            if segno == '1' and gc > 0 and gt > 0: combo_stats["1 + Goal"] += p
            if segno == '1' and (gc + gt) > 2.5: combo_stats["1 + Over 2.5"] += p
            if segno == 'X' and (gc + gt) < 2.5: combo_stats["X + Under 2.5"] += p
            if segno == '2' and gc > 0 and gt > 0: combo_stats["2 + Goal"] += p

    if tot_p > 0:
        f = 100.0 / tot_p
        prob_1, prob_x, prob_2 = prob_1*f, prob_x*f, prob_2*f
        prob_goal, prob_nogoal = prob_goal*f, prob_nogoal*f
        prob_under = {l: v*f for l, v in prob_under.items()}
        multigol_casa = {k: v*f for k, v in multigol_casa.items()}
        multigol_trasf = {k: v*f for k, v in multigol_trasf.items()}
        combo_stats = {k: v*f for k, v in combo_stats.items()}
        for r in griglia_risultati:
            r["p"] *= f

    return {
        "prob_1": prob_1, "prob_X": prob_x, "prob_2": prob_2,
        "prob_goal": prob_goal, "prob_nogoal": prob_nogoal,
        "prob_under": prob_under, "multigol_casa": multigol_casa, "multigol_trasf": multigol_trasf,
        "combo": combo_stats, "griglia": griglia_risultati,
        "angoli_stimati": f"{corner_casa_finale + corner_trasf_finale:.1f}",
        "tiri_stimati": f"{tiri_casa_finale + tiri_trasf_finale:.1f}",
        "n_casa": n_casa, "n_trasf": n_trasf,
    }

def classifica_colonne_quote(colonne):
    apertura_h = [c for c in colonne if c.endswith('H') and not c.endswith('CH') and c not in ['FTHG', 'HTHG', 'PTHG']]
    apertura_d = [c for c in colonne if c.endswith('D') and not c.endswith('CD') and c not in ['FTHG', 'FTAG', 'HTHG', 'HTAG']]
    apertura_a = [c for c in colonne if c.endswith('A') and not c.endswith('CA') and c not in ['FTAG', 'HTAG', 'PTAG']]
    return apertura_h, apertura_d, apertura_a

def classifica_colonne_over_under(colonne, soglia="2.5"):
    over_cols = [c for c in colonne if c.endswith(f'>{soglia}')]
    under_cols = [c for c in colonne if c.endswith(f'<{soglia}')]
    return over_cols, under_cols

def quote_medie_grezze_1x2(riga, colonne_h, colonne_d, colonne_a):
    def media_valori(cols):
        vals = []
        for c in cols:
            v = riga.get(c)
            if pd.notna(v) and isinstance(v, (int, float)) and v > 1.0:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None
    qh, qd, qa = media_valori(colonne_h), media_valori(colonne_d), media_valori(colonne_a)
    if qh is None or qd is None or qa is None:
        return None
    return {"1": qh, "X": qd, "2": qa,
            "n_bookmakers": min(sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_h),
                                  sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_d),
                                  sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_a))}

def quote_medie_grezze_ou(riga, colonne_over, colonne_under):
    def media_valori(cols):
        vals = []
        for c in cols:
            v = riga.get(c)
            if pd.notna(v) and isinstance(v, (int, float)) and v > 1.0:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None
    qo, qu = media_valori(colonne_over), media_valori(colonne_under)
    if qo is None or qu is None:
        return None
    return {"Over": qo, "Under": qu,
            "n_bookmakers": min(sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_over),
                                  sum(pd.notna(riga[c]) and isinstance(riga.get(c), (int, float)) and riga.get(c) > 1.0 for c in colonne_under))}

def _v5_raw_history(dati_completi, rho, ewma_span, emivita):
    """Previsioni raw walk-forward per tutto lo storico con quote disponibili."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    ch, cd, ca = classifica_colonne_quote(tutte.columns)
    co, cu = classifica_colonne_over_under(tutte.columns, '2.5')
    righe = []
    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(
            prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
            emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed')
        )
        if m is None:
            continue
        q12 = quote_medie_grezze_1x2(partita, ch, cd, ca)
        qou = quote_medie_grezze_ou(partita, co, cu)
        if q12 is None or qou is None:
            continue
        esito12 = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        tot = partita['FTHG'] + partita['FTAG']
        esito_ou = 'Over' if tot > 2.5 else 'Under'
        p1 = float(m['prob_1']) / 100.0
        px = float(m['prob_X']) / 100.0
        p2 = float(m['prob_2']) / 100.0
        po = (100.0 - float(m['prob_under'][2.5])) / 100.0
        pu = 1.0 - po
        righe.append({
            'indice': i, 'data': partita.get('Date_parsed'),
            'casa': partita['HomeTeam'], 'trasferta': partita['AwayTeam'],
            'p1': p1, 'px': px, 'p2': p2, 'po': po, 'pu': pu,
            'esito12': esito12, 'esito_ou': esito_ou,
            'q1': q12['1'], 'qx': q12['X'], 'q2': q12['2'],
            'qo': qou['Over'], 'qu': qou['Under'],
        })
    return pd.DataFrame(righe)

rho_val = -0.10

ewma_span_val = 6

emivita_val = 180

from sklearn.linear_model import LogisticRegression

from scipy.optimize import minimize_scalar

V12_CAL_MIN_TRAIN = 100

V12_METHODS = ['RAW', 'ISOTONIC', 'PLATT', 'BETA', 'SHRINK50']

def _v12_clip(p):
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))

def _v12_logloss_binary(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

def _v12_fit_isotonic(p, y):
    if len(p) < V12_CAL_MIN_TRAIN or len(set(y)) < 2:
        return None
    cal = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    cal.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    return cal

def _v12_fit_platt(p, y):
    if len(p) < V12_CAL_MIN_TRAIN or len(set(y)) < 2:
        return None
    x = np.asarray(p, dtype=float).reshape(-1, 1)
    yy = np.asarray(y, dtype=int)
    # Platt scaling: logistic regression on the raw probability.
    cal = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    cal.fit(x, yy)
    return cal

def _v12_fit_beta(p, y):
    if len(p) < V12_CAL_MIN_TRAIN or len(set(y)) < 2:
        return None
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    x = np.column_stack([np.log(p), np.log1p(-p)])
    yy = np.asarray(y, dtype=int)
    # Beta calibration in the canonical logistic form:
    # logit(p_cal) = a*log(p) + b*log(1-p) + c.
    cal = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    cal.fit(x, yy)
    return cal

def _v12_apply_binary(cal, p):
    if cal is None:
        return float(p)
    p = _v12_clip(p)
    if isinstance(cal, tuple) and cal and cal[0] == 'shrink50':
        alpha = float(cal[1])
        return float(np.clip(0.5 + alpha*(p-0.5), 0.001, 0.999))
    if isinstance(cal, tuple) and cal and cal[0] == 'beta':
        model = cal[1]
        x = np.array([[np.log(p), np.log1p(-p)]], dtype=float)
        return float(np.clip(model.predict_proba(x)[0,1], 0.001, 0.999))
    # Isotonic or Platt
    if isinstance(cal, IsotonicRegression):
        return float(np.clip(cal.predict([p])[0], 0.001, 0.999))
    return float(np.clip(cal.predict_proba(np.array([[p]], dtype=float))[0,1], 0.001, 0.999))

def _v12_fit_shrink50(p, y):
    if len(p) < V12_CAL_MIN_TRAIN or len(set(y)) < 2:
        return None
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # alpha=0 -> 50%; alpha=1 -> RAW.
    def objective(alpha):
        pc = np.clip(0.5 + alpha*(p-0.5), 1e-6, 1-1e-6)
        return _v12_logloss_binary(pc, y)

    res = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded',
                          options={'xatol': 1e-4})
    alpha = float(np.clip(res.x if res.success else 1.0, 0.0, 1.0))
    return ('shrink50', alpha)

def _v12_fit_method(name, p, y):
    if name == 'ISOTONIC':
        return _v12_fit_isotonic(p, y)
    if name == 'PLATT':
        return _v12_fit_platt(p, y)
    if name == 'BETA':
        fitted = _v12_fit_beta(p, y)
        return ('beta', fitted) if fitted is not None else None
    if name == 'SHRINK50':
        return _v12_fit_shrink50(p, y)
    return None

def _v12_apply_1x2(rawp, models):
    if models is None or not all(models.get(s) is not None for s in ['1','X','2']):
        return rawp.copy()
    cp = {s: _v12_apply_binary(models[s], rawp[s]) for s in ['1','X','2']}
    tot = sum(cp.values())
    if tot <= 0:
        return rawp.copy()
    return {s: cp[s]/tot for s in ['1','X','2']}

def _v12_build_paired(dati_completi, rho, ewma_span, emivita,
                      soglia_ev=0.10, warmup=100):
    """Costruisce un dataset OOS paired per RAW + 4 calibratori."""
    hist = _v5_raw_history(dati_completi, rho, ewma_span, emivita)
    if hist.empty:
        return None

    rows12, rowsou = [], []

    prior12_p = {'1': [], 'X': [], '2': []}
    prior12_y = {'1': [], 'X': [], '2': []}
    prior_ou_p, prior_ou_y = [], []

    for j, r in hist.iterrows():
        if j < warmup:
            prior12_p['1'].append(float(r.p1)); prior12_y['1'].append(int(r.esito12 == '1'))
            prior12_p['X'].append(float(r.px)); prior12_y['X'].append(int(r.esito12 == 'X'))
            prior12_p['2'].append(float(r.p2)); prior12_y['2'].append(int(r.esito12 == '2'))
            prior_ou_p.append(float(r.po)); prior_ou_y.append(int(r.esito_ou == 'Over'))
            continue

        raw12 = {'1': float(r.p1), 'X': float(r.px), '2': float(r.p2)}

        models12 = {
            method: {
                s: _v12_fit_method(method, prior12_p[s], prior12_y[s])
                for s in ['1','X','2']
            }
            for method in V12_METHODS if method != 'RAW'
        }
        probs12 = {'RAW': raw12.copy()}
        for method in V12_METHODS:
            if method == 'RAW':
                continue
            probs12[method] = _v12_apply_1x2(raw12, models12[method])

        rq = {'1': float(r.q1), 'X': float(r.qx), '2': float(r.q2)}
        row12 = {
            'data': r.data, 'esito': r.esito12,
            'p1_raw': raw12['1'], 'px_raw': raw12['X'], 'p2_raw': raw12['2']
        }

        for method in V12_METHODS:
            pp = probs12[method]
            choice = max(pp, key=pp.get)
            ev = pp[choice] * rq[choice] - 1.0
            bet = bool(ev >= soglia_ev)
            profit = ((rq[choice]-1.0) if choice == r.esito12 else -1.0) if bet else 0.0
            row12.update({
                f'{method.lower()}_p1': pp['1'],
                f'{method.lower()}_px': pp['X'],
                f'{method.lower()}_p2': pp['2'],
                f'{method.lower()}_conf': pp[choice],
                f'{method.lower()}_choice': choice,
                f'{method.lower()}_hit': bool(choice == r.esito12),
                f'{method.lower()}_ev': ev,
                f'{method.lower()}_in': bet,
                f'{method.lower()}_profit': profit,
                f'{method.lower()}_quota': rq[choice],
            })
        rows12.append(row12)

        rawou = {'Over': float(r.po), 'Under': float(r.pu)}
        ou_models = {
            method: _v12_fit_method(method, prior_ou_p, prior_ou_y)
            for method in V12_METHODS if method != 'RAW'
        }
        probsou = {'RAW': rawou.copy()}
        for method in V12_METHODS:
            if method == 'RAW':
                continue
            po_cal = _v12_apply_binary(ou_models[method], rawou['Over'])
            probsou[method] = {'Over': po_cal, 'Under': 1.0-po_cal}

        oq = {'Over': float(r.qo), 'Under': float(r.qu)}
        rowou = {'data': r.data, 'esito': r.esito_ou,
                  'po_raw': rawou['Over'], 'pu_raw': rawou['Under']}

        for method in V12_METHODS:
            pp = probsou[method]
            choice = max(pp, key=pp.get)
            ev = pp[choice] * oq[choice] - 1.0
            bet = bool(ev >= soglia_ev)
            profit = ((oq[choice]-1.0) if choice == r.esito_ou else -1.0) if bet else 0.0
            rowou.update({
                f'{method.lower()}_po': pp['Over'],
                f'{method.lower()}_pu': pp['Under'],
                f'{method.lower()}_conf': pp[choice],
                f'{method.lower()}_choice': choice,
                f'{method.lower()}_hit': bool(choice == r.esito_ou),
                f'{method.lower()}_ev': ev,
                f'{method.lower()}_in': bet,
                f'{method.lower()}_profit': profit,
                f'{method.lower()}_quota': oq[choice],
            })
        rowsou.append(rowou)

        # IMPORTANT: current observation enters training only AFTER evaluation.
        prior12_p['1'].append(float(r.p1)); prior12_y['1'].append(int(r.esito12 == '1'))
        prior12_p['X'].append(float(r.px)); prior12_y['X'].append(int(r.esito12 == 'X'))
        prior12_p['2'].append(float(r.p2)); prior12_y['2'].append(int(r.esito12 == '2'))
        prior_ou_p.append(float(r.po)); prior_ou_y.append(int(r.esito_ou == 'Over'))

    return {'1X2': pd.DataFrame(rows12), 'O/U 2.5': pd.DataFrame(rowsou)}

def _v12_metriche(df, merc, common_only=False):
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[df.raw_in & df.cal_dummy].copy() if False else df.copy()
    if common_only:
        mask = np.ones(len(df), dtype=bool)
        for m in V12_METHODS:
            mask &= df[f'{m.lower()}_in'].to_numpy(dtype=bool)
        d = df.loc[mask].copy()
    if d.empty:
        return pd.DataFrame()

    out = []
    for method in V12_METHODS:
        key = method.lower()
        hits = d[f'{key}_hit'].to_numpy(dtype=bool)
        n = len(d)
        probs = []
        true_probs = []
        brier = []
        logloss = []
        for _, r in d.iterrows():
            if merc == '1X2':
                ps = np.array([r[f'{key}_p1'], r[f'{key}_px'], r[f'{key}_p2']], dtype=float)
                y = np.array([r.esito == '1', r.esito == 'X', r.esito == '2'], dtype=float)
                true_p = float(ps[int(np.argmax(y))])
                brier.append(float(np.sum((ps-y)**2)))
            else:
                po = float(r[f'{key}_po'])
                ps = {'Over': po, 'Under': 1-po}
                true_p = ps[r.esito]
                brier.append(float((po-(r.esito == 'Over'))**2 +
                                   ((1-po)-(r.esito == 'Under'))**2))
            true_probs.append(true_p)
            logloss.append(-np.log(_v12_clip(true_p)))

        profit = d[f'{key}_profit'].to_numpy(dtype=float)
        bets = d[f'{key}_in'].to_numpy(dtype=bool)
        nb = int(bets.sum())
        out.append({
            'metodo': method,
            'n': n,
            'accuracy_pct': float(hits.mean()*100),
            'brier': float(np.mean(brier)),
            'log_loss': float(np.mean(logloss)),
            'calibration_mae_pp': float(np.mean(np.abs(
                np.asarray(true_probs)*100 -
                np.asarray(hits, dtype=float)*100
            ))),
            'bets_ev10': nb,
            'profit_ev10_u': float(profit.sum()),
            'roi_ev10_pct': float(100*profit.sum()/nb) if nb else np.nan
        })
    return pd.DataFrame(out)

def _v12_band_table(df, merc, common_only=False):
    if df is None or df.empty:
        return pd.DataFrame()
    if common_only:
        mask = np.ones(len(df), dtype=bool)
        for m in V12_METHODS:
            mask &= df[f'{m.lower()}_in'].to_numpy(dtype=bool)
        d = df.loc[mask].copy()
    else:
        d = df.copy()
    if d.empty:
        return pd.DataFrame()

    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0000001]
    labels = ['0-40%','40-50%','50-60%','60-70%','70-80%','80-100%']
    out = []

    for method in V12_METHODS:
        key = method.lower()
        conf = np.asarray(d[f'{key}_conf'], dtype=float)
        hit = np.asarray(d[f'{key}_hit'], dtype=bool)
        idx = np.digitize(conf, bins[1:-1], right=False)
        for k, label in enumerate(labels):
            mask = idx == k
            n = int(mask.sum())
            if not n:
                continue
            p = conf[mask]
            h = hit[mask]
            profit = np.asarray(d.loc[mask, f'{key}_profit'], dtype=float)
            bets = np.asarray(d.loc[mask, f'{key}_in'], dtype=bool)
            nb = int(bets.sum())
            out.append({
                'metodo': method,
                'fascia': label,
                'n_opportunita': n,
                'prob_media': float(p.mean()*100),
                'frequenza_reale': float(h.mean()*100),
                'errore_cal_pp': float(h.mean()*100-p.mean()*100),
                'bets_ev10': nb,
                'roi_ev10_pct': float(100*profit.sum()/nb) if nb else np.nan,
                'profit_ev10_u': float(profit.sum())
            })
    return pd.DataFrame(out)

def _v12_build_all(warmup=100):
    per_metriche = []
    per_bands = []
    dfs = {'1X2': {}, 'O/U 2.5': {}}
    errors = []

    for camp, info in CAMPIONATI_DOMESTICI.items():
        try:
            dati = carica_dati_campionato(info['id_fd'])
            paired = _v12_build_paired(
                dati, rho_val, ewma_span_val, emivita_val, 0.10, warmup
            )
            if paired is None:
                raise ValueError('dati OOS vuoti')

            for merc, df in paired.items():
                dfs[merc][camp] = df
                for common in [False, True]:
                    m = _v12_metriche(df, merc, common)
                    if not m.empty:
                        m.insert(0, 'campionato', camp)
                        m.insert(1, 'mercato', merc)
                        m.insert(2, 'universo',
                                  'COMMON BETS' if common else 'ALL OOS')
                        per_metriche.append(m)

                    b = _v12_band_table(df, merc, common)
                    if not b.empty:
                        b.insert(0, 'campionato', camp)
                        b.insert(1, 'mercato', merc)
                        b.insert(2, 'universo',
                                  'COMMON BETS' if common else 'ALL OOS')
                        per_bands.append(b)
        except Exception as e:
            errors.append((camp, f'{type(e).__name__}: {e}'))

    metrics = pd.concat(per_metriche, ignore_index=True) if per_metriche else pd.DataFrame()
    bands = pd.concat(per_bands, ignore_index=True) if per_bands else pd.DataFrame()
    return metrics, bands, errors

def mostra_v12_validation():
    st.divider()
    st.markdown('## 🧬 V12.1 — CONFRONTO 5 METODI DI CALIBRAZIONE OOS · CPU OPTIMIZED')
    st.caption(
        'RAW vs Isotonic vs Platt vs Beta vs Shrink50. Cache attiva e V11 non viene eseguito automaticamente. '
        'Tutti i calibratori sono walk-forward: per ogni partita usano solo il passato. '
        '1X2 e O/U 2.5 sono valutati separatamente.'
    )

    c1, c2 = st.columns(2)
    with c1:
        wu = st.number_input('Warm-up V12', 50, 300, 100, 10, key='v12_warm')
    with c2:
        st.metric('Metodi', '5')

    st.info(
        'Brier e Log Loss più bassi indicano probabilità più accurate. '
        'Errore calibrazione vicino a 0 indica migliore allineamento tra probabilità '
        'e frequenza osservata. ROI/profit sono proxy storiche sulle selezioni EV≥10%. '
        'Le fasce alte possono avere campioni piccoli.'
    )

    if st.button('🧬 ESEGUI V12 — CONFRONTO 5 LEGHE', key='v12_run'):
        try:
            with st.spinner('V12.1 in esecuzione: 5 metodi × 5 leghe... primo run può essere pesante; i successivi usano cache.'):
                metrics, bands, errors = _v12_build_all(int(wu))

            if not metrics.empty:
                st.markdown('### 1. Metriche OOS per lega')
                st.dataframe(
                    metrics[
                        ['campionato','mercato','universo','metodo','n',
                         'accuracy_pct','brier','log_loss',
                         'calibration_mae_pp','bets_ev10',
                         'roi_ev10_pct','profit_ev10_u']
                    ],
                    use_container_width=True, hide_index=True
                )
                st.download_button(
                    '⬇️ Scarica V12 metriche',
                    data=metrics.to_csv(index=False).encode('utf-8'),
                    file_name='V12_metriche_metodi.csv',
                    mime='text/csv', key='v12_dl_metrics'
                )

            if not bands.empty:
                st.markdown('### 2. Reliability per fasce di probabilità')
                st.dataframe(bands, use_container_width=True, hide_index=True)
                st.download_button(
                    '⬇️ Scarica V12 fasce',
                    data=bands.to_csv(index=False).encode('utf-8'),
                    file_name='V12_reliability_fasce.csv',
                    mime='text/csv', key='v12_dl_bands'
                )

            if errors:
                st.error('Campionati non completati:')
                for nome, msg in errors:
                    st.write(f'- **{nome}**: {msg}')

        except Exception as e:
            st.error(f'Errore V12: {type(e).__name__}: {e}')

try:
    mostra_v12_validation()
except Exception as _v12_err:
    st.error(f"V12 CLEAN non disponibile: {type(_v12_err).__name__}: {_v12_err}")
