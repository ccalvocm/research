
import geopandas as gpd
import shapely
import os
import pandas as pd
from unidecode import unidecode
import numpy as np
from pyproj import Transformer
import geopandas as gpd
import pandas as pd
import geopandas as gpd
from unidecode import unidecode
import locale
locale.setlocale(locale.LC_NUMERIC, "es_ES")
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import os
import numpy as np
import re
import contextily as ctx

CRS = {'1984': {'18': 'EPSG:32718', '19': 'EPSG:32719'},
       '1969': {'18': 'EPSG:29188', '19': 'EPSG:29189'},
       '1956': {'18': 'EPSG:24878', '19': 'EPSG:24879'},
       '84': {'18': 'EPSG:32718', '19': 'EPSG:32719'}}

CRSg = {'1984': 'EPSG:4326',
        '1956': 'EPSG:4248',
        '1969': 'EPSG:4724'}

def clear_spaces(x):
    x = x.rstrip()
    return x

def load_DAA_xls(path, **kwargs):
    df = pd.read_excel(path, **kwargs)

    return df

def clean_headers(column_list, char_list):
    column_list2 = [unidecode(x) for x in column_list]

    for char in char_list:
        column_list2 = [x.replace(char, '_') for x in column_list2]

    return column_list2

def replace_empty(s):
    if len(s) < 1:
        s = '0'
    else:
        pass
    return s

def latlon2dd(num):
    invalid = ['nan', np.nan, 0, '', ' ', '0', 'S/N']
    if num not in invalid:
        num = str(num)
        if num not in "nan":
            num = str(num)
            grados = float(num[0:2])
            minutos = float(num[2:4])/60
            segundos = float(num[4:6])/3600
            dd = -(grados+minutos+segundos)
            return dd
        else:
            return num
    else:
        return np.nan


def get_georreferenciables_UTM(df, utm_tuples):
    for tup in utm_tuples:
        wdatum = df.iloc[:, tup[2]].astype(str).isin(['1984', '1956', '1969'])
        whuso = df.iloc[:, tup[3]].astype(str).isin(['18', '19'])
        wx = (df.iloc[:, tup[0]] > 0) & df.iloc[:, tup[0]].notna()
        wy = (df.iloc[:, tup[1]] > 0) & df.iloc[:, tup[1]].notna()
        filtro = wdatum & whuso & wx & wy
    return filtro


def get_georreferenciables_latlon(df, geo_tuples):
    for tup in geo_tuples:
        for element in tup:
            if df.iloc[:, element].astype(str).isin(['1984', '1956', '1969']).any():
                df.iloc[:, element] = df.iloc[:, element].astype(
                    str).replace(' ', '')
                df.iloc[:, element] = df.iloc[:,
                                              element].astype(str).replace('S/N', '')
                df.iloc[:, element] = df.iloc[:, element].apply(replace_empty)
            else:
                pass
        wdatum = df.iloc[:, tup[2]].isin(['1984', '1956', '1969'])
        # df.iloc[:, tup[0]] = df.iloc[:, tup[0]].str.strip().astype(int)
        # df.iloc[:, tup[1]] = df.iloc[:, tup[1]].str.strip().astype(int)
        df.iloc[:, tup[0]] = pd.to_numeric(
            df.iloc[:, tup[0]], errors='coerce').astype('float')
        df.iloc[:, tup[1]] = pd.to_numeric(
            df.iloc[:, tup[1]], errors='coerce').astype('float')
        wx = (df.iloc[:, tup[0]] > 0) & df.iloc[:,
                                                tup[0]].astype(float).notna()
        wy = (df.iloc[:, tup[1]] > 0) & df.iloc[:,
                                                tup[1]].astype(float).notna()
        filtro = wdatum & wx & wy
    return filtro


def x_non_negative(df, col):
    df_c = df.copy()
    idx = df_c[col][df_c[col] < 0].index
    df_c.loc[idx, col] = -df_c.loc[idx, col].values
    return df_c


def drop_infinity(df, fields):
    # df[fields].replace([np.inf,-np.inf], np.nan, inplace=True)
    df[fields] = df[fields].replace([np.inf, -np.inf], np.nan)
    newdf = df.dropna(subset=fields, how="all")
    return newdf


def multiply(list_N, column, gdf_, fac):
    for col in column:
        gdf_.loc[list_N, col] = gdf_.loc[list_N, col]*fac
    return gdf_


def Huso19(list_N, column, gdf_):
    for col in column:
        gdf_.loc[list_N, col] = '19'
    return gdf_


def change_first_digit(list_N, column, gdf_):
    for col in column:
        for index in list_N:
            coord = str(gdf_.loc[index, col])
            first_coord = coord[0]
            list_coord = list(coord)
            if first_coord in ['5', '6']:
                list_coord[0] = '7'
                coord = "".join(list_coord)
                coord = "".join(list_coord)
            else:
                list_coord[0] = '5'
                coord = "".join(list_coord)
            gdf_.loc[index, col] = float(coord)
    return gdf_


def remove_ln(df_):
    df_.columns = [x.replace('\n', '') for x in df_.columns]
    return df_


def pad(df, col, n):
    df[col].apply(lambda x: x.str.ljust(7, '0'))
    return df

def padNum(serie,n):
    serie=serie.copy()
    ret=serie.replace(0,1.1).apply(lambda x: x*10**(math.ceil(n-np.log10(x))))
    return ret

def df_to_gdf(minE, maxE, minN, maxN, **kwargs):
    # Paso 1: cargar el archivo
    # los encabezados estan en la fila 7 (6 en indexacion python)
    name = kwargs.get('name')

    if len(kwargs.get('df_path')):
        path = os.path.join('..', 'Datos', 'DAA', kwargs.get('df_path'))
        kwargs = {'index_col': 0,
                  'skiprows': 6}
        df = load_DAA_xls(path, **kwargs)
    else:
        df = kwargs.get('df')
        print(df.head())

    # Paso 2: limpiar encabezados
    df.columns = clean_headers(df.columns, ['\n', ' ', '?', '!', '/', '.'])

    # Paso 3: revisar los georreferenciables
    # en este caso en particular tenemos los siguientes campos

    # 39: UTM Norte Captacion
    # 40: UTM Este Captacion
    # 41: Huso
    # 42: Datum
    # 43: Latitud Captacion
    # 44: Longitud Captacion
    # 45: Datum 1
    # 46: UTM Norte Restitucion
    # 47: UTM Este Restitucion
    # 48: Latitud Restitucion
    # 49: Longitud Restitucion

    # Condiciones para ser georreferenciable

    # 39 y 40 y 41 y 42
    # o bien
    # 43 y 44 y (42 o 45)

    # Paso 4: filtrar UTM y coordenadas geograficas

    x_ = np.where(df.columns.str.contains(r'(?=.*Este)(?=.*Captacion)',
                                          regex=True))[0][0]
    y_ = np.where(df.columns.str.contains(r'(?=.*Norte)(?=.*Captacion)',
                                          regex=True))[0][0]
    datum_ = np.where(df.columns.str.contains('Datum'))[0][0]
    huso_ = np.where(df.columns.str.contains('Huso'))[0][0]

    xlon_ = np.where(df.columns.str.contains(
        r'(?=.*Longitud)(?=.*Captacion)'))[0][0]
    ylon_ = np.where(df.columns.str.contains(
        r'(?=.*Latitud)(?=.*Captacion)'))[0][0]
    datumg_ = np.where(df.columns.str.contains('Datum'))[0][-1]

    utm_tuples = [(x_, y_, datum_, huso_)]  # (x,y,datum,huso)
    geo_tuples = [(xlon_, ylon_, datumg_)]  # (xlon,ylat,datum)

    filtroUTM = get_georreferenciables_UTM(df, utm_tuples)
    filtrogeo = get_georreferenciables_latlon(df, geo_tuples)

    # Paso intermedio: rescatar los indices
    df['ID'] = df.index

    # Paso 5: crear nuevo dataframe con solo lo georreferenciable
    df_new = df[filtroUTM | filtrogeo].copy()

    xcol = df_new.columns[x_]
    ycol = df_new.columns[y_]
    datcol = df_new.columns[datum_]
    huscol = df_new.columns[huso_]

    latcol = df_new.columns[ylon_]
    loncol = df_new.columns[xlon_]
    datgcol = df_new.columns[datumg_]

    for idx in df_new[filtroUTM].index:
        d = df_new.loc[idx, datcol]
        h = df_new.loc[idx, huscol]
        x = df_new.loc[idx, xcol]
        y = df_new.loc[idx, ycol]
        df_new.loc[idx, 'x'] = x
        df_new.loc[idx, 'y'] = y
        df_new.loc[idx, 'CRS_og'] = CRS[d][h]

    for idx in df_new[~filtroUTM & filtrogeo].index:
        lat = df_new.loc[idx, latcol]
        lon = df_new.loc[idx, loncol]
        lat = latlon2dd(lat)
        lon = latlon2dd(lon)
        dat = str(df_new.loc[idx, datgcol])
        df_new.loc[idx, 'x'] = lon
        df_new.loc[idx, 'y'] = lat
        df_new.loc[idx, 'CRS_og'] = CRSg[dat]

    projectionsR = [x for x in df_new['CRS_og'].unique() if x not in [
        'none', 'EPSG:32719']]
    transformers = {}
    for proj in projectionsR:
        transformers[proj] = Transformer.from_crs(
            proj, 'EPSG:32719', always_xy=True)

    to_transform = ~df_new['CRS_og'].isin(['EPSG:32719'])
    df_new['x_32719'] = df_new[xcol].copy()
    df_new['y_32719'] = df_new[ycol].copy()

    for idx in df_new[to_transform].index:
        CRSog = df_new.loc[idx, 'CRS_og']
        if CRSog not in ['EPSG:32719', 'NONE']:
            x = df_new.loc[idx, 'x']
            y = df_new.loc[idx, 'y']
            newx, newy = transformers[CRSog].transform(x, y)
            df_new.loc[idx, 'x_32719'] = newx
            df_new.loc[idx, 'y_32719'] = newy
        elif CRSog == 'EPSG:32719':
            x = df_new.loc[idx, 'x']
            y = df_new.loc[idx, 'y']
            df_new.loc[idx, 'x_32719'] = x
            df_new.loc[idx, 'y_32719'] = y
        else:
            pass

    # Paso : rescatar las coordenadas leidas como negativas

    df_new = x_non_negative(df_new, 'x_32719')

    # Paso : filtrar las coordenadas buenas del nuevo Dataframe

    filx = (df_new['x_32719'] > minE) & (df_new['x_32719'] < maxE)
    fily = (df_new['y_32719'] > minN) & (df_new['y_32719'] < maxN)
    df_export = df_new[filx & fily].copy()

    df_export = drop_infinity(df_export, ['x_32719', 'y_32719'])

    gdf = gpd.GeoDataFrame(df_export,
                           crs='EPSG:32719',
                           geometry=gpd.points_from_xy(df_export['x_32719'],
                                                       df_export['y_32719']))

    gdf['Tipo_Derecho'] = gdf['Tipo_Derecho'].apply(clear_spaces)
    gdf.to_file(name.replace('.xls', '')+'.geojson',
                driver='GeoJSON', encoding='utf-8')

def fill_column(df_,cols_,column_):
    """
    

    Parameters
    ----------
    df_ : Pandas Dataframe
        Dataframe de entrada.
    cols_ : list
        lista de columnas con información.

    Returns
    -------
    df_ : Pandas Dataframe
        Dataframe de salida.

    """
    for col in cols_:
        if not any(df_[col].dropna()=='SI'):
            df_[column_].fillna(df_[col].str.strip().replace('',np.nan),inplace=True)
    return df_

def homogenCols(dfDAA,dfUCH):
    dfUCH=dfUCH.copy()
    keys=['Fecha_de_Resolucion','Tipo_Derec','Naturaleza',
    'Uso_del_Agua','Ejercicio','Caudal_AnualProm','Unidad_de_Caudal','Huso','Datum_1',
    'Codigo_de_Expediente','Nombre_Sol','Fuente']
    
    dictKeys={'Caudal_AnualProm':'Q_Ano_pr',
              'Unidad_de_Caudal':'Unidad_Q',
              'Fuente':'Fuente','Uso_del_Agua':'Uso_Agua',
              'Codigo_de_Expediente':'Cod_Expedi',
              'Fecha_de_Resolucion':'Fecha_Reso'}
    
    for nom in keys:
        if nom in list(dictKeys.keys()):
            colUCH=dfUCH.columns[dfUCH.columns.str.contains(dictKeys[nom].lower(),
            case=False)][0]
        else:
            colUCH=dfUCH.columns[dfUCH.columns.str.contains(nom.lower(),
            case=False)][0]
        
        colDAA=dfDAA.columns[dfDAA.columns.str.contains(nom.lower(),
        case=False)][0]
        dfUCH.rename(columns={colUCH:colDAA},inplace=True)

    dfRet=dfUCH.loc[:,~dfUCH.columns.duplicated()].copy()
        
    return dfRet

def fixColsUnderScore(df):
    import re
    df=df.copy()
    df.columns=[re.sub('__','_',x) for x in df.columns]
    df.columns=[re.sub('_$','',x) for x in df.columns]
    df.columns=[re.sub('$','',x) for x in df.columns]
    return df

def add_daa_uch(df,df_uch):
    import shapely.wkt
    import re
    """
    

    Parameters
    ----------
    df : GeoDataFrame
        geodataframe de daa actualizados.
    df_uchile : GeoDataFrame
        geodataframe de los daa de Uch 2018.
    reg : str
        region.
        
    Returns
    -------
    None.

    """
    # tag de los DAA UCh
    df_uch['SOURCE']='UCh'
    
    # tag de los DAA DGA
    df['SOURCE']='DGA'
    
    df=df.copy()
    # arreglar las columnas para que calce el concat
    df=fixColsUnderScore(df)
    # cargar los daa de la Uchile
    
    # columnas para merge
    str_exp='Expedi'
    
    # expediente
    cols_exp=df.columns[df.columns.str.contains(str_exp)]
    cols_exp=[x for x in cols_exp if 'Antiguo' not in x]
    
    # nombre
    # cols_nombre=df.columns[df.columns.str.contains(str_nom,regex=True)]
    
    # antes del link completar los expedienetes que están en varias columnas
    df=fill_column(df,cols_exp,cols_exp[0])
        
    # match de nombres
    cols_link=df_uch.columns[df_uch.columns.str.contains(str_exp,
case=False)]
 # | df_uch.columns.str.contains(str_nom,case=False)]
    cols_uchile=df_uch[cols_link].apply(lambda row: '-'.join(
    row.str.strip().apply(lambda y: re.sub('[^A-Za-z0-9]+','',y))),axis=1)

    cols_link=[df.columns[df.columns.str.contains(str_exp,case=False)][1]]
# ,df.columns[df.columns.str.contains(str_nom,case=False)][0]]
    cols_link=[x for x in cols_link if 'Antiguo' not in x]
    back=df[cols_link].copy().apply(
    lambda x: ','.join(x.dropna().astype(str)),axis=1).copy()
    df=df[[x for x in df.columns if x not in cols_link]]
    df[cols_link[0]]=back
    cols_shac=df[cols_link].apply(lambda row: '-'.join(
    row.str.strip().apply(lambda y: re.sub('[^A-Za-z0-9]+','',y))),axis=1)
    
    # guardar los adicionales que tienen distinto expediente y nombre
    daa_uch_extra=df_uch[~cols_uchile.isin(cols_shac)]
   
    # ahora filtrar los que tengan las mismas coordenadas, mismo caudal
    presicion=0
    idx_daa_uch_same_xy=daa_uch_extra[daa_uch_extra.geometry.apply(lambda x: shapely.wkt.dumps(x,rounding_precision=presicion)).isin(df.geometry.apply(lambda x: shapely.wkt.dumps(x, 
rounding_precision=presicion)))].index
                                                                            
#     geometria=daa_uch_extra.loc[idx_daa_uch_same_xy].geometry.apply(lambda x: shapely.wkt.dumps(x, 
# rounding_precision=presicion))
    idx_daa_df_same_xy=df[df.geometry.apply(lambda x:shapely.wkt.dumps(x, 
rounding_precision=presicion)).isin(daa_uch_extra.geometry.apply(lambda x:shapely.wkt.dumps(x, 
rounding_precision=presicion)))].index
    
    # dentro de los que tengan las mismas coordenadas, remover los que presenten
    # el mismo tipo, naturaleza, ejercicio, caudal y unidad de caudal
    lista_match_df=['Tipo_Derecho','Naturaleza_del_Agua',
                    'Ejercicio_del_Derecho']
    lista_match_uch=['Tipo_Derec','Naturaleza','Ejercicio']
     
    # primero completar las columnas multiples de la DGA
    for strr in lista_match_df:
        string=strr.split('_')[0]
        df=fill_column(df,df.columns[df.columns.str.contains(string)],strr)
    
#     # identificar los que tienen misma coordenada, mismo tipo, naturaleza,
#     # ejercicio, caudal y unidad       
    df_check=df_uch.loc[idx_daa_uch_same_xy,
lista_match_uch].apply(lambda row: '-'.join(row.str.replace('.',
',').str.strip().values.astype(str)),axis=1).isin(df.loc[idx_daa_df_same_xy,
lista_match_df].apply(lambda row: '-'.join(row.str.strip().values.astype(str)),
axis=1).str.strip())
    idx_identicos=df_check[df_check].index
    # remover los daa con igual ubicación y mismo caudal
    daa_uch_extra.drop(index=idx_identicos,inplace=True)
        
    df = df.loc[:,~df.columns.duplicated()].copy()
    gdf=gpd.GeoDataFrame(pd.concat([df,homogenCols(df,daa_uch_extra)],
                                   ignore_index=True))
    
    return gdf

def load():
    gdf=gpd.read_file(r'G:\OneDrive - ciren.cl\Licitación CNR OH Zona Centro\IFI V3 DIGITAL\05_SIG_CNR_IFI\SIG CNR IFI Cuenca 01 Maipo\02_Capas\01_CUENCA RIO MAIPO\12_DEMANDA DE AGUA\01_INSUMOS\03_VARIOS INSUMOS\AR_Maipo.shp')
    ar=gdf[['NOM_CANAL','COD_CANAL', 'FUENTE_ABA','geometry']]

    uso=gpd.read_file(r'C:\Users\ccalvo\Downloads\solicitud_230407(uso_actual_bgoffin)\cut_rm_2021.shp')
    usoAg=uso[uso['CLASE'].str.contains('agricola',case=False,na=False)]

    return usoAg,ar

def sanitizeSjoin(gdf,field='ID'):
    gdfRet=gdf.copy()
    gdfRet['area']=gdfRet.geometry.area
    #Sort by area so largest area is last
    gdfRet.sort_values(by='area',inplace=True,ascending=True)

    #Drop duplicates, keep last/largest
    gdfRet.drop_duplicates(subset=field, keep='last', inplace=True)
    return gdfRet

def filterArea(uso,ar):
    usoJoin=gpd.sjoin(uso, ar, how='left', op='intersects')
    usoJoin['ID']=usoJoin.reset_index().index
    uso2=sanitizeSjoin(usoJoin)
    usoSis=uso2.dissolve(by='NOM_CANAL')

    usoSis['area']=usoSis.geometry.area/1e4
    # los sistemas mayores a 1000 ha

    uso3=usoSis[usoSis['area']>890]
    uso3.index.name='NOM_CANAL'
    uso3.reset_index(inplace=True)
    return uso3

def getDiff(df):
    gdf=gpd.read_file('Maipo.geojson')
    #get df Ndeg columns is not in gdf Ndeg columns
    diff=df[~df['Ndeg'].isin(list(gdf['Ndeg']))]
    return diff

def filterNoCoordinates(df):
    fieldsC=['UTM_Este_Captacion(m)','UTM_Norte_Captacion(m)']
    fieldsR=['UTM_Este_Restitucion(m)','UTM_Norte_Restitucion(m)']
    df=df.copy()
    df[fieldsC+fieldsR]=df[fieldsC+fieldsR].replace(0,np.nan)
    df=df[(df[fieldsC].notna().all(axis=1)) | (df[fieldsR].notna().all(axis=1))]

    # complete missing 'UTM_Este_Captacion(m)' with 'UTM_Este_Restitucion(m)'
    df.loc[df['UTM_Este_Captacion(m)'].isna(),'UTM_Este_Captacion(m)']=df.loc[df['UTM_Este_Captacion(m)'].isna(),'UTM_Este_Restitucion(m)']
    df.loc[df['UTM_Norte_Captacion(m)'].isna(),'UTM_Norte_Captacion(m)']=df.loc[df['UTM_Norte_Captacion(m)'].isna(),'UTM_Norte_Restitucion(m)']
    return df

def DAA():
    # -----------------------------------------------------------------------------
    # minE,maxE,minN,maxN = 1e5,1e6,1e6,1e7
    # -----------------------------------------------------------------------------   
    # script referenciacion
    df = pd.read_excel(os.path.join('.','data',
                    'Derechos_Concedidos_XIII_Region.xls'), skiprows=6)
    df = remove_ln(df)
    global rm 
    rm=gpd.read_file(os.path.join('.', 'data', 'rm.shp'))
    reg_buffer = rm.buffer(1e3)
    o, s, e, n = reg_buffer.bounds.loc[0, ['minx', 'miny', 'maxx', 'maxy']]
    df_to_gdf(o,e,s,n,df_path='',df=df,name='Maipo')
    # %% DAA no georreferenciables
    gdf_no_georref_raw=getDiff(df)

    # filtrar sin coordenadas
    gdf_no_georref=filterNoCoordinates(gdf_no_georref_raw)

    # uniform the different orders of magnitud  in 'UTM_Este_Captacion(m)' column so every one of them is of the order of 10^5
    gdf_no_georref['UTM_Este_Captacion(m)']=padNum(gdf_no_georref['UTM_Este_Captacion(m)'],5)
    gdf_no_georref['UTM_Norte_Captacion(m)']=padNum(gdf_no_georref['UTM_Norte_Captacion(m)'],6)


    # %% DAA no georreferenciables
    gdf_no_georref_final1=gdf_no_georref[(gdf_no_georref['UTM_Este_Captacion(m)']>0) & (gdf_no_georref['UTM_Norte_Captacion(m)']>0)]

    gdf_no_georref_final=gdf_no_georref_final1.copy()
    gdf_no_georref_final['Datum']=gdf_no_georref['Datum'].str.strip().replace('',np.nan).fillna('1984')
    gdf_no_georref_final['Huso']=gdf_no_georref['Huso'].str.strip().replace('',np.nan).fillna('19')

#=============================================================================
#                                reproyectar
# =============================================================================

    # %%
    gdf_no_georref_final["CRS"] = gdf_no_georref_final.apply(
        lambda x: CRS[x['Datum']][x['Huso']], axis=1)

    gdf_no_georref_final['geometry'] = gdf_no_georref_final.apply(lambda x:
                                                                  Transformer.from_crs(x["CRS"], 'EPSG:32719', always_xy=True).transform(x['UTM_Este_Captacion(m)'],
                                                                                                                                         x['UTM_Norte_Captacion(m)']), axis=1)
    x = gdf_no_georref_final['geometry'].apply(lambda x: x[0])
    y = gdf_no_georref_final['geometry'].apply(lambda x: x[1])

    gdf_no_georref_final = gpd.GeoDataFrame(gdf_no_georref_final,
                                            geometry=gpd.points_from_xy(x, y), crs='EPSG:32719')


# =============================================================================
#                                DAA que originalmente faltaban
# =============================================================================

    DAA_orig = gpd.read_file(os.path.join('.', 'Maipo.geojson'))

    # %%
    DAA_to_fix = DAA_orig[(DAA_orig.geometry.x>rm.bounds.loc[0,'maxx']) | (DAA_orig.geometry.y > rm.bounds.loc[0,'maxy'])
| (DAA_orig.geometry.y < rm.bounds.loc[0,'miny'])| 
(DAA_orig.geometry.x<rm.bounds.loc[0,'minx'])]

    DAA_orig_final = DAA_orig.copy()

    DAA_to_fix['UTM_Este_Captacion(m)']=padNum(DAA_to_fix['UTM_Este_Captacion(m)'],5)
    DAA_to_fix['UTM_Norte_Captacion(m)']=padNum(DAA_to_fix['UTM_Norte_Captacion(m)'],7)


    # %% DAA no georreferenciables

    DAA_to_fix['Datum']=DAA_to_fix['Datum'].str.strip().replace('',np.nan).fillna('1984')
    DAA_to_fix['Huso']=DAA_to_fix['Huso'].str.strip().replace('',np.nan).fillna('19')

    DAA_to_fix["CRS"] = DAA_to_fix.apply(
        lambda x: CRS[x['Datum']][x['Huso']], axis=1)

    DAA_to_fix['geometry'] = DAA_to_fix.apply(lambda x:
                                                                  Transformer.from_crs(x["CRS"], 'EPSG:32719', always_xy=True).transform(x['UTM_Este_Captacion(m)'],
                                                                                                                                         x['UTM_Norte_Captacion(m)']), axis=1)
    x = DAA_to_fix['geometry'].apply(lambda x: x[0])
    y = DAA_to_fix['geometry'].apply(lambda x: x[1])

    DAA_to_fix = gpd.GeoDataFrame(DAA_to_fix,
                                geometry=gpd.points_from_xy(x, y), crs='EPSG:32719')



# =============================================================================
#                                geográficas
# =============================================================================
    filtro = (DAA_to_fix['x'] < 0)
    idx = DAA_to_fix[filtro].index
    DAA_orig_final.drop(index=idx, inplace=True)

    DAA_orig_final = DAA_orig_final.append(gdf_no_georref_final, sort=True)
    DAA_orig_final = DAA_orig_final.append(DAA_to_fix, sort=True)
    # drop duplicated Ndeg columns form DAA_orig_final
    DAA_orig_final = DAA_orig_final.loc[:,~DAA_orig_final.columns.duplicated()]

    DAA_orig_final = overlay(DAA_orig_final,reg_buffer)
    # DAA_orig_final.drop(index=DAA_out.index, inplace=True)

        # agregar daa uchile
    rutaUCH=r'G:\OneDrive - ciren.cl\2022_Ficha_Atacama\07_Datos\DAA\DAA_Uchile\DAA_uchile.shp'
    df_uchile = gpd.read_file(rutaUCH)

    daa_uchile=gpd.overlay(df_uchile,rm)

    daa_vf=DAA_orig_final
    daa_vf=overlay(daa_vf,reg_buffer)

    daaRev0=fixUnits(daa_vf)

    daaRiego=consuntiveIrr(daaRev0)
    daaRiego.to_file(os.path.join('.','data','DAA_Maipo_rev0_Riego.geopkg'),driver='GPKG')
    # daaRev0.to_file(os.path.join('.','data','DAA_Maipo_rev0.geopkg'),driver='GPKG')
    return daaRiego

def consuntiveIrr(gdf):
    gdf=gdf[~gdf['Tipo_Derecho'].str.contains('No ')]
    gdf=gdf[(gdf['Uso_del_Agua'].str.contains('Riego')) | (gdf['Uso_del_Agua'].str.contains('Otros'))]

    gdfRiego=gdf[gdf['Uso_del_Agua'].str.contains('Riego')]

    blacklist=['AES GENER','AGUAS ANDINAS','AGUAS ANDINAS.','AGUAS CORDILLERA','AGUAS BATUCO','AGUAS DE LAS LILAS','AGUAS DEL SOLAR',
    'AGUAS LOS DOMINICOS',
    'AGUAS MANQUEHUE',
    'AGUAS PIRQUE',
    'AGUAS SAN PEDRO',
    'AGUAS SANTIAGO NORTE',
    'AGUAS SANTIAGO PONIENTE',
    'AGUAS SUBSTRATUM',
    'AGUA POTABLE',
    'PETROLEOS DE CHILE',
    'MINERA','CONSTRUCTORA',
    'INMOBILIARIA','CODELCO',
    'FERROCARILLES',
    'SERVICIOS SANITARIOS',
    'COPEC','SANITARIAS',
    'MINERA','UNIVERSIDAD',
    'INDUSTRIALES','BANCO DE',
    'CENTRO DE INVESTIGACION',
    'SOLDADURAS','SEGUROS'
    'CORPORACION NACIONAL',
    'EMBOTELLADORA',
    'EXPLORACIONES, INVERSIONES Y ASESORIAS HUTURI','FISCO',
    'TEXTIL','CLINICA','ALUMINIOS','ALIMENTOS',
    'BANCO','BCC S.A.','BODEGAS',
    'DEPORTE','CERVECERIA','COMERCIAL','POTABLE',
    'CERVECERIAS','INDUSTRIAL','COSMETICOS','CURTIDOS','DERCO S.A.','FIBROCEMENTOS',
    'FISCO','GOODYEAR','MUNICIPALIDAD','INDUSTRIA','INDUSTRIAL','INDUSTRIALES','INMOBILIARIA','INVERSIONES',
    'LABORATORIO','MELON HORMIGONES','MIERA','NESTLE','UNIVERSIDAD','QUIMICOS','HORMIGONES','VIVIENDA',
    'TURISMO','SUPERMERCADO','SALUD','SALMONES','INVERSIONES','HOTEL','HOSPITAL','HORMIGONES','BCC','B.M.S','CEMENTO','CLUB DEPORTIVO','COMITE DE AGUA POTABLE','CONSTRUCTORA','DISTRIB','EMPRESA','FUNDICION','FIBROCEMENTOS',
    'GOODYEAR','HIGH RETURN',
    'ILUSTRE MUNICIPALIDAD','INDUSTRIAL','INMOBILIARIA',
    'INVERSIONES SANITARIAS',
    'SERVICIO DE VIVIENDA','SOPROLE','SUPERMERCADO','VITAL S.A.','WALMART']
    # remove gdf['Nombre_Solicitante'] that contains blacklist as a substring
    filter=~gdf['Nombre_Solicitante'].str.contains('|'.join(blacklist),case=False,na=False) | (gdf['Uso_del_Agua'].str.contains('Riego'))
    gdf=gdf[filter]
    return gdf

def overlay(gdf,reg_buffer):
    gdf=gdf.copy()
    gdfBuffer=gpd.GeoDataFrame([],geometry=reg_buffer)
    gdfRet=gpd.overlay(gdf,gdfBuffer,how='intersection')
    return gdfRet

def fixUnits(gdf):
    gdf.columns=gdf.columns.str.replace('Caudal_AnualProm_','Caudal_AnualProm')
    gdf['Unidad_de_Caudal']=units=gdf['Unidad_de_Caudal'].str.strip().str.lower()

    dictUnits={'lt/s':1,'m3/s':1000,'acciones':0,'lt/min':1/60.,
    'm3/año':1e3/86400/365,'lt/h':1/3600, 'm3/h':1e3/3600,
       'mm3/año':1e-6/86400, '%':1, None:1}

    gdf['Caudal_AnualProm']=gdf['Caudal_AnualProm'].astype(str).str.strip().replace('',np.nan).str.replace(',','.').fillna(0).replace('None',np.nan).astype(float).apply(lambda x: abs(x))
    gdf['Caudal_AnualProm']=gdf['Caudal_AnualProm'].mul(gdf['Unidad_de_Caudal'].apply(lambda x: dictUnits[x]).values)

    gdf['Caudal_AnualProm'][gdf['Caudal_AnualProm']>1e5]=gdf['Caudal_AnualProm'][gdf['Caudal_AnualProm']>1e5]/1e3

    return gdf

def sumOverlayGeometry(gdf1,gdf2):
    # spatial join of gdf1 and gdf2, and sum col in gdf2 by gdf1 geometry
    sjoin=gpd.sjoin(gdf1,gdf2,how='left',op='intersects')
    # dissolve by 'NOM_CANAL' and sum col Caudal_AnualProm by 'NOM_CANAL'
    suma=sjoin.dissolve(by='NOM_CMA',aggfunc='sum')
    return suma['Caudal_AnualProm'].values

def sumDAA(ar,daa):
    daa2=daa.copy()
    daa2=gpd.GeoDataFrame(daa2['Caudal_AnualProm'],geometry=daa2.geometry)
    gw=gpd.GeoDataFrame(daa[~daa['Naturaleza_del_Agua'].str.contains('Superficial')],geometry=daa.geometry)

    # overlay sw in ar geometries and sum each by 'Caudal_AnualProm' in sw

    ar['GW']=sumOverlayGeometry(ar,gw)
    return ar

def fixSW(ar):
    # arregla los daa superficiales en l/s
    dictSW={'ACUIFERO':1,
    'CANCHA DE PIEDRA': 265.4,
    'CACHAPOAL':3680, 
    # fuente: asociacion de canalistas
    'CARMEN ALTO':8000,
    # TESIS
    'CASTILLO':3425,
    # informe
    'CHACABUCO':1,
    # informe
    'CHADA TRONCO':1928.9812000000002,
    # informe
    'CHINIHUE':2700,
    #buscar mas del chinihue
    'CHOCALAN':5000,
    'CHOLQUI':2000,
    'CLARILLO':500,
    'CODIGUA':4800,
    'COLINA':550,
    'COMUN ASOCIACION CANALES DEL MAIPO':35000,
    'CULIPRAN':5000,
    'D EL CARMEN UNO': 7000,
    'D SAN FRANCISCO':1700,
    'EL PAICO':1600,
    'ESPERANZA BAJO':1800,
    'EYZAGUIRRE':11470,
    'FERNANDINO':2.901577*1e3,
    'HOSPITAL':583.28,
    'HUALEMU':2000,
    'HUECHUN':4200,
    'HUIDOBRO':16070,
    'ISLA DE HUECHUN':2290,
    'LA ISLA':1500,
    'LAS MERCEDES':10200,
    # 'LO ESPEJO':6420.3387288,
    'LO HERRERA':1080,
    'LONQUEN ISLA':610,
    'LUCANO':3410,
    # https://canallucano.cl/caudales-historicos/
    'MALLARAUCO':8692.54,
    'PAINE':1.903561*1e3,
    'POLPAICO':1,
    'PUANGUE':3600,
    'QUINTA':3.382360*1e3,
    'SAN ANTONIO DE NALTAHUA':2460,
    'SAN DIEGO':1980,
    'SAN JOSE':3260,
    'SAN MIGUEL':4000,
    'SANTA RITA UNO':5343,
    'SANTA RITA':0.557519*1e3,
    'VILUCO':2.754984*1e3,
    'WODEHOUSE':2300,
    'SANTA EMILIA O RULANO':883,
    'TOMA DEL TORO':58,
    'UNIFICADO AGUILA NORTE AGUILA SUR':583
    }
    ar['SW']=0
    ar['SW']=ar['NOM_CMA'].apply(lambda x: dictSW[x] if x in dictSW.keys() else 0)/ar.area*1e4
    ar['SGW']=ar['GW'].astype(float)/ar.area*1e4+ar['SW'].astype(float)
    return ar

def acub(ar):
    lista=['QUINTA','PAINE','SANTA RITA','FERNANDINO','VILUCO']
    areas=ar[ar['NOM_CANAL'].isin(lista)]
    areasDis=areas.dissolve(by='NOM_CANAL')
    11.5*areasDis.area/areasDis.area.sum()
    return None

def mapCANMA(gdf1,gdf2):
    gdf1=gdf1.copy()
    for index in gdf1.index:
        x=gdf1.loc[index,'COD_CMA']
        try:
            idx=gdf2[gdf2['COD_CMA']==x].index
            gdf1.loc[index,'NOM_CMA']=gdf2.loc[idx,'NOM_CANAL']
        except:
            gdf1.loc[index,'NOM_CMA']=gdf1.loc[index,'NOM_CANAL']
    return gdf1['NOM_CMA']

def canMA(ar):
    can=gpd.read_file(os.path.join('.','data','Maipo','Canales_Maipo.shp'))
    ar['COD_CMA']=ar['COD_CANAL'].apply(lambda x: x[:14].ljust(20, '0'))
    arCANMA=ar.dissolve(by='COD_CMA')
    arCANMA.reset_index(inplace=True)

    # lookup codMa in can['COD_CANAL']
    can['COD_CMA']=can['COD_CANAL'].apply(lambda x: x[:20])
    arCANMA['NOM_CMA']=mapCANMA(arCANMA,can)
    dictCanMa={'CALERA':'COMUN ASOCIACION CANALES DEL MAIPO',
    'SAN VICENTE':'COMUN ASOCIACION CANALES DEL MAIPO',
    'SANTA CRUZ':'COMUN ASOCIACION CANALES DEL MAIPO',
    'LO ESPEJO':'COMUN ASOCIACION CANALES DEL MAIPO'}
    # rename arCANMA arCANMA according to dictCanMa
    arCANMA['NOM_CMA']=arCANMA['NOM_CMA'].apply(lambda x: dictCanMa[x] if x in dictCanMa.keys() else x)
    arCANMA=arCANMA.dissolve(by='NOM_CMA')
    arCANMA.reset_index(inplace=True)
    return arCANMA

def fixAR(ar):
    idx=ar[ar['NOM_CMA'].isin(['TOMA RIO MAIPO','SIN INFORMACION',
    'DERRAMES','COLINA SUR',
    'UNIFICADO RINCONADA Y LA LOMA'])].index
    ar.drop(index=idx,inplace=True)
    del ar['index_right']
    return ar

def main():
    daa=DAA()
    uso,ar=load()
    uso2=filterArea(uso,ar)
    arCANMA=canMA(uso2)
    arCANMAfix=fixAR(arCANMA)
    arDAA=sumDAA(arCANMAfix,daa)
    arDAAsw=fixSW(arDAA)
    arDAAsw.to_file(os.path.join('.','data','arDAA.gpkg'),
    driver='GPKG')

    # now merge ar and arCANMA con COD_CANAL

if __name__=='__main__':
    main()