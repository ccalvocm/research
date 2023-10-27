import os
import ee
import pandas as pd
import datetime
import requests
import zipfile
import xarray as xr


def time_index_from_filenames(filenames):
    '''helper function to create a pandas DatetimeIndex
       Filename example: 20150520_0164.tif'''
    return pd.DatetimeIndex([pd.Timestamp(f[:8]) for f in filenames])


def files2NETCDF(folderName):
    filenames = os.listdir(os.path.join('.', folderName))
    files = sorted([x for x in filenames if ('.tif' in x) & ('xml' not in x)])
    time = xr.Variable('time', time_index_from_filenames(files))
    chunks = {'x': 5490, 'y': 5490, 'band': 1}
    da = xr.concat([xr.open_dataset(os.path.join(
        '.', folderName, f), chunks=chunks, engine='h5netcdf') for f in files], dim=time)
    da.to_netcdf(os.path.join('.', 'folderName', 'netcdf.nc'))


def pythonDate2eeDate(date=datetime.datetime):
    eeDate = date.strftime("%Y-%m-%d")
    return eeDate


def outFolder(folderName):
    try:
        os.mkdir(os.path.join('.', folderName))
    except:
        error = 'directorio ya existe'
    return None


def download(collection, band, folderName):
    # login()

    # 1. authenticate
    ee.Initialize()

    # 2. define python dates
    fechaIni = datetime.datetime(2000, 1, 1)
    fechaFin = datetime.datetime(2020, 12, 31)
    # print(fechaIni, fechaFin)
    # print(pythonDate2eeDate(fechaFin))

    # 3. define the collection
    # collection = "ECMWF/ERA5_LAND/DAILY_AGGR"
    imageCollection = ee.ImageCollection(collection)

    # 4. define the bounding box and filter accordingly
    llx, lly, urx, ury = -72.8719644121910193, - \
        45.1156153853877626, -71.0575803364210401, -44.3781096308613030
    rectangle = ee.Geometry.Rectangle([llx, lly, urx, ury])
    imageCollectionRegion = imageCollection.filterBounds(rectangle)

    # 5. select the band
    # band = "evaporation_from_open_water_surfaces_excluding_oceans_sum"
    imageCollectionBand = imageCollectionRegion.select(band)
    f = open("log.txt", "w")

    # 6. output folder
    outFolder(folderName)
    rutaDl = os.path.join('.', folderName)

    # 7. select date
    for date in pd.date_range(fechaIni, fechaFin):
        date = pythonDate2eeDate(date)
        print(f"Attempting download image for {date}")
        try:
            imageCollectionDate = imageCollectionBand.filterDate(date).first()

            url = imageCollectionDate.getDownloadURL({
                'scale': 10000,
                'crs': 'EPSG:32718',
                'fileFormat': 'GeoTIFF',
                'region': rectangle})
            print(url)
            # print(f"Attempting download image for {date} OK!")
            response = requests.get(url, stream=True)
            # filePath = os.path.join('..', 'raw', 'tmax', f"tmax{date}.tif")

            # zip
            with open(os.path.join(rutaDl, band+'_'+date+'.zip'), "wb") as fd:
                for chunk in response.iter_content(chunk_size=1024):
                    fd.write(chunk)
            fd.close()

        except:
            # print(f"Attempting download image for {date} FAILED!")
            f.write(f"{date}\n")

    # 8. extract files
    files = os.listdir(rutaDl)
    files = [x for x in files if x.endswith('.zip')]
    for file in files:
        zip_ref = zipfile.ZipFile(os.path.join(
            rutaDl, file))  # create zipfile object
        zip_ref.extractall(rutaDl)  # extract file to dir
        zip_ref.close()  # close file
        os.remove(os.path.join(rutaDl, file))  # delete zipped file

def NDVI(image):
    return image.normalizedDifference(['B5', 'B4'])

def filterCloud(images):
    return images.filter(ee.Filter.lt("CLOUD_COVER", 30))

if __name__ == '__main__':
    collection = "LANDSAT/LC08/C02/T1_TOA'"
    # outFolder(folderName=folderName)
    # download(collection=collection, band=band, folderName=folderName)
    files2NETCDF(folderName=folderName)
