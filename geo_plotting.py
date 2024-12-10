import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)


def city(city_coords):
    south_carolina = gpd.read_file(r'tl_2011_45_taz10.shp')
    geometry = gpd.GeoDataFrame([city_coords], geometry=[Point(city_coords['Longitude'], city_coords['Latitude'])])
    geometry.crs = 'EPSG:4326'
    fig, ax = plt.subplots()
    south_carolina.plot(ax=ax, color='lightblue', edgecolor='black')
    geometry.plot(ax=ax, color='red', marker='o', markersize=50)

    ax.set_title('{}, SC'.format(city_coords['City']))
    ax.axis('off')
    plt.show()


def community(city, args):
    df = pd.read_excel(r'SC.xlsx', sheet_name=city)
    retirement = df[df.Type == 1]
    hospital = df[df.Type == 2]
    mall = df[df.Type == 3]
    dental = df[df.Type == 4]
    service = df[df.Type == 5]
    transport = df[df.Type == 6]

    south_carolina = gpd.read_file(r'tl_2011_45_taz10.shp')
    county_map = {'Charleston': ['0115', '019'], 'Clemson': ['077'], 'Columbia': ['063', '079'], 'Greenville': ['045'], 'Myrtle Beach': ['051']}
    limits = {'Charleston': [[-80.07, -79.75], [32.75, 32.93]], 'Clemson': [[-82.85, -82.77], [34.67, 34.72]], 'Columbia': [[-81.172, -81], [33.93, 34.09]], 'Greenville': [[-82.42, -82.328], [34.789, 34.868]], 'Myrtle Beach': [[-79, -78.85], [33.58, 33.76]]}
    counties = south_carolina[south_carolina.COUNTYFP10.isin(county_map[city])]
    
    fig, ax = plt.subplots(figsize=(15, 9))
    
    counties.plot(ax=ax, color='white', edgecolor='black')
    for i in args:
        if i == 1:
            retirement = gpd.GeoDataFrame(retirement, crs='EPSG:4326', geometry=[Point(xy) for xy in zip(retirement['Longitude'], retirement['Latitude'])])
            retirement.plot(ax=ax,marker='*', markersize=70, label='Retirement Community')
        if i == 2:
            hospital = gpd.GeoDataFrame(hospital, crs='EPSG:4326', geometry=[Point(xy) for xy in zip(hospital['Longitude'], hospital['Latitude'])])
            hospital.plot(ax=ax, marker='o', markersize=70, label='Hospital')
        if i == 3:
            mall = gpd.GeoDataFrame(mall, crs='EPSG:4326', geometry=[Point(xy) for xy in zip(mall['Longitude'], mall['Latitude'])])
            mall.plot(ax=ax, marker='^', markersize=70, label='Shopping Mall')
        if i == 4:
            dental = gpd.GeoDataFrame(dental, crs='EPSG:4326', geometry=[Point(xy) for xy in zip(dental['Longitude'], dental['Latitude'])])
            dental.plot(ax=ax, marker='p', markersize=70, label='Dental Care')
        if i == 5:
            service = gpd.GeoDataFrame(service, crs='EPSG:4326', geometry=[Point(xy) for xy in zip(service['Longitude'], service['Latitude'])])
            service.plot(ax=ax, marker='D', markersize=70, label='Senior Service Center')
        if i == 6:
            transport = gpd.GeoDataFrame(transport, crs='EPSG:4326', geometry=[Point(xy) for xy in zip(transport['Longitude'], transport['Latitude'])])
            transport.plot(ax=ax, marker='s', markersize=70, label='Transportation Hub')

    ax.set_title('Points of Interests (POIs) in {}, SC'.format(city))
    ax.legend(bbox_to_anchor=(1.2, 0.01),loc=4, framealpha=1)
    ax.set_xlim(limits[city][0])
    ax.set_ylim(limits[city][1])
    ax.axis('off')
    plt.show()

                
if __name__ == '__main__':
    cities = {1: {'City': 'Charleston', 'Latitude': 32.7833, 'Longitude': -79.932},
              2: {'City': 'Columbia', 'Latitude': 34.0007, 'Longitude': -81.0348},
              3: {'City': 'Greenville', 'Latitude': 34.8526, 'Longitude': -82.394},
              4: {'City': 'Myrtle Beach', 'Latitude': 33.6954, 'Longitude': -78.8802},
              5: {'City': 'Clemson', 'Latitude': 34.675, 'Longitude': -82.9406}
        }
    for c in cities:
        city(cities[c])
        
