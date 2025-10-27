import requests
import json
import pyproj
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict, LiteralString
from matplotlib.patches import Polygon
import shapely
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.visualizer.drawable import IDrawable
from matplotlib.axes import Axes
from python_vehicle_simulator.lib.path import PWLPath
import os

class HelsingborgMap(IDrawable):
    def __init__(self, bbox_utm: Tuple = (350000, 6210000, 358000, 6215000), verbose_level:int=0):
        self.bbox_utm = bbox_utm
        self.transformer = pyproj.Transformer.from_proj(
            '+proj=longlat +datum=WGS84',
            '+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs'
        )
        self.bbox_latlon = self._utm_to_latlon_bbox(bbox_utm)
        self.data = self._load_data()
        self.verbose_level = verbose_level
    
    def _utm_to_latlon_bbox(self, bbox_utm: Tuple) -> Tuple:
        min_x, min_y, max_x, max_y = bbox_utm
        min_lon, min_lat = self.transformer.transform(min_x, min_y, direction='INVERSE')
        max_lon, max_lat = self.transformer.transform(max_x, max_y, direction='INVERSE')
        return (min_lat, min_lon, max_lat, max_lon)
    
    @property
    def bbox_utm_coordinates(self) -> List[Tuple]:
        # upper left corner, upper right corner, lower right corner, lower left corner
        return [
                (self.bbox_utm[0], self.bbox_utm[3]),
                (self.bbox_utm[2], self.bbox_utm[3]),
                (self.bbox_utm[2], self.bbox_utm[1]),
                (self.bbox_utm[0], self.bbox_utm[1]),
            ]
    
    def _load_data(self, filename: LiteralString = os.path.join('data', 'ferry_data.json')) -> Dict:
        try:
            print(f"Trying to load shore from {filename}..")
            with open(filename, 'r') as f:
                osm_data = json.load(f)
            print(f"Successfully loaded data from {filename}!")
            return osm_data
        except:
            print(f"No data in cache, sending query..")
            osm_data = self._get_osm_data()
            return self._parse_osm_data(osm_data) if osm_data else {}
    
    def _get_osm_data(self) -> Dict:
        bbox = self.bbox_latlon
        query = f"""
        [out:json][timeout:60];
        (
          way["route"="ferry"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          way["natural"="coastline"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          way["seamark:type"~"separation_zone|traffic_lane|roundabout"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          node["amenity"="ferry_terminal"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          way["seamark:type"="traffic_separation_scheme"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          rel["seamark:type"="traffic_separation_scheme"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          way["seamark:traffic_separation_scheme:category"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
        );
        out geom;
        """
        
        response = requests.get("http://overpass-api.de/api/interpreter", 
                              params={'data': query})
        return response.json() if response.status_code == 200 else None
    
    def _convert_coords(self, lat_lon_coords: List[Tuple]) -> List[Tuple]:
        return [self.transformer.transform(lon, lat) for lon, lat in lat_lon_coords]
    
    def _parse_osm_data(self, osm_data: Dict) -> Dict:
        data = {
            'ferry_routes': [],
            'coastlines': [],
            'terminals': [],
            'tss': []
        }
        
        for element in osm_data.get('elements', []):
            if element['type'] == 'way':
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                coords_utm = self._convert_coords(coords)
                
                tags = element.get('tags', {})
                
                if tags.get('route') == 'ferry':
                    data['ferry_routes'].append({
                        'name': tags.get('name', 'Unnamed'),
                        'coords': coords_utm
                    })
                elif tags.get('natural') == 'coastline':
                    data['coastlines'].append({'coords': coords_utm})
                elif 'seamark:type' in tags:
                    data['tss'].append({
                        'type': tags['seamark:type'],
                        'coords': coords_utm
                    })
            
            elif element['type'] == 'node' and element.get('tags', {}).get('amenity') == 'ferry_terminal':
                coord_utm = self._convert_coords([(element['lon'], element['lat'])])[0]
                data['terminals'].append({
                    'name': element.get('tags', {}).get('name', 'Terminal'),
                    'coords': coord_utm
                })
        
        return data
    
    
    def get_ferry_routes(self) -> Dict[str, PWLPath]:
        routes = {}
        for route in self.data.get('ferry_routes', []):
            routes.update({route['name']: PWLPath(route['coords'], input_format='east-north')})
        return routes
    
    def get_tss_zones(self) -> List[Dict]:
        return self.data.get('tss', [])
    
    def get_shore_as_polygons(self) -> List[shapely.Polygon]:
        # Stitch regions -> Ugly hard-coded but works :)
        data = self.data.get('coastlines', [])
        # print(data)
        data[4]['coords'] = data[4]['coords'][40::]

        left_side = np.concatenate([
            [(data[4]['coords'][0][0], data[7]['coords'][-1][1])],
            data[4]['coords'],
            data[7]['coords'],
            [(data[4]['coords'][0][0], data[7]['coords'][-1][1])],
        ])

        right_side = np.concatenate([
            [(358_000 , data[2]['coords'][0][1])],
            data[2]['coords'],
            data[5]['coords'],
            data[0]['coords'],
            data[1]['coords'],
            [(358_000, data[1]['coords'][-1][1])],
            [(358_000, data[2]['coords'][0][1])]
        ])

        islands = [data[3]['coords'], data[6]['coords'], data[8]['coords'], right_side.tolist(), left_side.tolist()]
        return [shapely.Polygon(island) for island in islands]
    
    def get_shore_as_obstacles(self) -> List[Obstacle]:
        return [Obstacle(*zip(poly.exterior.coords.xy)) for poly in self.get_shore_as_polygons()]

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

    def __plot__(self, ax:Axes, *args, verbose:int=0, terminals:bool=False, routes:bool=False, **kwargs) -> Axes:
        ax.set_facecolor('lightblue')
        polygons = self.get_shore_as_polygons()
        for polygon in polygons:
            ax.plot(*polygon.exterior.coords.xy, 'black', linewidth=2)
            ax.fill(*polygon.exterior.coords.xy, 'forestgreen')

        # Plot ferry routes
        if routes:
            for route in self.data.get('ferry_routes', []):
                if route is not None and route['name'] in ['Helsingør (DK) - Helsingborg (SE)']: # 'Helsingør (DK) - Helsingborg (S)', 
                    coords = np.array(route['coords'])
                    ax.plot(coords[:, 0], coords[:, 1], '--r', linewidth=3, 
                        label=f"Ferry: {route['name']}")
            
        # Plot terminals
        if terminals:
            for terminal in self.data.get('terminals', []):
                if terminal is not None and terminal['name'] in ['Helsingborg-Helsingør', 'Helsingør-Helsingborg']:
                    x, y = terminal['coords']
                    ax.scatter(x, y, color='orange', s=100, zorder=5)
            
        # Plot TSS zones
        colors = {'separation_zone': 'purple', 'traffic_lane': 'orange', 'roundabout': 'pink'}
        for tss in self.data.get('tss', []):
            coords = np.array(tss['coords'])
            color = colors.get(tss['type'], 'gray')
            
            if len(coords) > 2:
                polygon = Polygon(coords, alpha=0.4, facecolor=color, edgecolor='black')
                ax.add_patch(polygon)
            else:
                ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2)
        
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title('Ferry Environment - UTM Zone 33N')
        ax.set_xlim(self.bbox_utm[0], self.bbox_utm[2])
        ax.set_ylim(self.bbox_utm[1], self.bbox_utm[3])
        ax.legend()
        ax.grid(False)
        ax.set_aspect('equal')
        plt.tight_layout()
        return ax        
    
    def save_data(self, filename: LiteralString = os.path.join('data', 'ferry_data.json')):
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)


if __name__ == "__main__":
    env = HelsingborgMap()
    ax = env.plot(routes=False, terminals=False)
    routes = env.get_ferry_routes()
    for name in routes.keys():
        routes[name].plot(ax=ax, c='red' if name in ['Helsingør (DK) - Helsingborg (S)', 'Helsingør (DK) - Helsingborg (D)'] else 'grey')
    plt.show()

