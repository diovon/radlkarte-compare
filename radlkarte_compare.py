import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString, MultiPoint
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import warnings
import time
import urllib.request
import math
from io import BytesIO
from PIL import Image
from shapely.errors import GEOSException

warnings.filterwarnings('ignore')

# ================= KONFIGURATION =================
DEFAULT_OUTPUT_DIR = "ausgabe"
TILE_CACHE_DIR = "tile_cache_17" 
ZOOM_LEVEL = 17
MAX_RENDERS = 0 
BG_COLOR = '#f0f0f0'

# Clustering:
# Wenn zwei Änderungsbereiche näher als dieser Abstand sind,
# werden sie zu einem Cluster zusammengefasst.
#
# Grobe Orientierung in WGS84:
# 0.001 Grad ≈ 100 m
# 0.003 Grad ≈ 300 m
# 0.005 Grad ≈ 500 m
# 0.010 Grad ≈ 1000 m
CLUSTER_BUFFER_DEG = 0.005


def setup_args():
    if len(sys.argv) < 3:
        print(
            "Verwendung: python3 radlkarte_diff.py "
            "<datei_a.geojson> <datei_b.geojson> "
            "[ausgabe_ordner] [max_renders] [cluster_buffer_deg]"
        )
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT_DIR
    max_renders = int(sys.argv[4]) if len(sys.argv) > 4 else MAX_RENDERS

    # Optional: Cluster-Buffer als 5. Argument übergeben
    if len(sys.argv) > 5:
        global CLUSTER_BUFFER_DEG
        CLUSTER_BUFFER_DEG = float(sys.argv[5])

    return file_a, file_b, output_dir, max_renders


def get_josm_style(props):
    main_color = '#51A4B6' 
    main_width = 3
    casing_color = 'none'
    casing_width = 0
    linestyle = '-'
    
    if not isinstance(props, dict):
        return {
            'main_color': main_color,
            'main_width': main_width,
            'casing_color': casing_color,
            'casing_width': casing_width,
            'linestyle': linestyle
        }

    if 'stress' in props and props['stress'] is not None:
        try:
            stress = int(props['stress'])
            if stress == 0:
                main_color = '#004B67'
            elif stress == 1:
                main_color = '#51A4B6'
            elif stress == 2:
                main_color = '#FF6600'
        except:
            pass

    if 'priority' in props and props['priority'] is not None:
        try:
            priority = int(props['priority'])
            if priority == 0:
                main_width = 12
            elif priority == 1:
                main_width = 3
            elif priority == 2: 
                main_width = 3
                linestyle = (0, (5, 5))
        except:
            pass

    if props.get('unpaved') == 'yes':
        casing_color = '#00ff18'
        casing_width = 7
    elif props.get('steep') == 'yes':
        casing_color = '#ff00f0'
        casing_width = 7
    elif 'fixme' in props:
        casing_color = '#FF0'
        casing_width = 7

    return {
        'main_color': main_color,
        'main_width': main_width,
        'casing_color': casing_color,
        'casing_width': casing_width,
        'linestyle': linestyle
    }


def lonlat_to_tile(lon, lat, zoom):
    """
    Konvertiert WGS84 (Lon/Lat) in slippy map tile coordinates (x, y).
    Quelle: http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    """
    n = 2 ** zoom
    lon_deg = lon
    lat_rad = math.radians(lat)
    x = int((lon_deg + 180.0) / 360.0 * n)
    # Y-Koordinate: 0 ist oben (Nordpol)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lonlat(x, y, zoom):
    """
    Konvertiert Tile-Koordinaten (x, y) zurück in WGS84 (Lon/Lat)
    der TOP-LEFT Ecke des Tiles.
    """
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def get_tile_image(x, y, zoom, cache_dir=TILE_CACHE_DIR, verbose=False):
    """
    Lädt eine Kachel. Prüft zuerst den lokalen Cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{zoom}_{x}_{y}.png")
    
    if os.path.exists(cache_file):
        try:
            img = Image.open(cache_file)
            if verbose:
                print(f"   [Cache] {x}/{y}")
            return img
        except Exception as e:
            print(f"      Cache-Datei beschädigt: {e}")
            try:
                os.remove(cache_file)
            except:
                pass

    # URL für OpenStreetMap Standard Tiles
    # Wichtig: OSM verlangt einen User-Agent
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) RadlkarteDiff/1.0'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()
            img = Image.open(BytesIO(data))
            
            # Im Cache speichern
            img.save(cache_file)
            if verbose:
                print(f"   [Download] {x}/{y}")
            return img
    except Exception as e:
        print(f"      Fehler beim Laden von Tile {x}/{y} (URL: {url}): {e}")
        # Fallback: Graues Bild
        img = Image.new('RGB', (256, 256), color=(200, 200, 200))
        return img


def draw_basemap_on_ax(ax, xmin, ymin, xmax, ymax, zoom=ZOOM_LEVEL):
    """
    Zeichnet die Basemap-Kacheln auf dem Axes-Objekt.
    """
    # 1. Bestimme die benötigten Kacheln
    # xmin/ymin ist Bottom-Left, xmax/ymax ist Top-Right
    # Wir brauchen die Tile-Indizes für diesen Bereich.
    
    # Top-Left Tile (max Lat, min Lon)
    x_tl, y_tl = lonlat_to_tile(xmin, ymax, zoom)
    # Bottom-Right Tile (min Lat, max Lon)
    x_br, y_br = lonlat_to_tile(xmax, ymin, zoom)
    
    # Sicherstellen, dass x_tl <= x_br und y_tl <= y_br
    # (Bei Web Mercator ist y=0 oben, also ist y_tl < y_br)
    
    # 2. Iteriere über alle Tiles im Bereich
    for x in range(x_tl, x_br + 1):
        for y in range(y_tl, y_br + 1):
            # Hole die geografischen Grenzen dieses einzelnen Tiles
            # tile_to_lonlat gibt die TOP-LEFT Ecke zurück
            tile_lon_tl, tile_lat_tl = tile_to_lonlat(x, y, zoom)
            tile_lon_br, tile_lat_br = tile_to_lonlat(x + 1, y + 1, zoom)
            
            # Schnellerer Schnitt: Liegt der Tile wirklich in unserem Viewport?
            if tile_lon_br < xmin or tile_lon_tl > xmax:
                continue
            if tile_lat_br < ymin or tile_lat_tl > ymax:
                continue

            try:
                tile_img = get_tile_image(x, y, zoom)
                if tile_img:
                    # extent = [left, right, bottom, top]
                    # left = tile_lon_tl
                    # right = tile_lon_br
                    # bottom = tile_lat_br  (da y=1 ist unten)
                    # top = tile_lat_tl     (da y=0 ist oben)
                    ax.imshow(
                        tile_img, 
                        extent=[tile_lon_tl, tile_lon_br, tile_lat_br, tile_lat_tl], 
                        origin='upper', 
                        zorder=0, 
                        aspect='auto'
                    )
            except Exception as e:
                print(f"      Fehler beim Zeichnen von Tile {x}/{y}: {e}")
                continue


def draw_geometry_on_ax(ax, gdf_subset, style_override=None):
    if gdf_subset.empty:
        return
        
    for _, row in gdf_subset.iterrows():
        geom = row.geometry
        props = row.to_dict()
        
        if style_override:
            style = style_override
        else:
            style = get_josm_style(props)
            
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            
            if style['casing_color'] != 'none':
                ax.plot(
                    lons,
                    lats,
                    color=style['casing_color'], 
                    linewidth=style['casing_width'] + style['main_width'], 
                    alpha=0.75,
                    zorder=1
                )
            
            ax.plot(
                lons,
                lats,
                color=style['main_color'], 
                linewidth=style['main_width'], 
                linestyle=style['linestyle'],
                zorder=2
            )

        elif geom.geom_type == "Point":
            x, y = geom.x, geom.y
            color = 'red'
            if 'dismount' in props and props.get('dismount') == 'yes':
                color = 'blue'
            ax.plot(
                x,
                y,
                marker='o',
                color=color,
                markersize=8,
                zorder=3
            )

        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                coords = list(part.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                if style['casing_color'] != 'none':
                    ax.plot(
                        lons,
                        lats,
                        color=style['casing_color'], 
                        linewidth=style['casing_width'] + style['main_width'], 
                        alpha=0.75,
                        zorder=1
                    )
                ax.plot(
                    lons,
                    lats,
                    color=style['main_color'], 
                    linewidth=style['main_width'], 
                    linestyle=style['linestyle'],
                    zorder=2
                )


def _get_buffered_geometries(diff_parts, buffer_deg):
    """
    Erzeugt für jede Diff-Geometrie eine leicht aufgeblähte Version.
    Diese wird nur für das Clustering verwendet.
    """
    buffered = []
    for geom in diff_parts:
        if geom.is_empty:
            continue

        if buffer_deg > 0:
            try:
                # cap_style=2 = Round
                # join_style=2 = Round
                buf = geom.buffer(buffer_deg, cap_style=2, join_style=2)
            except Exception:
                # Fallback, falls Buffer bei exotischen Geometrien fehlschlägt
                buf = geom
        else:
            buf = geom

        if not buf.is_empty:
            buffered.append(buf)
        else:
            buffered.append(geom)

    return buffered


def _cluster_with_strtree(diff_parts, buffered, buffer_deg):
    """
    Versucht, Clustering über Shapely STRtree durchzuführen.
    Fallback, falls nicht möglich, gibt None zurück.
    """
    try:
        from shapely.strtree import STRtree
    except ImportError:
        return None

    if not buffered:
        return []

    try:
        # Shapely 2.x
        tree = STRtree(buffered)

        visited = [False] * len(buffered)
        clusters = []

        for i in range(len(buffered)):
            if visited[i]:
                continue

            queue = [i]
            visited[i] = True
            cluster_indices = []

            while queue:
                idx = queue.pop(0)
                cluster_indices.append(idx)

                try:
                    # Shapely 2.x
                    neighbors = tree.query(buffered[idx], predicate='intersects')
                except TypeError:
                    # Ältere Shapely-Versionen
                    neighbors = tree.query(buffered[idx])

                for j in neighbors:
                    j = int(j)
                    if not visited[j]:
                        visited[j] = True
                        queue.append(j)

            cluster_geoms = [diff_parts[j] for j in cluster_indices]
            clusters.append(cluster_geoms)

        return clusters

    except Exception as e:
        print(f"   STRtree-Clustering nicht möglich: {e}")
        return None


def _cluster_simple(diff_parts, buffered, buffer_deg):
    """
    Einfaches Fallback-Clustering ohne STRtree.
    Läuft langsamer, ist aber robust.
    """
    if not buffered:
        return []

    visited = [False] * len(buffered)
    clusters = []

    for i in range(len(buffered)):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        cluster_indices = []

        while queue:
            idx = queue.pop(0)
            cluster_indices.append(idx)

            for j in range(len(buffered)):
                if not visited[j]:
                    try:
                        if buffered[idx].intersects(buffered[j]):
                            visited[j] = True
                            queue.append(j)
                    except Exception:
                        pass

        cluster_geoms = [diff_parts[j] for j in cluster_indices]
        clusters.append(cluster_geoms)

    return clusters


def cluster_diff_parts(diff_parts, buffer_deg):
    """
    Führt ein Clustering der gefundenen Diff-Geometrien durch.

    Zwei Diff-Geometrien werden zusammengefasst, wenn ihr
    gegenseitiger Abstand kleiner oder gleich buffer_deg ist.
    """
    if not diff_parts:
        return []

    print(f"  Führe Clustering durch (Buffer: {buffer_deg} Grad)...")

    # Leere Geometrien entfernen
    clean_parts = [g for g in diff_parts if not g.is_empty]

    if not clean_parts:
        return []

    buffered = _get_buffered_geometries(clean_parts, buffer_deg)

    # Erst versuchen, mit STRtree zu clustern
    clusters = _cluster_with_strtree(clean_parts, buffered, buffer_deg)

    # Fallback, falls STRtree nicht funktioniert
    if clusters is None:
        print("   Fallback: Einfaches Clustering ohne STRtree...")
        clusters = _cluster_simple(clean_parts, buffered, buffer_deg)

    print(f"   Clustering abgeschlossen: {len(clean_parts)} Diff-Teile -> {len(clusters)} Cluster")

    return clusters


def main():
    file_a, file_b, output_dir, max_renders = setup_args()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"  Lade {file_a} ...")
    gdf_a = gpd.read_file(file_a)
    print(f"  Lade {file_b} ...")
    gdf_b = gpd.read_file(file_b)

    if gdf_a.crs is None:
        gdf_a = gdf_a.set_crs(4326)
    if gdf_b.crs is None:
        gdf_b = gdf_b.set_crs(4326)
    if gdf_a.crs != gdf_b.crs:
        gdf_a = gdf_a.to_crs(gdf_b.crs)

    valid_geoms = ['LineString', 'Point', 'MultiLineString', 'MultiPoint']
    gdf_a = gdf_a[gdf_a.geometry.type.isin(valid_geoms)].copy()
    gdf_b = gdf_b[gdf_b.geometry.type.isin(valid_geoms)].copy()

    if gdf_a.empty and gdf_b.empty:
        print("Beide Dateien sind leer.")
        sys.exit(0)

    print("  Berechne symmetrische Differenz...")
    start_time = time.time()
    
    try:
        union_a = unary_union(gdf_a.geometry)
        union_b = unary_union(gdf_b.geometry)
        diff_geom = union_a.symmetric_difference(union_b)
    except GEOSException as e:
        print(f"GEOS Fehler: {e}")
        sys.exit(1)
        
    calc_time = time.time() - start_time
    print(f"   (Berechnung Zeit: {calc_time:.2f}s)")
    
    if diff_geom.is_empty:
        print("  Keine geometrischen Unterschiede gefunden.")
        sys.exit(0)
        
    diff_parts = []
    if isinstance(diff_geom, (LineString, Point)):
        diff_parts.append(diff_geom)
    elif isinstance(diff_geom, (MultiLineString, MultiPoint)):
        for part in diff_geom.geoms:
            diff_parts.append(part)
    else:
        if hasattr(diff_geom, 'geoms'):
            for part in diff_geom.geoms:
                diff_parts.append(part)
        else:
            diff_parts.append(diff_geom)

    # Leere Diff-Teile entfernen
    diff_parts = [g for g in diff_parts if not g.is_empty]

    print(f"  Gefundene Änderungsbereiche: {len(diff_parts)}")

    # NEU: Clustering der Diff-Teile
    diff_parts = cluster_diff_parts(diff_parts, CLUSTER_BUFFER_DEG)

    if max_renders > 0:
        diff_parts = diff_parts[:max_renders]
        print(f"   (Limitiert auf {max_renders} Bilder)")

    print(f"    Erstelle Vergleichsbilder (Vorher/Nachher)...")
    
    for i, cluster in enumerate(diff_parts):
        # Cluster ist jetzt eine Liste von Geometrien
        if not cluster:
            continue

        # Gemeinsame Bounds für das ganze Cluster berechnen
        minx = min(g.bounds[0] for g in cluster)
        miny = min(g.bounds[1] for g in cluster)
        maxx = max(g.bounds[2] for g in cluster)
        maxy = max(g.bounds[3] for g in cluster)

        xmin = minx
        ymin = miny
        xmax = maxx
        ymax = maxy

        width = xmax - xmin
        height = ymax - ymin
        pad_x = max(width * 0.3, 0.001)
        pad_y = max(height * 0.3, 0.001)
        
        xmin = xmin - pad_x
        ymin = ymin - pad_y
        xmax = xmax + pad_x
        ymax = ymax + pad_y
        
        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
        
        common_xlim = (xmin, xmax)
        common_ylim = (ymin, ymax)
        
        # --- VORHER (Datei A) ---
        # Basemap zeichnen (nutzt Cache)
        draw_basemap_on_ax(ax_before, xmin, ymin, xmax, ymax)
        ax_before.set_xlim(common_xlim)
        ax_before.set_ylim(common_ylim)
        
        # Schnellerer Filter
        gdf_a_subset = gdf_a.cx[xmin:xmax, ymin:ymax]
        
        draw_geometry_on_ax(ax_before, gdf_a_subset)
        
        ax_before.set_title("VORHER (Datei A)", fontsize=12, loc='left')
        ax_before.set_xticklabels([])
        ax_before.set_yticklabels([])

        # --- NACHHER (Datei B) ---
        draw_basemap_on_ax(ax_after, xmin, ymin, xmax, ymax)
        ax_after.set_xlim(common_xlim)
        ax_after.set_ylim(common_ylim)
        
        gdf_b_subset = gdf_b.cx[xmin:xmax, ymin:ymax]
        
        draw_geometry_on_ax(ax_after, gdf_b_subset)
        
        ax_after.set_title("NACHHER (Datei B)", fontsize=12, loc='left')
        ax_after.set_xticklabels([])
        ax_after.set_yticklabels([])
        
        fig.suptitle(f"Änderung #{i+1} ({len(cluster)} Teil(e))", fontsize=14)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"vergleich_{i+1:04d}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        
        if (i + 1) % 5 == 0:
            print(f"  ... {i+1}/{len(diff_parts)}")

    total_time = time.time() - start_time
    print(f"  Fertig! Total Zeit: {total_time:.2f}s. Ordner: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
