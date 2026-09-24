import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString, MultiPoint
from shapely.ops import unary_union, transform as shp_transform
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
from pyproj import Transformer

warnings.filterwarnings('ignore')

# ================= KONFIGURATION =================
DEFAULT_OUTPUT_DIR = "ausgabe"
TILE_CACHE_DIR = "tile_cache_17"
ZOOM_LEVEL = 17
MAX_RENDERS = 0
BG_COLOR = '#f0f0f0'

# Clustering:
# Wenn zwei Änderungsbereiche näher als diese Distanz (in Metern) sind,
# werden sie zu einem Cluster zusammengefasst.
CLUSTER_DISTANCE_M = 500

# "Geändert"-Erkennung:
# Ein entfernter und ein hinzugefügter Teil werden als "geändert" (gelb)
# markiert, wenn sie näher als diese Distanz beieinander liegen.
CHANGE_THRESHOLD_M = 10

# Farben für die Diff-Überlagerung
COLOR_ADDED = '#00AA00'      # grün  = hinzugefügt
COLOR_REMOVED = '#CC0000'    # rot   = entfernt
COLOR_CHANGED = '#FFC800'    # gelb  = geändert


def setup_args():
    if len(sys.argv) < 3:
        print(
            "Verwendung: python3 radlkarte_diff.py "
            "<datei_a.geojson> <datei_b.geojson> "
            "[ausgabe_ordner] [max_renders] [cluster_distance_m]"
        )
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT_DIR
    max_renders = int(sys.argv[4]) if len(sys.argv) > 4 else MAX_RENDERS

    if len(sys.argv) > 5:
        global CLUSTER_DISTANCE_M
        CLUSTER_DISTANCE_M = float(sys.argv[5])

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
        except (ValueError, TypeError):
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
        except (ValueError, TypeError):
            pass

    if props.get('unpaved') == 'yes':
        casing_color = '#00ff18'
        casing_width = 7
    elif props.get('steep') == 'yes':
        casing_color = '#ff00f0'
        casing_width = 7
    else:
        value = props.get("fixme")
        if pd.notna(value) and value:
            casing_color = '#FF0'
            casing_width = 7

    return {
        'main_color': main_color,
        'main_width': main_width,
        'casing_color': casing_color,
        'casing_width': casing_width,
        'linestyle': linestyle
    }


# ================= TILE-HELPER =================

def lonlat_to_tile(lon, lat, zoom):
    n = 2 ** zoom
    lat_rad = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lonlat(x, y, zoom):
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def get_tile_image(x, y, zoom, cache_dir=TILE_CACHE_DIR, verbose=False):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{zoom}_{x}_{y}.png")

    if os.path.exists(cache_file):
        try:
            img = Image.open(cache_file)
            return img
        except Exception:
            try:
                os.remove(cache_file)
            except OSError:
                pass

    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) RadlkarteDiff/1.0'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()
            img = Image.open(BytesIO(data))
            img.save(cache_file)
            return img
    except Exception as e:
        print(f"      Fehler beim Laden von Tile {x}/{y}: {e}")
        return Image.new('RGB', (256, 256), color=(200, 200, 200))


# ================= ZEICHNEN =================

def draw_basemap_on_ax(ax, xmin, ymin, xmax, ymax, zoom=ZOOM_LEVEL):
    x_tl, y_tl = lonlat_to_tile(xmin, ymax, zoom)
    x_br, y_br = lonlat_to_tile(xmax, ymin, zoom)

    for x in range(x_tl, x_br + 1):
        for y in range(y_tl, y_br + 1):
            tile_lon_tl, tile_lat_tl = tile_to_lonlat(x, y, zoom)
            tile_lon_br, tile_lat_br = tile_to_lonlat(x + 1, y + 1, zoom)

            if tile_lon_br < xmin or tile_lon_tl > xmax:
                continue
            if tile_lat_br < ymin or tile_lat_tl > ymax:
                continue

            try:
                tile_img = get_tile_image(x, y, zoom)
                if tile_img:
                    ax.imshow(
                        tile_img,
                        extent=[tile_lon_tl, tile_lon_br, tile_lat_br, tile_lat_tl],
                        origin='upper', zorder=0, aspect='auto'
                    )
            except Exception as e:
                print(f"      Fehler beim Zeichnen von Tile {x}/{y}: {e}")


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
                ax.plot(lons, lats, color=style['casing_color'],
                        linewidth=style['casing_width'] + style['main_width'],
                        alpha=0.75, zorder=1)

            ax.plot(lons, lats, color=style['main_color'],
                    linewidth=style['main_width'], linestyle=style['linestyle'], zorder=2)

        elif geom.geom_type == "Point":
            x, y = geom.x, geom.y
            color = 'blue' if props.get('dismount') == 'yes' else 'red'
            ax.plot(x, y, marker='o', color=color, markersize=8, zorder=3)

        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                coords = list(part.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]

                if style['casing_color'] != 'none':
                    ax.plot(lons, lats, color=style['casing_color'],
                            linewidth=style['casing_width'] + style['main_width'],
                            alpha=0.75, zorder=1)
                ax.plot(lons, lats, color=style['main_color'],
                        linewidth=style['main_width'], linestyle=style['linestyle'], zorder=2)


def draw_diff_on_ax(ax, diff_parts_colored):
    """Zeichnet die klassifizierten Diff-Teile als farbiges Overlay."""
    for geom, color in diff_parts_colored:
        if geom.is_empty:
            continue

        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            ax.plot(lons, lats, color=color, linewidth=3.5, alpha=0.9,
                    zorder=4, solid_capstyle='round')

        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                coords = list(part.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                ax.plot(lons, lats, color=color, linewidth=3.5, alpha=0.9,
                        zorder=4, solid_capstyle='round')

        elif geom.geom_type == "Point":
            ax.plot(geom.x, geom.y, marker='o', color=color, markersize=10,
                    alpha=0.9, zorder=4, markeredgecolor='white', markeredgewidth=1)


# ================= UTM-CRS & CLUSTERING =================

def _get_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return f"EPSG:{epsg}"


def _extract_coords(geom):
    pts = []
    if hasattr(geom, 'coords'):
        pts.extend(list(geom.coords))
    elif hasattr(geom, 'geoms'):
        for sub in geom.geoms:
            pts.extend(_extract_coords(sub))
    return pts


def _extract_parts(geom):
    """Extrahiert Einzelteile aus einer (Multi-)Geometrie."""
    if geom.is_empty:
        return []
    parts = []
    if isinstance(geom, (LineString, Point)):
        parts.append(geom)
    elif isinstance(geom, (MultiLineString, MultiPoint)):
        for part in geom.geoms:
            parts.append(part)
    else:
        if hasattr(geom, 'geoms'):
            for part in geom.geoms:
                parts.append(part)
        else:
            parts.append(geom)
    return [p for p in parts if not p.is_empty]


def classify_diff_parts(union_a, union_b):
    """
    Klassifiziert die Diff-Teile:
      - nur in A (entfernt)  → rot
      - nur in B (hinzugef.) → grün
      - beides in der Nähe   → gelb (geändert)

    Rückgabe: Liste von (geometry, color)-Tupeln.
    """
    only_in_a = union_a.difference(union_b)
    only_in_b = union_b.difference(union_a)

    removed_parts = _extract_parts(only_in_a)
    added_parts = _extract_parts(only_in_b)

    if not removed_parts and not added_parts:
        return []

    # UTM-Projektion für Distanzvergleich
    all_pts = []
    for g in removed_parts + added_parts:
        all_pts.extend(_extract_coords(g))

    centroid = MultiPoint(all_pts).centroid
    utm_crs = _get_utm_crs(centroid.x, centroid.y)
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    removed_utm = []
    for g in removed_parts:
        try:
            removed_utm.append(shp_transform(lambda x, y: transformer.transform(x, y), g))
        except Exception:
            removed_utm.append(g)

    added_utm = []
    for g in added_parts:
        try:
            added_utm.append(shp_transform(lambda x, y: transformer.transform(x, y), g))
        except Exception:
            added_utm.append(g)

    # "Geändert"-Erkennung: entfernt + hinzugefügt in der Nähe
    from shapely.strtree import STRtree

    removed_is_changed = [False] * len(removed_parts)
    added_is_changed = [False] * len(added_parts)

    if removed_utm and added_utm:
        added_tree = STRtree(added_utm)
        for i, rm in enumerate(removed_utm):
            neighbors = added_tree.query(rm, predicate='dwithin', distance=CHANGE_THRESHOLD_M)
            if len(neighbors) > 0:
                removed_is_changed[i] = True
                for j in neighbors:
                    added_is_changed[int(j)] = True

    # Ergebnis zusammenstellen
    results = []
    for i, g in enumerate(removed_parts):
        color = COLOR_CHANGED if removed_is_changed[i] else COLOR_REMOVED
        results.append((g, color))

    for i, g in enumerate(added_parts):
        color = COLOR_CHANGED if added_is_changed[i] else COLOR_ADDED
        results.append((g, color))

    n_green = sum(1 for _, c in results if c == COLOR_ADDED)
    n_red = sum(1 for _, c in results if c == COLOR_REMOVED)
    n_yellow = sum(1 for _, c in results if c == COLOR_CHANGED)
    print(f"   Klassifikation: {n_green} hinzugefügt, {n_red} entfernt, {n_yellow} geändert")

    return results


def cluster_colored_parts(colored_parts, distance_m):
    """
    Clustert (geometry, color)-Tupel über UTM + dwithin.
    Rückgabe: Liste von Clustern, jedes Cluster ist eine Liste von (geom, color).
    """
    if not colored_parts:
        return []

    print(f"  Führe Clustering durch (Abstand: {distance_m:.0f} m)...")

    geoms = [g for g, _ in colored_parts]
    clean_indices = [i for i, g in enumerate(geoms) if not g.is_empty]

    if not clean_indices:
        return []

    # UTM transformieren
    all_pts = []
    for i in clean_indices:
        all_pts.extend(_extract_coords(geoms[i]))

    centroid = MultiPoint(all_pts).centroid
    utm_crs = _get_utm_crs(centroid.x, centroid.y)
    print(f"   Verwende Projektion: {utm_crs}")

    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    clean_geoms = []
    for i in clean_indices:
        try:
            clean_geoms.append(shp_transform(lambda x, y: transformer.transform(x, y), geoms[i]))
        except Exception:
            clean_geoms.append(geoms[i])

    # Clustering via STRtree + dwithin
    from shapely.strtree import STRtree

    tree = STRtree(clean_geoms)
    visited = [False] * len(clean_geoms)
    clusters = []

    for i in range(len(clean_geoms)):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        cluster_indices = []

        while queue:
            idx = queue.pop(0)
            cluster_indices.append(idx)
            neighbors = tree.query(clean_geoms[idx], predicate='dwithin', distance=distance_m)
            for j in neighbors:
                j = int(j)
                if not visited[j]:
                    visited[j] = True
                    queue.append(j)

        # Auf Original-Indizes mappen
        cluster_parts = [colored_parts[clean_indices[k]] for k in cluster_indices]
        clusters.append(cluster_parts)

    print(f"   Clustering abgeschlossen: {len(clean_indices)} Teile -> {len(clusters)} Cluster")
    return clusters


# ================= HAUPTPROGRAMM =================

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
    except GEOSException as e:
        print(f"GEOS Fehler: {e}")
        sys.exit(1)

    calc_time = time.time() - start_time
    print(f"   (Berechnung Zeit: {calc_time:.2f}s)")

    # Klassifizierung der Änderungen
    colored_parts = classify_diff_parts(union_a, union_b)

    if not colored_parts:
        print("  Keine geometrischen Unterschiede gefunden.")
        sys.exit(0)

    print(f"  Gefundene Änderungsteile: {len(colored_parts)}")

    # Clustering
    clusters = cluster_colored_parts(colored_parts, CLUSTER_DISTANCE_M)

    if max_renders > 0:
        clusters = clusters[:max_renders]
        print(f"   (Limitiert auf {max_renders} Bilder)")

    print("    Erstelle Vergleichsbilder (Vorher/Nachher)...")

    for i, cluster in enumerate(clusters):
        if not cluster:
            continue

        # Bounds für das Cluster
        minx = min(g.bounds[0] for g, _ in cluster)
        miny = min(g.bounds[1] for g, _ in cluster)
        maxx = max(g.bounds[2] for g, _ in cluster)
        maxy = max(g.bounds[3] for g, _ in cluster)

        width = maxx - minx
        height = maxy - miny
        pad_x = max(width * 0.3, 0.001)
        pad_y = max(height * 0.3, 0.001)

        xmin = minx - pad_x
        ymin = miny - pad_y
        xmax = maxx + pad_x
        ymax = maxy + pad_y

        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
        common_xlim = (xmin, xmax)
        common_ylim = (ymin, ymax)

        # --- VORHER (Datei A) mit Diff-Overlay ---
        draw_basemap_on_ax(ax_before, xmin, ymin, xmax, ymax)
        ax_before.set_xlim(common_xlim)
        ax_before.set_ylim(common_ylim)
        gdf_a_subset = gdf_a.cx[xmin:xmax, ymin:ymax]
        draw_geometry_on_ax(ax_before, gdf_a_subset)
        draw_diff_on_ax(ax_before, cluster)
        ax_before.set_title("VORHER (Datei A)", fontsize=12, loc='left')
        ax_before.set_xticklabels([])
        ax_before.set_yticklabels([])

        # --- NACHHER (Datei B) ohne Diff-Overlay ---
        draw_basemap_on_ax(ax_after, xmin, ymin, xmax, ymax)
        ax_after.set_xlim(common_xlim)
        ax_after.set_ylim(common_ylim)
        gdf_b_subset = gdf_b.cx[xmin:xmax, ymin:ymax]
        draw_geometry_on_ax(ax_after, gdf_b_subset)
        ax_after.set_title("NACHHER (Datei B)", fontsize=12, loc='left')
        ax_after.set_xticklabels([])
        ax_after.set_yticklabels([])

        fig.suptitle(f"Änderung #{i + 1} ({len(cluster)} Teil(e))", fontsize=14)

        # OSM-Attribution
        fig.text(0.99, 0.01, '© OpenStreetMap contributors',
                 ha='right', va='bottom', fontsize=7, alpha=0.85, color='#333333')

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        filename = os.path.join(output_dir, f"vergleich_{i + 1:04d}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        if (i + 1) % 5 == 0:
            print(f"  ... {i + 1}/{len(clusters)}")

    total_time = time.time() - start_time
    print(f"  Fertig! Total Zeit: {total_time:.2f}s. Ordner: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
