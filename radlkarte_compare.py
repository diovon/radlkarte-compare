#!/usr/bin/env python3
"""
Radlkarte Compare – Vorher/Nachher-Vergleich zweier Radlkarte-GeoJSON-Dateien.

Verglichen werden Geometrie UND alle Attribute (stress, priority, oneway,
unpaved, steep, fixme, dismount, description, ...). Grundlage ist ein
Abgleich auf Ebene einzelner Liniensegmente bzw. Punkte:

  * Jede Linie wird in ihre Segmente (Stützpunkt -> Stützpunkt) zerlegt.
    Ein Segment ist "gleich", wenn es in der anderen Datei an exakt derselben
    Stelle mit denselben Attributen vorkommt. Dadurch werden Teilungen und
    Zusammenführungen von Wegen, neue IDs oder eine andere Reihenfolge in der
    Datei NICHT als Änderung gemeldet – echte Änderungen aber immer.
  * Gleiche Lage, andere Attribute          -> "Werte geändert"   (violett)
  * Nur in alt, neu liegt etwas daneben     -> "Geometrie geändert" (gelb)
  * Nur in alt                              -> "Entfernt"          (rot)
  * Nur in neu                              -> "Hinzugefügt"       (grün)
  * Bei Einbahnen (oneway) wird auch die Fahrtrichtung verglichen.

Benötigt nur matplotlib und Pillow (kein geopandas/shapely/pyproj mehr).
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

# ================= KONFIGURATION (Standardwerte) =================
DEFAULT_OUTPUT_DIR = "ausgabe"
TILE_CACHE_DIR = "tile_cache"
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
# Die OSM-Tile-Policy verlangt einen eindeutigen User-Agent (kein Browser-Fake).
USER_AGENT = "radlkarte-compare/2.0 (+https://github.com/diovon/radlkarte-compare)"
MAX_ZOOM = 18

CLUSTER_DISTANCE_M = 500     # Änderungen näher als das -> ein Bild
MAX_CLUSTER_EXTENT_M = 3000  # größere Cluster werden weiter aufgeteilt
CHANGE_THRESHOLD_M = 10      # entfernt+neu innerhalb davon -> "Geometrie geändert"
MIN_PADDING_M = 80           # Mindest-Rand um eine Änderung im Bild
PRECISION = 5                # Nachkommastellen (wie prepare_geojson.py der Radlkarte)
IGNORE_KEYS = {"id"}         # technische Attribute, die nicht verglichen werden

PANEL_INCH = 6.0
DPI = 110
MAX_TEXT_LINES = 14

COLOR_ADDED = "#00AA00"      # grün   = hinzugefügt
COLOR_REMOVED = "#CC0000"    # rot    = entfernt
COLOR_GEOM = "#FFC800"       # gelb   = Geometrie geändert
COLOR_ATTR = "#9B30FF"       # violett = nur Werte geändert

STATUS_INFO = {
    "removed": ("Entfernt", COLOR_REMOVED),
    "added": ("Hinzugefügt", COLOR_ADDED),
    "geom": ("Geometrie geändert", COLOR_GEOM),
    "attr": ("Werte geändert", COLOR_ATTR),
}

R_MERC = 6378137.0
R_MEAN = 6371008.8
DIR_KEY = "_fahrtrichtung"   # abgeleitetes Attribut für die Einbahn-Richtung


# ================= DATEN LADEN =================

class Feature:
    __slots__ = ("kind", "coords", "props", "fid", "attrs", "status", "_merc", "_mbbox")

    def __init__(self, kind, coords, props, fid):
        self.kind = kind          # "line" | "point"
        self.coords = coords      # Liste von (lon, lat)
        self.props = props        # normalisierte Attribute (ohne ignorierte)
        self.fid = fid            # Radlkarte-id (nur für Berichte)
        self.attrs = tuple(sorted(props.items()))
        self.status = None        # pro Segment (Linie) bzw. einzeln (Punkt)


def normalize_value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    v = str(v).strip()
    return v if v != "" else None


def load_features(path, ignore_keys, precision, geometry_only=False):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        raw = data.get("features") or []
    elif data.get("type") == "Feature":
        raw = [data]
    else:
        raise ValueError(f"{path}: keine GeoJSON FeatureCollection")

    def rnd(c):
        return (round(float(c[0]), precision), round(float(c[1]), precision))

    lines, points = [], []
    skipped = Counter()
    for feat in raw:
        geom = feat.get("geometry") or {}
        props_raw = feat.get("properties") or {}
        fid = props_raw.get("id")
        props = {}
        if not geometry_only:
            for k, v in props_raw.items():
                if k in ignore_keys:
                    continue
                nv = normalize_value(v)
                if nv is not None:
                    props[k] = nv

        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype in ("LineString", "MultiLineString"):
            parts = [coords] if gtype == "LineString" else coords
            for part in parts:
                pts = []
                for c in part:
                    p = rnd(c)
                    if not pts or pts[-1] != p:
                        pts.append(p)
                if len(pts) >= 2:
                    lines.append(Feature("line", pts, props, fid))
        elif gtype in ("Point", "MultiPoint"):
            for c in ([coords] if gtype == "Point" else coords):
                points.append(Feature("point", [rnd(c)], props, fid))
        else:
            skipped[str(gtype)] += 1

    if skipped:
        print(f"   Hinweis: ignorierte Geometrietypen in {path}: {dict(skipped)}")
    return lines, points


def oneway_direction(props):
    v = props.get("oneway")
    if v in ("yes", "true", "1"):
        return 1
    if v in ("-1", "reverse"):
        return -1
    return 0


# ================= VERGLEICH =================

def diff_attrs(attrs_a, attrs_b):
    """Liste von (key, alt, neu) für alle unterschiedlichen Attribute."""
    da, db = dict(attrs_a), dict(attrs_b)
    out = []
    for k in sorted(set(da) | set(db)):
        if da.get(k) != db.get(k):
            out.append((k, da.get(k), db.get(k)))
    return tuple(out)


def _pair_records(recs_a, recs_b):
    """
    recs_*: Listen von (attrs, fid, ref). Paart zuerst identische Attribute, dann
    die ähnlichsten. Liefert (gleich, geändert, nur_a, nur_b).
    """
    rest_b = list(recs_b)
    same, changed, only_a = [], [], []
    rest_a = []
    for ra in recs_a:
        # gleiche Attribute; bei Duplikaten bevorzugt dieselbe id
        cand = [j for j, rb in enumerate(rest_b) if rb[0] == ra[0]]
        if cand:
            j = next((j for j in cand if rest_b[j][1] == ra[1]), cand[0])
            same.append((ra, rest_b.pop(j)))
        else:
            rest_a.append(ra)
    while rest_a and rest_b:
        best = None
        for i, ra in enumerate(rest_a):
            for j, rb in enumerate(rest_b):
                n = len(diff_attrs(ra[0], rb[0]))
                if best is None or n < best[0]:
                    best = (n, i, j)
        _, i, j = best
        changed.append((rest_a.pop(i), rest_b.pop(j)))
    return same, changed, rest_a, rest_b


def compare_lines(lines_a, lines_b):
    """Setzt feature.status = Liste (pro Segment) für beide Seiten."""
    index = defaultdict(lambda: ([], []))
    for side, feats in ((0, lines_a), (1, lines_b)):
        for f in feats:
            f.status = [None] * (len(f.coords) - 1)
            ow = oneway_direction(f.props)
            for i in range(len(f.coords) - 1):
                p, q = f.coords[i], f.coords[i + 1]
                key = (p, q) if p <= q else (q, p)
                attrs = f.attrs
                if ow:
                    forward = (p <= q) == (ow == 1)
                    attrs = attrs + ((DIR_KEY, "+" if forward else "-"),)
                index[key][side].append((attrs, f.fid, (f, i)))

    for recs_a, recs_b in index.values():
        if not recs_a:
            for _, _, (f, i) in recs_b:
                f.status[i] = ("added",)
            continue
        if not recs_b:
            for _, _, (f, i) in recs_a:
                f.status[i] = ("removed",)
            continue
        _, changed, only_a, only_b = _pair_records(recs_a, recs_b)
        for (attrs_a, _, (fa, ia)), (attrs_b, _, (fb, ib)) in changed:
            fa.status[ia] = ("attr", diff_attrs(attrs_a, attrs_b), fb.fid)
        for _, _, (f, i) in only_a:
            f.status[i] = ("removed",)
        for _, _, (f, i) in only_b:
            f.status[i] = ("added",)


def compare_points(points_a, points_b):
    index = defaultdict(lambda: ([], []))
    for side, feats in ((0, points_a), (1, points_b)):
        for f in feats:
            f.status = None
            index[f.coords[0]][side].append((f.attrs, f.fid, f))
    for recs_a, recs_b in index.values():
        _, changed, only_a, only_b = _pair_records(recs_a, recs_b)
        for (attrs_a, _, fa), (attrs_b, _, fb) in changed:
            fa.status = ("attr", diff_attrs(attrs_a, attrs_b), fb.fid)
        for _, _, f in only_a:
            f.status = ("removed",)
        for _, _, f in only_b:
            f.status = ("added",)


# ================= LOKALE METRISCHE KOORDINATEN =================

class LocalProjection:
    """Einfache Äquirektangular-Projektion um den Datenmittelpunkt (Meter)."""

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.ky = math.pi / 180.0 * R_MEAN
        self.kx = self.ky * math.cos(math.radians(lat0))

    def xy(self, c):
        return ((c[0] - self.lon0) * self.kx, (c[1] - self.lat0) * self.ky)


def pt_seg_dist(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L = dx * dx + dy * dy
    if L == 0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _ccw(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def seg_seg_dist(a, b, c, d):
    d1, d2 = _ccw(a, b, c), _ccw(a, b, d)
    d3, d4 = _ccw(c, d, a), _ccw(c, d, b)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 and d2 and d3 and d4:
        return 0.0
    return min(pt_seg_dist(a, c, d), pt_seg_dist(b, c, d),
               pt_seg_dist(c, a, b), pt_seg_dist(d, a, b))


class SegmentGrid:
    """Gitter-Index für Segmente (in Metern)."""

    def __init__(self, cell):
        self.cell = max(cell, 1.0)
        self.cells = defaultdict(list)
        self.items = []

    def _range(self, x0, y0, x1, y1):
        c = self.cell
        return (range(math.floor(x0 / c), math.floor(x1 / c) + 1),
                range(math.floor(y0 / c), math.floor(y1 / c) + 1))

    def add(self, a, b, ref):
        idx = len(self.items)
        self.items.append((a, b, ref))
        rx, ry = self._range(min(a[0], b[0]), min(a[1], b[1]),
                             max(a[0], b[0]), max(a[1], b[1]))
        for ix in rx:
            for iy in ry:
                self.cells[(ix, iy)].append(idx)

    def query(self, x0, y0, x1, y1):
        seen = set()
        rx, ry = self._range(x0, y0, x1, y1)
        for ix in rx:
            for iy in ry:
                for idx in self.cells.get((ix, iy), ()):
                    if idx not in seen:
                        seen.add(idx)
                        yield self.items[idx]


# ================= ÄNDERUNGSTEILE =================

class Piece:
    """Zusammenhängender geänderter Abschnitt (oder einzelner Punkt)."""

    def __init__(self, side, status, kind, coords, feature, changes=(), fid_other=None):
        self.side = side            # "A" (alt) oder "B" (neu)
        self.status = status        # removed | added | geom | attr
        self.kind = kind            # line | point
        self.coords = coords
        self.feature = feature
        self.changes = changes      # Tupel (key, alt, neu)
        self.fid_other = fid_other
        self.xy = None
        self.length_m = 0.0
        self.partner = None

    def segments_xy(self):
        if len(self.xy) == 1:
            return [(self.xy[0], self.xy[0])]
        return list(zip(self.xy[:-1], self.xy[1:]))

    def bbox_xy(self):
        xs = [p[0] for p in self.xy]
        ys = [p[1] for p in self.xy]
        return min(xs), min(ys), max(xs), max(ys)


def build_pieces(lines_a, lines_b, points_a, points_b):
    pieces = []
    for side, feats in (("A", lines_a), ("B", lines_b)):
        for f in feats:
            i, n = 0, len(f.status)
            while i < n:
                st = f.status[i]
                if st is None or (side == "B" and st[0] == "attr"):
                    i += 1
                    continue
                j = i
                while j + 1 < n and f.status[j + 1] == st:
                    j += 1
                coords = f.coords[i:j + 2]
                if st[0] == "attr":
                    pieces.append(Piece(side, "attr", "line", coords, f, st[1], st[2]))
                else:
                    pieces.append(Piece(side, st[0], "line", coords, f))
                i = j + 1
    for side, feats in (("A", points_a), ("B", points_b)):
        for f in feats:
            st = f.status
            if st is None or (side == "B" and st[0] == "attr"):
                continue
            if st[0] == "attr":
                pieces.append(Piece(side, "attr", "point", f.coords, f, st[1], st[2]))
            else:
                pieces.append(Piece(side, st[0], "point", f.coords, f))
    return pieces


def _sample_points(xy, step):
    if len(xy) == 1:
        return [xy[0]]
    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(L / step))
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def detect_geometry_changes(pieces, threshold):
    """
    Ein entfernter Teil gilt als "Geometrie geändert", wenn er vollständig
    (jeder Punkt) innerhalb von `threshold` Metern neben neu hinzugekommener
    Geometrie liegt – und umgekehrt. Ein bloßes Berühren an einer Kreuzung
    reicht damit nicht mehr (das war vorher ein häufiger Fehlalarm).
    """
    removed = [p for p in pieces if p.status == "removed"]
    added = [p for p in pieces if p.status == "added"]
    if not removed or not added:
        return

    def make_grid(group):
        grid = SegmentGrid(threshold)
        for p in group:
            for a, b in p.segments_xy():
                grid.add(a, b, p)
        return grid

    grids = {"removed": make_grid(removed), "added": make_grid(added)}
    new_status = {}
    for group, other in ((removed, "added"), (added, "removed")):
        grid = grids[other]
        for p in group:
            votes = Counter()
            refs = {}
            ok = True
            for s in _sample_points(p.xy, max(threshold / 2.0, 1.0)):
                best, best_ref = None, None
                for a, b, ref in grid.query(s[0] - threshold, s[1] - threshold,
                                            s[0] + threshold, s[1] + threshold):
                    if ref.kind != p.kind:
                        continue
                    d = pt_seg_dist(s, a, b)
                    if best is None or d < best:
                        best, best_ref = d, ref
                if best is None or best > threshold:
                    ok = False
                    break
                votes[id(best_ref)] += 1
                refs[id(best_ref)] = best_ref
            if ok:
                new_status[id(p)] = True
                p.partner = refs[votes.most_common(1)[0][0]]

    for p in removed + added:
        if new_status.get(id(p)):
            p.status = "geom"
            if p.partner is not None:
                if p.side == "A":
                    p.changes = diff_attrs(p.feature.attrs, p.partner.feature.attrs)
                    p.fid_other = p.partner.feature.fid
                else:
                    p.changes = diff_attrs(p.partner.feature.attrs, p.feature.attrs)
                    p.fid_other = p.partner.feature.fid


# ================= CLUSTERING =================

def cluster_pieces(pieces, distance):
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grid = SegmentGrid(distance)
    for i, p in enumerate(pieces):
        for a, b in p.segments_xy():
            grid.add(a, b, i)

    for i, p in enumerate(pieces):
        for a, b in p.segments_xy():
            x0, x1 = min(a[0], b[0]) - distance, max(a[0], b[0]) + distance
            y0, y1 = min(a[1], b[1]) - distance, max(a[1], b[1]) + distance
            for c, d, j in grid.query(x0, y0, x1, y1):
                if find(i) == find(j):
                    continue
                if seg_seg_dist(a, b, c, d) <= distance:
                    parent[find(i)] = find(j)

    groups = defaultdict(list)
    for i, p in enumerate(pieces):
        groups[find(i)].append(p)
    return list(groups.values())


def cluster_extent(cluster):
    boxes = [p.bbox_xy() for p in cluster]
    return max(max(b[2] for b in boxes) - min(b[0] for b in boxes),
               max(b[3] for b in boxes) - min(b[1] for b in boxes))


def _sub_piece(p, i, j):
    """Teilstück der Segmente i..j (inklusive) eines Linien-Pieces."""
    q = Piece(p.side, p.status, p.kind, p.coords[i:j + 2], p.feature, p.changes, p.fid_other)
    q.xy = p.xy[i:j + 2]
    q.length_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(q.xy[:-1], q.xy[1:]))
    q.partner = p.partner
    return q


def grid_split(cluster, cell):
    """
    Teilt einen zusammenhängenden Riesen-Cluster in Gitterzellen der Größe `cell`
    (jedes Segment kommt in die Zelle seines Mittelpunkts).
    """
    groups = defaultdict(list)

    def cell_of(a, b):
        return (math.floor((a[0] + b[0]) / 2 / cell), math.floor((a[1] + b[1]) / 2 / cell))

    for p in cluster:
        if p.kind == "point":
            groups[cell_of(p.xy[0], p.xy[0])].append(p)
            continue
        segs = list(zip(p.xy[:-1], p.xy[1:]))
        start = 0
        for k in range(1, len(segs) + 1):
            if k == len(segs) or cell_of(*segs[k]) != cell_of(*segs[start]):
                groups[cell_of(*segs[start])].append(
                    p if (start == 0 and k == len(segs)) else _sub_piece(p, start, k - 1))
                start = k
    return list(groups.values())


def cluster_with_limit(pieces, distance, max_extent, min_distance):
    """
    Clustert; zu große Cluster (Ketteneffekt) werden zuerst mit kleinerer Distanz
    neu geclustert und, falls die Änderungen lückenlos zusammenhängen, in
    Gitterzellen geteilt.
    """
    result = []
    for cl in cluster_pieces(pieces, distance):
        if max_extent <= 0 or cluster_extent(cl) <= max_extent:
            result.append(cl)
        elif len(cl) > 1 and distance / 2.0 >= min_distance:
            result.extend(cluster_with_limit(cl, distance / 2.0, max_extent, min_distance))
        else:
            result.extend(grid_split(cl, max_extent))
    return result


# ================= TEXT / BERICHT =================

def fmt_value(v, maxlen=40):
    if v is None:
        return "–"
    return v if len(v) <= maxlen else v[:maxlen - 1] + "…"


def describe_changes(changes):
    keys = {k for k, _, _ in changes}
    out = []
    for k, va, vb in changes:
        if k == DIR_KEY:
            if va and vb:
                out.append("Einbahn-Richtung umgedreht")
            elif "oneway" not in keys:
                out.append("Einbahn-Richtung geändert")
            continue
        out.append(f"{k}: {fmt_value(va, 25)} → {fmt_value(vb, 25)}")
    return out


def describe_attrs(attrs, maxlen=60, value_len=20):
    s = ", ".join(f"{k}={fmt_value(v, value_len)}" for k, v in attrs)
    return fmt_value(s, maxlen) if s else "(keine Attribute)"


def fid_str(fid):
    return "?" if fid is None else str(fid)


def summarize_cluster(cluster):
    """Aggregierte, lesbare Zeilen für die Änderungsliste unter dem Bild."""
    agg = {}
    for p in cluster:
        f = p.feature
        if p.status == "geom" and p.side == "B" and p.partner is not None:
            continue  # wird über die alte Seite gemeldet
        if p.side == "A":
            ids = (f.fid, p.fid_other)
        else:
            ids = (p.fid_other, f.fid)
        if p.status in ("removed", "added"):
            detail = describe_attrs(f.attrs)
        else:
            detail = "; ".join(describe_changes(p.changes))
        key = (p.status, p.kind, ids, detail)
        agg.setdefault(key, 0.0)
        agg[key] += p.length_m

    order = {"attr": 0, "geom": 1, "removed": 2, "added": 3}
    lines = []
    for (status, kind, (ida, idb), detail), length in sorted(
            agg.items(), key=lambda kv: (order[kv[0][0]], -kv[1])):
        label = STATUS_INFO[status][0]
        if status == "removed":
            idtxt = f"id {fid_str(ida)}"
        elif status == "added":
            idtxt = f"id {fid_str(idb)}"
        elif ida == idb or idb is None:
            idtxt = f"id {fid_str(ida)}"
        else:
            idtxt = f"id {fid_str(ida)}→{fid_str(idb)}"
        what = "Punkt" if kind == "point" else f"{length:.0f} m"
        if status == "geom" and not detail:
            detail = "nur Lage verändert"
        lines.append((status, f"{label} ({idtxt}, {what}): {detail}"))
    return lines


def write_csv(path, clusters, filenames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["bild", "art", "objekt", "id_alt", "id_neu", "laenge_m",
                    "attribut", "wert_alt", "wert_neu", "lon", "lat"])
        for cluster, fname in zip(clusters, filenames):
            for p in cluster:
                if p.status == "geom" and p.side == "B" and p.partner is not None:
                    continue
                f_ = p.feature
                id_a, id_b = (f_.fid, p.fid_other) if p.side == "A" else (p.fid_other, f_.fid)
                lon, lat = p.coords[len(p.coords) // 2]
                base = [fname, STATUS_INFO[p.status][0],
                        "Punkt" if p.kind == "point" else "Linie",
                        fid_str(id_a) if id_a is not None else "",
                        fid_str(id_b) if id_b is not None else "",
                        f"{p.length_m:.1f}" if p.kind == "line" else ""]
                tail = [lon, lat]
                if p.status == "removed":
                    w.writerow(base + ["", describe_attrs(f_.attrs, 10**6, 10**6), ""] + tail)
                elif p.status == "added":
                    w.writerow(base + ["", "", describe_attrs(f_.attrs, 10**6, 10**6)] + tail)
                elif not p.changes:
                    w.writerow(base + ["", "", ""] + tail)
                else:
                    for k, va, vb in p.changes:
                        if k == DIR_KEY:
                            if not (va and vb):
                                continue  # steckt bereits in der oneway-Änderung
                            k, va, vb = "oneway (Richtung)", "", "umgedreht"
                        w.writerow(base + [k, va or "", vb or ""] + tail)


def feature_level_summary(lines_a, points_a, lines_b, points_b):
    """Kurze Übersicht auf Ebene der Radlkarte-ids (nur Info in der Konsole)."""
    def by_id(feats):
        d = defaultdict(list)
        for f in feats:
            if f.fid is not None and str(f.fid) not in ("-1", ""):
                d[str(f.fid)].append((f.kind, tuple(f.coords), f.attrs))
        return {k: sorted(v) for k, v in d.items()}

    a = by_id(lines_a + points_a)
    b = by_id(lines_b + points_b)
    only_a = len(set(a) - set(b))
    only_b = len(set(b) - set(a))
    changed = sum(1 for k in set(a) & set(b) if a[k] != b[k])
    print(f"   Objekte nach id: {changed} geändert, {only_a} nur in alt, {only_b} nur in neu "
          f"(Teilungen/Neunummerierungen erscheinen hier, aber nicht als Kartenänderung)")


# ================= KARTE / DARSTELLUNG =================

def merc(c):
    lon, lat = c
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = math.radians(lon) * R_MERC
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R_MERC
    return x, y


def get_josm_style(props):
    """Darstellung wie josm-radlkarte-style.mapcss (spätere Regeln gewinnen)."""
    main_color = "#51A4B6"
    main_width = 3
    linestyle = "-"
    casing_color, casing_width, casing_alpha = None, 0, 0.75

    stress = props.get("stress")
    if stress == "0":
        main_color = "#004B67"
    elif stress == "1":
        main_color = "#51A4B6"
    elif stress == "2":
        main_color = "#FF6600"

    priority = props.get("priority")
    if priority == "0":
        main_width = 12
    elif priority == "2":
        linestyle = (0, (5, 5))

    # Reihenfolge wie im MapCSS: unpaved < steep < fixme
    if props.get("unpaved") == "yes":
        casing_color, casing_width, casing_alpha = "#00ff18", 7, 0.75
    if props.get("steep") == "yes":
        casing_color, casing_width, casing_alpha = "#ff00f0", 7, 0.75
    if props.get("fixme"):
        casing_color, casing_width, casing_alpha = "#FFFF00", 7, 0.5

    return {"main_color": main_color, "main_width": main_width, "linestyle": linestyle,
            "casing_color": casing_color, "casing_width": casing_width,
            "casing_alpha": casing_alpha}


def point_style(props):
    if props.get("dismount") == "yes":
        return dict(marker="o", color="#1f4fd1", size=8)
    if props.get("warning") == "yes":
        return dict(marker="^", color="#E8A000", size=9)
    if props.get("nocargo") == "yes":
        return dict(marker="s", color="#7a4a1a", size=7)
    return dict(marker="o", color="#CC0000", size=7)


class Renderer:
    def __init__(self, args):
        self.args = args
        self.tile_count = 0

    # --- Kacheln ---
    def get_tile(self, z, x, y):
        n = 2 ** z
        x = x % n
        if y < 0 or y >= n:
            return None
        path = os.path.join(self.args.tile_cache, str(z), str(x), f"{y}.png")
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                try:
                    os.remove(path)
                except OSError:
                    pass
        url = self.args.tile_url.format(z=z, x=x, y=y)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            img = Image.open(BytesIO(data)).convert("RGB")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)
            self.tile_count += 1
            return img
        except Exception as e:
            print(f"      Kachel {z}/{x}/{y} nicht ladbar: {e}")
            return None

    def basemap(self, x0, y0, x1, y1, z):
        """Mosaik aus Kacheln, die den Bereich (Web-Mercator) vollständig abdecken."""
        world = 2 * math.pi * R_MERC
        n = 2 ** z
        tx0 = math.floor((x0 + world / 2) / world * n)
        tx1 = math.floor((x1 + world / 2) / world * n)
        ty0 = math.floor((world / 2 - y1) / world * n)
        ty1 = math.floor((world / 2 - y0) / world * n)
        mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (230, 230, 230))
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                tile = self.get_tile(z, tx, ty)
                if tile is not None:
                    mosaic.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))
        size = world / n
        extent = (tx0 * size - world / 2, (tx1 + 1) * size - world / 2,
                  world / 2 - (ty1 + 1) * size, world / 2 - ty0 * size)
        return mosaic, extent

    # --- Zeichnen ---
    @staticmethod
    def draw_features(ax, feats, bbox):
        x0, y0, x1, y1 = bbox
        for f in feats:
            m = f._merc
            fb = f._mbbox
            if fb[2] < x0 or fb[0] > x1 or fb[3] < y0 or fb[1] > y1:
                continue
            xs = [p[0] for p in m]
            ys = [p[1] for p in m]
            if f.kind == "line":
                st = get_josm_style(f.props)
                if st["casing_color"]:
                    ax.plot(xs, ys, color=st["casing_color"],
                            linewidth=st["casing_width"] + st["main_width"],
                            alpha=st["casing_alpha"], zorder=1, solid_capstyle="round")
                ax.plot(xs, ys, color=st["main_color"], linewidth=st["main_width"],
                        linestyle=st["linestyle"], zorder=2, solid_capstyle="round")
            else:
                ps = point_style(f.props)
                ax.plot(xs, ys, marker=ps["marker"], color=ps["color"], markersize=ps["size"],
                        markeredgecolor="white", markeredgewidth=1, linestyle="none", zorder=3)

    @staticmethod
    def draw_overlay(ax, pieces):
        """
        Markiert Änderungen wie ein Textmarker als breites, halbtransparentes Band
        UNTER den Radlkarte-Linien – so bleibt deren Farbe/Stil (z. B. eine
        geänderte stress-Stufe) in beiden Bildern sichtbar.
        """
        for p in pieces:
            color = STATUS_INFO[p.status][1]
            xs = [c[0] for c in p._merc]
            ys = [c[1] for c in p._merc]
            if p.kind == "line":
                ax.plot(xs, ys, color=color, linewidth=22, alpha=0.45, zorder=0.5,
                        solid_capstyle="butt", solid_joinstyle="round")
                ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.9, zorder=0.6,
                        linestyle=(0, (2, 2)))
            else:
                ax.plot(xs, ys, marker="o", markersize=22, markerfacecolor=color, alpha=0.45,
                        markeredgecolor=color, markeredgewidth=2.5, linestyle="none",
                        zorder=0.5)

    def render(self, nr, cluster, feats_a, feats_b, filename):
        args = self.args
        pts = [c for p in cluster for c in p._merc]
        minx = min(p[0] for p in pts)
        maxx = max(p[0] for p in pts)
        miny = min(p[1] for p in pts)
        maxy = max(p[1] for p in pts)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

        # Web-Mercator-Meter sind um 1/cos(lat) gestreckt -> Puffer in echten Metern
        lat_c = math.degrees(2 * math.atan(math.exp(cy / R_MERC)) - math.pi / 2)
        scale = 1.0 / math.cos(math.radians(lat_c))
        size = max(maxx - minx, maxy - miny)
        pad = max(size * 0.25, args.padding_m * scale)
        half = size / 2 + pad
        bbox = (cx - half, cy - half, cx + half, cy + half)

        panel_px = PANEL_INCH * DPI
        world = 2 * math.pi * R_MERC
        z = round(math.log2(world * panel_px / (256 * 2 * half)))
        z = max(1, min(args.max_zoom, z))

        summary = summarize_cluster(cluster)
        shown = summary[:MAX_TEXT_LINES]
        n_text = len(shown) + (1 if len(summary) > MAX_TEXT_LINES else 0)
        text_h = 0.25 + 0.2 * n_text

        fig = plt.figure(figsize=(2 * PANEL_INCH, PANEL_INCH + 0.9 + text_h), dpi=DPI)
        top = 1 - 0.6 / fig.get_figheight()
        bottom_map = (text_h + 0.3) / fig.get_figheight()
        gs = fig.add_gridspec(1, 2, left=0.01, right=0.99, top=top, bottom=bottom_map, wspace=0.03)
        ax_before = fig.add_subplot(gs[0, 0])
        ax_after = fig.add_subplot(gs[0, 1])

        mosaic = None
        if not args.no_basemap:
            mosaic, extent = self.basemap(*bbox, z)

        for ax, feats, title in ((ax_before, feats_a, "VORHER (alt)"),
                                 (ax_after, feats_b, "NACHHER (neu)")):
            ax.set_facecolor("#f0f0f0")
            if mosaic is not None:
                ax.imshow(mosaic, extent=extent, origin="upper", zorder=0, interpolation="bilinear")
            self.draw_features(ax, feats, bbox)
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=12, loc="left")

            self.draw_overlay(ax, cluster)

        statuses = [s for s in ("removed", "added", "geom", "attr")
                    if any(p.status == s for p in cluster)]
        handles = [Line2D([0], [0], color=STATUS_INFO[s][1], lw=8, alpha=0.6,
                          label=STATUS_INFO[s][0])
                   for s in statuses]
        if handles:
            ax_before.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

        lon_c = math.degrees(cx / R_MERC)
        fig.suptitle(f"Änderung #{nr}  –  {lat_c:.5f}, {lon_c:.5f}", fontsize=13,
                     x=0.01, ha="left", y=1 - 0.15 / fig.get_figheight(), va="top")

        y = (text_h + 0.05) / fig.get_figheight()
        dy = 0.2 / fig.get_figheight()
        for status, line in shown:
            fig.text(0.012, y, "■", color=STATUS_INFO[status][1], fontsize=9, va="top")
            fig.text(0.028, y, line, fontsize=8.5, va="top", family="DejaVu Sans")
            y -= dy
        if len(summary) > MAX_TEXT_LINES:
            fig.text(0.028, y, f"… und {len(summary) - MAX_TEXT_LINES} weitere "
                     f"(siehe aenderungen.csv)", fontsize=8.5, va="top", style="italic")

        if mosaic is not None:
            fig.text(0.99, 0.005, "Karte: © OpenStreetMap-Mitwirkende", ha="right",
                     va="bottom", fontsize=7, color="#333333")

        fig.savefig(filename)
        plt.close(fig)


# ================= HAUPTPROGRAMM =================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Vorher/Nachher-Vergleich zweier Radlkarte-GeoJSON-Dateien "
                    "(Geometrie und alle Attribute).")
    ap.add_argument("datei_alt")
    ap.add_argument("datei_neu")
    ap.add_argument("ausgabe_ordner", nargs="?", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("max_renders", nargs="?", type=int, default=0,
                    help="max. Anzahl Bilder (0 = alle)")
    ap.add_argument("cluster_distance_m", nargs="?", type=float, default=CLUSTER_DISTANCE_M,
                    help=f"Änderungen näher als X m kommen in ein Bild (Std. {CLUSTER_DISTANCE_M})")
    ap.add_argument("--max-ausdehnung", dest="max_extent", type=float,
                    default=MAX_CLUSTER_EXTENT_M,
                    help="größere Cluster werden weiter geteilt (m, 0 = aus)")
    ap.add_argument("--schwelle", dest="change_threshold", type=float, default=CHANGE_THRESHOLD_M,
                    help="Abstand in m, bis zu dem entfernt+neu als 'Geometrie geändert' gilt")
    ap.add_argument("--rand", dest="padding_m", type=float, default=MIN_PADDING_M,
                    help="Mindestrand um die Änderung im Bild (m)")
    ap.add_argument("--ignoriere", default=",".join(sorted(IGNORE_KEYS)),
                    help="Komma-getrennte Attribute, die nicht verglichen werden (Std.: id)")
    ap.add_argument("--nur-geometrie", dest="geometry_only", action="store_true",
                    help="nur Geometrie vergleichen (altes Verhalten)")
    ap.add_argument("--genauigkeit", dest="precision", type=int, default=PRECISION,
                    help="Nachkommastellen beim Koordinatenvergleich (Std. 5)")
    ap.add_argument("--ohne-karte", dest="no_basemap", action="store_true",
                    help="keine OSM-Hintergrundkarte laden")
    ap.add_argument("--max-zoom", type=int, default=MAX_ZOOM)
    ap.add_argument("--tile-url", default=TILE_URL)
    ap.add_argument("--tile-cache", default=TILE_CACHE_DIR)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ignore = {k.strip() for k in args.ignoriere.split(",") if k.strip()}
    os.makedirs(args.ausgabe_ordner, exist_ok=True)
    t0 = time.time()

    print(f"  Lade {args.datei_alt} ...")
    lines_a, points_a = load_features(args.datei_alt, ignore, args.precision, args.geometry_only)
    print(f"  Lade {args.datei_neu} ...")
    lines_b, points_b = load_features(args.datei_neu, ignore, args.precision, args.geometry_only)
    print(f"   alt: {len(lines_a)} Linien, {len(points_a)} Punkte | "
          f"neu: {len(lines_b)} Linien, {len(points_b)} Punkte")

    if not args.geometry_only:
        feature_level_summary(lines_a, points_a, lines_b, points_b)

    print("  Vergleiche Segmente und Punkte ...")
    compare_lines(lines_a, lines_b)
    compare_points(points_a, points_b)
    pieces = build_pieces(lines_a, lines_b, points_a, points_b)
    if not pieces:
        print("  Keine Unterschiede gefunden.")
        return 0

    all_coords = [c for f in lines_a + lines_b + points_a + points_b for c in f.coords]
    proj = LocalProjection(sum(c[0] for c in all_coords) / len(all_coords),
                           sum(c[1] for c in all_coords) / len(all_coords))
    for p in pieces:
        p.xy = [proj.xy(c) for c in p.coords]
        p.length_m = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                         for a, b in zip(p.xy[:-1], p.xy[1:]))

    detect_geometry_changes(pieces, args.change_threshold)
    counts = Counter(p.status for p in pieces)
    print("   Änderungsteile: " + ", ".join(
        f"{counts.get(s, 0)} {STATUS_INFO[s][0].lower()}" for s in STATUS_INFO))

    print(f"  Clustering (Abstand {args.cluster_distance_m:.0f} m, "
          f"max. Ausdehnung {args.max_extent:.0f} m) ...")
    clusters = cluster_with_limit(pieces, args.cluster_distance_m, args.max_extent,
                                  max(2 * args.change_threshold, 20.0))
    # Westen -> Osten, damit die Nummerierung reproduzierbar ist
    clusters.sort(key=lambda cl: (min(c[0] for p in cl for c in p.coords),
                                  min(c[1] for p in cl for c in p.coords)))
    print(f"   {len(pieces)} Teile -> {len(clusters)} Bilder")
    if args.max_renders > 0 and len(clusters) > args.max_renders:
        clusters = clusters[:args.max_renders]
        print(f"   (begrenzt auf {args.max_renders} Bilder)")

    for f in lines_a + lines_b + points_a + points_b:
        f._merc = [merc(c) for c in f.coords]
        xs = [p[0] for p in f._merc]
        ys = [p[1] for p in f._merc]
        f._mbbox = (min(xs), min(ys), max(xs), max(ys))
    for cl in clusters:
        for p in cl:
            p._merc = [merc(c) for c in p.coords]

    renderer = Renderer(args)
    feats_a = lines_a + points_a
    feats_b = lines_b + points_b
    filenames = []
    print("  Erstelle Vergleichsbilder (Vorher/Nachher) ...")
    for i, cluster in enumerate(clusters, 1):
        fname = f"vergleich_{i:04d}.png"
        renderer.render(i, cluster, feats_a, feats_b, os.path.join(args.ausgabe_ordner, fname))
        filenames.append(fname)
        if i % 5 == 0:
            print(f"   ... {i}/{len(clusters)}")

    csv_path = os.path.join(args.ausgabe_ordner, "aenderungen.csv")
    write_csv(csv_path, clusters, filenames)
    if renderer.tile_count:
        print(f"   {renderer.tile_count} Kacheln neu geladen")
    print(f"  Fertig in {time.time() - t0:.1f}s. Ordner: {os.path.abspath(args.ausgabe_ordner)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
