# ===================== Streamlit Cloud–friendly build ======================
# Features:
# - Places (New) autocomplete + details (HTTP)
# - KMeans clustering
# - VRPTW per cluster (OR-Tools), with safe parameters + heuristic fallback
# - TSP fallback (pure Python heuristic) when no TW or infeasible
# - Priced time-slot offers per run for a new address
# - Google Routes polylines (chunked) + Maps JS render
# - Works with ONE key (uses same for JS + HTTP) or TWO keys (secrets)
# - Thread/env caps to reduce native segfaults on Cloud

# -------------------- Set env caps BEFORE heavy imports --------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")

# -------------------- Standard imports ------------------------------------
import sys, faulthandler
faulthandler.enable()

import json
import math
import time
from typing import List, Dict, Any, Tuple
from datetime import time as dtime

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# -------------------- App header & keys -----------------------------------
st.set_page_config(page_title="VRPTW + Priced Time Slots (Cloud Stable)", page_icon="⏱️", layout="wide")
st.title("⏱️ Cluster + VRPTW + Priced Time Slots (cloud-stable)")

# Keys: support either 2-key setup OR a single key named GOOGLE_API_KEY.
MAPS_JS_KEY   = st.secrets.get("GOOGLE_MAPS_JS_KEY") or os.getenv("GOOGLE_MAPS_JS_KEY") or ""
BACKEND_KEY   = st.secrets.get("GOOGLE_BACKEND_KEY") or os.getenv("GOOGLE_BACKEND_KEY") or ""
SINGLE_KEY    = st.secrets.get("GOOGLE_API_KEY")     or os.getenv("GOOGLE_API_KEY")     or ""

if not MAPS_JS_KEY and SINGLE_KEY:
    MAPS_JS_KEY = SINGLE_KEY
if not BACKEND_KEY and SINGLE_KEY:
    BACKEND_KEY = SINGLE_KEY

PLACES_AUTOCOMPLETE_V1 = "https://places.googleapis.com/v1/places:autocomplete"
PLACE_DETAILS_V1_TMPL  = "https://places.googleapis.com/v1/places/{place_id}"
ROUTES_COMPUTE         = "https://routes.googleapis.com/directions/v2:computeRoutes"

# -------------------- Session state ---------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("chosen", [])
    ss.setdefault("suggestions_main", [])
    ss.setdefault("suggestions_new", [])
    ss.setdefault("clusters", None)       # {'labels','k','target_size','points','sizes','centroids'}
    ss.setdefault("routes", [])           # [{'cluster','run_id','order','ordered_points','poly_points','steps','arrivals'}]
    ss.setdefault("new_candidate", None)
    ss.setdefault("slot_options", None)   # slots for new order
    ss.setdefault("bias_au", True)
    ss.setdefault("map_version", 0)
    ss.setdefault("speed_kmh", 40.0)

    # Pricing model
    ss.setdefault("cost_labour_per_h", 40.0)
    ss.setdefault("cost_vehicle_per_km", 0.7)
    ss.setdefault("cost_fixed_stop", 3.0)
    ss.setdefault("slot_len_min", 15)

    # Time windows per place_id
    ss.setdefault("tw", {})
init_state()

def dbg(msg: str):
    print(f"[VRPTW] {msg}", file=sys.stderr, flush=True)

# -------------------- Helpers ---------------------------------------------
def minutes_to_hhmm(day_start_sec: int, secs_from_daystart: int) -> str:
    t = day_start_sec + int(secs_from_daystart)
    hh = (t // 3600) % 24
    mm = (t % 3600) // 60
    return f"{hh:02d}:{mm:02d}"

def latlng_to_local_xy_m(points_latlng: np.ndarray) -> np.ndarray:
    if len(points_latlng) == 0: return points_latlng
    mean_lat = np.deg2rad(np.mean(points_latlng[:,0]))
    m_per_deg_lat = 111_132.954
    m_per_deg_lng = 111_132.954 * math.cos(mean_lat)
    x = points_latlng[:,1] * m_per_deg_lng
    y = points_latlng[:,0] * m_per_deg_lat
    return np.column_stack([x,y])

def build_distance_matrix_m(points_ll: List[Tuple[float,float]]) -> List[List[int]]:
    pts = np.array(points_ll, dtype=float)
    xy = latlng_to_local_xy_m(pts)
    n = len(xy)
    M = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: M[i][j]=0
            else:
                dx=xy[i,0]-xy[j,0]; dy=xy[i,1]-xy[j,1]
                M[i][j] = int(round(math.hypot(dx,dy)))
    return M

def meters_to_seconds(meters: int, speed_kmh: float) -> int:
    speed_mps = max(1e-6, speed_kmh*1000.0/3600.0)
    return int(round(meters/speed_mps))

# -------------------- Google helpers --------------------------------------
def autocomplete_new(query: str, bias_au: bool, min_chars: int = 3) -> Dict[str, Any]:
    if not query or len(query) < min_chars or not BACKEND_KEY:
        return {"status": "IDLE", "error": "", "suggestions": []}
    headers = {
        "Content-Type":"application/json",
        "X-Goog-Api-Key":BACKEND_KEY,
        "X-Goog-FieldMask":"suggestions.placePrediction.text.text,suggestions.placePrediction.placeId",
    }
    body = {"input": query, "includeQueryPredictions": False}
    if bias_au: body["includedRegionCodes"] = ["au"]
    try:
        r = requests.post(PLACES_AUTOCOMPLETE_V1, headers=headers, json=body, timeout=12)
        data = r.json()
        if r.status_code != 200:
            return {"status": f"HTTP_{r.status_code}",
                    "error": data.get("error", {}).get("message", ""),
                    "suggestions": []}
        out = []
        for s in data.get("suggestions", []):
            pp = s.get("placePrediction")
            if not pp: continue
            label = (pp.get("text") or {}).get("text")
            pid   = pp.get("placeId")
            if label and pid: out.append({"label": label, "place_id": pid})
        return {"status":"OK","error":"","suggestions":out}
    except Exception as e:
        return {"status":"REQUEST_ERROR","error":str(e),"suggestions":[]}

def place_details_new(place_id: str) -> Dict[str, Any]:
    headers = {
        "Content-Type":"application/json",
        "X-Goog-Api-Key":BACKEND_KEY,
        "X-Goog-FieldMask":"id,formattedAddress,location",
    }
    url = PLACE_DETAILS_V1_TMPL.format(place_id=place_id)
    r = requests.get(url, headers=headers, timeout=12)
    data = r.json()
    if r.status_code != 200:
        return {"error": data.get("error",{}).get("message","Unknown error")}
    loc = data.get("location", {})
    return {"address": data.get("formattedAddress",""),
            "lat": loc.get("latitude"),
            "lng": loc.get("longitude"),
            "place_id": data.get("id"),
            "error": ""}

# -------------------- TSP (fallback) --------------------------------------
def tsp_fast_order(points_ll, route_type="loop"):
    n = len(points_ll)
    if n <= 1: return list(range(n))
    D = build_distance_matrix_m(points_ll)
    clat = sum(p[0] for p in points_ll)/n
    clng = sum(p[1] for p in points_ll)/n
    start = min(range(n), key=lambda i: (points_ll[i][0]-clat)**2 + (points_ll[i][1]-clng)**2)
    unv = set(range(n)); unv.remove(start)
    path = [start]; cur = start
    while unv:
        nxt = min(unv, key=lambda j: D[cur][j]); unv.remove(nxt); path.append(nxt); cur = nxt
    if route_type == "loop": path = path + [path[0]]
    return path

# -------------------- VRPTW (OR-Tools) + heuristic fallback ---------------
def vrptw_ortools(points_ll, service_sec, tw_start_sec, tw_end_sec,
                  route_type, speed_kmh, horizon_sec, time_limit_sec=12):
    """Single-vehicle VRPTW via OR-Tools (guarded). Returns (order, arrivals) or ([], [])."""
    n=len(points_ll)
    if n<=1: return list(range(n)), [0]
    D = build_distance_matrix_m(points_ll)
    T = [[meters_to_seconds(D[i][j], speed_kmh) for j in range(n)] for i in range(n)]
    start_idx = int(np.argmin(tw_start_sec))
    end_idx   = start_idx if route_type=="loop" else int(np.argmax(tw_end_sec))
    if route_type=="path" and end_idx==start_idx and n>1: end_idx=(start_idx+1)%n

    manager = pywrapcp.RoutingIndexManager(n, 1, [start_idx], [start_idx] if route_type=="loop" else [end_idx])
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(i,j):
        a=manager.IndexToNode(i); b=manager.IndexToNode(j)
        return T[a][b] + service_sec[a]
    transit = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    dim="Time"
    routing.AddDimension(transit, horizon_sec, horizon_sec, True, dim)
    time_dim = routing.GetDimensionOrDie(dim)
    time_dim.SetSpanCostCoefficientForAllVehicles(1)

    # Normalize windows
    ns=[]; ne=[]
    for s,e in zip(tw_start_sec, tw_end_sec):
        s2=max(0,int(s)); e2=int(horizon_sec if e is None else e)
        if e2 < s2: s2, e2 = 0, int(horizon_sec)
        ns.append(s2); ne.append(e2)

    def set_range(routing_index, lb, ub):
        if routing_index is None or routing_index<0: return
        v = time_dim.CumulVar(routing_index)
        if v is not None: v.SetRange(int(lb), int(ub))

    for node in range(n):
        if node==start_idx: continue
        if route_type=="path" and node==end_idx: continue
        set_range(manager.NodeToIndex(node), ns[node], ne[node])
    set_range(routing.Start(0), 0, horizon_sec)
    set_range(routing.End(0),   0, horizon_sec)

    for node in range(n):
        v = time_dim.CumulVar(manager.NodeToIndex(node))
        routing.AddVariableMinimizedByFinalizer(v)

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
    p.time_limit.FromSeconds(time_limit_sec)

    try:
        dbg(f"VRPTW OR-Tools: n={n}, horizon={horizon_sec}, route_type={route_type}")
        sol = routing.SolveWithParameters(p)
    except Exception as e:
        dbg(f"VRPTW OR-Tools exception: {e}")
        return [], []

    if not sol: return [], []

    idx = routing.Start(0); order, arrivals = [], []
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        order.append(node); arrivals.append(sol.Value(time_dim.CumulVar(idx)))
        idx = sol.Value(routing.NextVar(idx))
    node = manager.IndexToNode(idx)
    order.append(node); arrivals.append(sol.Value(time_dim.CumulVar(idx)))
    return order, arrivals

def vrptw_greedy(points_ll, service_sec, tw_start_sec, tw_end_sec,
                 route_type, speed_kmh, horizon_sec):
    """Fast pure-Python fallback (earliest-due-date + min added time)."""
    n = len(points_ll)
    if n == 0: return [], []
    if n == 1: return ([0,0] if route_type=="loop" else [0], [0,0] if route_type=="loop" else [0])

    D = build_distance_matrix_m(points_ll)
    T = [[meters_to_seconds(D[i][j], speed_kmh) for j in range(n)] for i in range(n)]
    ns = [max(0, int(s)) for s in tw_start_sec]
    ne = [int(horizon_sec if e is None else e) for e in tw_end_sec]

    start = int(np.argmin(ns))
    order = [start]; t = max(0, ns[start]); arrivals=[int(t)]
    t += int(service_sec[start]); cur = start
    unv = set(range(n)); unv.remove(start)

    while unv:
        best=None; best_key=None; best_arr=None
        for j in unv:
            arr = t + T[cur][j]
            if arr < ns[j]: arr = ns[j]
            if arr > ne[j]: continue
            added = arr + service_sec[j] - t
            key = (ne[j], added)
            if best is None or key < best_key:
                best=j; best_key=key; best_arr=arr
        if best is None: return [], []
        cur=best; unv.remove(best); order.append(best); arrivals.append(int(best_arr))
        t = int(best_arr + service_sec[best])
        if t > horizon_sec: return [], []

    if route_type == "loop":
        arrivals.append(int(t))  # arrival back to start (approx)
        order.append(order[0])

    return order, arrivals

# -------------------- Route metrics (simulation) ---------------------------
def route_metrics(points_ll: List[Tuple[float,float]],
                  order_idxs: List[int],
                  service_sec: List[int],
                  tw_start_sec: List[int],
                  tw_end_sec: List[int],
                  route_type: str,
                  speed_kmh: float) -> Tuple[int, int, List[int]]:
    if not order_idxs: return 0, 0, []
    D = build_distance_matrix_m(points_ll)
    ns = [max(0,int(s)) for s in tw_start_sec]
    ne = [int(e if e is not None else 10**9) for e in tw_end_sec]
    order = list(order_idxs)
    if route_type == "loop" and (len(order) < 2 or order[0] != order[-1]):
        order = order + [order[0]]
    t = 0; travel_m = 0; arrivals = []
    for k in range(len(order)):
        i = order[k]
        if t < ns[i]: t = ns[i]
        arrivals.append(int(t))
        if k == len(order)-1 and route_type=="loop" and i==order[0]:
            break
        t += int(service_sec[i])
        if k < len(order)-1:
            j = order[k+1]
            dm = D[i][j]; travel_m += dm; t += meters_to_seconds(dm, speed_kmh)
    return int(t), int(travel_m), arrivals

# -------------------- Polylines (Routes API) ------------------------------
def decode_polyline(enc: str) -> List[Tuple[float,float]]:
    pts=[]; idx=0; lat=0; lng=0
    while idx < len(enc):
        shift=result=0
        while True:
            b=ord(enc[idx])-63; idx+=1
            result|=(b&0x1f)<<shift; shift+=5
            if b<0x20: break
        dlat=~(result>>1) if (result&1) else (result>>1); lat+=dlat
        shift=result=0
        while True:
            b=ord(enc[idx])-63; idx+=1
            result|=(b&0x1f)<<shift; shift+=5
            if b<0x20: break
        dlng=~(result>>1) if (result&1) else (result>>1); lng+=dlng
        pts.append((lat/1e5, lng/1e5))
    return pts

def _compute_polyline_one(origin, destination, inters):
    if not BACKEND_KEY: return [], []
    headers = {
        "Content-Type":"application/json",
        "X-Goog-Api-Key":BACKEND_KEY,
        "X-Goog-FieldMask":"routes.polyline.encodedPolyline,routes.legs.steps.navigationInstruction",
    }
    body = {
        "origin":{"location":{"latLng":{"latitude":origin[0],"longitude":origin[1]}}},
        "destination":{"location":{"latLng":{"latitude":destination[0],"longitude":destination[1]}}},
        "travelMode":"DRIVE","routingPreference":"TRAFFIC_AWARE",
        "polylineQuality":"OVERVIEW","languageCode":"en-AU","units":"METRIC",
    }
    if inters:
        body["intermediates"]=[{"location":{"latLng":{"latitude":p[0],"longitude":p[1]}}} for p in inters]
    try:
        r = requests.post(ROUTES_COMPUTE, headers=headers, json=body, timeout=30)
        if r.status_code!=200: return [], []
        resp=r.json()
        if not resp.get("routes"): return [], []
        enc=resp["routes"][0]["polyline"]["encodedPolyline"]
        pts=decode_polyline(enc)
        steps=[]
        for leg in resp["routes"][0].get("legs",[]):
            for stp in leg.get("steps",[]):
                t=(stp.get("navigationInstruction") or {}).get("instructions")
                if t: steps.append(t)
        return pts, steps
    except Exception:
        return [], []

def compute_google_route_polyline_chunked(seq_ll: List[Tuple[float,float]]) -> Tuple[List[Tuple[float,float]], List[str]]:
    if len(seq_ll) < 2: return [], []
    MAX_WP=25
    result_pts=[]; result_steps=[]
    start=0
    while start < len(seq_ll)-1:
        end=min(len(seq_ll)-1, start+(MAX_WP-1))
        origin=seq_ll[start]; dest=seq_ll[end]
        inters=seq_ll[start+1:end]
        pts, steps=_compute_polyline_one(origin, dest, inters)
        if pts:
            if result_pts and result_pts[-1]==pts[0]:
                result_pts.extend(pts[1:])
            else:
                result_pts.extend(pts)
        result_steps.extend(steps)
        start=end
    return result_pts, result_steps

# -------------------- Map renderer (Maps JS) ------------------------------
def render_google_map(markers, polylines, center_lat, center_lng, api_key, map_version, height_px=650):
    mv = int(map_version); ts = int(time.time()); map_div_id = "map_canvas"
    html = f"""
    <!doctype html><html><head><meta charset="utf-8"/>
    <style>
      html, body {{ margin:0; padding:0; }} #{map_div_id} {{ width:100%; height:{height_px}px; }}
      .legend {{ background:#fff; padding:8px 10px; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,.3); font:12px Arial; }}
      .legend div {{ margin-bottom:6px; }}
      .err {{ position:absolute; left:8px; bottom:8px; background:rgba(255,0,0,.08); color:#900; padding:6px 8px; border-radius:6px; font:12px Arial; }}
    </style>
    <meta name="x-map-version" content="{mv}-{ts}">
    <script>
      window.onerror = function(msg) {{ const d=document.createElement('div'); d.className='err'; d.textContent='Map JS error: '+msg; document.body.appendChild(d); }};
      window.gm_authFailure = function() {{ const d=document.createElement('div'); d.className='err'; d.textContent='Google Maps auth failure: check API key, restrictions, and billing.'; document.body.appendChild(d); }};
    </script>
    <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&v=weekly&channel=st_{mv}_{ts}"></script>
    </head><body>
      <div id="{map_div_id}"></div>
      <script>
        const markersData = {json.dumps(markers)};
        const polylines   = {json.dumps(polylines)};
        function icon(color){{
          const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28">
            <circle cx="14" cy="14" r="10" fill="${{color||'#1f77b4'}}" stroke="#fff" stroke-width="2"/></svg>`;
          return {{url:'data:image/svg+xml;charset=UTF-8,'+encodeURIComponent(svg),
                  scaledSize:new google.maps.Size(28,28), anchor:new google.maps.Point(14,14)}};
        }}
        function init(){{
          const map = new google.maps.Map(document.getElementById('{map_div_id}'), {{
            center: {{lat: {center_lat}, lng: {center_lng}}}, zoom: 10, mapTypeId: 'roadmap'
          }});
          const bounds=new google.maps.LatLngBounds(); const info=new google.maps.InfoWindow(); let hasAny=false;
          (markersData||[]).forEach(m=>{{ hasAny=true; const pos=new google.maps.LatLng(m.lat,m.lng); bounds.extend(pos);
            const mk=new google.maps.Marker({{position:pos,map,title:m.title||'',icon:icon(m.color)}}); 
            mk.addListener('click',()=>{{ const qpid=m.place_id?('&query_place_id='+encodeURIComponent(m.place_id)):'';
              const url='https://www.google.com/maps/search/?api=1&query='+m.lat+','+m.lng+qpid;
              info.setContent(`<div><strong><a target="_blank" href="${{url}}">${{m.title||'(no title)'}}<\/a><\/strong><br/>Cluster: <code>${{m.cluster ?? '-'}}<\/code><\/div>`); info.open(map,mk);
            }});
          }});
          (polylines||[]).forEach(pl=>{{ if(!pl.path||!pl.path.length) return; hasAny=true;
            const r=new google.maps.Polyline({{path:pl.path,geodesic:true,strokeColor:pl.color||'#1f77b4',strokeOpacity:.9,strokeWeight:5}});
            r.setMap(map); pl.path.forEach(pt=>bounds.extend(new google.maps.LatLng(pt.lat,pt.lng)));
          }});
          if(hasAny) map.fitBounds(bounds);
          const legend=document.createElement('div'); legend.className='legend'; legend.innerHTML='<strong>Clusters & runs</strong>';
          const colors=new Map(); (markersData||[]).forEach(m=>{{ if(!colors.has(m.cluster)) colors.set(m.cluster,m.color||'#1f77b4'); }});
          Array.from(colors.entries()).sort((a,b)=>a[0]-b[0]).forEach(([cl,col])=>{{ const row=document.createElement('div');
            row.innerHTML=`<span style="display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:50%;background:${{col}}"></span>Cluster ${{cl}}`; legend.appendChild(row);
          }}); map.controls[google.maps.ControlPosition.LEFT_TOP].push(legend);
        }} try{{init();}}catch(e){{ const d=document.createElement('div'); d.className='err'; d.textContent='Map init failed: '+(e&&e.message?e.message:e); document.body.appendChild(d); }}
      </script>
      <!-- MAP_VERSION:{mv}-{ts} -->
    </body></html>
    """
    cache_bust = f"<!-- cache_bust:{mv}_{ts}_{len(markers)}_{sum(len(pl.get('path',[])) for pl in polylines)} -->"
    components.html(html + cache_bust, height=height_px, scrolling=False)

# -------------------- Sidebar --------------------------------------------
with st.sidebar:
    st.header("Options")
    st.session_state.bias_au = st.checkbox("Bias Autocomplete to Australia", value=st.session_state.bias_au)
    target_size   = st.number_input("Target points per cluster (≈)", 3, 100, 10, 1)
    random_state  = st.number_input("Random seed", 0, 9999, 42, 1)
    route_type    = st.selectbox("Route type (base runs)", ["Return to start (loop)", "End at farthest (open path)"], 0)
    route_type_key= "loop" if route_type.startswith("Return") else "path"
    st.session_state.speed_kmh = float(st.number_input("Avg speed (km/h)", 10, 120, int(st.session_state.speed_kmh), 1))

    st.divider()
    st.subheader("Time windows")
    default_service_min = st.number_input("Default service time (min)", 0, 120, 5)
    day_start           = st.time_input("Route starts at", value=dtime(8, 0))
    horizon_hours       = st.number_input("Time horizon (hours)", 1, 24, 12)
    enforce_all_tw      = st.checkbox("Always enforce TWs", value=True)
    st.caption("If OFF: only Priority stops enforce windows. If ON: all stops use their windows.")

    st.divider()
    st.subheader("Cost model (slot pricing)")
    st.session_state.cost_labour_per_h   = float(st.number_input("Labour $/hour", 0.0, 500.0, float(st.session_state.cost_labour_per_h), 1.0, format="%.2f"))
    st.session_state.cost_vehicle_per_km = float(st.number_input("Vehicle $/km", 0.0, 10.0, float(st.session_state.cost_vehicle_per_km), 0.1, format="%.2f"))
    st.session_state.cost_fixed_stop     = float(st.number_input("Fixed cost per stop", 0.0, 200.0, float(st.session_state.cost_fixed_stop), 1.0, format="%.2f"))
    st.session_state.slot_len_min        = int(st.number_input("Display slot length (min)", 5, 60, int(st.session_state.slot_len_min), 5))

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        if st.button("Force map refresh"):
            st.session_state.map_version += 1
    with colB:
        if st.button("Clear all"):
            st.session_state.chosen = []
            st.session_state.suggestions_main = []
            st.session_state.suggestions_new = []
            st.session_state.clusters = None
            st.session_state.routes = []
            st.session_state.new_candidate = None
            st.session_state.slot_options = None
            st.session_state.tw = {}
            st.session_state.map_version += 1

if not BACKEND_KEY or not MAPS_JS_KEY:
    st.info("Add **GOOGLE_BACKEND_KEY** (Places + Routes) and **GOOGLE_MAPS_JS_KEY** (Maps JS) to secrets. If you only have one key, set **GOOGLE_API_KEY** and it will be used for both.")

day_start_sec = int(day_start.hour*3600 + day_start.minute*60)
horizon_sec   = int(horizon_hours*3600)

# -------------------- 1) Search & pick -----------------------------------
st.subheader("1) Search and pick locations")
with st.form("place_search"):
    q = st.text_input("Search place", placeholder="e.g., 699 Collins St, Docklands VIC 3008")
    submitted = st.form_submit_button("Get suggestions")
if submitted:
    resp = autocomplete_new(q, st.session_state.bias_au)
    st.session_state.suggestions_main = resp["suggestions"]
    if resp["status"] != "OK":
        st.warning(f"Autocomplete status: {resp['status']}{' - ' + resp['error'] if resp['error'] else ''}")

if st.session_state.suggestions_main:
    labels = [s["label"] for s in st.session_state.suggestions_main]
    pick = st.selectbox("Suggestions", labels, index=0, key="pick_main")
    if pick:
        sel = next((s for s in st.session_state.suggestions_main if s["label"] == pick), None)
        if sel:
            det = place_details_new(sel["place_id"])
            if det.get("error"):
                st.error(det["error"])
            else:
                st.success("Selected:")
                st.write(f"**{det['address']}**")
                st.write(f"Lat/Lng: {det['lat']}, {det['lng']}")
                if st.button("➕ Add to list"):
                    st.session_state.chosen = list(st.session_state.chosen) + [det]
                    pid = det.get("place_id")
                    st.session_state.tw.setdefault(pid, {"priority": False, "start_min": 9*60, "end_min": 12*60, "service_min": default_service_min})
                    st.session_state.clusters = None
                    st.session_state.routes = []
                    st.session_state.map_version += 1

# -------------------- 2) Chosen + Time Windows ----------------------------
st.subheader("2) Chosen locations & Time Windows")
if not st.session_state.chosen:
    st.info("No locations yet.")
else:
    for i, c in enumerate(st.session_state.chosen, start=1):
        st.write(f"{i}. {c['address']}  \n({c['lat']:.6f}, {c['lng']:.6f})")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Remove last"):
            last = st.session_state.chosen[-1]
            st.session_state.chosen = st.session_state.chosen[:-1]
            st.session_state.tw.pop(last.get("place_id",""), None)
            st.session_state.clusters = None
            st.session_state.routes = []
            st.session_state.map_version += 1
    with c2:
        if st.button("Clear list"):
            st.session_state.chosen = []
            st.session_state.tw = {}
            st.session_state.clusters = None
            st.session_state.routes = []
            st.session_state.map_version += 1

    st.markdown("**Set windows (Priority enforced if global toggle OFF; otherwise all enforced).**")
    for c in st.session_state.chosen:
        pid=c.get("place_id")
        cfg=st.session_state.tw.setdefault(pid, {"priority": False, "start_min": 9*60, "end_min": 12*60, "service_min": default_service_min})
        with st.expander(f"Constraints — {c['address']}", expanded=False):
            colA,colB,colC,colD = st.columns(4)
            with colA:
                cfg["priority"] = st.checkbox("Priority", value=bool(cfg.get("priority",False)), key=f"prio_{pid}")
            with colB:
                s_min=int(cfg.get("start_min", 9*60))
                t=st.time_input("Earliest", value=dtime(s_min//60, s_min%60), key=f"s_{pid}")
                cfg["start_min"]=t.hour*60 + t.minute
            with colC:
                e_min=int(cfg.get("end_min", 12*60))
                t2=st.time_input("Latest", value=dtime(e_min//60, e_min%60), key=f"e_{pid}")
                cfg["end_min"]=t2.hour*60 + t2.minute
            with colD:
                cfg["service_min"]=st.number_input("Service (min)",0,120,int(cfg.get("service_min",default_service_min)), key=f"svc_{pid}")
            st.session_state.tw[pid]=cfg

# -------------------- 3) Cluster -----------------------------------------
st.subheader("3) Cluster — Aim for ~target points per cluster")
if st.session_state.chosen and st.button("Run clustering"):
    pts_ll = np.array([[c["lat"], c["lng"]] for c in st.session_state.chosen], float)
    n = len(pts_ll)
    k = max(1, min(n, (n + int(target_size) - 1) // int(target_size)))  # ceil(n/target)
    pts_xy = latlng_to_local_xy_m(pts_ll)
    km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=int(random_state))
    labels = km.fit_predict(pts_xy).tolist()
    mean_lat = np.deg2rad(np.mean(pts_ll[:,0]))
    m_per_deg_lat = 111_132.954
    m_per_deg_lng = 111_132.954 * math.cos(mean_lat)
    centers_xy = km.cluster_centers_
    centers_ll = np.column_stack([centers_xy[:,1]/m_per_deg_lat, centers_xy[:,0]/m_per_deg_lng]).tolist()
    sizes = {int(cl): int(np.sum(np.array(labels)==cl)) for cl in sorted(set(labels))}
    centroids = [{"cluster": i, "lat": float(centers_ll[i][0]), "lng": float(centers_ll[i][1])} for i in range(k)]
    st.session_state.clusters = {
        "labels": labels,
        "k": k,
        "target_size": int(target_size),
        "points": [{"address": c["address"], "lat": float(c["lat"]), "lng": float(c["lng"]), "place_id": c.get("place_id","")} for c in st.session_state.chosen],
        "sizes": sizes,
        "centroids": centroids,
    }
    st.session_state.routes = []
    st.session_state.map_version += 1
    st.success(f"Clustering complete: K={k}")

# -------------------- 4) Base runs (VRPTW/TSP + polylines) ----------------
st.subheader("4) Make base runs — VRPTW/TSP per cluster + Google polylines")
clusters = st.session_state.clusters
if clusters and st.button("Solve routes per cluster"):
    pts = clusters["points"]; labs = clusters["labels"]; k = clusters["k"]
    routes=[]
    for cl_id in range(k):
        idxs=[i for i, l in enumerate(labs) if int(l) == cl_id]
        if len(idxs) <= 1:
            routes.append({"cluster": cl_id, "run_id": 1, "order": idxs[:],
                           "ordered_points": [pts[i] for i in idxs],
                           "poly_points": [], "steps": [], "arrivals": []})
            continue

        cl_pts = [pts[i] for i in idxs]
        cl_ll  = [(p["lat"], p["lng"]) for p in cl_pts]

        pri_present = any((st.session_state.tw.get(p.get("place_id","")) or {}).get("priority", False) for p in cl_pts)
        enforce_windows = enforce_all_tw or pri_present

        svc,tws,twe=[],[],[]
        for p in cl_pts:
            cfg = st.session_state.tw.get(p.get("place_id","")) or {}
            svc.append(int(cfg.get("service_min", default_service_min))*60)
            if enforce_windows:
                tws.append(max(0, cfg.get("start_min", 9*60)*60))
                twe.append(max(0, cfg.get("end_min", 12*60)*60))
            else:
                tws.append(0); twe.append(horizon_sec)

        if enforce_windows:
            order_local, arrivals = vrptw_ortools(
                cl_ll, svc, tws, twe,
                "loop" if route_type_key=="loop" else "path",
                float(st.session_state.speed_kmh), int(horizon_sec), time_limit_sec=12
            )
            if not order_local:
                order_local, arrivals = vrptw_greedy(
                    cl_ll, svc, tws, twe,
                    "loop" if route_type_key=="loop" else "path",
                    float(st.session_state.speed_kmh), int(horizon_sec)
                )
                if not order_local:
                    order_local = tsp_fast_order(cl_ll, "loop" if route_type_key=="loop" else "path")
                    arrivals = []
        else:
            order_local = tsp_fast_order(cl_ll, "loop" if route_type_key=="loop" else "path")
            arrivals = []

        order_global = [idxs[i] for i in order_local]
        ordered_points = [pts[i] for i in order_global]
        seq_ll = [(p["lat"], p["lng"]) for p in ordered_points]
        poly, steps = compute_google_route_polyline_chunked(seq_ll)
        routes.append({"cluster": cl_id, "run_id": 1, "order": order_global,
                       "ordered_points": ordered_points, "poly_points": poly, "steps": steps, "arrivals": arrivals})

    st.session_state.routes = routes
    st.session_state.map_version += 1
    st.success("Base runs created.")

# -------------------- 5) Results & Map -----------------------------------
st.subheader("5) Results & Map")
routes   = st.session_state.routes
clusters = st.session_state.clusters
if clusters:
    if routes:
        for rr in routes:
            title=f"Cluster {rr['cluster']} — run {rr.get('run_id',1)} — ordered stops ({len(rr['order'])} nodes)"
            with st.expander(title):
                for pos,p in enumerate(rr["ordered_points"], start=1):
                    st.write(f"{pos}. {p['address']}  ({p['lat']:.6f}, {p['lng']:.6f})")
                if rr["steps"]:
                    st.markdown("**Driving steps (Google)**")
                    for i,sline in enumerate(rr["steps"], start=1):
                        st.write(f"{i}. {sline}")
    else:
        st.caption("Click **Solve routes per cluster** to make runs first.")

    pts = clusters["points"]; labs = list(clusters["labels"]); k = clusters["k"]
    if len(labs)<len(pts) and len(labs)>0: labs = labs + [labs[-1]]*(len(pts)-len(labs))
    elif len(labs)>len(pts): labs = labs[:len(pts)]

    palette=[
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#393b79","#637939","#8c6d31","#843c39","#7b4173",
        "#3182bd","#31a354","#756bb1","#636363","#e6550d"
    ]
    label_to_color={i:palette[i%len(palette)] for i in range(k)}
    markers=[{
        "lat":p["lat"],"lng":p["lng"],"title":p["address"],"place_id":p.get("place_id",""),
        "cluster":int(l),"color":label_to_color.get(int(l),"#1f77b4")
    } for p,l in zip(pts,labs)]

    polylines=[]
    if routes:
        for rr in routes:
            col=label_to_color.get(rr["cluster"], "#1f77b4")
            if rr["poly_points"]:
                polylines.append({"color":col,"path":[{"lat":float(a),"lng":float(b)} for (a,b) in rr["poly_points"]],"cluster":rr["cluster"],"run_id":rr.get("run_id",1)})
            elif len(rr["ordered_points"])>=2:
                path=[{"lat":float(p["lat"]),"lng":float(p["lng"])} for p in rr["ordered_points"]]
                polylines.append({"color":col,"path":path,"cluster":rr["cluster"],"run_id":rr.get("run_id",1)})

    lats=[m["lat"] for m in markers]; lngs=[m["lng"] for m in markers]
    clat=float(np.mean(lats)) if lats else -37.8136
    clng=float(np.mean(lngs)) if lngs else 144.9631

    # (Fixed f-string parens)
    st.caption(
        f"DEBUG • runs={len(routes or [])} • points={len(pts)} • markers={len(markers)} "
        f"• polylines={sum(len(pl.get('path',[])) for pl in polylines)} • map_v={st.session_state.map_version}"
    )
    render_google_map(markers, polylines, clat, clng, MAPS_JS_KEY, st.session_state.map_version, height_px=650)

# -------------------- 6) New order → PRICED TIME SLOTS --------------------
st.subheader("6) New order → choose a PRICED time slot (updates one run)")
if not st.session_state.routes:
    st.caption("Make base runs first (section 4), then add a new order here.")
else:
    with st.form("new_order_form"):
        q2 = st.text_input("Search new order", placeholder="e.g., New delivery address")
        submitted2 = st.form_submit_button("Get suggestions")
    if submitted2:
        resp2 = autocomplete_new(q2, st.session_state.bias_au)
        st.session_state.suggestions_new = resp2["suggestions"]
        if resp2["status"]!="OK":
            st.warning(f"Autocomplete status: {resp2['status']}{' - ' + resp2['error'] if resp2['error'] else ''}")

    selected2 = None
    if st.session_state.suggestions_new:
        labels2 = [s["label"] for s in st.session_state.suggestions_new]
        pick2 = st.selectbox("New-order suggestions", labels2, index=0, key="new_order_pick")
        if pick2:
            selected2 = next((s for s in st.session_state.suggestions_new if s["label"]==pick2), None)

    if selected2:
        det2 = place_details_new(selected2["place_id"])
        if det2.get("error"):
            st.error(det2["error"])
        else:
            st.session_state.new_candidate = {"address":det2["address"],"lat":float(det2["lat"]),"lng":float(det2["lng"]),"place_id":det2["place_id"]}
            pid = det2["place_id"]
            st.session_state.tw.setdefault(pid, {"priority": False,"start_min": 9*60,"end_min": 12*60,"service_min": default_service_min})
            st.success(f"Staged new order: {det2['address']}")

    cand = st.session_state.new_candidate

    def money_cost(delta_sec: float, delta_km: float) -> float:
        labour = (delta_sec/3600.0) * float(st.session_state.cost_labour_per_h)
        vehicle = max(0.0, delta_km) * float(st.session_state.cost_vehicle_per_km)
        fixed = float(st.session_state.cost_fixed_stop)
        return max(0.0, labour + vehicle + fixed)

    def build_slots_for_candidate(cand_point: Dict[str,Any]) -> List[Dict[str,Any]]:
        clusters = st.session_state.clusters
        routes   = st.session_state.routes
        pts_all  = clusters["points"]
        labs_all = clusters["labels"]

        slots=[]
        for rr in routes:
            cl_id = int(rr["cluster"])
            idxs=[i for i,l in enumerate(labs_all) if int(l)==cl_id]
            cl_pts = [pts_all[i] for i in idxs]
            cl_ll  = [(p["lat"],p["lng"]) for p in cl_pts]

            pri_present = any((st.session_state.tw.get(p.get("place_id","")) or {}).get("priority", False) for p in cl_pts+[cand_point])
            enforce_windows = enforce_all_tw or pri_present

            # TW arrays for cluster (+ candidate)
            svc=[]; tws=[]; twe=[]
            for p in cl_pts:
                cfg = st.session_state.tw.get(p.get("place_id","")) or {}
                svc.append(int(cfg.get("service_min", default_service_min))*60)
                if enforce_windows:
                    tws.append(max(0, cfg.get("start_min", 9*60)*60))
                    twe.append(max(0, cfg.get("end_min", 12*60)*60))
                else:
                    tws.append(0); twe.append(horizon_sec)

            cfg_c = st.session_state.tw.get(cand_point.get("place_id","")) or {}
            svc_c = int(cfg_c.get("service_min", default_service_min))*60
            tws_c = max(0, cfg_c.get("start_min", 9*60)*60) if enforce_windows else 0
            twe_c = max(0, cfg_c.get("end_min", 12*60)*60) if enforce_windows else horizon_sec

            # Base metrics from current run
            base_points = rr["ordered_points"]
            base_ll     = [(p["lat"],p["lng"]) for p in base_points]
            svc_base=[]; tws_base=[]; twe_base=[]
            for p in base_points:
                cfg = st.session_state.tw.get(p.get("place_id","")) or {}
                svc_base.append(int(cfg.get("service_min", default_service_min))*60)
                if enforce_windows:
                    tws_base.append(max(0, cfg.get("start_min", 9*60)*60))
                    twe_base.append(max(0, cfg.get("end_min", 12*60)*60))
                else:
                    tws_base.append(0); twe_base.append(horizon_sec)
            base_total_sec, base_travel_m, _ = route_metrics(
                base_ll, list(range(len(base_ll))),
                svc_base, tws_base, twe_base,
                "loop" if route_type_key=="loop" else "path",
                float(st.session_state.speed_kmh)
            )

            # Solve with candidate included
            cl_pts_plus = cl_pts + [cand_point]
            cl_ll_plus  = cl_ll + [(cand_point["lat"], cand_point["lng"])]
            svc_plus    = svc + [svc_c]
            tws_plus    = tws + [tws_c]
            twe_plus    = twe + [twe_c]

            order_local, arrivals = vrptw_ortools(
                cl_ll_plus, svc_plus, tws_plus, twe_plus,
                "loop" if route_type_key=="loop" else "path",
                float(st.session_state.speed_kmh), int(horizon_sec), time_limit_sec=10
            )
            if not order_local:
                # Try heuristic fallback (if infeasible with OR-Tools)
                order_local, arrivals = vrptw_greedy(
                    cl_ll_plus, svc_plus, tws_plus, twe_plus,
                    "loop" if route_type_key=="loop" else "path",
                    float(st.session_state.speed_kmh), int(horizon_sec)
                )
                if not order_local:
                    continue  # no feasible slot in this run

            # Candidate position & arrival
            cand_local_idx = len(cl_ll_plus)-1
            try:
                pos_in_sol = order_local.index(cand_local_idx)
                cand_arrival = int(arrivals[pos_in_sol])
            except ValueError:
                continue

            # New route metrics on solved order
            new_points_order = [cl_pts_plus[i] for i in order_local]
            new_ll_order     = [(p["lat"],p["lng"]) for p in new_points_order]
            svc_new=[]; tws_new=[]; twe_new=[]
            for p in new_points_order:
                cfg = (st.session_state.tw.get(p.get("place_id","")) or {})
                svc_new.append(int(cfg.get("service_min", default_service_min))*60)
                if enforce_windows:
                    tws_new.append(max(0, cfg.get("start_min", 9*60)*60))
                    twe_new.append(max(0, cfg.get("end_min", 12*60)*60))
                else:
                    tws_new.append(0); twe_new.append(horizon_sec)

            new_total_sec, new_travel_m, _ = route_metrics(
                new_ll_order, list(range(len(new_ll_order))),
                svc_new, tws_new, twe_new,
                "loop" if route_type_key=="loop" else "path",
                float(st.session_state.speed_kmh)
            )

            delta_sec = max(0, new_total_sec - base_total_sec)
            delta_km  = max(0.0, (new_travel_m - base_travel_m)/1000.0)
            price     = (delta_sec/3600.0)*float(st.session_state.cost_labour_per_h) \
                        + delta_km*float(st.session_state.cost_vehicle_per_km) \
                        + float(st.session_state.cost_fixed_stop)

            slots.append({
                "cluster": cl_id,
                "run_id": int(rr.get("run_id",1)),
                "price": float(round(price,2)),
                "delta_sec": int(delta_sec),
                "delta_km": float(round(delta_km,2)),
                "arrival_sec": int(cand_arrival),
                "end_sec": int(cand_arrival + max(60, svc_c)),
                "order_local_with_cand": order_local,
                "new_points_order": new_points_order,
            })
        return slots

    if cand:
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Generate priced time slots"):
                slots = build_slots_for_candidate(cand)
                if not slots:
                    st.warning("No feasible slots found with current time windows. Try relaxing windows or horizon.")
                else:
                    slots.sort(key=lambda x: (x["price"], x["arrival_sec"]))
                    st.session_state.slot_options = slots
                    st.success(f"Built {len(slots)} slot(s). Choose one to book.")

        slots = st.session_state.slot_options
        if slots:
            st.markdown("**Available slots (sorted by price):**")
            labels=[]
            for i, sl in enumerate(slots):
                t1 = minutes_to_hhmm(day_start_sec, sl["arrival_sec"])
                t2 = minutes_to_hhmm(day_start_sec, sl["end_sec"])
                labels.append(
                    f"{t1}–{t2}  •  ${sl['price']:.2f}  •  +{int(round(sl['delta_sec']/60.0))} min, +{sl['delta_km']:.2f} km  •  Cluster {sl['cluster']}"
                )
            pick_idx=st.selectbox("Pick a slot to book", list(range(len(labels))), format_func=lambda i: labels[i], key="slot_pick")

            if st.button("Book selected slot (updates that run only)"):
                chosen = slots[pick_idx]
                target_cluster = int(chosen["cluster"])
                chosen_run_id  = int(chosen["run_id"])

                clusters = st.session_state.clusters
                routes   = st.session_state.routes

                # add candidate to that cluster
                clusters["points"].append(cand)
                clusters["labels"].append(target_cluster)

                # recompute ONLY that cluster
                idxs = [i for i, l in enumerate(clusters["labels"]) if int(l) == target_cluster]
                cl_pts = [clusters["points"][i] for i in idxs]
                cl_ll  = [(p["lat"], p["lng"]) for p in cl_pts]

                pri_present = any((st.session_state.tw.get(p.get("place_id","")) or {}).get("priority", False) for p in cl_pts)
                enforce_windows = enforce_all_tw or pri_present

                svc,tws,twe=[],[],[]
                for p in cl_pts:
                    cfg = st.session_state.tw.get(p.get("place_id","")) or {}
                    svc.append(int(cfg.get("service_min", default_service_min))*60)
                    if enforce_windows:
                        tws.append(max(0, cfg.get("start_min", 9*60)*60))
                        twe.append(max(0, cfg.get("end_min", 12*60)*60))
                    else:
                        tws.append(0); twe.append(horizon_sec)

                order_local, arrivals = vrptw_ortools(
                    cl_ll, svc, tws, twe,
                    "loop" if route_type_key=="loop" else "path",
                    float(st.session_state.speed_kmh), int(horizon_sec), time_limit_sec=12
                )
                if not order_local:
                    order_local, arrivals = vrptw_greedy(
                        cl_ll, svc, tws, twe,
                        "loop" if route_type_key=="loop" else "path",
                        float(st.session_state.speed_kmh), int(horizon_sec)
                    )
                    if not order_local:
                        order_local = tsp_fast_order(cl_ll, "loop" if route_type_key=="loop" else "path")
                        arrivals = []

                order_global  = [idxs[i] for i in order_local]
                ordered_points= [clusters["points"][i] for i in order_global]
                seq_ll_final  = [(p["lat"], p["lng"]) for p in ordered_points]
                poly, steps   = compute_google_route_polyline_chunked(seq_ll_final)

                # replace only that run
                replaced = False
                for i, rr in enumerate(routes):
                    if int(rr.get("cluster", -1)) == target_cluster and int(rr.get("run_id",1)) == chosen_run_id:
                        routes[i] = {"cluster": target_cluster, "run_id": chosen_run_id, "order": order_global,
                                     "ordered_points": ordered_points, "poly_points": poly, "steps": steps, "arrivals": arrivals}
                        replaced = True
                        break
                if not replaced:
                    routes.append({"cluster": target_cluster, "run_id": chosen_run_id, "order": order_global,
                                   "ordered_points": ordered_points, "poly_points": poly, "steps": steps, "arrivals": arrivals})

                # update UI stats
                sizes = clusters.get("sizes", {})
                sizes[target_cluster] = len([1 for l in clusters["labels"] if int(l)==target_cluster])
                clusters["sizes"] = sizes
                lat_mean = float(np.mean([p["lat"] for p in cl_pts] or [0.0]))
                lng_mean = float(np.mean([p["lng"] for p in cl_pts] or [0.0]))
                centroids = clusters.get("centroids", [])
                if not centroids or len(centroids) < clusters["k"]:
                    centroids = [{"cluster": c, "lat": lat_mean, "lng": lng_mean} for c in range(clusters["k"])]
                centroids[target_cluster] = {"cluster": target_cluster, "lat": lat_mean, "lng": lng_mean}
                clusters["centroids"] = centroids

                st.session_state.clusters = clusters
                st.session_state.routes   = routes
                st.session_state.new_candidate = None
                st.session_state.slot_options = None
                st.session_state.map_version += 1
                st.success(f"Booked slot • Cluster {target_cluster} updated.")
