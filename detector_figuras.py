#!/usr/bin/env python3
"""
Reconocimiento de figuras como problema de busqueda.
Figuras: circulo, cuadrado, triangulo, trapecio, estrella (o "null" si ninguna coincide).

Uso:
    pip install opencv-python numpy
    python detector_figuras.py
Se abre el navegador en http://127.0.0.1:8000 con la interfaz grafica (HTML).

Formulacion de busqueda
  Estado inicial : imagen cruda, ningun rasgo extraido.
  Estado         : conjunto de rasgos extraidos hasta el momento (+ candidatos que siguen vivos).
  Acciones       : extraer un rasgo (lados, curvatura, convexidad, angulos rectos, simetrias).
  Prueba objetivo: queda un unico candidato consistente (o ninguno -> "null").
  Costo          : 1 por cada rasgo extraido/comparado.
"""
import base64
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Base de conocimiento: rasgos ideales de cada figura
# ejes = -1 significa "infinitos"
# ----------------------------------------------------------------------------
FIGURAS = {
    "circulo":   {"lados": 0,  "curvo": 1, "convexo": 1, "rectos": 0, "ejes": -1},
    "cuadrado":  {"lados": 4,  "curvo": 0, "convexo": 1, "rectos": 4, "ejes": 4},
    "triangulo": {"lados": 3,  "curvo": 0, "convexo": 1, "rectos": 0, "ejes": 3},
    "trapecio":  {"lados": 4,  "curvo": 0, "convexo": 1, "rectos": 0, "ejes": 1},
    "estrella":  {"lados": 10, "curvo": 0, "convexo": 0, "rectos": 0, "ejes": 5},
}
RASGOS = ["lados", "curvo", "convexo", "rectos", "ejes"]


# ----------------------------------------------------------------------------
# Vision: extraccion de rasgos (el "mundo" sobre el que busca el algoritmo)
# ----------------------------------------------------------------------------
def segmentar(img_bgr):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    _, bw = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Fondo = color predominante en el borde de la imagen
    borde = np.concatenate([bw[0, :], bw[-1, :], bw[:, 0], bw[:, -1]])
    if np.mean(borde) > 127:
        bw = cv2.bitwise_not(bw)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.005 * img_bgr.shape[0] * img_bgr.shape[1]:
        return None
    return c


def contar_angulos_rectos(approx):
    pts = approx.reshape(-1, 2).astype(float)
    n = len(pts)
    rectos = 0
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = a - b, c - b
        den = np.linalg.norm(v1) * np.linalg.norm(v2)
        if den == 0:
            continue
        ang = math.degrees(math.acos(np.clip(np.dot(v1, v2) / den, -1, 1)))
        if abs(ang - 90) <= 10:
            rectos += 1
    return rectos


def contar_ejes(contorno, n=360, tol=0.04):
    """Simetrias de reflexion usando la firma radial r(theta) alrededor del centroide."""
    m = cv2.moments(contorno)
    if m["m00"] == 0:
        return 0
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    p = contorno.reshape(-1, 2).astype(float)
    # densificar el contorno para no dejar huecos en la firma
    dens = []
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        k = max(2, int(np.linalg.norm(b - a)))
        for t in np.linspace(0, 1, k, endpoint=False):
            dens.append(a + t * (b - a))
    d = np.array(dens)
    ang = (np.degrees(np.arctan2(d[:, 1] - cy, d[:, 0] - cx)) % 360)
    rad = np.hypot(d[:, 0] - cx, d[:, 1] - cy)
    firma = np.full(n, np.nan)
    idx = (ang * n / 360).astype(int) % n
    for i, r in zip(idx, rad):
        if np.isnan(firma[i]) or r > firma[i]:
            firma[i] = r
    vac = np.isnan(firma)
    if vac.all():
        return 0
    xs = np.arange(n)
    firma[vac] = np.interp(xs[vac], xs[~vac], firma[~vac], period=n)
    firma = firma / firma.mean()
    pasa = np.zeros(n, bool)
    for k in range(n):
        refl = firma[(k - xs) % n]
        pasa[k] = np.mean(np.abs(firma - refl)) < tol
    if pasa.all():
        return -1  # infinitos
    # contar racimos circulares de k consecutivos que pasan 
    racimos = sum(1 for k in range(n) if pasa[k] and not pasa[k - 1])
    return racimos


class Extractor:
    """Calcula cada rasgo solo cuando la busqueda lo pide (cada llamada = 1 de costo)."""

    def __init__(self, contorno):
        self.c = contorno
        self.area = cv2.contourArea(contorno)
        self.per = cv2.arcLength(contorno, True)
        self.approx = cv2.approxPolyDP(contorno, 0.02 * self.per, True)
        self.circ = 4 * math.pi * self.area / (self.per ** 2) if self.per else 0
        self.llamadas = 0

    def medir(self, rasgo):
        self.llamadas += 1
        n = len(self.approx)
        if rasgo == "curvo":
            return int(self.circ > 0.85 and n >= 8)
        if rasgo == "lados":
            return 0 if (self.circ > 0.85 and n >= 8) else n
        if rasgo == "convexo":
            return int(cv2.isContourConvex(self.approx) or self.circ > 0.85)
        if rasgo == "rectos":
            return contar_angulos_rectos(self.approx) if n < 8 else 0
        if rasgo == "ejes":
            return contar_ejes(self.c)
        raise ValueError(rasgo)


# ----------------------------------------------------------------------------
# Busqueda
# ----------------------------------------------------------------------------
def candidatos_tras(medidos):
    return [f for f, r in FIGURAS.items() if all(r[k] == v for k, v in medidos.items())]


def busqueda_ciega(ext):
    """Busqueda en anchura sobre el espacio de rasgos: SIN informacion, extrae
    todos los rasgos en orden fijo y recien al final prueba el objetivo."""
    medidos, traza = {}, []
    for r in RASGOS:
        medidos[r] = ext.medir(r)
        traza.append({"accion": f"extraer {r}", "valor": medidos[r],
                      "candidatos": candidatos_tras(medidos), "g": len(medidos)})
    cand = candidatos_tras(medidos)
    return medidos, cand, traza, len(medidos)


def margen(c, extraidos):
    """Minimo (sobre las demas figuras) de rasgos extraidos en que c difiere de ellas.
    Margen >= 2 significa que un solo rasgo ruidoso no basta para confundir a c con otra."""
    return min(sum(FIGURAS[c][k] != FIGURAS[f][k] for k in extraidos)
               for f in FIGURAS if f != c)


def suma_margen(c, extraidos):
    return sum(sum(FIGURAS[c][k] != FIGURAS[f][k] for k in extraidos)
               for f in FIGURAS if f != c)


def busqueda_heuristica(ext, margen_min=2):
    """Busqueda voraz (best-first) guiada por heuristica.
    h(n) = (candidatos vivos - 1) + deficit de margen (margen_min - margen del unico candidato).
    En cada paso se elige la accion con mejor h esperada segun la tabla de rasgos ideales
    (peor caso de candidatos restantes y, si queda uno, mayor margen frente a las demas figuras).
    Prueba objetivo: ningun candidato (null) o un unico candidato con margen suficiente."""
    medidos, traza = {}, []
    while True:
        cand = candidatos_tras(medidos)
        pendientes = [r for r in RASGOS if r not in medidos]
        if not cand:
            break
        if len(cand) == 1 and (margen(cand[0], list(medidos)) >= margen_min or not pendientes):
            break
        if not pendientes:
            break

        def clave(r):
            peor = max(sum(1 for c in cand if FIGURAS[c][r] == v)
                       for v in {FIGURAS[c][r] for c in cand})
            ganancia = sum(suma_margen(c, list(medidos) + [r]) for c in cand)
            return (peor, -ganancia, RASGOS.index(r))

        r = min(pendientes, key=clave)
        medidos[r] = ext.medir(r)
        traza.append({"accion": f"extraer {r}", "valor": medidos[r],
                      "candidatos": candidatos_tras(medidos), "g": len(medidos)})
    return medidos, candidatos_tras(medidos), traza, len(medidos)


def reconocer(img_bgr, modo):
    c = segmentar(img_bgr)
    if c is None:
        return {"figura": "null", "motivo": "No se encontro ninguna figura", "traza": [], "costo": 0}
    ext = Extractor(c)
    algoritmo = busqueda_ciega if modo == "ciega" else busqueda_heuristica
    medidos, cand, traza, costo = algoritmo(ext)
    figura = cand[0] if len(cand) == 1 else "null"
    # mejor coincidencia aproximada (para informar cuando es null)
    if figura == "null" and medidos:
        mejor = min(FIGURAS, key=lambda f: sum(FIGURAS[f][k] != v for k, v in medidos.items()))
        motivo = f"Ninguna figura coincide exactamente. Mas parecida: {mejor}"
    else:
        motivo = "Coincidencia exacta de rasgos"
    # dibujar contorno en la imagen resultado
    salida = img_bgr.copy()
    cv2.drawContours(salida, [c], -1, (0, 160, 0), 3)
    cv2.drawContours(salida, [ext.approx], -1, (0, 0, 255), 2)
    ok, buf = cv2.imencode(".png", salida)
    return {
        "figura": figura, "motivo": motivo, "medidos": {k: int(v) for k, v in medidos.items()},
        "traza": traza, "costo": int(costo), "modo": modo,
        "imagen": "data:image/png;base64," + base64.b64encode(buf).decode(),
    }




class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, tipo):
        self.send_response(code)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/detect":
            return self._send(404, b"{}", "application/json")
        n = int(self.headers.get("Content-Length", 0))
        datos = json.loads(self.rfile.read(n))
        b64 = datos["image"].split(",", 1)[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        res = reconocer(img, datos.get("modo", "heuristica"))
        self._send(200, json.dumps(res).encode(), "application/json")


def main(puerto=8000):
    srv = HTTPServer(("127.0.0.1", puerto), Handler)
    url = f"http://127.0.0.1:{puerto}"
    print("Interfaz en", url, "(Ctrl+C para salir)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()