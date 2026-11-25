#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vues interactives (Plotly + ipywidgets) pour Hermitte_NL_Classes
================================================================

Complète ``pbEF_poutre.Affichage`` (matplotlib, figures statiques) sans le modifier :
le module ne fait que LIRE l'objet problème (noeuds, éléments, ddls, modes).

Utilisation dans un notebook ::

    import affichage_interactif as AI
    AI.vue(pb)                      # statique : choix du champ, survol des valeurs
    AI.vue_modes(pb)                # flambage / vibration : choix du mode + animation
    AI.compare([pb2, pb4, pb8], ['2 élts', '4 élts', '8 élts'], mode=1)
    AI.tableau_noeuds(pb).show()    # ddls et réactions aux noeuds
    AI.tableau_elements(pb).show()  # géométrie, charges et efforts internes des éléments
    AI.tableau_valeurs_propres([pb0, pb1], ['initial', 'modifié']).show()

Sans ipywidgets, les fonctions ``figure``, ``figure_mode`` et ``compare`` renvoient
des figures Plotly utilisables seules (``fig.show()``, ``fig.write_html(...)``).

Interpolation d'Hermite
-----------------------
La déformée est tracée avec les fonctions de forme de l'élément, et non par des segments
droits entre noeuds. Dans le repère local (t, n) de l'élément, pour xi = x/le dans [0, 1] :

    u(xi) = (1 - xi) u_1 + xi u_2                                   (linéaire)
    v(xi) = H1 v_1 + le H2 theta_1 + H3 v_2 + le H4 theta_2           (cubique)

    H1 = 1 - 3 xi^2 + 2 xi^3     H2 = xi - 2 xi^2 + xi^3
    H3 = 3 xi^2 - 2 xi^3         H4 = -xi^2 + xi^3

Les champs le long de l'élément s'en déduisent (convention du code) :
    N = ES du/dx (constant),  M = EI d2v/dx2 (linéaire),  T = -dM/dx (constant),  K = d2v/dx2.
"""

import copy

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale, qualitative

try:
    import ipywidgets as W
    from IPython.display import display
except ImportError:  # le module reste utilisable sans ipywidgets (figures seules)
    W = None
    import warnings
    warnings.warn("ipywidgets non installé : vue() et vue_modes() affichent une figure fixe "
                  "(mode 1 pour vue_modes), sans contrôles. Installer : pip install 'ipywidgets>=8'",
                  stacklevel=2)

# clé : (libellé, unité, échelle de couleur divergente centrée sur 0 ?)
CHAMPS = {
    'u':  ('Déplacement |u|', 'm', False),
    'N':  ('Effort normal N', 'N', True),
    'T':  ('Effort tranchant T', 'N', True),
    'M':  ('Moment fléchissant M', 'N.m', True),
    'K':  ('Courbure K', '1/m', True),
    'e':  ('Déformation de membrane e', '-', True),
    'ES': ('Rigidité de membrane ES', 'N', False),
    'EI': ('Rigidité de flexion EI', 'N.m²', False),
}
ECHELLE_DIV, ECHELLE_SEQ = 'RdBu_r', 'Viridis'
COUL_INIT, COUL_DEF, COUL_APPUI, COUL_EFFORT = '#9aa5b1', '#c0392b', '#1e8449', '#1f4e9c'


# =====================================================================================
#                                     CALCULS
# =====================================================================================
def ddls_noeuds(pb, q=None):
    """Tableau (nb_noeuds, 3) des ddls (u1, u2, theta) dans le repère de structure.
    q = None : solution statique pb.dofs ; sinon un vecteur de ddls (mode propre...)."""
    q = pb.dofs if q is None else q
    return np.asarray(q, dtype=float).reshape(-1, 3)


def interpole_elem(e, q, n_pts=15, hermite=True):
    """Interpolation sur un élément.

    *Retourne* : dict avec
        - xi : abscisses réduites (n_pts)
        - X  : positions initiales (n_pts, 2)
        - d  : déplacements dans le repère de structure (n_pts, 2)
        - n  : normale locale (2,) = (-S, C)
        - champs : dict des champs le long de l'élément (toujours calculés avec Hermite)
    """
    C, S, L = e.C, e.S, e.le
    a, b = q[e.node1.position], q[e.node2.position]
    # ddls dans le repère local : u = C u1 + S u2 ; v = -S u1 + C u2
    u1, v1, t1 = C*a[0] + S*a[1], -S*a[0] + C*a[1], a[2]
    u2, v2, t2 = C*b[0] + S*b[1], -S*b[0] + C*b[1], b[2]

    xi = np.linspace(0., 1., n_pts if hermite else 2)
    P1 = np.array([e.node1.x1, e.node1.x2])
    P2 = np.array([e.node2.x1, e.node2.x2])
    X = np.outer(1 - xi, P1) + np.outer(xi, P2)

    u = (1 - xi)*u1 + xi*u2
    if hermite:
        v = ((1 - 3*xi**2 + 2*xi**3)*v1 + L*(xi - 2*xi**2 + xi**3)*t1
             + (3*xi**2 - 2*xi**3)*v2 + L*(-xi**2 + xi**3)*t2)
    else:
        v = (1 - xi)*v1 + xi*v2
    d = np.column_stack([C*u - S*v, S*u + C*v])

    # dérivées des fonctions d'Hermite => courbure linéaire, d3v/dx3 constant
    d2v = ((-6 + 12*xi)*v1 + L*(-4 + 6*xi)*t1 + (6 - 12*xi)*v2 + L*(-2 + 6*xi)*t2) / L**2
    d3v = (12*v1 + 6*L*t1 - 12*v2 + 6*L*t2) / L**3
    eps = (u2 - u1) / L
    un = np.ones_like(xi)
    champs = {
        'u': np.hypot(d[:, 0], d[:, 1]),
        'e': eps*un, 'N': e.ES*eps*un,
        'K': d2v, 'M': e.EI*d2v, 'T': -e.EI*d3v*un,
        'ES': e.ES*un, 'EI': e.EI*un,
    }
    return dict(xi=xi, X=X, d=d, n=np.array([-S, C]), champs=champs)


def _taille_structure(pb):
    xy = np.array([[n.x1, n.x2] for n in pb.nodes_global])
    return max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-12)


def amplification_auto(pb, q=None, fraction=0.1):
    """Facteur tel que le plus grand déplacement nodal vaille `fraction` de la taille de la structure."""
    umax = np.abs(ddls_noeuds(pb, q)[:, :2]).max()
    return fraction*_taille_structure(pb)/umax if umax > 0 else 0.


def normalise_mode(q):
    """Mode normé : plus grande translation = 1, et positive (le signe d'un mode est arbitraire)."""
    q = np.asarray(q, dtype=float)
    Q = q.reshape(-1, 3)
    i, j = np.unravel_index(np.argmax(np.abs(Q[:, :2])), Q[:, :2].shape)
    return q / Q[i, j] if Q[i, j] != 0 else q


# =====================================================================================
#                                BRIQUES DE FIGURE
# =====================================================================================
def _lignes(segments):
    """Concatène des polylignes en une seule trace (séparateurs None)."""
    x, y = [], []
    for P in segments:
        x += list(P[:, 0]) + [None]
        y += list(P[:, 1]) + [None]
    return x, y


def _trace_maillage_initial(pb):
    segs = [np.array([[e.node1.x1, e.node1.x2], [e.node2.x1, e.node2.x2]]) for e in pb.elem_global]
    x, y = _lignes(segs)
    return go.Scatter(x=x, y=y, mode='lines', name='Maillage initial', hoverinfo='skip',
                      line=dict(color=COUL_INIT, dash='dash', width=1.5))


def _traces_appuis(pb):
    """Symboles des conditions de Dirichlet : carré = encastrement, triangle = articulation,
    triangle vide = appui simple ; étoile = déplacement imposé non nul."""
    noms_ddl = {1: 'u1', 2: 'u2', 3: 'θ'}
    par_noeud = {}
    for (num, ddl, val) in pb.CL_deplacements:
        par_noeud.setdefault(num, {})[ddl] = val
    x, y, sym, txt = [], [], [], []
    for num, ddls in par_noeud.items():
        nd = next(n for n in pb.nodes_global if n.numero == num)
        x.append(nd.x1); y.append(nd.x2)
        if any(v != 0 for v in ddls.values()):
            sym.append('star')
        elif len(ddls) == 3:
            sym.append('square')
        elif 1 in ddls and 2 in ddls:
            sym.append('triangle-up')
        else:
            sym.append('triangle-up-open')
        txt.append(f"Noeud {num} — CL Dirichlet<br>" +
                   "<br>".join(f"{noms_ddl[k]} = {v:g}" for k, v in sorted(ddls.items())))
    return [go.Scatter(x=x, y=y, mode='markers', name='Appuis', text=txt, hoverinfo='text',
                       marker=dict(symbol=sym, size=13, color=COUL_APPUI,
                                   line=dict(color=COUL_APPUI, width=2)))]


def _annotations_efforts(pb, lref, Q=None, ampli=0.):
    """Flèches des efforts concentrés (CL de Neumann), longueur proportionnelle à l'intensité,
    placées sur les noeuds déplacés de ampli*Q."""
    forces = [f for f in pb.CL_forces if f[2] != 0]
    if not forces:
        return []
    fmax = max(abs(f[2]) for f in forces if f[1] in (1, 2)) if any(f[1] in (1, 2) for f in forces) else 1.
    ann = []
    for (num, ddl, val) in forces:
        i = Cl_index(pb, num)
        nd = pb.nodes_global[i]
        x0 = nd.x1 + (ampli*Q[i, 0] if Q is not None else 0.)
        y0 = nd.x2 + (ampli*Q[i, 1] if Q is not None else 0.)
        if ddl in (1, 2):
            long = 0.18*lref*abs(val)/fmax
            dx, dy = (np.sign(val)*long, 0.) if ddl == 1 else (0., np.sign(val)*long)
            ann.append(dict(x=x0, y=y0, ax=x0 - dx, ay=y0 - dy, xref='x', yref='y',
                            axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1.3,
                            arrowwidth=2, arrowcolor=COUL_EFFORT,
                            text=f"F{ddl} = {val:.3g} N", font=dict(color=COUL_EFFORT, size=11)))
        else:
            ann.append(dict(x=x0, y=y0, xref='x', yref='y', showarrow=False, yshift=14,
                            text=f"{'↺' if val > 0 else '↻'} {val:.3g} N.m",
                            font=dict(color=COUL_EFFORT, size=12)))
    return ann


def Cl_index(pb, num):
    """Position du noeud de numéro `num` dans pb.nodes_global."""
    for i, nd in enumerate(pb.nodes_global):
        if nd.numero == num:
            return i
    raise ValueError(f"Le noeud {num} n'existe pas.")


def _hover_noeuds(pb, q):
    R = getattr(pb, 'reactions', None)
    a_reac = R is not None and np.ndim(R) == 1 and len(R) == 3*len(pb.nodes_global) and np.any(R)
    txt = []
    for i, nd in enumerate(pb.nodes_global):
        t = (f"<b>Noeud {nd.numero}</b><br>u1 = {q[i, 0]:.3e} m<br>u2 = {q[i, 1]:.3e} m"
             f"<br>θ = {q[i, 2]:.3e} rad")
        if a_reac and nd.numero in [c[0] for c in pb.CL_deplacements]:
            t += f"<br><i>Réactions</i> : R1 = {R[3*i]:.3e}, R2 = {R[3*i+1]:.3e}, Mr = {R[3*i+2]:.3e}"
        txt.append(t)
    return txt


def _mise_en_page(fig, titre, lref, hauteur=560):
    fig.update_layout(title=titre, template='plotly_white', height=hauteur,
                      margin=dict(l=20, r=20, t=60, b=20), hovermode='closest',
                      legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1))
    fig.update_xaxes(title_text='x1 [m]', zeroline=False)
    # même échelle sur x1 et x2 : la géométrie n'est pas déformée
    fig.update_yaxes(title_text='x2 [m]', scaleanchor='x', scaleratio=1, zeroline=False)
    return fig


# =====================================================================================
#                                  FIGURES
# =====================================================================================
def figure(pb, champ='u', q=None, ampli=None, hermite=True, representation='couleur',
           maillage=True, numeros=False, efforts=True, appuis=True, n_pts=15, titre=None):
    """Figure Plotly d'un champ sur la structure.

    - champ : 'u', 'N', 'T', 'M', 'K', 'e', 'ES', 'EI' (cf. CHAMPS)
    - representation : 'couleur' (champ en couleur sur la déformée) ou
      'diagramme' (diagramme du champ tracé perpendiculairement aux poutres, côté normale locale n
      pour les valeurs positives — n = t tourné de +90°, t orienté du noeud 1 vers le noeud 2)
    - ampli : amplification de la déformée (None = automatique)
    - hermite : déformée interpolée par les fonctions d'Hermite (sinon segments droits)
    """
    Q = ddls_noeuds(pb, q)
    lref = _taille_structure(pb)
    if ampli is None:
        ampli = amplification_auto(pb, q)
    lib, unite, divergent = CHAMPS[champ]
    elems = [(e, interpole_elem(e, Q, n_pts, hermite)) for e in pb.elem_global]
    # champs toujours calculés avec Hermite, même si la déformée est tracée en segments
    vals_h = {e.num_elem: interpole_elem(e, Q, n_pts, True)['champs'] for e, _ in elems}

    allv = np.concatenate([vals_h[e.num_elem][champ] for e, _ in elems])
    vmin, vmax = float(allv.min()), float(allv.max())
    if divergent:
        m = max(abs(vmin), abs(vmax)) or 1.
        cmin, cmax, echelle = -m, m, ECHELLE_DIV
    else:
        cmin, cmax, echelle = vmin, (vmax if vmax > vmin else vmin + 1.), ECHELLE_SEQ

    fig = go.Figure()
    if maillage:
        fig.add_trace(_trace_maillage_initial(pb))

    diagramme = representation == 'diagramme' and champ != 'u'
    if diagramme:
        smax = max(abs(vmin), abs(vmax)) or 1.
        s = 0.15*lref/smax
    for e, I in elems:
        ch = vals_h[e.num_elem]
        if diagramme:  # diagramme sur la géométrie initiale
            Xh = interpole_elem(e, Q, n_pts, True)
            pointe = Xh['X'] + s*np.outer(ch[champ], Xh['n'])
            poly = np.vstack([Xh['X'], pointe[::-1], Xh['X'][:1]])
            moy = ch[champ].mean()
            # np.clip : pour un champ constant (N, T), la moyenne arrondie peut dépasser cmax de 1 ulp
            coul = sample_colorscale(echelle, [float(np.clip((moy - cmin)/(cmax - cmin), 0, 1))])[0]
            fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], mode='lines', fill='toself',
                                     fillcolor=coul.replace('rgb', 'rgba').replace(')', ',0.45)'),
                                     line=dict(color=coul, width=1), hoverinfo='skip', showlegend=False))
            P, xi_h, chp = pointe, Xh['xi'], ch
        else:          # couleur sur la déformée amplifiée
            P = I['X'] + ampli*I['d']
            xi_h = I['xi']
            idx = np.linspace(0, len(ch[champ]) - 1, len(xi_h)).round().astype(int)
            chp = {k: v[idx] for k, v in ch.items()}
        custom = np.column_stack([np.full(len(xi_h), e.num_elem), xi_h, chp['N'], chp['T'],
                                  chp['M'], chp['K'], chp['u']])
        survol = ("<b>Élément %{customdata[0]:.0f}</b>  ξ = %{customdata[1]:.2f}<br>"
                  "N = %{customdata[2]:.3e} N<br>T = %{customdata[3]:.3e} N<br>"
                  "M = %{customdata[4]:.3e} N.m<br>K = %{customdata[5]:.3e} 1/m<br>"
                  "|u| = %{customdata[6]:.3e} m"
                  f"<br>ES = {e.ES:.3e}  EI = {e.EI:.3e}<extra></extra>")
        if diagramme:
            fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode='lines', showlegend=False,
                                     customdata=custom, line=dict(color=coul, width=2),
                                     hovertemplate=survol))
        else:
            # un tronçon par sous-segment : la couleur suit la variation du champ dans l'élément
            # (ex. M linéaire), le survol donne la valeur exacte au point
            vp = chp[champ]
            for j in range(len(P) - 1):
                c = sample_colorscale(echelle, [np.clip((0.5*(vp[j] + vp[j+1]) - cmin)/(cmax - cmin), 0, 1)])[0]
                fig.add_trace(go.Scatter(x=P[j:j+2, 0], y=P[j:j+2, 1], mode='lines', showlegend=False,
                                         customdata=custom[j:j+2], line=dict(color=c, width=6),
                                         hovertemplate=survol))
        if numeros:
            mid = P[len(P)//2]
            fig.add_annotation(x=mid[0], y=mid[1], text=str(e.num_elem), showarrow=False,
                               font=dict(size=10, color='#555'), bgcolor='rgba(255,255,255,0.7)')

    # barre de couleur (trace fantôme)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', showlegend=False, hoverinfo='skip',
                             marker=dict(colorscale=echelle, cmin=cmin, cmax=cmax, color=[cmin],
                                         showscale=True,
                                         colorbar=dict(title=f"{champ} [{unite}]", exponentformat='e'))))

    # noeuds (sur la déformée, ou sur la géométrie initiale pour un diagramme)
    a = 0. if diagramme else ampli
    xn = np.array([n.x1 for n in pb.nodes_global]) + a*Q[:, 0]
    yn = np.array([n.x2 for n in pb.nodes_global]) + a*Q[:, 1]
    fig.add_trace(go.Scatter(x=xn, y=yn, mode='markers+text' if numeros else 'markers', name='Noeuds',
                             text=[str(n.numero) for n in pb.nodes_global] if numeros else None,
                             textposition='bottom right', textfont=dict(size=9, color=COUL_DEF),
                             hovertext=_hover_noeuds(pb, Q), hoverinfo='text',
                             marker=dict(size=6, color='white', line=dict(color='#333', width=1.2))))
    if appuis and pb.CL_deplacements:
        for t in _traces_appuis(pb):
            fig.add_trace(t)
    if efforts:  # flèches appliquées aux noeuds dans la configuration tracée
        for an in _annotations_efforts(pb, lref, Q, a):
            fig.add_annotation(**an)

    if titre is None:
        titre = f"{lib} — min {vmin:.3e}, max {vmax:.3e} {unite}"
        titre += (" (diagramme, valeurs > 0 côté normale locale n)" if diagramme
                  else f"  ·  déformée ×{ampli:.3g}" + ("" if hermite else " (segments)"))
    return _mise_en_page(fig, titre, lref)


def _titre_mode(pb, k):
    vp = pb.vp_min[k - 1]
    if pb.vibration:
        return f"Mode {k} — pulsation ω = {vp:.4e} rad/s  (f = {vp/(2*np.pi):.4e} Hz)"
    return f"Mode {k} — charge critique λ = {vp:.4e} × chargement de référence"


def figure_mode(pb, k=1, hermite=True, maillage=True, numeros=False, appuis=True,
                n_images=24, ampli=0.15, n_pts=15):
    """Mode propre k (1 = premier) de pb.mode_min, animé (bouton ▶).

    Le mode est normé (plus grande translation = 1) puis amplifié à `ampli` × taille de la structure."""
    if len(pb.vp_min) == 0:
        raise ValueError("Pas de modes : appeler d'abord pb.Calcul_Modes_Propres('Vibration' ou 'Flambage', n)")
    q = normalise_mode(pb.mode_min[:, k - 1])
    Q = ddls_noeuds(pb, q)
    lref = _taille_structure(pb)
    a0 = ampli*lref
    elems = [interpole_elem(e, Q, n_pts, hermite) for e in pb.elem_global]
    X0 = np.array([[n.x1, n.x2] for n in pb.nodes_global])

    def traces_def(c):
        x, y = _lignes([I['X'] + c*a0*I['d'] for I in elems])
        tr_def = go.Scatter(x=x, y=y, mode='lines', name='Mode', hoverinfo='skip',
                            line=dict(color=COUL_DEF, width=3))
        P = X0 + c*a0*Q[:, :2]
        tr_nd = go.Scatter(x=P[:, 0], y=P[:, 1], mode='markers+text' if numeros else 'markers',
                           name='Noeuds', hovertext=_hover_noeuds(pb, Q), hoverinfo='text',
                           text=[str(n.numero) for n in pb.nodes_global] if numeros else None,
                           textposition='bottom right', textfont=dict(size=9),
                           marker=dict(size=6, color='white', line=dict(color='#333', width=1.2)))
        return [tr_def, tr_nd]

    fig = go.Figure()
    if maillage:
        fig.add_trace(_trace_maillage_initial(pb))
    i0 = len(fig.data)
    for t in traces_def(1.):
        fig.add_trace(t)
    if appuis and pb.CL_deplacements:
        for t in _traces_appuis(pb):
            fig.add_trace(t)

    # animation : amplitude cos(2 pi j / n)
    coefs = np.cos(2*np.pi*np.arange(n_images)/n_images)
    fig.frames = [go.Frame(data=traces_def(c), traces=[i0, i0 + 1], name=str(j))
                  for j, c in enumerate(coefs)]
    # cadre fixe pendant l'animation
    tout = np.vstack([X0 + s*a0*Q[:, :2] for s in (-1, 1)] + [X0])
    marge = 0.1*lref
    fig.update_xaxes(range=[tout[:, 0].min() - marge, tout[:, 0].max() + marge])
    fig.update_yaxes(range=[tout[:, 1].min() - marge, tout[:, 1].max() + marge])
    fig.update_layout(updatemenus=[dict(
        type='buttons', direction='left', x=0, y=-0.14, xanchor='left', yanchor='top',
        buttons=[dict(label='▶ Animer', method='animate',
                      args=[None, dict(frame=dict(duration=60, redraw=False), transition=dict(duration=0),
                                       fromcurrent=True, mode='immediate')]),
                 dict(label='⏸ Pause', method='animate',
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])])
    titre = _titre_mode(pb, k) + ("" if hermite else "  (segments)")
    _mise_en_page(fig, titre, lref)
    fig.update_layout(margin=dict(b=80))  # place pour les boutons d'animation
    return fig


def compare(liste_pb, noms=None, mode=None, hermite=True, ampli=None, n_pts=15, titre=None):
    """Superpose les déformées (mode=None) ou le mode `mode` de plusieurs problèmes,
    par exemple plusieurs maillages pour illustrer la convergence en h."""
    noms = noms or [f"Problème {i+1}" for i in range(len(liste_pb))]
    lref = _taille_structure(liste_pb[0])
    if mode is None and ampli is None:
        ampli = amplification_auto(liste_pb[0])
    elif ampli is None:
        ampli = 0.15*lref
    fig = go.Figure()
    fig.add_trace(_trace_maillage_initial(liste_pb[0]))
    lignes_titre = []
    for i, (pb, nom) in enumerate(zip(liste_pb, noms)):
        if mode is None:
            Q = ddls_noeuds(pb)
        else:
            Q = ddls_noeuds(pb, normalise_mode(pb.mode_min[:, mode - 1]))
            lignes_titre.append(f"{nom} : {pb.vp_min[mode - 1]:.4e}")
        coul = qualitative.Plotly[i % len(qualitative.Plotly)]
        x, y = _lignes([I['X'] + ampli*I['d'] for I in (interpole_elem(e, Q, n_pts, hermite)
                                                         for e in pb.elem_global)])
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=nom, line=dict(color=coul, width=2.5),
                                 hoverinfo='skip'))
        X = np.array([[n.x1, n.x2] for n in pb.nodes_global]) + ampli*Q[:, :2]
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False,
                                 marker=dict(size=7, color=coul), hovertext=_hover_noeuds(pb, Q),
                                 hoverinfo='text'))
    if titre is None:
        quoi = "Déformées" if mode is None else f"Mode {mode}"
        titre = f"{quoi} ({'Hermite' if hermite else 'segments'}, ×{ampli:.3g})"
        if lignes_titre:
            titre += "<br><sup>" + "  ·  ".join(lignes_titre) + "</sup>"
    return _mise_en_page(fig, titre, lref)


# =====================================================================================
#                                  TABLEAUX
# =====================================================================================
def _tableau(entetes, colonnes, titre, formats):
    """Tableau Plotly (défilement intégré au-delà d'une vingtaine de lignes)."""
    n = len(colonnes[0]) if colonnes else 0
    cellules = [[(f if isinstance(v, (int, float, np.floating, np.integer)) else '{}').format(v)
                 for v in col] for col, f in zip(colonnes, formats)]
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{e}</b>" for e in entetes], fill_color='#dfe6ee', align='center'),
        cells=dict(values=cellules, align='right', height=24,
                   fill_color=[['white', '#f5f7fa']*(n//2 + 1)])))
    fig.update_layout(title=dict(text=titre, x=0.5), margin=dict(l=10, r=10, t=50, b=10),
                      height=min(110 + 26*n, 620))
    return fig


def tableau_noeuds(pb, titre="Noeuds : déplacements et réactions (repère de structure)"):
    """Tableau des noeuds : coordonnées, ddls (u1, u2, theta) et réactions (R1, R2, Mr).
    Les réactions sont celles de pb.reactions (nulles tant que pb.Calcul_Reactions() n'a pas été appelé)."""
    q = ddls_noeuds(pb)
    R = np.asarray(pb.reactions, dtype=float).reshape(-1, 3) if len(pb.reactions) else np.zeros_like(q)
    cols = [[n.numero for n in pb.nodes_global], [n.x1 for n in pb.nodes_global], [n.x2 for n in pb.nodes_global],
            q[:, 0], q[:, 1], q[:, 2], R[:, 0], R[:, 1], R[:, 2]]
    return _tableau(['Noeud', 'x1', 'x2', 'u1', 'u2', 'θ', 'R1', 'R2', 'Mr'], cols, titre,
                    ['{:d}', '{:.4g}', '{:.4g}'] + ['{:.4e}']*6)


def tableau_elements(pb, titre="Éléments : géométrie, charges réparties (repère local) et efforts internes"):
    """Tableau des éléments : noeuds, angle (degrés), longueur, (px, py) locaux, N, T, M au milieu de l'élément.
    Les efforts sont ceux de pb.Calcul_Def_Eff() (convention du code : M = EI v'', T = -dM/dx)."""
    E = pb.elem_global
    cols = [[e.num_elem for e in E], [f"{e.node1.numero} → {e.node2.numero}" for e in E],
            [e.angle for e in E], [e.le for e in E], [e.px for e in E], [e.py for e in E],
            [e.def_eff[2] for e in E], [e.def_eff[3] for e in E], [e.def_eff[4] for e in E]]
    return _tableau(['Élément', 'Noeuds', 'Angle (°)', 'Longueur', 'px', 'py', 'N', 'T', 'M (milieu)'], cols, titre,
                    ['{:d}', '{}', '{:.2f}', '{:.4g}', '{:.3g}', '{:.3g}', '{:.4e}', '{:.4e}', '{:.4e}'])


def tableau_valeurs_propres(liste_pb, noms=None, reference=None, nom_reference='Référence', titre=None):
    """Valeurs propres (pulsations en rad/s ou multiplicateurs de flambage) de un ou plusieurs problèmes,
    mode par mode. reference : liste de valeurs (analytiques...) ajoutée en dernière colonne, avec
    l'écart relatif du premier problème ; à n'utiliser que si elle correspond au même problème."""
    noms = noms or [f"Problème {i+1}" for i in range(len(liste_pb))]
    vps = [np.asarray(p.vp_min, dtype=float) for p in liste_pb]
    n = max(len(v) for v in vps)
    col = lambda v: [v[k] if k < len(v) else float('nan') for k in range(n)]
    entetes, cols, fmts = ['Mode'], [list(range(1, n + 1))], ['{:d}']
    for nom, v in zip(noms, vps):
        entetes.append(nom); cols.append(col(v)); fmts.append('{:.5e}')
    if len(vps) == 2:
        entetes.append(f"{noms[1]} / {noms[0]}")
        cols.append([b/a if a else float('nan') for a, b in zip(col(vps[0]), col(vps[1]))]); fmts.append('{:.5g}')
    if reference is not None:
        ref = col(np.asarray(reference, dtype=float))
        entetes += [nom_reference, f"Écart {noms[0]}"]
        cols += [ref, [abs(a - r)/abs(r) if r else float('nan') for a, r in zip(col(vps[0]), ref)]]
        fmts += ['{:.5e}', '{:.2e}']
    if titre is None:
        titre = "Pulsations propres (rad/s)" if liste_pb[0].vibration else "Charges critiques (multiplicateurs du chargement)"
    return _tableau(entetes, cols, titre, fmts)


# =====================================================================================
#                           VUES INTERACTIVES (ipywidgets)
# =====================================================================================
def _sortie(construire, controles, lignes):
    """Affiche les contrôles (``lignes`` : liste de listes de widgets) + la figure,
    reconstruite à chaque changement.

    Le widget Output est affiché AVANT le premier tracé : sinon (VS Code notamment)
    la première figure n'est pas capturée et sort sous la cellule, hors du widget."""
    out = W.Output()

    def maj(*_):
        with out:
            out.clear_output(wait=True)
            try:
                construire().show()
            except Exception as err:  # message lisible au lieu d'une trace dans le widget
                print(f"Erreur : {err}")

    for c in controles:
        c.observe(maj, names='value')
    display(W.VBox([W.HBox(l) for l in lignes] + [out]))
    maj()


def vue(pb, champ='u', ampli=None, representation='couleur'):
    """Vue interactive d'un calcul statique : champ, représentation, amplification, options.

    - ampli : amplification initiale de la déformée (None : automatique, plus grand déplacement
      = 10 % de la taille de la structure) ; réglable ensuite avec le curseur.
    - representation : 'couleur' (champ en couleur sur la déformée) ou 'diagramme' (N, T, M).

    La vue travaille sur une COPIE de pb faite à l'appel : la suite du notebook peut modifier pb
    (propriétés, nouveau calcul) sans changer ce qu'affichent les contrôles."""
    pb = copy.deepcopy(pb)
    if W is None:
        return figure(pb, champ, ampli=ampli, representation=representation).show()
    a0 = ampli or amplification_auto(pb) or 1.
    w_champ = W.Dropdown(options=[(v[0], k) for k, v in CHAMPS.items()], value=champ,
                         description='Champ')
    w_rep = W.ToggleButtons(options=[('Couleur sur déformée', 'couleur'), ('Diagramme', 'diagramme')],
                            value=representation, description='Vue')
    w_amp = W.FloatLogSlider(value=a0, base=10, min=np.log10(a0) - 2, max=np.log10(a0) + 1,
                             step=0.05, description='Amplif.', readout_format='.3g',
                             continuous_update=False, layout=W.Layout(width='420px'))
    w_herm = W.Checkbox(value=True, description="Interpolation d'Hermite", indent=False)
    w_mail = W.Checkbox(value=True, description='Maillage initial', indent=False)
    w_num = W.Checkbox(value=False, description='Numéros', indent=False)
    w_eff = W.Checkbox(value=True, description='Efforts et appuis', indent=False)
    ctrl = [w_champ, w_rep, w_amp, w_herm, w_mail, w_num, w_eff]
    _sortie(lambda: figure(pb, w_champ.value, ampli=w_amp.value, hermite=w_herm.value,
                           representation=w_rep.value, maillage=w_mail.value,
                           numeros=w_num.value, efforts=w_eff.value, appuis=w_eff.value), ctrl,
            [[w_champ, w_rep], [w_amp, w_herm], [w_mail, w_num, w_eff]])


def vue_modes(pb, mode=1):
    """Vue interactive des modes (flambage ou vibration) : choix du mode, animation dans la figure.

    mode : rang du mode affiché au départ (1 = premier), comme 'vp' dans pb.Affichage('X', ..., vp).

    La vue travaille sur une COPIE de pb faite à l'appel (cf. vue)."""
    pb = copy.deepcopy(pb)
    n = len(pb.vp_min)
    if n == 0:
        raise ValueError("Pas de modes : appeler d'abord pb.Calcul_Modes_Propres('Vibration' ou 'Flambage', n)")
    if not 1 <= mode <= n:
        raise ValueError(f"mode = {mode} : seuls les modes 1 à {n} ont été calculés")
    if W is None:
        return figure_mode(pb, mode).show()
    unite = 'rad/s' if pb.vibration else '×F'
    w_mode = W.Dropdown(options=[(f"Mode {k} : {pb.vp_min[k-1]:.3e} {unite}", k) for k in range(1, n + 1)],
                        value=mode, description='Mode', layout=W.Layout(width='340px'))
    w_amp = W.FloatSlider(value=0.15, min=0.02, max=0.5, step=0.01, description='Amplitude',
                          readout_format='.2f', continuous_update=False)
    w_herm = W.Checkbox(value=True, description="Interpolation d'Hermite", indent=False)
    w_num = W.Checkbox(value=False, description='Numéros', indent=False)
    ctrl = [w_mode, w_amp, w_herm, w_num]
    _sortie(lambda: figure_mode(pb, w_mode.value, hermite=w_herm.value, numeros=w_num.value,
                                ampli=w_amp.value), ctrl,
            [[w_mode, w_amp], [w_herm, w_num]])
