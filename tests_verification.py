#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests de vérification de Hermitte_NL_Classes par comparaison à des solutions analytiques.

Lancement :  python tests_verification.py      (ou : pytest tests_verification.py)

Chaque test affiche la valeur EF, la valeur analytique et l'écart relatif ;
il échoue si l'écart dépasse la tolérance indiquée.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas de fenêtre graphique pendant les tests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Hermitte_NL_Classes as Cl

# Données communes : poutre rectangulaire en aluminium
E, b, h, rho, L = 70e9, 0.15, 0.3, 2.6e3, 2.0
ES, EI, m = E*b*h, E*b*h**3/12, rho*b*h
ENCASTREMENT = [[1, 1, 0], [1, 2, 0], [1, 3, 0]]


def probleme_poutre(n_elem, forces=(), deplacements=ENCASTREMENT, angle=0.0, num_mult=1,
                    px_py_local=(0., 0.), p1_p2_noeuds=(0., 0.)):
    """Poutre de longueur L maillée avec n_elem éléments (numérotés num_mult*k), CL appliquées.
    px_py_local  : charge répartie constante par élément, dans le REPERE LOCAL
    p1_p2_noeuds : charge répartie (identique à tous les noeuds), dans le REPERE DE STRUCTURE"""
    nodes, conn = Cl.fonct_Maille_Poutre(L, n_elem, angle)
    for nd in nodes:
        nd.p1, nd.p2 = p1_p2_noeuds
    conn = [[num_mult*c[0], c[1], c[2]] for c in conn]
    props = [np.repeat([[ES, EI, m]], n_elem, 0), np.repeat([px_py_local], n_elem, 0)]
    pb = Cl.pbEF_poutre(nodes, conn, props)
    pb.Maillage_Assemblage()
    pb.CL_forces = [list(f) for f in forces]
    pb.CL_deplacements = [list(d) for d in deplacements]
    pb.Application_CL()
    return pb


def verifie(nom, ef, ana, tol):
    ecart = abs(ef - ana) / abs(ana)
    print(f"  {nom:<55s} EF = {ef: .6e}   ana = {ana: .6e}   écart = {ecart:.1e}")
    assert ecart < tol, f"{nom} : écart {ecart:.2e} > {tol:.0e}"


# ------------------------------------------------------------------ matrices élémentaires
def test_matrices_elementaires_symetriques():
    for (C, S) in [(1, 0), (0, 1), (np.cos(0.7), np.sin(0.7))]:
        for nom, Me in [("kel", Cl.calcul_kel(ES/0.3, EI/0.3**3, C, S, 0.3)),
                        ("kg", Cl.calcul_kg(C, S, 0.3, 1.0)),
                        ("M", Cl.calcul_M(m, C, S, 0.3))]:
            assert np.allclose(Me, Me.T, rtol=0, atol=1e-12*np.abs(Me).max()), f"{nom} non symétrique"
    print("  kel, kg, M symétriques")


def test_angle_elements_en_degres():
    """Angle des éléments calculé en degrés dans ]-180, 180], selon l'orientation noeud 1 -> noeud 2."""
    for angle, attendu in [(0, 0), (30, 30), (90, 90), (120, 120), (180, 180), (210, -150), (-90, -90), (-45, -45)]:
        pb = probleme_poutre(2, angle=angle)
        for e in pb.elem_global:
            assert abs(e.angle - attendu) < 1e-9, f"angle {e.angle} au lieu de {attendu}"
            assert abs(e.C - np.cos(np.radians(attendu))) < 1e-12 and abs(e.S - np.sin(np.radians(attendu))) < 1e-12
    print("  angles des éléments en degrés, dans ]-180, 180]")


def test_mouvements_de_corps_rigide():
    """K * (translation ou rotation rigide) = 0 pour un élément incliné."""
    th, le = 0.7, 0.3
    C, S = np.cos(th), np.sin(th)
    k = Cl.calcul_kel(ES/le, EI/le**3, C, S, le)
    x2 = np.array([le*C, le*S])
    rotation = np.array([0, 0, 1, -x2[1], x2[0], 1])
    for q in (np.array([1, 0, 0, 1, 0, 0]), np.array([0, 1, 0, 0, 1, 0]), rotation):
        assert np.abs(k @ q).max() < 1e-6*np.abs(k).max()
    print("  modes rigides : K q = 0")


# ------------------------------------------------------------------ statique
def test_console_effort_bout():
    P = -1e3
    for angle in (0.0, 60.0):  # degrés
        C, S = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        pb = probleme_poutre(4, forces=[[5, 1, -P*S], [5, 2, P*C]], angle=angle)
        pb.Resol()
        n = pb.nodes_global[-1]
        fleche = -n.u1*S + n.u2*C
        verifie(f"console : flèche PL^3/3EI (angle {angle:.0f}°)", fleche, P*L**3/(3*EI), 1e-9)
        verifie(f"console : rotation PL^2/2EI (angle {angle:.0f}°)", n.theta, P*L**2/(2*EI), 1e-9)


def test_console_charge_repartie():
    q = -2e3
    pb = probleme_poutre(4, px_py_local=(0., q))
    pb.Resol()
    verifie("console charge répartie : flèche qL^4/8EI", pb.nodes_global[-1].u2, q*L**4/(8*EI), 1e-9)
    R = pb.Calcul_Reactions()
    verifie("console charge répartie : réaction -qL", R[1], -q*L, 1e-9)
    verifie("console charge répartie : moment encastrement -qL^2/2", R[2], -q*L**2/2, 1e-9)


def test_charge_repartie_aux_noeuds():
    """Charge triangulaire (0 -> q en bout) définie aux noeuds : flèche 11 q L^4 / 120 EI."""
    q, n_elem = -3e3, 4
    nodes, conn = Cl.fonct_Maille_Poutre(L, n_elem, 0.0)
    for nd in nodes:
        nd.p2 = q * nd.x1 / L
    pb = Cl.pbEF_poutre(nodes, conn, [np.repeat([[ES, EI, m]], n_elem, 0), np.zeros((n_elem, 2))])
    pb.Maillage_Assemblage()
    pb.CL_deplacements = ENCASTREMENT
    pb.Application_CL(); pb.Resol()
    verifie("charge triangulaire aux noeuds : flèche 11qL^4/120EI", pb.nodes_global[-1].u2, 11*q*L**4/(120*EI), 1e-9)


ANGLES_CONSOLE = (0.0, 30.0, 120.0, -45.0, 210.0)  # degrés ; 210° : poutre parcourue vers x1 < 0


def resultats_console(pb):
    """Grandeurs comparées entre deux mises en données d'un même problème."""
    pb.Resol(); pb.Calcul_Def_Eff()
    return {"F_rep": pb.F_rep.copy(),
            "déplacements": pb.dofs.copy(),
            "réactions": pb.Calcul_Reactions().copy(),
            "N, T, M": np.array([e.def_eff[2:5] for e in pb.elem_global])}


def compare(nom, A, B, tol=1e-12):
    ecart = np.abs(A - B).max() / np.abs(A).max()
    print(f"  {nom:<55s} écart max relatif = {ecart:.1e}")
    assert ecart < tol, f"{nom} : écart {ecart:.2e} > {tol:.0e}"


def test_charges_reparties_noeuds_vs_element():
    """Console inclinée d'un angle theta (degrés) sous charge répartie constante (px, py) :
    cas A : (px, py) définis par élément dans le REPERE LOCAL ;
    cas B : même charge définie aux noeuds dans le REPERE DE STRUCTURE, (p1, p2) = R (px, py).
    Les deux mises en données doivent donner exactement les mêmes résultats,
    et la solution analytique de la console dans le repère local."""
    px, py, n_elem = 1.5e3, -2e3, 4
    for theta in ANGLES_CONSOLE:
        C, S = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        deg = f"{theta:.0f}°"
        A = resultats_console(probleme_poutre(n_elem, angle=theta, px_py_local=(px, py)))
        B = resultats_console(probleme_poutre(n_elem, angle=theta, p1_p2_noeuds=(C*px - S*py, S*px + C*py)))
        for cle in A:
            compare(f"angle {deg} : {cle} élément / noeuds", A[cle], B[cle])
        u1, u2 = B["déplacements"][-3:-1]
        verifie(f"angle {deg} : allongement px L^2/2ES", C*u1 + S*u2, px*L**2/(2*ES), 1e-9)
        verifie(f"angle {deg} : flèche py L^4/8EI", -S*u1 + C*u2, py*L**4/(8*EI), 1e-9)


def test_superposition_charges_element_et_noeuds():
    """La moitié de la charge par élément (repère local) + l'autre moitié aux noeuds (repère de structure)
    doit donner le même résultat que la charge complète définie par élément."""
    px, py, theta = 1.5e3, -2e3, 30.0
    C, S = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    A = resultats_console(probleme_poutre(4, angle=theta, px_py_local=(px, py)))
    B = resultats_console(probleme_poutre(4, angle=theta, px_py_local=(px/2, py/2),
                                          p1_p2_noeuds=((C*px - S*py)/2, (S*px + C*py)/2)))
    for cle in A:
        compare(f"superposition élément + noeuds : {cle}", A[cle], B[cle])


def test_deplacement_impose_et_efforts():
    """Barre encastrée-libre en traction par déplacement imposé : N = ES*ud/L.
    Le déplacement est imposé par PENALITE (10^5 * max K) : erreur relative attendue ~1e-5."""
    ud = 1e-3
    pb = probleme_poutre(3, deplacements=ENCASTREMENT + [[4, 1, ud]])
    pb.Resol(); pb.Calcul_Def_Eff()
    verifie("déplacement imposé : u en bout", pb.nodes_global[-1].u1, ud, 1e-4)
    for e in pb.elem_global:
        verifie(f"déplacement imposé : N élément {e.num_elem}", e.def_eff[2], ES*ud/L, 1e-4)


def test_efforts_internes_console():
    """Convention du code : M = EI v'' et T = -dM/dx ; pour la console M(x) = P (L - x), T = P."""
    P = -1e3
    pb = probleme_poutre(4, forces=[[5, 2, P]])
    pb.Resol(); pb.Calcul_Def_Eff()
    for e in pb.elem_global:
        x_mil = 0.5*(e.node1.x1 + e.node2.x1)
        verifie(f"console : M au milieu de l'élément {e.num_elem}", e.def_eff[4], P*(L - x_mil), 1e-9)
        verifie(f"console : T dans l'élément {e.num_elem}", e.def_eff[3], P, 1e-9)


# ------------------------------------------------------------------ modification / destruction d'éléments
def test_modif_proprietes_reactions():
    """Après modification d'un élément, les réactions restent nulles hors encastrement."""
    P = -1e3
    pb = probleme_poutre(10, forces=[[11, 2, P]])
    Cl.Modif_Proprietes_Elem(pb, 3, 2*ES, 2*EI, m)
    pb.Resol()
    R = pb.Calcul_Reactions()
    verifie("modif. propriétés : réaction encastrement", R[1], -P, 1e-9)
    assert np.abs(R[3:]).max() < 1e-6*abs(P), "réactions parasites sur les noeuds libres"
    print("  modif. propriétés : pas de réaction parasite sur les noeuds libres")


def test_detruit_elem():
    pb = probleme_poutre(3)
    Cl.Detruit_Elem(pb, 3)
    assert [e.num_elem for e in pb.elem_global] == [1, 2]
    # le noeud 4 n'est plus relié : ses lignes de K sont nulles
    assert np.abs(pb.K_global[9:12, :]).max() < 1e-6
    print("  Detruit_Elem : élément retiré, contributions annulées")


# ------------------------------------------------------------------ valeurs propres
def test_vibration_console_convergence():
    """Pulsations propres de la console : (beta_n L)^2 sqrt(EI / m L^4) ; convergence en h."""
    beta_L = [1.87510407, 4.69409113, 7.85475744]
    ecarts = []
    for n_elem in (2, 4, 8):
        pb = probleme_poutre(n_elem)
        w, _ = pb.Calcul_Modes_Propres('Vibration', 3)
        w_ana = beta_L[0]**2*np.sqrt(EI/(m*L**4))
        ecarts.append(abs(w[0] - w_ana)/w_ana)
        assert w[0] >= w_ana*(1 - 1e-12), "la masse cohérente doit donner une borne supérieure"
    print(f"  vibration : écarts 1er mode pour 2/4/8 élts = {ecarts[0]:.1e} / {ecarts[1]:.1e} / {ecarts[2]:.1e}")
    assert ecarts[1] < ecarts[0]/8 and ecarts[2] < ecarts[1]/8, "convergence en h insuffisante"
    # le 3e mode de la console (h/L = 0.15) est le 1er mode AXIAL : (pi/2) sqrt(ES/m) / L
    pb = probleme_poutre(10)
    w, _ = pb.Calcul_Modes_Propres('Vibration', 3)
    verifie("vibration console 10 élts : 1er mode de flexion", w[0], beta_L[0]**2*np.sqrt(EI/(m*L**4)), 1e-3)
    verifie("vibration console 10 élts : 2e mode de flexion", w[1], beta_L[1]**2*np.sqrt(EI/(m*L**4)), 1e-3)
    verifie("vibration console 10 élts : 1er mode axial", w[2], np.pi/2*np.sqrt(ES/m)/L, 5e-3)  # interpolation axiale linéaire : convergence plus lente


def test_flambage_console():
    for num_mult in (1, 10):  # 10 : éléments numérotés 10, 20, 30... (numérotation non consécutive)
        pb = probleme_poutre(6, forces=[[7, 1, -1.0]], num_mult=num_mult)
        pb.Resol(); pb.Calcul_Def_Eff(); pb.MAJ_Rigidite_Geom()
        lam, _ = pb.Calcul_Modes_Propres('Flambage', 1)
        verifie(f"flambage console pi^2EI/4L^2 (numéros x{num_mult})", lam[0], np.pi**2*EI/(4*L**2), 1e-4)


def flambage_console_inclinee(angle, n_elem, n_modes=3):
    """Console inclinée de angle (degrés), comprimée en bout par un effort unitaire selon son axe."""
    C, S = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    pb = probleme_poutre(n_elem, forces=[[n_elem + 1, 1, -C], [n_elem + 1, 2, -S]], angle=angle)
    pb.Resol(); pb.Calcul_Def_Eff(); pb.MAJ_Rigidite_Geom()
    lam, modes = pb.Calcul_Modes_Propres('Flambage', n_modes)
    return pb, np.asarray(lam), np.asarray(modes), C, S


def test_flambage_console_inclinee():
    """Console inclinée (15° et autres quadrants) : charges critiques d'Euler (2k-1)^2 pi^2 EI / 4L^2
    pour les 3 premiers modes, indépendantes de l'orientation ; N = -1 ; modes purement transverses ;
    convergence en h^4 de la 1re charge critique."""
    euler = [(2*k - 1)**2*np.pi**2*EI/(4*L**2) for k in (1, 2, 3)]
    tol = (1e-5, 2e-4, 1e-3)  # 10 éléments : les modes plus ondulés convergent moins vite
    for angle in (15.0, 105.0, 195.0, -75.0):
        pb, lam, modes, C, S = flambage_console_inclinee(angle, 10)
        N = np.array([e.def_eff[2] for e in pb.elem_global])
        assert np.abs(N + 1).max() < 1e-9, f"angle {angle}° : N = {N} au lieu de -1"
        for k in range(3):
            verifie(f"flambage console inclinée {angle:.0f}° : mode {k+1} (Euler)", lam[k], euler[k], tol[k])
        q = (modes[:, 0] if modes.ndim == 2 else modes[0]).reshape(-1, 3)
        axial, transverse = np.abs(q[:, 0]*C + q[:, 1]*S).max(), np.abs(-q[:, 0]*S + q[:, 1]*C).max()
        assert axial < 1e-10*transverse, f"angle {angle}° : 1er mode non transverse ({axial/transverse:.1e})"
    ecarts = [abs(flambage_console_inclinee(15.0, n, 1)[1][0] - euler[0])/euler[0] for n in (2, 4, 8)]
    print(f"  flambage console inclinée 15° : écarts mode 1 pour 2/4/8 élts = {ecarts[0]:.1e} / {ecarts[1]:.1e} / {ecarts[2]:.1e}")
    assert ecarts[1] < ecarts[0]/10 and ecarts[2] < ecarts[1]/10, "convergence en h insuffisante"


# ------------------------------------------------------------------ erreurs de mise en données
def test_erreurs_lisibles():
    nodes, conn = Cl.fonct_Maille_Poutre(L, 2, 0.0)
    pb = Cl.pbEF_poutre(nodes, conn, [np.repeat([[ES, EI, m]], 2, 0), np.zeros((2, 2))])
    pb.Maillage_Assemblage()
    pb.CL_deplacements = [[99, 1, 0]]
    try:
        pb.Application_CL()
    except ValueError as e:
        print(f"  CL sur noeud inexistant -> ValueError : {e}")
    else:
        raise AssertionError("une CL sur un noeud inexistant doit lever ValueError")
    try:
        Cl.Renvoi_index_elem(42, pb.elem_global)
    except ValueError as e:
        print(f"  élément inexistant -> ValueError : {e}")
    else:
        raise AssertionError("un élément inexistant doit lever ValueError")


def test_affichage():
    pb = probleme_poutre(4, forces=[[5, 2, -1e3]])
    pb.Resol(); pb.Calcul_Def_Eff()
    for champ in ('u', 'N', 'T', 'M'):
        pb.Affichage(champ, 'coolwarm', 1)
    matplotlib.pyplot.close('all')
    print("  affichages u, N, T, M sans erreur")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    echecs = 0
    for nom, f in tests:
        print(f"\n{nom}")
        try:
            f()
        except AssertionError as e:
            echecs += 1
            print(f"  ECHEC : {e}")
    print(f"\n{len(tests) - echecs}/{len(tests)} tests réussis")
    sys.exit(1 if echecs else 0)
