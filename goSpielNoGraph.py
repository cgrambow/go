#!/usr/kai/anaconda3/python
# -*- coding: utf-8 -*-

# Nicht-Graphisches Go 9*9 Spiel: für NN
# V1: setzt auf auf goSpiel V4
# V2: inRedGruppen in fhUpdate
# V3: testing
# V4: berechneFhGr, auszählen implementiert
# V5: toten Stein schlagen im auszählen
# V6: fehler im auszählen: and not in neueLeereFelder, beenden mit zugNr 81, ko bei vonBeginn=False
# V7: tote schlagen mit 2 fh bei auszählen
# V8: ZUG_MAX als class par, schlageStein: fh=0, doppelt reihe var in fhUpdate
# V9: sgfWrite
#

import os, copy, gameGo

KOMI = 6.5

class PlayGo:
    """
    Play one single game or part of
    :param b7: Startposition incl History als Board7
                Board7 = Board * 7 ist NN Input
    """
    col = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j')

    def __init__(self, b7, vonBeginn=True, zugMax=81):
        self.zugMax = max(4, zugMax)
        self.gepasst = False
        self.zugNr = 0
        self.anzPass = 0    # for debug
        self.spielBeendet = False
        self.toteSchlagen = False
        self.pktSchwarz, self.pktWeiss = 0, 0
        self.gewinner = 9                   # 9: unbestimmt, 0 unentschieden , sonst 1 | -1
        self.farbeActual = 1-2*b7[6][0][0]  # 1 | -1, Farbe Actual ist vom letzten Zug
                                            #           und wird erst in SetzZug weitergeschrieben
        self.koGefahr = False              # wird True, wenn genau 1 Stein geschlagen wird
        self.koStein = [-1, -1]
        self.koPerLastMove = vonBeginn  # Ko-Überprüfung mittels letzten Zügen, oder Stellung = b1
        self.gefangenS, self.gefangenW = 0, 0   # zählt Anzahl Gefangene
        # instance variable für Go Board, sowie letzte 2 Boards:
            # 9*9 Matrix mit Besetzung:
            # 0,1,-1 für leer,schwarz,weiß (0,0) ist oben links
            # fh: Anzahl Freiheiten
            # gr: verbundene Gruppe von gleichfarbigen Steinen, mind. 2
        self.b  = [[0] * 9 for i in range(9)]
        self.b1 = [[0] * 9 for i in range(9)]
        self.b2 = [[0] * 9 for i in range(9)]
        b, b1, b2 = gameGo.b7To3(b7)
        for i in range(9):
            for j in range(9):
                self.b[i][j]  = {'farbe': b[i][j],  'fh': 0, 'gr':[]}
                self.b1[i][j] = {'farbe': b1[i][j], 'fh': 0, 'gr':[]}
                self.b2[i][j] = {'farbe': b2[i][j], 'fh': 0, 'gr':[]}
        if not vonBeginn:
            self.berechneFhGr()

    def b12Farbe(self):
        b  = [[0] * 9 for i in range(9)]
        b1 = [[0] * 9 for i in range(9)]
        b2 = [[0] * 9 for i in range(9)]
        for row in range(9):
            for col in range(9):
                b[row][col]  = self.b[row][col]['farbe']
                b1[row][col] = self.b1[row][col]['farbe']
                b2[row][col] = self.b2[row][col]['farbe']
        return b, b1, b2

    def printB(self, b1=False, b2=False):
        board = [[0] * 9 for i in range(9)]
        for row in range(9):
            for col in range(9):
                if b1:
                    board[row][col] = self.b1[row][col]['farbe']
                elif b2:
                    board[row][col] = self.b2[row][col]['farbe']
                else:
                    board[row][col] = self.b[row][col]['farbe']

        gameGo.printBrett(board)
        print('')

    def sgfWrite(self, zuege, values):
        # Writes sgfFile including values in root
        # parameter zuege: list of actions
        # parameter values: list of strings 0.xx
        os.chdir('/Users/kai/Documents/Python/ML/go9x9test')
        i = 1
        while os.path.isfile('go' + '-' + str(i) + '.sgf'):
                i += 1
        file = open('go' + '-' + str(i) + '.sgf', 'w')
        if self.gewinner == 1:
            file.write('(;RE[B+0]\n')
        elif self.gewinner == -1:
            file.write('(;RE[W+0]\n')

        file.write('EV[')
        for i in range(len(values)):
            file.write(values[i])
            if i < len(values)-1:
                file.write(',')
        file.write(']\n\n')

        if self.zugNr > self.zugMax:
            zuege.append(81)
            zuege.append(81)

        for i in range(len(zuege)):
            if i%2 == 0:
                farbe = 'B'
            else:
                farbe = 'W'
            if zuege[i] == 81:  # pass
                file.write(';' + farbe + '[]')
            else:
                file.write(';' + farbe + '[' + PlayGo.col[zuege[i]%9] + PlayGo.col[zuege[i]//9] + ']')

        file.write('\n)')
        file.close()

    def auszaehlen(self):
        # return: Punkte für schwarz, weiß
        def schlage1fh():
            # Steine, Gruppen mit nur 1 fh schlagen
            for reihe in range(9):
                for spalte in range(9):
                    if self.b[reihe][spalte]['farbe'] == 0:
                        for nb in range(4):  # Nachbarn oben, rechts, unten, links
                            nbReihe = reihe + nb - 1
                            if nb == 3:
                                nbReihe = reihe
                            nbSpalte = spalte - nb + 2
                            if nb == 0:
                                nbSpalte = spalte
                            if nbReihe in range(9) and nbSpalte in range(9):
                                if self.b[nbReihe][nbSpalte]['fh'] == 1:
                                    if self.farbeActual != self.b[nbReihe][nbSpalte]['farbe']:
                                        self.setzZug(81) # pass, andere Farbe schlägt tote Steine
                                    self.setzZug(9*reihe+spalte)
                                    break
        def schlage2fh():
            # Steine, Gruppen mit 2 fh: eine fh besetzen, aber nur wenn selbst dann nicht nur 1 fh
            #   Gegner besetzt danach die andere Freiheit
            # return T/F je nachdem ob 2 fh ausgeführt wurden
            ret = False
            # Steine, Gruppen mit 2 fh: eine fh besetzen, aber nur wenn selbst dann nicht nur 1 fh
            #   Gegner besetzt danach die andere Freiheit
            for reihe in range(9):
                for spalte in range(9):
                    if self.b[reihe][spalte]['farbe'] == 0:
                        for nb in range(4):  # Nachbarn oben, rechts, unten, links
                            nbReihe = reihe + nb - 1
                            if nb == 3:
                                nbReihe = reihe
                            nbSpalte = spalte - nb + 2
                            if nb == 0:
                                nbSpalte = spalte
                            if nbReihe in range(9) and nbSpalte in range(9):
                                if self.b[nbReihe][nbSpalte]['fh'] == 2:
                                    if self.farbeActual != self.b[nbReihe][nbSpalte]['farbe']:
                                        self.setzZug(81) # pass, andere Farbe besetzt fh
                                    b2Safe = copy.deepcopy(self.b2)
                                    gefangenS, gefangenW = self.gefangenS, self.gefangenW
                                    if not self.setzZug(9*reihe+spalte):
                                        break   # illegaler Zug
                                    if self.b[reihe][spalte]['fh'] == 1:
                                        # Fh nach Update 1 --> Zug rückgängig machen
                                        self.b = copy.deepcopy(self.b1)
                                        self.b1 = copy.deepcopy(self.b2)
                                        self.b2 = copy.deepcopy(b2Safe)
                                        self.farbeActual = - self.farbeActual
                                        self.gefangenS, self.gefangenW = gefangenS, gefangenW
                                    else:
                                        if self.b[nbReihe][nbSpalte]['gr'] == []:
                                            # finde andere Freiheit des einzelnen Steins
                                            for nb2 in range(4):  # Nachbarn oben, rechts, unten, links
                                                nb2Reihe = nbReihe + nb2 - 1
                                                if nb2 == 3:
                                                    nb2Reihe = nbReihe
                                                nb2Spalte = nbSpalte - nb2 + 2
                                                if nb2 == 0:
                                                    nb2Spalte = nbSpalte
                                                if nb2Reihe in range(9) and nb2Spalte in range(9):
                                                    if self.b[nb2Reihe][nb2Spalte]['farbe'] == 0:
                                                        fhStein = (nb2Reihe, nb2Spalte)
                                                        found = True
                                                        break
                                        else:
                                            _, fhStein = self.fhZaehlen(self.b[nbReihe][nbSpalte]['gr'])
                                        self.setzZug(9*fhStein[0]+fhStein[1])
                                        ret = True
                                    break
            schlage1fh()
            return ret

        pktSchwarz, pktWeiss = 0, 0
        if self.spielBeendet:
            self.toteSchlagen = True
            # teils die toten Steine rausschlagen:
            schlage1fh()
            while schlage2fh():
                None

        # Tromp Taylor counting
        for row in range(9):
            for col in range(9):
                farbe = self.b[row][col]['farbe']
                if farbe == 1:
                    pktSchwarz += 1
                elif farbe == -1:
                    pktWeiss += 1
                else:
                    # only reach black or white from here (--> Point) or both
                    reachSchwarz, reachWeiss = False, False
                    neueLeereFelder = [[row, col]]
                    bearbeiteteLeereFelder = []
                    while neueLeereFelder != []:
                        feld = neueLeereFelder[0]
                        reihe, spalte = feld[0], feld[1]
                        for nb in range(4):  # Nachbarn oben, rechts, unten, links
                            nbReihe = reihe + nb - 1
                            if nb == 3:
                                nbReihe = reihe
                            nbSpalte = spalte - nb + 2
                            if nb == 0:
                                nbSpalte = spalte
                            if nbReihe in range(9) and nbSpalte in range(9):
                                farbeAngrenzend = self.b[nbReihe][nbSpalte]['farbe']
                                if farbeAngrenzend == 0:
                                    if [nbReihe, nbSpalte] not in bearbeiteteLeereFelder\
                                        and [nbReihe, nbSpalte] not in neueLeereFelder:
                                        neueLeereFelder.append([nbReihe, nbSpalte])
                                elif farbeAngrenzend == 1:
                                    reachSchwarz = True
                                else:
                                    reachWeiss = True
                        if reachSchwarz and reachWeiss:
                            break
                        bearbeiteteLeereFelder.append([reihe, spalte])
                        del neueLeereFelder[0]
                    if reachSchwarz and not reachWeiss:
                        pktSchwarz += 1
                    if not reachSchwarz and reachWeiss:
                        pktWeiss += 1
        return pktSchwarz, pktWeiss+KOMI

    def berechneFhGr(self):
        # für angrenzende Felder:
        #   falls leer: fh erhöhen
        #   falls selbe Farbe: Gruppe erweitern
        for reihe in range(9):
            for spalte in range(9):
                farbe = self.b[reihe][spalte]['farbe']
                if farbe == 0:
                    continue
                erweiterGruppe = False
                gruppe = [[reihe, spalte]]
                for nb in range(4):  # Nachbarn oben, rechts, unten, links
                    nbReihe = reihe + nb - 1
                    if nb == 3:
                        nbReihe = reihe
                    nbSpalte = spalte - nb + 2
                    if nb == 0:
                        nbSpalte = spalte
                    if nbReihe in range(9) and nbSpalte in range(9):
                        if self.b[nbReihe][nbSpalte]['farbe'] == 0:
                            self.b[reihe][spalte]['fh'] += 1
                        elif self.b[nbReihe][nbSpalte]['farbe'] == farbe:
                            # erweitere Gruppe
                            erweiterGruppe = True
                            if self.b[nbReihe][nbSpalte]['gr'] == []:
                                gruppe.append([nbReihe, nbSpalte])
                            else:
                                for g in self.b[nbReihe][nbSpalte]['gr']:
                                    if not g in gruppe:
                                        gruppe.append(g)
                if erweiterGruppe:
                    self.fhZaehlen(gruppe)
                    # update Gruppe
                    for gruppenStein in gruppe:
                        self.b[gruppenStein[0]][gruppenStein[1]]['gr'] = gruppe

    def schlageStein(self, reihe, spalte):
        self.b[reihe][spalte]['farbe'] = 0
        self.b[reihe][spalte]['fh'] = 0
        self.b[reihe][spalte]['gr'] = []
        # angrenzende Steine anderer Farbe bekommen eine fh mehr, bzw wenn Gruppe, dann fh neu zählen
        for nb in range(4):   # Nachbarn oben, rechts, unten, links
            nbReihe = reihe + nb - 1
            if nb == 3:
                nbReihe = reihe
            nbSpalte = spalte - nb + 2
            if nb == 0:
                nbSpalte = spalte
            if nbReihe in range(9) and nbSpalte in range(9):
                if self.b[nbReihe][nbSpalte]['farbe'] == self.farbeActual:
                    if self.b[nbReihe][nbSpalte]['gr'] == []:
                        self.b[nbReihe][nbSpalte]['fh'] += 1
                    else:
                        self.fhZaehlen(self.b[nbReihe][nbSpalte]['gr'])
        if self.farbeActual == 1:
            self.gefangenS += 1
        else:
            self.gefangenW += 1

    def fhZaehlen(self, gruppe):
        # zähle fh im Rechteck um das Gruppenfeld
        # setze ermitteltes fh für alle Gruppensteine
        # return fh und Stein der letzt gefundenen fh
        stein = (0, 0)
        fh, reiheMin, spalteMin = 0, 8, 8
        gruppenFarbe = self.b[gruppe[0][0]][gruppe[0][1]]['farbe']
        for gruppenStein in gruppe:
            if gruppenStein[0] < reiheMin:
                reiheMin = gruppenStein[0]
            if gruppenStein[1] < spalteMin:
                spalteMin = gruppenStein[1]
        reiheMax, spalteMax = reiheMin, spalteMin
        for gruppenStein in gruppe:
            if gruppenStein[0] > reiheMax:
                reiheMax = gruppenStein[0]
            if gruppenStein[1] > spalteMax:
                spalteMax = gruppenStein[1]
        for r in range(max(0, reiheMin - 1), min(8, reiheMax + 1) + 1):
            for s in range(max(0, spalteMin - 1), min(8, spalteMax + 1) + 1):
                if self.b[r][s]['farbe'] == 0:
                    # prüfe ob Gruppe von hier erreichbar
                    for nb in range(4):  # Nachbarn oben, rechts, unten, links
                        nbReihe = r + nb - 1
                        if nb == 3:
                            nbReihe = r
                        nbSpalte = s - nb + 2
                        if nb == 0:
                            nbSpalte = s
                        if nbReihe in range(9) and nbSpalte in range(9):
                            if self.b[nbReihe][nbSpalte]['farbe'] == gruppenFarbe \
                                    and [nbReihe, nbSpalte] in gruppe:
                                fh += 1
                                stein = (r, s)
                                break
        # update fh der Gruppe
        for gruppenStein in gruppe:
            self.b[gruppenStein[0]][gruppenStein[1]]['fh'] = fh
        return fh, stein

    def fhUpdate(self, reihe, spalte):
        # für angrenzende Felder:
        #   falls leer: fh erhöhen
        #   falls andere Farbe: von diesem Stein fh vermindern und schlagen wenn fh = 0, Ko berücksichtigen
        #   falls selbe Farbe: Gruppe erweitern
        # wenn fh 0 bleibt, dann Zug rückgängig machen (über return code)
        fh = 0
        erweiterGruppe = False
        mitKoGefahr = self.koGefahr
        gruppe = [[reihe, spalte]]
        fhReduziertGruppe = []
        for nb in range(4):   # Nachbarn oben, rechts, unten, links
            nbReihe = reihe + nb - 1
            if nb == 3:
                nbReihe = reihe
            nbSpalte = spalte - nb + 2
            if nb == 0:
                nbSpalte = spalte
            if nbReihe in range(9) and nbSpalte in range(9):
                if self.b[nbReihe][nbSpalte]['farbe'] == 0:
                    self.b[reihe][spalte]['fh'] += 1
                    fh = 1
                elif self.b[nbReihe][nbSpalte]['farbe'] != self.farbeActual:
                    if self.b[nbReihe][nbSpalte]['gr'] == []:
                        self.b[nbReihe][nbSpalte]['fh'] -= 1
                        if self.b[nbReihe][nbSpalte]['fh'] == 0:
                            if self.koPerLastMove:
                                if not self.koGefahr or self.koStein != [reihe, spalte]:
                                    self.koGefahr = True
                                    mitKoGefahr = False # damit bleibt self.koGefahr True
                                    self.koStein = [nbReihe, nbSpalte]
                                    self.schlageStein(nbReihe, nbSpalte)
                                elif self.koGefahr:
                                    self.schlageStein(nbReihe, nbSpalte)
                                    if self.koStein == [reihe, spalte]:
                                        # Ko darf nicht direkt zurückgeschlagen werden
                                        fh = -1
                                        return fh
                            else:
                                self.schlageStein(nbReihe, nbSpalte)
                                gleich = True
                                for row in range(9):
                                    for col in range(9):
                                        if self.b[row][col]['farbe'] != self.b1[row][col]['farbe']:
                                            gleich = False
                                            break
                                if gleich:
                                    # Ko darf nicht direkt zurückgeschlagen werden
                                    fh = -1
                                    return fh
                            fh = 1
                    else:
                        # vermeiden, dass dieselbe Gruppe mehrmals fh reduziert bekommt,
                        #   wenn sie mehrfach an den gesetzten Stein angrenzt
                        if fhReduziertGruppe == []:
                            fhReduziertGruppe.append(self.b[nbReihe][nbSpalte]['gr'])
                            for grStein in self.b[nbReihe][nbSpalte]['gr']:
                                self.b[grStein[0]][grStein[1]]['fh'] -= 1
                                if self.b[grStein[0]][grStein[1]]['fh'] == 0:
                                    self.schlageStein(grStein[0], grStein[1])
                                    # vermeiden, dass fh mit direkter Nachbarschaft zweifach erhöht wird
                                    if nb == 0:
                                        if reihe == grStein[0] and (
                                                spalte + 1 == grStein[1] or spalte - 1 == grStein[1]) \
                                                or reihe + 1 == grStein[0] and spalte == grStein[1]:
                                            self.b[reihe][spalte]['fh'] -= 1
                                    elif nb == 1:
                                        if reihe + 1 == grStein[0] and spalte == grStein[1] \
                                                or reihe == grStein[0] and spalte - 1 == grStein[1]:
                                            self.b[reihe][spalte]['fh'] -= 1
                                    elif nb == 2:
                                        if reihe == grStein[0] and spalte - 1 == grStein[1]:
                                            self.b[reihe][spalte]['fh'] -= 1
                                    fh = 1
                        else:
                            inRedGruppen = False
                            for g in fhReduziertGruppe:
                                if [nbReihe, nbSpalte] in g:
                                    inRedGruppen = True
                            if not inRedGruppen:
                                fhReduziertGruppe.append(self.b[nbReihe][nbSpalte]['gr'])
                                for grStein in self.b[nbReihe][nbSpalte]['gr']:
                                    self.b[grStein[0]][grStein[1]]['fh'] -= 1
                                    if self.b[grStein[0]][grStein[1]]['fh'] == 0:
                                        self.schlageStein(grStein[0], grStein[1])
                                        fh = 1
                else:   # erweitere Gruppe
                    erweiterGruppe = True
                    if self.b[nbReihe][nbSpalte]['gr'] == []:
                        gruppe.append([nbReihe, nbSpalte])
                    else:
                        for g in self.b[nbReihe][nbSpalte]['gr']:
                            if not g in gruppe:
                                gruppe.append(g)
        if erweiterGruppe:
            fhZ, _ = self.fhZaehlen(gruppe)
            if fhZ > 0:
                fh = 1
            # update Gruppe
            for gruppenStein in gruppe:
                self.b[gruppenStein[0]][gruppenStein[1]]['gr'] = gruppe
        if mitKoGefahr:
            self.koGefahr = False
            self.koStein = [-1, -1]
        return fh

    def evalZug(self):  # evtl nützlich um auszählen zu optimieren
        #   Bewertung: Summe aller fh der eigenen Farbe minus der der Gegner Farbe
        #       plus eigene Gefangene minus gegnerische Gefangene
        #   Dabei zählen fh einer Gruppe nur einmal, nicht pro Stein,
        #       mit Multiplikator zur Egalisierung von nur Einzelsteinen
        fhS, fhW = 0, 0
        grS, grW = [], []
        for i in range(9):
            for j in range(9):
                if self.b[i][j]['farbe'] == 1:
                    if [i, j] not in grS:
                        if self.b[i][j]['gr'] == []:
                            fhS = fhS + self.b[i][j]['fh']
                        else:
                            fhS = fhS + 2 * self.b[i][j]['fh'] - 4
                        for g in self.b[i][j]['gr']:
                            grS.append(g)
                elif self.b[i][j]['farbe'] == -1:
                    if [i, j] not in grW:
                        if self.b[i][j]['gr'] == []:
                            fhW = fhW + self.b[i][j]['fh']
                        else:
                            fhW = fhW + 2 * self.b[i][j]['fh'] - 4
                        for g in self.b[i][j]['gr']:
                            grW.append(g)
        if self.farbeActual == 1:
            return fhS - fhW + self.gefangenS - self.gefangenW
        else:
            return fhW - fhS + self.gefangenW - self.gefangenS

    def setzZug(self, position):
        # return: possible True/False
        possible = True
        bSafe = copy.deepcopy(self.b)
        koGefahr = self.koGefahr
        koStein = self.koStein
        gefangenS, gefangenW = self.gefangenS, self.gefangenW
        self.farbeActual = - self.farbeActual
        if position == 81:   # pass
            if not self.toteSchlagen:
                self.anzPass +=1
            if self.gepasst:
                self.spielBeendet = True
                if not self.toteSchlagen:
                    self.pktSchwarz, self.pktWeiss = self.auszaehlen()
                    if self.pktSchwarz == self.pktWeiss:
                        self.gewinner = 0
                    elif self.pktSchwarz > self.pktWeiss:
                        self.gewinner = 1
                    else:
                        self.gewinner = -1
#                    if self.koPerLastMove:
#                        print('Play_game: S:', self.pktSchwarz, ' W:', self.pktWeiss,
#                              ' #Pass:', self.anzPass, 'Zug: ', self.zugNr)
#                    else:
#                        print('Find_leaf: S:', self.pktSchwarz, ' W:', self.pktWeiss,
#                              ' #Pass:', self.anzPass, 'Zug: ', self.zugNr)
            else:
                self.gepasst = True
                self.koGefahr = False
        elif position == 82: # aufg
            self.spielBeendet = True
            self.gewinner = - self.farbeActual
            self.pktSchwarz, self.pktWeiss = self.auszaehlen()
        else:
            reihe = position // 9
            spalte = position % 9
            self.b[reihe][spalte]['farbe'] = self.farbeActual
            fhReturn = self.fhUpdate(reihe, spalte)
            if fhReturn > 0:
                self.gepasst = False
            else:
                # Fh bleibt nach Update 0, oder unerlaubtes Ko-Zurückschlagen --> Zug rückgängig machen
                self.b = copy.deepcopy(bSafe)   # nötig? oder nur Zuweisung reicht?
                self.farbeActual = - self.farbeActual
                self.koGefahr = koGefahr
                self.koStein = koStein
                self.gefangenS, self.gefangenW = gefangenS, gefangenW
                possible = False
        if possible:
            self.b2 = copy.deepcopy(self.b1)
            self.b1 = copy.deepcopy(bSafe)
            if not self.toteSchlagen:
                self.zugNr += 1
            if self.zugNr > self.zugMax and not self.spielBeendet:
                self.spielBeendet = True
                self.pktSchwarz, self.pktWeiss = self.auszaehlen()
                if self.pktSchwarz == self.pktWeiss:
                    self.gewinner = 0
                elif self.pktSchwarz > self.pktWeiss:
                    self.gewinner = 1
                else:
                    self.gewinner = -1
#                if self.koPerLastMove:
#                    print('play_game mit ', self.zugMax, ' Zügen beendet. S:', self.pktSchwarz, ' W:', self.pktWeiss,
#                          ' #Pass:', self.anzPass)
#                else:
#                    print('find_leaf mit ', self.zugMax, ' Zügen beendet. S:', self.pktSchwarz, ' W:', self.pktWeiss,
#                          ' #Pass:', self.anzPass)
        return possible
