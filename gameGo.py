#!/usr/kai/anaconda3/python
# -*- coding: utf-8 -*-

# Hilfs-Funktionen:
#   zum Spielen von Go9x9 (Brettdrehung, Print)
#   zur Konvertierung des NN Input Formats (B7)
#   zur Speicherung des MCTS Trees
# V1: setzt auf auf V4: gameT3
# V2: b7 convert functions
# V3: testing
# V4: drehung
#

PLAYER_BLACK = 1
PLAYER_WHITE = -1

size = 9    # Board Größe
size2 = size*size
ANZ_POSITIONS = size2+1
boardSchwarz = [[1] * size for i in range(size)]
boardWeiss = [[0] * size for i in range(size)]
b7Initial = [[[0] * size for i in range(size)] for j in range(6)] + [boardSchwarz]

# Codierung Board als b = [[0] * size for i in range(size)]
            # 9*9 Matrix,(0,0) ist oben links
            # schwarz = PLAYER_BLACK, weiss = PLAYER_WHITE, leer = 0
            # Board7 = Board * 7 ist NN Input
        # sowie Board7 für Performance als Int mit 2**x Konvertierung (0,0), (0,1), ...(8,8), 6-fach,
#           dann 0|1 als negativ, positiv

class BoardValues:
    """
    Speichert und Retrieves für Board7-States: Count, Value, Probs (ValueAvg wird berechnet)
    """
    def __init__(self):
        # count of visits, state -> [N(s, a)]
        # total value of the state's action, state -> [W(s, a)]
        # average value of actions, state -> [Q(s, a)]: berechnet
        # prior probability of actions, state -> [P(s,a)]
        # Dictionaries mit Key Int-of-Board7
        self.b = {}

    def clear(self):
        self.b.clear()

    def __len__(self):
        return len(self.b)

    def expand(self, b7Int, probs):
        self.b[b7Int] = [[0] * ANZ_POSITIONS, [0] * ANZ_POSITIONS, probs]

    def backup(self, b7Int, action, val):
        self.b[b7Int][0][action] += 1
        self.b[b7Int][1][action] += val

def b7To3(b7):
    b  = [[0] * size for i in range(size)]
    b1 = [[0] * size for i in range(size)]
    b2 = [[0] * size for i in range(size)]
    f01 = b7[6][0][0]
    for i in range(size):
        for j in range(size):
            if b7[1-f01][i][j] == 1:
                b[i][j] = 1
            else:
                b[i][j] = - b7[f01][i][j]
            if b7[3-f01][i][j] == 1:
                b1[i][j] = 1
            else:
                b1[i][j] = - b7[2+f01][i][j]
            if b7[5-f01][i][j] == 1:
                b2[i][j] = 1
            else:
                b2[i][j] = - b7[4+f01][i][j]
    return b, b1, b2

def b12To7(b, b1, b2, whoMoves):
    bRet = [[[0] * size for i in range(size)] for j in range(7)]
    if whoMoves == 0:
        bRet[6] = boardWeiss
    else:
        bRet[6] = boardSchwarz
    for i in range(size):
        for j in range(size):
            if b[i][j] == 1:
                bRet[1-whoMoves][i][j] = 1
            else:
                bRet[whoMoves][i][j] = -b[i][j]
            if b1[i][j] == 1:
                bRet[3-whoMoves][i][j] = 1
            else:
                bRet[2+whoMoves][i][j] = -b1[i][j]
            if b2[i][j] == 1:
                bRet[5-whoMoves][i][j] = 1
            else:
                bRet[4+whoMoves][i][j] = -b2[i][j]
    return bRet

def printBrett(b, istFlat=False, mitFloat=False):
    for row in range(size):
        for col in range(size):
            if istFlat:
                cell = b[row*size+col]
            else:
                cell = b[row][col]
            if mitFloat:
                print('%4.3f ' % (cell), end='')
            else:
                print(str(cell).rjust(3), end='')
        print('')
    if istFlat:
        print('Wert für pass: ',b[81])

def printBufferEntry(buf):
    # buf ist List mit board7, whoMoves, probs, value
    b = [[0] * size for i in range(size)]
    whoMoves = buf[1]
    buf0 = intToB(buf[0])
    print(whoMoves, ' moves')
    print('Probs:')
    printBrett(buf[2], istFlat=True, mitFloat=True)
    print('Value: ', buf[3])
    for i in range(size):
        for j in range(size):
            if buf0[1-whoMoves][i][j] == 1:
                b[i][j] = 1
            else:
                b[i][j] = - buf0[whoMoves][i][j]
    print('current Board:')
    printBrett(b)
    print('')

def intToB(num):
    # Board Convertierung von Int Darstellung zu 9x9er Boards
    board7 = [[[0] * size for i in range(size)] for j in range(7)]
    if num >= 0:
        board7[6] = boardSchwarz
    else:
        board7[6] = boardWeiss
        num = - num
    for board in range(6):
        for row in range(size):
            for col in range(size):
                board7[board][row][col] = num % 2
                num = num // 2
    return board7

def bToInt(board7):
    num = 0
    for board in range(6):
        for row in range(size):
            for col in range(size):
                num += board7[board][row][col] * 2 ** (size2*board+(size*row+col))
    if board7[6][0][0] == 0:
        return - num
    else:
        return num

def dreh(reihe, spalte, drehung):
    # return gespiegelte/gedrehte row, col
    if drehung % 90 == 0:
        for i in range(drehung // 90):
            reiheAlt = reihe
            reihe = spalte
            spalte = size - 1 - reiheAlt
    elif drehung == 1:
        reihe = size - 1 - reihe
    elif drehung == 2:
        spalte = size - 1 - spalte
    elif drehung == 3:
        reihe = size - 1 - reihe
        reiheAlt = reihe
        reihe = spalte
        spalte = size - 1 - reiheAlt
    else:   # drehung == 4
        spalte = size - 1 - spalte
        reiheAlt = reihe
        reihe = spalte
        spalte = size - 1 - reiheAlt
    return reihe, spalte

def drehPosition(position, drehung):
    # return gespiegelte/gedrehte Position Liste
    posRet = [0] * ANZ_POSITIONS
    posRet[81] = position[81]
    for i in range(size2):
        reihe, spalte = dreh(i//9, i%9, drehung)
        posRet[reihe*size+spalte] = position[i]
    return posRet

def drehB7(b7int, drehung):
    # return gespiegelte/gedrehte b7int
    b7 = intToB(b7int)
    b7ret = [[[0] * size for i in range(size)] for j in range(7)]
    b7ret[6] = b7[6]
    for row in range(size):
        for col in range(size):
            reihe, spalte = dreh(row, col, drehung)
            for board in range(6):
                b7ret[board][reihe][spalte] = b7[board][row][col]
    return bToInt(b7ret)