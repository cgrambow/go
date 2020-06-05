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
# V5: Performance: b, b1 und neues Int statt b7
# V6: timer
#
import torch, time
import numpy as np

PLAYER_BLACK = 1
PLAYER_WHITE = -1

size = 9    # Board Größe
size2 = size*size
ANZ_POSITIONS = size2+1
boardSchwarzNP = np.ones((size, size), dtype=float)
boardWeissNP = np.zeros((size, size), dtype=float)
boardSchwarz = [[1] * size for i in range(size)]
boardWeiss = [[0] * size for i in range(size)]
b2Initial = [[[0] * size for i in range(size)] for j in range(2)]

# Codierung Board als b = [[0] * size for i in range(size)]
            # 9*9 Matrix,(0,0) ist oben links
            # schwarz = PLAYER_BLACK, weiss = PLAYER_WHITE, leer = 0
            # Board5 = Board * 5 ist NN Input (hier jeweils Codierung Black, White mit 1/0
            #   für aktuell, vorheriges Board und BlackOrWhite Board
        # sowie Board2 für Performance als Int mit 2**x und 3**x Konvertierung (0,0), (0,1), ...(8,8),
        #           für aktuell, vorheriges Board mit -1/0/1
        #           Ermittlung vorheriges als 0/1 Delta: 0=gleicher Stein, 1=wurde gesetzt oder geschlagen

class BoardValues:
    """
    Speichert und Retrieves für Board2-States: Count, Value, Probs (ValueAvg wird berechnet)
    """
    def __init__(self):
        # count of visits, state -> [N(s, a)]
        # total value of the state's action, state -> [W(s, a)]
        # average value of actions, state -> [Q(s, a)]: berechnet
        # prior probability of actions, state -> [P(s,a)]
        # Dictionaries mit Key Int-of-Board2
        self.b = {}

    def clear(self):
        self.b.clear()

    def __len__(self):
        return len(self.b)

    def expand(self, b2Int, probs):
        self.b[b2Int] = [[0] * ANZ_POSITIONS, [0.0] * ANZ_POSITIONS, probs]

    def backup(self, b2Int, action, val):
        self.b[b2Int][0][action] += 1
        self.b[b2Int][1][action] += val
#        if self.b[b2Int][0][action] > 40 and abs(self.b[b2Int][1][action] - val) > 0.001:
#            print('backup mit action: ', action, ' value: ', val, ' bei:')
#            printBrett(intToB(b2Int)[0])
#            print('neues count: ', self.b[b2Int][0][action], 'neues value: ', self.b[b2Int][1][action])

class GoTimer:
    # startet und stoppt die Zeit, mehrfach
    def __init__(self, routine, mitGesamt=False):
        self.t = 0
        self.tUsed = 0
        self.anz = 0
        self.routine = routine
        self.mitGesamt = mitGesamt
    def start(self):
        self.t = time.time()
    def stop(self):
        self.tUsed += time.time() - self.t
        self.anz += 1
    def timerPrint(self):
        if self.mitGesamt:
            stdUsed = self.tUsed // 3600
            minUsed = (self.tUsed - 3600 * stdUsed) // 60
            secUsed = (self.tUsed - 3600 * stdUsed) % 60
            print('Routine: '+self.routine+', Zeit insg.: %02d:%02d:%02d' % (stdUsed, minUsed, secUsed))

        self.tUsed = round(self.tUsed / self.anz)
        minUsed = self.tUsed // 60
        secUsed = self.tUsed % 60
        print('Routine: '+self.routine+', Zeit pro Step: %02d:%02d' % (minUsed, secUsed))

def b5To2(b5):
    b  = [[0] * size for i in range(size)]
    b1 = [[0] * size for i in range(size)]
    f01 = b5[4][0][0]
    for i in range(size):
        for j in range(size):
            if b5[1-f01][i][j] == 1:
                b[i][j] = 1
            else:
                b[i][j] = - b5[f01][i][j]
            if b5[3-f01][i][j] == 1:
                b1[i][j] = 1
            else:
                b1[i][j] = - b5[2+f01][i][j]
    return b, b1

def b1To5(b, b1, whoMoves):
    bRet = [[[0] * size for i in range(size)] for j in range(5)]
    if whoMoves == 0:
        bRet[4] = boardWeiss
    else:
        bRet[4] = boardSchwarz
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
    # buf ist List mit board2-Int, whoMoves, probs, value
    whoMoves = buf[1]
    print(whoMoves, ' moves')
    print('Probs:')
    printBrett(buf[2], istFlat=True, mitFloat=True)
    print('Value: ', buf[3])
    print('current Board:')
    printBrett(intToB(buf[0])[0])
    print('')

def bToInt(board2):
    # analog der Methode von goSpielNoGraph
    #   wird nur bei Drehung benötigt
    num, numB = 0, 0
    bDelta  = [[0] * 9 for i in range(9)]
    for row in range(9):
        for col in range(9):
            if board2[0][row][col] != board2[1][row][col]:
                bDelta[row][col] = 1
    for row in range(9):
        for col in range(9):
            num += bDelta[row][col] * 2**(9*row+col)
    for row in range(9):
        for col in range(9):
            numB += (board2[0][row][col]+1) * 3**(9*row+col)
    return numB*2**81+num

def intToB(num):
    # Board Convertierung von Int Darstellung zu 9x9er Boards
    board2 = [[[0] * size for i in range(size)] for j in range(2)]
    bDelta = [[0] * 9 for i in range(size)]
    numDelta = num % 2**size2
    numB = num // 2**size2
    for row in range(size):
        for col in range(size):
            bDelta[row][col] = numDelta % 2
            numDelta = numDelta // 2
    for row in range(size):
        for col in range(size):
            board2[0][row][col] = numB % 3 - 1
            numB = numB // 3
    found = False
    for row in range(size):
        if found:
            break
        for col in range(size):
            if bDelta[row][col] == 1 and board2[0][row][col] != 0:
                farbeAct = board2[0][row][col]
                found = True
                break
    for row in range(size):
        for col in range(size):
            if bDelta[row][col] == 0:
                board2[1][row][col] = board2[0][row][col]
            elif board2[0][row][col] == 0:
                board2[1][row][col] = -farbeAct
    return board2

def _encode_list_state(dest_np, board2, whoMoves):
    """
    In-place encodes list state into the zero numpy array
    :param dest_np: dest array, expected to be zero
    :param state_list: state of the game in the list form
    :param who_move: player index who to move
    """
    if whoMoves == 0:
        dest_np[4] = boardWeissNP
    else:
        dest_np[4] = boardSchwarzNP
    for i in range(size):
        for j in range(size):
            if board2[0][i][j] == 1:
                dest_np[1-whoMoves][i][j] = 1
            else:
                dest_np[whoMoves][i][j] = -board2[0][i][j]
            if board2[1][i][j] == 1:
                dest_np[3-whoMoves][i][j] = 1
            else:
                dest_np[2+whoMoves][i][j] = -board2[1][i][j]

def state_lists_to_batch(state_lists, whoMoves_lists, device="cpu"):
    """
    Convert list of list states to batch for network
    :param state_lists: list of 'list states'
    :param who_moves_lists: list of player index who moves
    :return Variable with observations
    """
#    assert isinstance(state_lists, list)
    batch_size = len(state_lists)
    batch = np.zeros((batch_size, 5, size, size), dtype=float)
    for idx, (state, whoMoves) in enumerate(zip(state_lists, whoMoves_lists)):
        _encode_list_state(batch[idx], state, whoMoves)
    return torch.FloatTensor(batch).to(device)

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

def drehB2(b2int, drehung):
    # return gespiegelte/gedrehte b
    if drehung == 0:
        return b2int
    b2 = intToB(b2int)
    b2ret = [[[0] * size for i in range(size)] for j in range(2)]
    for row in range(size):
        for col in range(size):
            reihe, spalte = dreh(row, col, drehung)
            for board in range(2):
                b2ret[board][reihe][spalte] = b2[board][row][col]
    return bToInt(b2ret)
