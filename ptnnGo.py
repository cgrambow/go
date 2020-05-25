#!/usr/kai/anaconda3/python
# -*- coding: utf-8 -*-

# Supervised and then Reinforcement Learning mit PyTorch
# für Go9x9
# Supervised learning der gespeicherten best practices sgf
# # für Gewinner: Input = alle Board Positionen in denen Gewinner am Zug. Label = sein Zug
# # Lösung mit 1 Convolution, 2 Res und 1 FullyConnected Layer
# # In   : 2 binary Map, eine für current player, eine für Opponent (gesetzt=1, empty=0) , 9x9 Feld.
#           Dies auch für Zug -1 und -2 (wegen Ko), und eine const. Map 1 oder 0 abh. ob S oder W current ist
#           Also insg. 7 9x9 binary Maps
# Reinforcement Learning: MCTS mit SelfPlay
# V1: weiterentwickelt von der V7 Lösung für Tic-Tac-Toe
# V2: sgf: pj und mj sind pass, setzZug mit möglichem Return False
# V3: NN Ausbau
# V4: calls zu mcts
# V5: SL SGF gewinner != Go Gewinner
# V6: falsches save dir, zugNr Par zu mcts search, endlos impossible action
# V7: drehung
# V8: ZUG_MAX bei SL 81
# V9: GPU
# V10:predict auf ANZ_POSITIONS geändert, sgfWrite bei playGame
#
import torch, torch.cuda
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import copy, os, sgf, sys, math, random, time, collections, mctsGo, gameGo, goSpielNoGraph, shutil

from tensorboardX import SummaryWriter
writer = SummaryWriter(comment="-Go")

col = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j')

if torch.cuda.is_available():
    DEVICE = 'cuda'
    dirSave = 'goNN'
else:
    DEVICE = 'cpu'
    dirSave = '/Users/kai/Documents/Python/ML/goNN'

PLAY_STATISTIK = 1  # 0: keine, 1: Summary, 2: Detail
GEWICHT_SL_MSE = 0.75
ZUG_MAX = 75
MAX_STEP = 25 # prod: 10000
PLAY_EPISODES = 1  # prod: 5000
MCTS_SEARCHES = 200  # prod: 40, bei kein Minibatch 300
MCTS_BATCH_SIZE = 0 # 0, wenn kein Minibatch, sonst 8
MCTS_SEARCHES_EVAL = MCTS_SEARCHES  # testing: 8, prod:
MCTS_BATCH_SIZE_EVAL = MCTS_BATCH_SIZE  # 0, wenn kein Minibatch
REPLAY_BUFFER = 10000 # prod: 100.000
LEARNING_RATE = 0.1    # auch 0.01 für MCTS Training probieren
BATCH_SIZE = 128    # prod: 128
TRAIN_ROUNDS = 250 # prod: 30 oder viel mehr
MIN_REPLAY_TO_TRAIN = 500   # muss größer BATCH_SIZE sein, prod: 1000
BEST_NET_WIN_RATIO = 0.55 # auch 0.6 probieren
EVALUATE_EVERY_STEP = 50 # prod: 200
PRINT_EVERY_STEP = EVALUATE_EVERY_STEP/50
EVALUATION_ROUNDS = 10   # prod: 80
STEPS_BEFORE_TAU_0 = 15

class NnGo(nn.Module):

    def __init__(self):
        ANZ_CONV_FILTER = 8 # auch 16 oder 32 probieren
        FC_HIDDEN = 16      # auch 8 oder 32 probieren
        super(NnGo, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(7, ANZ_CONV_FILTER, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ANZ_CONV_FILTER),
            nn.LeakyReLU()
        )
        # layers with residual
        self.convRes1 = nn.Sequential(
            nn.Conv2d(ANZ_CONV_FILTER, ANZ_CONV_FILTER, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ANZ_CONV_FILTER),
            nn.LeakyReLU()
        )
        self.convRes2 = nn.Sequential(
            nn.Conv2d(ANZ_CONV_FILTER, ANZ_CONV_FILTER, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ANZ_CONV_FILTER),
            nn.LeakyReLU()
        )
        self.policy1 = nn.Sequential(
            nn.Conv2d(ANZ_CONV_FILTER, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU()
        )
        self.policy2 = nn.Sequential(
            nn.Linear(2*gameGo.size2, gameGo.ANZ_POSITIONS)
        )
        self.value1 = nn.Sequential(
            nn.Conv2d(ANZ_CONV_FILTER, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.value2 = nn.Sequential(
            nn.Linear(gameGo.size2, FC_HIDDEN),
            nn.LeakyReLU(),
            nn.Linear(FC_HIDDEN, 1),
            nn.Tanh()   ### evtl
        )
    def forward(self, x):
        batch_size = x.size()[0]
        out = self.conv(x)
        out = out + self.convRes1(out)
        out = out + self.convRes2(out)
        outPol = self.policy1(out)
        outVal = self.value1(out)
        outPol = self.policy2(outPol.view(batch_size, -1))
        outVal = self.value2(outVal.view(batch_size, -1))
        return outPol, outVal

def trainSL(net):
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    os.chdir('/Users/kai/Documents/Python/ML/go9x9')
    dateien = os.listdir(".")
    buffer = collections.deque()    # mit board7, whoMoves, probs, value
    # teach aus einigen Spielen
    for datei in dateien:
        if datei.endswith(".sgf"):
            print('verarbeite '+datei)
            with open(datei) as f:
                collection = sgf.parse(f.read())
            try:
                gewinner = collection[0].root.properties['RE'][0][0]
            except:
                print('Noch kein Gewinner')
                continue
            for drehung in (0, 90, 180, 270, 1, 2, 3, 4):
                spiel = goSpielNoGraph.PlayGo(gameGo.b7Initial)
                passM1 = False
                for node in collection[0].rest:
                    for k, v in node.properties.items():
                        if k == 'B':
                            player = 1
                        elif k == 'W':
                            player = 0
                        else:
                            print('Falscher Key, weder B noch W!')
                            sys.exit()
                        if len(v[0]) == 0 or v[0] == 'pj' or v[0] == 'mj':
                            if not node.next and not passM1:
                                position = 82
                                spiel.setzZug(position)
                            else:
                                position = 81
                                # pass wird nicht trainiert
                                passM1 += 1
                                if passM1 <= 2:
                                    spiel.setzZug(position)
                        elif len(v[0]) == 1:
                            print('Falsche Zug-Syntax: ' + v[0])
                            sys.exit()
                        else:
                            passM1 = 0
                            reiheStr = v[0][1]
                            if reiheStr == 'i':
                                reiheStr = 'j'
                            spalteStr = v[0][0]
                            if spalteStr == 'i':
                                spalteStr = 'j'
                            if spalteStr not in col or reiheStr not in col:
                                print('Falsche Zug-Syntax: ' + v[0])
                                sys.exit()
                            spalte = col.index(spalteStr)
                            reihe = col.index(reiheStr)
                            reihe, spalte = gameGo.dreh(reihe, spalte, drehung)
                            probs = [0] * gameGo.ANZ_POSITIONS
                            probs[gameGo.size*reihe+spalte] = 1
                            b, b1, b2 = spiel.b12Farbe()
                            board7 = gameGo.b12To7(b, b1, b2, player)
                            if k == gewinner:
                                buffer.append((board7, player, probs, 1))
                            else:
                                buffer.append((board7, player, probs, -1))
                            position = gameGo.size * reihe + spalte
                            if not spiel.setzZug(position):
                                print('Unerlaubter Zug: Reihe='+str(reihe)+' Spalte='+str(spalte))
                                sys.exit()
                if spiel.gewinner == 1 and not gewinner == 'B' \
                        or spiel.gewinner == -1 and not gewinner == 'W':
                    print('Bei Drehung: ', drehung, ' Gewinner im SGF inconsistent zum Spiel !')
                    print('spiel.gewinner = '+str(spiel.gewinner)+' SGF gewinner = '+gewinner)
                    print('PktB: ', spiel.pktSchwarz, ' PktW: ', spiel.pktWeiss)
                    shutil.move(datei, '/Users/kai/Documents/Python/ML/go9x9test/'+datei)
                    break
    # train
    # bei m*m Instanzen m/2 Mini-Batches nehmen
    batch_size = int(math.sqrt(len(buffer))/2)
    print('Instances: ', len(buffer), ', Mini-Batch Size: ', batch_size)
    for _ in range(len(buffer)//batch_size):
        batch = random.sample(buffer, batch_size)
        batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
        batch_states = np.array(batch_states, dtype=float)
        states_v = torch.from_numpy(batch_states).float()
        optimizer.zero_grad()
        probs_v = torch.FloatTensor(batch_probs)
        values_v = torch.FloatTensor(batch_values)
        out_logits_v, out_values_v = net(states_v)

        loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
        loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
        loss_policy_v = loss_policy_v.sum(dim=1).mean()

        loss_v = loss_policy_v + GEWICHT_SL_MSE*loss_value_v
        loss_v.backward()
        optimizer.step()
        print('loss_total: ' + str(loss_v.item()) + ', loss_value: ' + str(loss_value_v.item())
              + ', loss_policy: ' + str(loss_policy_v.item()))
    torch.save(net, dirSave+'/go_SL.pt')

def play_game(mcts_stores, replay_buffer, net1, net2, steps_before_tau_0,
              mcts_searches, mcts_batch_size=0, stat='nicht', device='cpu'):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param mcts_stores: could be single MCTS or two MCTSes for individual net
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net1: player1
    :param net2: player2
    :param mcts_batch_size: Batch size for MCTS Minibatch, 0: no Minibatch Call
    :return: value for the game in respect to net1 (+1 if p1 won, -1 if lost, 0 if draw)
    Statistik: Anteil Leaf-Calls wird bei erstem Evaluate Spiel bestimmt
                Unterschiede MCTS vs NN wird bei letztem Evaluate Spiel bestimmt
                kann insg. über PLAY_STATISTIK gesteuert werden: aus, nur-Summary, detailliert
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_stores, (mctsGo.MCTS, type(None), list))
    assert isinstance(net1, NnGo)
    assert isinstance(net2, NnGo)
    if isinstance(mcts_stores, mctsGo.MCTS):
        mcts_stores = [mcts_stores, mcts_stores]
    spiel = goSpielNoGraph.PlayGo(gameGo.b7Initial, zugMax=ZUG_MAX)
    state = gameGo.bToInt(gameGo.b7Initial)
    nets = [net1, net2]
    cur_player = 1 # schwwarz beginnt immer, und das ist net1
    step = 0
    countDiff = 0
    countEnd = 0
    countSearch = mcts_searches * mcts_batch_size if mcts_batch_size > 0 else mcts_searches
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []
    values, zuege = [], []
    result = None
    net1_result = None
    while result is None:
        statEnd = mcts_stores[1-cur_player].search(mcts_searches, mcts_batch_size, state, cur_player,
                                        nets[1-cur_player], zugNr = step+1, zugMax=ZUG_MAX, device=device)
        countEnd += statEnd
        probs = mcts_stores[1-cur_player].get_policy(state, tau=tau)
        action = np.random.choice(gameGo.ANZ_POSITIONS, p=probs)
        if not spiel.setzZug(action):   # hier move: setzen eines Zuges
            print('Impossible action at step ', step, ', Player: ', cur_player, '. Action=', action, ' at:')
            spiel.printB()
            print('mit probs:')
            gameGo.printBrett(probs, istFlat=True, mitFloat=True)
            counts = mcts_stores[1-cur_player].stateStats.b[state][0]
            print('Counts:')
            gameGo.printBrett(counts, istFlat=True)
            spiel.setzZug(81)
        elif PLAY_STATISTIK == 1:
            zuege.append(action)
            values.append('%1.2f ' % (mcts_stores[1-cur_player].stateStats.b[state][2][action]))
        game_history.append((state, cur_player, probs))
        if PLAY_STATISTIK > 0 and stat != 'nicht':
            board7 = gameGo.intToB(state)
            board7 = np.array([board7], dtype=float)
            board7 = torch.from_numpy(board7).float().to(device)
            p_v, _ = nets[1-cur_player](board7)
            probs = p_v.detach().cpu().numpy()[0]
            position = np.argmax(probs)
            if position != action:
                countDiff += 1
                if PLAY_STATISTIK == 2:
                    print('play_game step ', step+1, ' action Unterschied!')
                    print('Action  MCTS: ', action, '  NN: ', position)
                    print('Anteil Leaf-Calls bis Spiel-Ende: '+str(statEnd)+' = '+str(int(statEnd*100/countSearch))+'%')
                    print('')
        if spiel.spielBeendet:
#            print('Gewinner:', spiel.gewinner, 'S:', spiel.pktSchwarz, 'W:', spiel.pktWeiss)
            if PLAY_STATISTIK == 1:
                spiel.sgfWrite(zuege, values)
            if spiel.gewinner == 1:
                net1_result = 1
                if cur_player == 1:
                    result = 1
                else:
                    result = -1
            elif spiel.gewinner == -1:
                net1_result = -1
                if cur_player == 1:
                    result = -1
                else:
                    result = 1
            else:
                result = 0
                net1_result = 0
            break
        cur_player = 1-cur_player
        b, b1, b2 = spiel.b12Farbe()
        state = gameGo.bToInt(gameGo.b12To7(b, b1, b2, cur_player))
        step += 1
        if step >= steps_before_tau_0:
            tau = 0
    if PLAY_STATISTIK > 0:
        if stat == 'Diff':
            print('play game Unterschiede MCTS zu NN: '+str(countDiff)+' = '+str(int(countDiff*100/(step+1)))+'%')
        elif stat == 'Leaf':
            print('Anteil Leaf-Calls bis Spiel-Ende insg: '
              + str(countEnd) + ' = ' + str(int(countEnd*100/(countSearch*(step+1)))) + '%')
    if replay_buffer is not None:
        for state, cur_player, probs in reversed(game_history):
            for drehung in (0, 90, 180, 270, 1, 2, 3, 4):
                replay_buffer.append((gameGo.drehB7(state, drehung), cur_player,
                                      gameGo.drehPosition(probs, drehung), result))
            result = -result
    return net1_result, step

def evaluate(net, netBest, rounds, device='cpu'):
    n1_win, n2_win = 0, 0
    mcts_stores = [mctsGo.MCTS(), mctsGo.MCTS()]
    netze_idx = np.random.choice(2)
    netze = [net, netBest]
    doStat = 'Leaf'
    schwarzGewinn = 0
    for r_idx in range(rounds):
        r, _ = play_game(mcts_stores, None, net, netBest, steps_before_tau_0=0,
                mcts_searches=MCTS_SEARCHES_EVAL, mcts_batch_size= MCTS_BATCH_SIZE_EVAL,
                stat=doStat, device=device)
        if r_idx < rounds - 2:
            doStat = 'nicht'
        else:
            doStat = 'Diff'
        if r != 0:
            if r == 2 * netze_idx - 1:
                n1_win += 1
            else:
                n2_win += 1
            if r == 1:
                schwarzGewinn += 1
        netze_idx = 1 - netze_idx
    n12_win = n1_win + n2_win
    print('aus Sicht neues NN: ', n1_win, ' gewonnen, ', rounds - n12_win, ' unentschieden, ', n2_win, ' verloren')
    if n12_win == 0:
        return 0.5
    else:
        print('                    Anteil Scharz am Gewinn: ', int(schwarzGewinn * 100 / n12_win), '%')
        return n1_win / n12_win

def trainMCTS(net, aufsatz, device):
    # input: Netz, Nr. des RL mit dem aufgesetzt wird (0: von Beginn an)
    # return Nr. des besten Netzes
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    if aufsatz > 0:
        model = 'RL' + str(aufsatz)
        with open(dirSave + '/go_' + model + '.txt', "r") as infile:
            for line in infile:
                items = line.split('_')
                i2 = items[2][1:-1]
                i2 = i2.split(',')
                i2List = []
                for i in range(gameGo.ANZ_POSITIONS):
                    i2List.append(float(i2[i]))
                replay_buffer.append((int(items[0]), int(items[1]), i2List, int(items[3])))
    mcts_store = mctsGo.MCTS()
    best_idx = aufsatz
    bestNet = NnGo()
    bestNet = bestNet.float()
    bestNet = copy.deepcopy(net)
    t = time.time()
    prev_nodes = 0
    for step_idx in range(1, MAX_STEP+1):    # endless
        game_steps = 0
#        print('Train step: ', step_idx)
        for _ in range(PLAY_EPISODES):
            _, steps = play_game(mcts_store, replay_buffer, bestNet, bestNet,
                        steps_before_tau_0=STEPS_BEFORE_TAU_0, mcts_searches=MCTS_SEARCHES,
                        mcts_batch_size= MCTS_BATCH_SIZE, device=device)
            game_steps += steps
#        tUsed = round(time.time() - t)
#        minUsed = tUsed // 60
#        secUsed = tUsed % 60
#        print('nach play_game %02d:%02d' % (minUsed, secUsed))
        if step_idx % PRINT_EVERY_STEP == 0:
            game_nodes = len(mcts_store) - prev_nodes
            prev_nodes = len(mcts_store)
            print("Step %d, steps %3d, new leaves %3d, best_idx %d, replay size %d" % (
                step_idx, game_steps, game_nodes, best_idx, len(replay_buffer)))
        if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
            continue
        # train
        sum_loss = 0.0
        sum_value_loss = 0.0
        sum_policy_loss = 0.0

        for _ in range(TRAIN_ROUNDS):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
            b7_states = []
            for s in batch_states:
                b7_states.append(gameGo.intToB(s))
            b7_states = np.array(b7_states, dtype=float)
            states_v = torch.from_numpy(b7_states).float().to(device)
            optimizer.zero_grad()
            probs_v = torch.FloatTensor(batch_probs).to(device)
            values_v = torch.FloatTensor(batch_values).to(device)
            out_logits_v, out_values_v = net(states_v)

            loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
            loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
            loss_policy_v = loss_policy_v.sum(dim=1).mean()

            loss_v = loss_policy_v + loss_value_v
            loss_v.backward()
            optimizer.step()
            sum_loss += loss_v.item()
            sum_value_loss += loss_value_v.item()
            sum_policy_loss += loss_policy_v.item()
        if step_idx % PRINT_EVERY_STEP == 0:
            lossTot = sum_loss/TRAIN_ROUNDS
            lossPol = sum_policy_loss/TRAIN_ROUNDS
            lossVal = sum_value_loss/TRAIN_ROUNDS
            print('loss_total: '+str(lossTot)+', loss_value: '+str(lossVal)+', loss_policy: '+str(lossPol))
            writer.add_scalar("loss_total",lossTot , step_idx)
            writer.add_scalar("loss_value",lossVal , step_idx)
            writer.add_scalar("loss_policy",lossPol , step_idx)

        # evaluate net
        if step_idx % EVALUATE_EVERY_STEP == 0:
            win_ratio = evaluate(net, bestNet, rounds=EVALUATION_ROUNDS, device=device)
            print("Net evaluated, win ratio = %.2f" % win_ratio)
            writer.add_scalar("eval_win_ratio", win_ratio, step_idx)
            if win_ratio > BEST_NET_WIN_RATIO:
                print("Net is better than cur best, sync")
                bestNet.load_state_dict(net.state_dict())
                best_idx += 1
                model = 'RL' + str(best_idx)
                torch.save(net, dirSave + '/go_' + model + '.pt')
                with open(dirSave + '/go_' + model + '.txt', "w") as outfile:
                    outfile.write("\n".join(["_".join([str(a[0]), str(a[1]), str(a[2]), str(a[3])])
                                             for a in replay_buffer]))
                ###showWeights(model)
                test(model, printNurSummary=True)
                mcts_store.clear()
    tUsed = round(time.time() - t)
    stdUsed = tUsed // 3600
    minUsed = (tUsed - 3600*stdUsed) // 60
    secUsed = (tUsed - 3600*stdUsed) % 60
    print('Verbrauchte Zeit für trainMCTS: %02d:%02d:%02d' % (stdUsed, minUsed, secUsed))
    tUsed = round(tUsed / MAX_STEP)
    minUsed = tUsed // 60
    secUsed = tUsed % 60
    print('                      pro Step:    %02d:%02d' % (minUsed, secUsed))
    return best_idx

def showWeights(model): # alter Stand
    # Überprüfung vanishing gradients
    net = torch.load(dirSave+'/go_'+model+'.pt')
    for key, module in net._modules.items():
        if key == 'policy' or key == 'value':
            break
        for param in module.parameters():
            weights = param.data.detach().numpy()
            for map in range(6):
                for input in range(2):
                    for row in range(3):
                        for col in range(3):
                            if abs(weights[map][input][row][col]) < 0.001:
                                print('Weight vanishing in conv at map '+str(map+1)+', input '
                                      +str(input+1)+', row/col '+str(row+1)+str(col+1))
            break

def predict(b7, model, mitPrint = False):
    net = torch.load(dirSave+'/go_'+model+'.pt', map_location=torch.device('cpu'))
    p, _ = net(torch.FloatTensor([b7]))
    p = p.detach().numpy()[0]
    board, _, __ = gameGo.b7To3(b7)
    b_reshape = np.asarray(board).reshape(1, gameGo.size2)[0]
    z = np.zeros(1, dtype=int)
    b = np.concatenate((b_reshape, z))
    maxInd = [-1] *  gameGo.ANZ_POSITIONS
    for i in range(gameGo.ANZ_POSITIONS):
        maxInd[i] = np.argmax(p)
        p[maxInd[i]] = -100
    for i in range(gameGo.ANZ_POSITIONS):
        if b[maxInd[i]] == 0:
            if i > 0 and mitPrint:
                print('nur '+str(i+1)+'-beste Prediction')
            return maxInd[i]
    print('Board ist voll!')

def test(model, printNurSummary=False):
    def testBoard(b, b1, b2, whoMoves, soll):
        # return Erfolg 1/0
        y_pred = predict(gameGo.b12To7(b, b1, b2, whoMoves), model=model, mitPrint=not printNurSummary)
        if not printNurSummary:
            gameGo.printBrett(b)
            print('Nächster Zug sollte sein: ', end='')
            for i in range(len(soll)-1):
                print(str(soll[i])+', ',end='')
            print(str(soll[-1]))
            print('NnGo predicts: ' + str(y_pred))
        if y_pred in soll:
            return 1
        else:
            return 0

    if not printNurSummary:
        print('\nPredictions:')

    erfolg = 0
    erfolg += testBoard([[0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0,  1,  0,  1,  0,  0,  0,  0,  0],
                         [0,  1,  1,  1,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                        [[0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0,  1,  0,  1,  0,  0,  0,  0,  0],
                         [0,  1,  0,  1,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                        [[0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  1,  0,  0,  0,  0,  0,  0],
                         [0,  1,  0,  1,  0,  0,  0,  0,  0],
                         [0,  1,  0,  1,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                         0, [38])
    erfolg += testBoard([[0,  1, -1, -1,  1,  0,  0,  0,  0],
                         [0,  1, -1,  0,  1,  0,  0,  0,  0],
                         [0,  0,  1, -1,  1,  0,  0,  0,  0],
                         [0,  0,  1,  1,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                        [[0,  1, -1, -1,  1,  0,  0,  0,  0],
                         [0,  1, -1,  0,  1,  0,  0,  0,  0],
                         [0,  0,  1,  0,  1,  0,  0,  0,  0],
                         [0,  0,  1,  1,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                        [[0,  1, -1, -1,  1,  0,  0,  0,  0],
                         [0,  1, -1,  0,  1,  0,  0,  0,  0],
                         [0,  0,  0,  0,  1,  0,  0,  0,  0],
                         [0,  0,  1,  1,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0, -1,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                         1, [12])
    erfolg += testBoard([[0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  1,  0,  0,  0,  0],
                         [1,  1,  1,  0,  1,  0,  0,  0,  0],
                         [0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0,  1,  0,  1,  0,  0,  0,  0,  0],
                         [0,  1,  1,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                        [[0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  1,  0,  0,  0,  0],
                         [1,  1,  1,  0,  1,  0,  0,  0,  0],
                         [0, -1, -1,  0,  0,  0,  0,  0,  0],
                         [0,  1,  0,  1,  0,  0,  0,  0,  0],
                         [0,  1,  1,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                        [[0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  1,  0,  0,  0,  0],
                         [1,  1,  0,  0,  1,  0,  0,  0,  0],
                         [0, -1, -1,  0,  0,  0,  0,  0,  0],
                         [0,  1,  0,  1,  0,  0,  0,  0,  0],
                         [0,  1,  1,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                         1, [21])
    erfolg += testBoard([[0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  1,  0,  0,  0,  0],
                         [1,  1,  1,  0,  1,  0,  0,  0,  0],
                         [0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0,  1,  1,  1,  0,  0,  0,  0,  0],
                         [0,  1,  1,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                        [[0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  1,  0,  0,  0,  0],
                         [1,  1,  1,  0,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0,  1,  1,  1,  0,  0,  0,  0,  0],
                         [0,  1,  1,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                        [[0, -1, -1, -1,  0,  0,  0,  0,  0],
                         [0, -1, -1, -1,  1,  0,  0,  0,  0],
                         [1,  1,  1,  0,  0,  0,  0,  0,  0],
                         [0, -1, -1,  0,  0,  0,  0,  0,  0],
                         [0,  1,  1,  1,  0,  0,  0,  0,  0],
                         [0,  1,  1,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0],
                         [0,  0,  0,  0,  0,  0,  0,  0,  0]],
                         0, [21])
    anzTests = 4
    print('Testerfolg = '+str(erfolg)+' von '+str(anzTests))

def main():
    net = NnGo()
    net = net.float()
    device = torch.device(DEVICE)
#    jnSL = input(' Training mit SL? (J), Use SL (U) oder nicht (N) : ')
    jnSL = sys.argv[1]
    if jnSL == 'J':
        trainSL(net)
        model = 'SL'
    else:
        if jnSL == 'U':
            net = torch.load(dirSave+'/go_SL.pt')
            net = net.to(device)
#        jnRL = input(' Training mit MCTS? (J, Fortsetzen mit i (RLi), oder N) : ')
        jnRL = sys.argv[2]
        if jnRL == 'J':
            nr = trainMCTS(net, 0, device)
            if nr == 0:
                print('RL hat Net nicht verbessert :-(')
                sys.exit()
            model = 'RL'+str(nr)
        elif jnRL == 'N':
            nr = input('welches existierende RL Modell soll predicted werden, oder 0 für SL: ')
            if nr == '0':
                model = 'SL'
            else:
                model = 'RL' + nr
        else:
            net = torch.load(dirSave+'/go_RL'+jnRL+'.pt')
            net = net.to(device)
            nr = trainMCTS(net, int(jnRL), device)
            if nr-int(jnRL) == 0:
                print('RL hat Net nicht verbessert :-(')
                sys.exit()
            model = 'RL'+str(nr)

    test(model)
    ### showWeights(model)

if __name__ == '__main__':
    main()
