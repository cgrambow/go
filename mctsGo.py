#!/usr/kai/anaconda3/python
# -*- coding: utf-8 -*-
# V1: setzt auf auf T3 V5
# V2: testing
# V3: find_leaf: spiel initialisierung nur einmal! return value not result. Diriclet nur auf innere 5*5
#       zugNr Par zu mcts search
# V4: zugMax
# V5: GPU
# V6: Performance: b, b1 und neues Int statt b7
# V7: find Leaf value Korrektur
#
"""
Monte-Carlo Tree Search
"""
import math as m
import numpy as np

import torch
import torch.nn.functional as F

import gameGo, goSpielNoGraph

class MCTS:
    """
    Class keeps statistics for every state encountered during the search
    """
    def __init__(self):
        self.stateStats = gameGo.BoardValues()

    def clear(self):
        self.stateStats.clear()

    def __len__(self):
        return self.stateStats.__len__()

    def boardPrint(self):
        self.stateStats.boardPrint()

    def is_leaf(self, s):
        return s not in self.stateStats.b

    def find_leaf(self, s, player, zugNr, zugMax):
        """
        Traverse the tree until the end of game or leaf node
        :param s: root node state
        :param player: player to move
        :param zugNr: bei welchem Zug eingesetzt wird
        :param zugMax: max. Anzahl Züge im Spiel bevor ausgezählt wird
        :return: tuple of (value, leaf_state, player, states, actions)
        1. value: None if leaf node, otherwise equals to the game outcome for the player at leaf
        2. leaf_state: state of the last state
        3. player: player at the leaf node
        4. states: list of states traversed
        5. list of actions taken
        """
        states = []
        actions = []
        cur_state = s
        cur_player = player
        spiel = goSpielNoGraph.PlayGo(gameGo.intToB(cur_state), cur_player=cur_player,
                                      vonBeginn=False, zugMax=zugMax-zugNr)
        value = None
        while not self.is_leaf(cur_state):
            # not leaf
            states.append(cur_state)
            stats = self.stateStats.b[cur_state]
            counts = stats[0]
            values = stats[1]
            values = [value/count if count > 0 else 0 for value, count in zip(values, counts)]
            probs = stats[2]
            total_sqrt = m.sqrt(sum(counts))
            # choose action to take, in the root node add the Dirichlet noise to the probs
            if cur_state == s:
                if zugNr < 20:
                    alpha = [0.03] * 25
                    noises = np.random.dirichlet(alpha)
                    n = np.zeros(2 * gameGo.size)
                    for i in range(5):
                        n = np.hstack((n, np.zeros(2), noises[i*5:(i+1)*5], np.zeros(2)))
                    n = np.hstack((n, np.zeros(2*gameGo.size+1)))
                elif zugNr < 40:
                    alpha = [0.03] * 49
                    noises = np.random.dirichlet(alpha)
                    n = np.zeros(gameGo.size)
                    for i in range(7):
                        n = np.hstack((n, np.zeros(1), noises[i*7:(i+1)*7], np.zeros(1)))
                    n = np.hstack((n, np.zeros(gameGo.size+1)))
                else:
                    alpha = [0.03] * gameGo.size2
                    noises = np.random.dirichlet(alpha)
                    n = np.hstack((noises, np.zeros(1)))
                probs = [0.75 * prob + 0.25 * noise for prob, noise in zip(probs, n)]
            score = [val + prob * total_sqrt / (1 + count)
                     for val, prob, count in zip(values, probs, counts)]
            # possible_moves: Feld nicht besetzt. Und dann iterativ ausprobieren
            for i in range(gameGo.size2):
                if spiel.b[i//gameGo.size][i%gameGo.size]['farbe'] != 0:
                    score[i] = -np.inf
            action = int(np.argmax(score))
            while not spiel.setzZug(action):    # hier move: setzen eines Zuges
                score[action] = -np.inf
                action = int(np.argmax(score))
            actions.append(action)
            if spiel.spielBeendet:
                value = spiel.gewinner*(1-2*cur_player) # value aus Sicht des Gegners, siehe backup)
                break
            cur_player = 1 - cur_player
            cur_state = spiel.bToInt()
        return value, cur_state, cur_player, states, actions

    def search(self, count, batch_size, s, player, net, zugNr, zugMax, device):
        # return: Anzahl von find_leaf calls, die Spiel bis Ende führten
        countEnd = 0
        if batch_size > 0:
            for _ in range(count):
                countEndMini = self.search_minibatch(batch_size, s, player, net, zugNr, zugMax, device)
                countEnd += countEndMini
        else:
            for _ in range(count):
                value, leaf_state, leaf_player, states, actions = self.find_leaf(s, player, zugNr, zugMax)
                if value is None:
                    # expand mit leaf_state, leaf_player, states, actions
                    batch_v = gameGo.state_lists_to_batch([gameGo.intToB(leaf_state)], [leaf_player], device)
                    logits_v, value_v = net(batch_v)
                    probs_v = F.softmax(logits_v, dim=1)
                    probs = probs_v.detach().cpu().numpy()[0]
                    value = value_v.data.cpu().numpy()[0][0]
                    # create the node
                    self.stateStats.expand(leaf_state, probs)
                else:
                    countEnd += 1
                    print('Leaf bis Spielende.')
                    cv = -value
                    cp = leaf_player
                    for state, action in zip(states[::-1], actions[::-1]):
                        print('backup mit action: ', action, 'player: ', cp, ' value: ', cv, ' bei:')
                        cv = -cv
                        cp = 1-cp
                        gameGo.printBrett(gameGo.intToB(state)[0])
                # backup mit value, states, actions
                # leaf state not stored in states + actions, so the value of the leaf will be the value of the opponent
                cur_value = -value
                for state, action in zip(states[::-1], actions[::-1]):
                    self.stateStats.backup(state, action, cur_value)
                    cur_value = -cur_value
        return countEnd

    def search_minibatch(self, count, s, player, net, zugNr, zugMax, device):
        """
        Perform several MCTS searches.
        """
        # return: Anzahl von find_leaf calls, die Spiel bis Ende führten
        countEnd = 0
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        for _ in range(count):
            value, leaf_state, leaf_player, states, actions = self.find_leaf(s, player, zugNr, zugMax)
            if value is not None:
                countEnd += 1
                backup_queue.append((value, states, actions))
            else:
                found = False
                for item in expand_states:
                    if item == leaf_state:
                        found = True
                        break
                if not found:
                    expand_states.append(gameGo.intToB(leaf_state))
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state, states, actions))
        # expansion of nodes
        if expand_queue:
            batch_v = gameGo.state_lists_to_batch(expand_states, expand_players, device)
            logits_v, values_v = net(batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            values = values_v.data.cpu().numpy()[:,0]
            probs = probs_v.detach().cpu().numpy()
            # create the nodes
            for (leaf_state, states, actions), value, prob in zip(expand_queue, values, probs):
                self.stateStats.expand(leaf_state, prob)
                backup_queue.append((value, states, actions))
        # backup mit value, states, actions
        for value, states, actions in backup_queue:
            # leaf state not stored in states + actions, so the value of the leaf will be the value of the opponent
            cur_value = -value
            for state, action in zip(states[::-1], actions[::-1]):
                self.stateStats.backup(state, action, cur_value)
                cur_value = -cur_value
        return countEnd

    def get_policy(self, s, tau=1):
        """
        Extract policy by the state
        :param state_int: state of the board
        :return: probs
        """
        counts = self.stateStats.b[s][0]
        if tau == 0:
            probs = [0.0] * gameGo.ANZ_POSITIONS
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count ** (1.0 / tau) for count in counts]
            total = sum(counts)
            if total == 0:  ### sollte NICHT passieren
                print('mcts.get_policy mit sum(count)=0, bei:')
                b2 = gameGo.intToB(s)
                gameGo.printBrett(b2[0])
                probs = self.stateStats.b[s][2]
            else:
                probs = [count / total for count in counts]
        return probs