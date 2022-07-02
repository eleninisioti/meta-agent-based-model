import os
from tournament import Tournament
import pickle
from visualize import plot_grid
import yaml
import numpy as np
import random


class MetaAgent:

    def __init__(self, args, idx, metastrategy):
        self.metastrategy = metastrategy
        if self.metastrategy == "rounds":
            self.strategy = 1
        elif self.metastrategy == "benefit":
            self.strategy = 0.2
        self.idx = idx
        self.args = args
        self.agent_project_dir = "../projects/" + args.project + "/agent_" + str(idx)

        if not os.path.exists(self.agent_project_dir + "/plots/grids"):
            os.makedirs(self.agent_project_dir + "/plots/grids")



        self.log_experiments = []
        self.memory = 50

    def simulate(self):
        # initialize project's subdirs
        log_perf = {"coop_perc": []}

        if self.args.metastrategy == "rounds":
            rounds = self.strategy
        elif self.args.metastrategy == "benefit":
            self.args.benefit = self.strategy
            rounds = self.args.rounds

        tournament = Tournament(self.args)
        logs = []

        for round in range(rounds):
            log_round = tournament.play_round()
            plot_grid(strat_transitions=log_round["strat_transitions"], round=round, project=self.agent_project_dir)
            logs.append(log_round)
            pop_log = tournament.pop_log()
            log_perf["coop_perc"].append(pop_log["coop_perc"])

        # ----- final saving for project ------

        with open(self.agent_project_dir + '/log.pickle', 'wb') as pfile:
            pickle.dump(log_perf, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        self.current_experiment = log_round["strat_transitions"]
        if len(self.log_experiments) < self.memory:
            self.log_experiments.append(log_round["strat_transitions"])
        else:
            self.log_experiments = [log_round["strat_transitions"]]

    def compute_fitness(self):
        # novelty
        novelty = 0
        for el in self.log_experiments[:-1]:
            diff = np.sum(el == self.current_experiment)
            novelty += diff
        novelty = novelty/len(self.log_experiments)
        novelty = novelty/(len(self.current_experiment)*len(self.current_experiment[0]))
        # pattern-recognition (static)
        # are there cooperators?
        num_coops = (self.current_experiment == 0).sum()
        num_coops += (self.current_experiment == 2).sum()

        num_coops = num_coops/(len(self.current_experiment)*len(self.current_experiment[0]))
        # are they clustered?
        flattened = self.current_experiment.flatten()
        pos = np.where(flattened == 0)[0]
        coops = np.ndarray.tolist(pos)
        coops.extend(np.ndarray.tolist(np.where(flattened == 2)[0]))
        clusters = []
        for el1 in coops:
            cluster = 0
            for el2 in coops:
                if np.abs(el1 - el2) == 1:
                    cluster += 1

            clusters.append(cluster)
        if len(coops):
            clustering = np.mean([el/len(coops) for el in clusters])
        else:
            clustering = 0
        self.fitness = novelty + num_coops + clustering
        print("agent", self.idx, novelty, num_coops, clustering)

    def update_strategy(self):
        max_fitness = self.fitness
        for neighbor in self.neighbors:
            if neighbor.fitness > max_fitness:
                max_fitness = neighbor.fitness
                self.strategy = neighbor.strategy

        # mutate strategy
        if self.metastrategy == "rounds":
            mutation = random.randint(-1,2)
            self.strategy = np.max([1, np.abs(self.strategy + mutation)])
        elif self.metastrategy == "benefit":
            mutation = np.random.uniform(low=-0.05, high=0.05, size=1)[0]
            self.strategy = np.max([0, np.abs(self.strategy + mutation)])

