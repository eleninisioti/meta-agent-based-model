import argparse
import networkx as nx
from meta_agent import MetaAgent
from visualize import plot_meta_model
import numpy as np
import os
import yaml
import pickle

def init_structure(shape, n_agents,  n_neighbors=1):
    """ Initializes the social network topology.

    Params
    ------
    shape: str
        the social network topology

    n_agents: int
        number of agents

    """
    if shape == "ring":
        structure = nx.cycle_graph(n_agents)

    elif shape == "fully-connected":
        structure = nx.complete_graph(n_agents)

    elif shape == "small-world":  # small-world
        structure = nx.watts_strogatz_graph(n=n_agents, k=n_neighbors, p=0.2)

    return structure


def evolve(agents):

    for agent in agents:
        agent.simulate()
        agent.compute_fitness()

    for agent in agents:
        agent.update_strategy()

    return agents

def main(args):
    project_dir = "../projects/" + args.project
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    with open(project_dir + '/config.yaml', 'w') as outfile:
        yaml.dump(args, outfile)

    for trial in range(args.trials):
        project_dir = "../projects/" + args.project + "/trial_" + str(trial) + "/plots"
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        # initialize network structure
        structure = init_structure(args.structure, args.num_meta_agents)
        agents = []

        for n in range(args.num_meta_agents):
            agents.append(MetaAgent(args, idx=n, metastrategy=args.metastrategy))

        for i, agent in enumerate(agents):
            neighbor_idxs = list(structure.neighbors(i))
            agent.neighbors = [agents[n] for n in neighbor_idxs]

        for round in range(args.metarounds):
            print("round", round)
            agents = evolve(agents)
            keep_log(agents)

        log["agents"] = agents

        with open(project_dir + '/log.pkl', 'wb') as outfile:
            pickle.dump(log, outfile)


        plot_meta_model(log, project_dir)

def keep_log(agents):
    # average fitness
    log["fitness"].append(np.mean([agent.fitness for agent in agents]))

    # histogram of strategies
    log["strategies"].append( [agent.strategy for agent in agents])

if __name__ == "__main__":
    log = {"fitness": [], "strategies": []}
    parser = argparse.ArgumentParser()

    parser.add_argument('--metastrategy',
                        help='Meta strategy. Choose between rounds and benefit',
                        type=str,
                        default="rounds")

    parser.add_argument('--structure',
                        help='Type of social structure',
                        type=str,
                        default="fully-connected")

    parser.add_argument('--num_meta_agents',
                        help='Number of meta-agents',
                        type=int,
                        default=20)

    parser.add_argument('--project',
                        help='Name of current project',
                        type=str,
                        default="temp")

    parser.add_argument('--order',
                        help='Choose between CDO (combat-diffusion-offspring) and COD ('
                             'combat-offspring-diffusion).',
                        type=str,
                        default="COD")

    parser.add_argument('--game',
                        help='Name of game. Choose between PD and Snow',
                        type=str,
                        default="PD")

    parser.add_argument('--grid_length',
                        help='Length of grid in tiles ',
                        type=int,
                        default=20)

    parser.add_argument('--radius',
                        help='Neighborhood radius ',
                        type=int,
                        default=1)

    parser.add_argument('--cost',
                        help='Cost of cooperation.',
                        type=float,
                        default=0)

    parser.add_argument('--benefit',
                        help='Benefit of cooperation.',
                        type=float,
                        default=10)

    parser.add_argument('--inter_per_round',
                        help='Interactions per round.',
                        type=int,
                        default=8)

    parser.add_argument('--init_coop',
                        help='Initial percentage of cooperators.',
                        type=float,
                        default=0.1)

    parser.add_argument('--prob_move',
                        help='Probability of moving during the day time.',
                        type=float,
                        default=0.1)

    parser.add_argument('--rounds',
                        help='Number of evolutionary rounds.',
                        type=int,
                        default=20)

    parser.add_argument('--metarounds',
                        help='Number of meta rounds.',
                        type=int,
                        default=30)

    parser.add_argument('--trials',
                        help='Number of independent trials.',
                        type=int,
                        default=10)

    parser.add_argument('--day_duration',
                        help='Number of trials a day consists of',
                        type=int,
                        default=5)

    parser.add_argument('--night_duration',
                        help='Number of trials a night consists of',
                        type=int,
                        default=5)

    parser.add_argument('--nagents',
                        help='Number of agents',
                        type=int,
                        default=10)

    parser.add_argument('--well_mixed',
                        help='Number of evolutionary rounds.',
                        default=False,
                        action="store_true")

    parser.add_argument('--bifurcation',
                        help='Whether a bifurcation plot will be made.',
                        default=False,
                        action="store_true")

    parser.add_argument('--eval_movement',
                        help='Evaluate movement during daytime.',
                        default=False,
                        action="store_true")

    parser.add_argument('--move_parametric',
                        help='Evaluate different values of movement.',
                        default=False,
                        action="store_true")

    args = parser.parse_args()
    main(args)