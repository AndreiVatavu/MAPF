import itertools
import multiprocessing
import pickle
import time
from random import choice
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mapf import Agent, CBS, print_solution


def generate_random_agents(height: int, width: int, num_agents: int) -> List[Agent]:
    cell_list = list(itertools.product(range(height), range(width)))
    agents = []
    for ag_id in range(num_agents):
        start = choice(cell_list)
        cell_list.remove(start)
        goal = choice(cell_list)
        cell_list.remove(goal)
        agents.append(Agent(ag_id, start, goal))
    return agents


def run_cbs(num_agents, queue):
    agents = generate_random_agents(8, 8, num_agents)
    cbs = CBS(8, 8, agents)
    start = time.time()
    solution = cbs.high_level()
    duration = time.time() - start
    duration *= 1000
    if not solution:
        queue.put((False, duration, cbs.ll_nodes, cbs.hl_nodes))
    else:
        queue.put((True, duration, cbs.ll_nodes, cbs.hl_nodes))


if __name__ == '__main__':
    NUM_INSTANCES = 100
    results = []
    for num_agents in range(3, 22):
        print(f"Num agents {num_agents}")
        count = 0
        run_times = []
        ll_nodes = []
        hl_nodes = []
        for inst_id in range(NUM_INSTANCES):
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=run_cbs, args=(num_agents, queue))
            p.start()
            # Wait for 5 minutes or until CBS finishes
            p.join(300)
            if p.is_alive():
                p.terminate()
            else:
                count += 1
                finish, run_time, ll_node, hl_node = queue.get()
                if not finish:
                    print("aici")
                run_times.append(run_time)
                ll_nodes.append(ll_node)
                hl_nodes.append(hl_node)
        run_time = np.mean(run_times)
        ll_count = np.mean(ll_nodes)
        hl_count = np.mean(hl_nodes)
        success_rate = count / NUM_INSTANCES * 100
        print(f"Success rate: {success_rate} | Run time: {run_time} | LL Count : {ll_count} | HL Count: {hl_count}")
        results.append([num_agents, success_rate, ll_count, hl_count, run_time])

    file_to_store = open("checkout_results.pickle", "wb")
    pickle.dump(results, file_to_store)
    file_to_store.close()

    success_rate_list = [success_rate for _, success_rate, _, _, _ in results]
    plt.plot(list(range(3, 18)), success_rate_list, label=f"CBS")
    plt.xlabel('Number of agents')
    plt.ylabel('Success rate')
    plt.title(f'Success rate vs number of agents')
    plt.legend()
    plt.savefig(f'success_rate.png')
    plt.clf()

    collabel = ("Number of agents", "success rate", "Low level nodes", "High level nodes", "Run time")
    data = {}
    for index in range(len(collabel)):
        data[collabel[index]] = [result[index] for result in results]
    df = pd.DataFrame(data)
    df.update(df[["Low level nodes", "High level nodes", "Run time"]].applymap('{:,.2f}'.format))
    plt.axis('off')
    plt.title("Nodes generated and running time on 8 × 8 grid", weight='bold')
    the_table = plt.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.savefig(f'table.png', dpi=300)
    plt.clf()

