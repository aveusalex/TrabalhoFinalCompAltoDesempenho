from RNA import NeuralNetwork
from ga import GA
import numpy as np
import multiprocessing as mp


def fitness(pesos, net):
    losses = []
    # recebemos toda a populacao e calculamos o fitness de cada individuo
    for individuo in pesos:
        net.weights = individuo  # passando o cromossomo para a rede neural
        net.pass_weights_to_net()  # passando os pesos para a rede neural (transformando o cromossomo em pesos)
        losses.append(net.run())

    losses = np.array(losses)
    return losses


def main(args):
    core, pop = args
    neural_net = NeuralNetwork()
    ga = GA(pop_size=pop, chrom_size=9550, cross_rate=0.8, mutation_rate=0.25, max_iter=1000, mutation_estrategy="gaussian",
            fitness_func=fitness, net=neural_net)
    ga.run(core)


if __name__ == '__main__':
    n_jobs = mp.cpu_count()  # aqui é o lugar que se determina o número de núcleos a executar o processamento.
    pop_size = 1000
    pop_size = int(np.ceil(pop_size / n_jobs))
    pool = mp.Pool(n_jobs)
    args = [[core, pop_size] for core in range(n_jobs)]
    pool.map(main, args)
