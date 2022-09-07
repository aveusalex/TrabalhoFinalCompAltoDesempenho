from RNA import NeuralNetwork
from ga import GA
import numpy as np
from time import time


def fitness(pesos, net):
    losses = []
    # recebemos toda a populacao e calculamos o fitness de cada individuo
    for individuo in pesos:
        net.weights = individuo  # passando o cromossomo para a rede neural
        net.pass_weights_to_net()  # passando os pesos para a rede neural (transformando o cromossomo em pesos)
        losses.append(net.run())

    losses = np.array(losses)
    return losses


neural_net = NeuralNetwork()
ga = GA(pop_size=10, chrom_size=9550, cross_rate=0.8, mutation_rate=0.5, max_iter=1000, mutation_estrategy="gaussian",
        fitness_func=fitness, net=neural_net)
ga.run()
