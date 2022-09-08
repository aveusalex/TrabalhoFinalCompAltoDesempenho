import numpy as np
import time


class GA:
    def __init__(self, pop_size, chrom_size, cross_rate, mutation_rate, max_iter, selection_estrategy="mi+lambda",
                 crossover_estrategy="uniforme", mutation_estrategy="bit_flip", fitness_func=None, net=None):
        # selection_estrategy pode ser "mi+lambda" ou "mi,lambda"
        # crossover_estrategy pode ser "one_point" ou "two_point" ou "uniforme"
        # mutation_estrategy pode ser "bit_flip" ou "gaussian"
        self.pop_size = pop_size
        self.chrom_size = chrom_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.selection_estrategy = selection_estrategy
        self.crossover_estrategy = crossover_estrategy
        self.mutation_estrategy = mutation_estrategy

        self.porcentagem_sucesso = 0.2
        ############################ Parametros manuais
        self.sigma = 1      # o quanto varia a mutacao gaussiana
        self.alcance = 1  # o alcance dos valores dos genes
        ############################
        self.history = []
        if not fitness_func:
            self.fitness = self.fitness_default
        else:
            self.fitness = fitness_func
        self.net = net

    def init_pop(self):
        pop = (np.random.rand(self.pop_size, self.chrom_size) - 0.5) * 2 * self.alcance  # estende os valores para [-alcance, alcance]
        return pop

    def fitness_default(self, pop):
        fitness = np.sum(pop, axis=1)
        return fitness

    def select(self, pop, fitness):
        # a funcao choice permite que tenhamos os indices dos individuos fornecidos para amostragem e associamos
        # a probabilidade de ser escolhido de acordo com seu fitness. Ou seja, implementacao da roleta.
        idx = np.random.choice(np.arange(len(pop)), size=self.pop_size, replace=False,
                               p=fitness / fitness.sum())
        return pop[idx], fitness[idx]

    def crossover(self, pop, fitness):
        # aqui onde dois genes sao cruzados para gerar dois novos genes
        qtd_filhos = self.pop_size * 7
        filhos = []
        while len(filhos) < qtd_filhos:
            transoes = np.random.choice(np.arange(len(pop)), size=2, replace=False, p=fitness / fitness.sum())
            pai1 = pop[transoes[0]]
            pai2 = pop[transoes[1]]
            filho1 = pai1.copy()
            filho2 = pai2.copy()

            if np.random.rand() < self.cross_rate:
                if self.selection_estrategy == "mi+lambda":
                    filhos.append(pai1)
                    filhos.append(pai2)

                if self.crossover_estrategy == "uniforme":
                    cross_points = np.random.randint(0, 2, size=self.chrom_size).astype(bool)

                    filho1[cross_points] = pai2[cross_points]
                    filho2[cross_points] = pai1[cross_points]

                elif self.crossover_estrategy == "one_point":
                    cross_point = np.random.randint(0, self.chrom_size)

                    filho1[cross_point:] = pai2[cross_point:]
                    filho2[cross_point:] = pai1[cross_point:]

                elif self.crossover_estrategy == "two_point":
                    cross_point1 = np.random.randint(0, self.chrom_size)
                    cross_point2 = np.random.randint(0, self.chrom_size)

                    filho1[cross_point1:cross_point2] = pai2[cross_point1:cross_point2]
                    filho2[cross_point1:cross_point2] = pai1[cross_point1:cross_point2]

            filhos.append(filho1)
            filhos.append(filho2)

        filhos = np.array(filhos)
        return filhos

    def mutate(self, pop):
        if self.net:
            fitness_0 = 1 / self.fitness(pop, self.net)
        else:
            fitness_0 = self.fitness(pop)

        # atualizando o sigma
        if self.porcentagem_sucesso > 0.12:
            self.sigma = self.sigma / 0.9
        elif self.porcentagem_sucesso < 0.12:
            self.sigma = self.sigma * 0.9

        for i in range(len(pop)):
            if np.random.rand() < self.mutation_rate:
                if self.mutation_estrategy == "bit_flip":
                    mutation_point = np.random.randint(0, self.chrom_size)
                    pop[i, mutation_point] = 1 if pop[i, mutation_point] == 0 else 0

                elif self.mutation_estrategy == "gaussian":
                    pop[i] = pop[i] + np.random.normal(0, self.sigma, self.chrom_size)

        if self.net:
            fitness_1 = 1 / self.fitness(pop, self.net)
        else:
            fitness_1 = self.fitness(pop)

        self.porcentagem_sucesso = self.sucesso(fitness_1, fitness_0)
        return pop, fitness_1

    def sucesso(self, fitness_1, fitness_0):
        diffs = fitness_1 - fitness_0
        bools = np.where(diffs > 0, 1, 0)
        # a sacada daqui foi verificar em quais lugares os fitness_1 foram maiores que os fitness_0 e verificar a porcentagem disso
        return np.sum(bools) / len(bools)

    def evolve(self, pop, fitness):
        fitness = 1/fitness
        pop = self.crossover(pop, fitness)  # cruzamento, o fitness dos cross são avaliados no inicio da funcao mutate
        pop, fitness = self.mutate(pop)  # já retorna o fitness dos cromossomos mutados

        pop, fitness = self.select(pop, fitness)
        return pop, fitness  # ja retorna a populacao escolhida dessa geracao e seus fitness

    def run(self, core):
        print(f"Iniciando a evolucao core - {core}")
        pop = self.init_pop()
        start = time.time()
        fitness = self.fitness(pop, self.net)  # calculo do primeiro fitness para a pop recem iniciada
        for i in range(self.max_iter):
            pop, fitness = self.evolve(pop, fitness)
            self.history.append(fitness)
            print('Core: {}, iter: {}, actual fitness: {:.6f}, best_fitness: {:.6f}'.format(core, i, float(fitness.min()), float(
                min([min(x) for x in self.history]))), end=", ")
            print("p_success: {:.6f}, tempo_exec: {:2f},  sigma: {}, pop: {}".format(self.porcentagem_sucesso, time.time() - start, self.sigma, self.pop_size))
            start = time.time()


if __name__ == '__main__':
    ga = GA(pop_size=1000, chrom_size=100, cross_rate=0.8, mutation_rate=0.5, max_iter=1000)
    ga.run()
