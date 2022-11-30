from random import randint, choice, uniform, sample, shuffle

try:
    import numpy as np
    import matplotlib.pyplot as plt

    matplot_installed = True
except ImportError:
    matplot_installed = False


class GA():
    def __init__(self, evaluator, bounds=None, num_genes=None, init=None, steps=100, stop_fitness=None,
                 stagnation=None, population_limit=20, survive_coef=0.25, productivity=4, cross_type="uniq_split",
                 mutate_genes=1, cata_mutate_genes=2, mutagen="1_step", cata_mutagen="1_step", verbose=True):
        assert type(bounds) is list or (type(bounds) is tuple and type(num_genes) is int)
        self.evaluator = evaluator
        self.init = init
        self.steps = steps
        self.stop_fitness = stop_fitness
        self.stagnation = stagnation
        self.population_limit = population_limit
        self.survive_coef = survive_coef
        self.productivity = productivity
        self.cross_type = cross_type
        self.mutate_genes = mutate_genes
        self.cata_mutate_genes = cata_mutate_genes
        self.mutagen = mutagen
        self.cata_mutagen = cata_mutagen
        self.verbose = verbose

        self.best = []  # История лучших
        self.fitness = []  # сохраняем рейтинги

        default_step = 0.01
        default_bounds = (-100, 100)

        if type(bounds) is list:
            self.bounds = bounds
        elif type(bounds) is tuple and num_genes:
            try:
                self.bounds = [(bounds[0], bounds[1], bounds[2])] * num_genes
            except IndexError:
                self.bounds = [(bounds[0], bounds[1], default_step)] * num_genes
        elif not bounds:
            self.bounds = self.gen_bounds(default_bounds[0], default_bounds[1],
                                          default_step, num_genes)

    def evolve(self, steps=None, newborns=None):
        if steps:
            self.steps = steps
        if type(newborns) != list:
            newborns = []  # новорожденные без фитнеса
        if self.init:
            newborns.append(self.init)
        best_ever = None
        for i in range(self.steps):
            population = self.generate_population(newborns)  # популяция с фитнесом
            survivors = self.survive(population)
            newborns = self.crossover(survivors)

            self.best.append(survivors[0])
            self.fitness.append([i[1] for i in population])
            if not best_ever:
                self.best_ever = self.best[-1]
            else:
                self.best_ever = max(best_ever, self.best[-1], key=lambda i: i[1])

            if self.verbose:
                print("- Step {:d} / {:d} results: best: {:.3f}".
                      format(i + 1, self.steps, self.best_ever[1]))

            # условие катаклизма
            if self.stagnation:
                best_fitness = [i[1] for i in self.best[-self.stagnation:]]
                if len(best_fitness) == self.stagnation and len(set(best_fitness)) == 1:
                    newborns = self.cataclysm(population)

            # условия досрочного завершения
            if self.stop_fitness != None and self.best_ever[1] >= self.stop_fitness:
                if self.verbose >= 1:
                    print("- Evolution completed: best fitness = {:.3f} <= {:.3f}".format(self.best_ever[1],
                                                                                          self.stop_fitness))
                break

        if self.verbose >= 1:
            print("Best: {} - {}".format(self.best_ever[1], self.best_ever[0]))
        return self.best_ever[0]

    def generate_population(self, newborns):
        population = []
        # добавляем мутации новорожденным
        for indiv in newborns:
            indiv = self.mutate(indiv, self.mutagen, self.mutate_genes)
            fitness = self.evaluator(indiv)
            population.append((indiv, fitness))

        # создаем случайных особей, если есть места в популяции
        for _ in range(self.population_limit - len(newborns)):
            indiv = []
            if "random" in self.mutagen or "change" in self.mutagen:
                for bounds in self.bounds:
                    gene = uniform(bounds[0], bounds[1])
                    indiv.append(gene)
            elif "step" in self.mutagen:
                for bounds in self.bounds:
                    step = bounds[2]
                    gene = choice(range(bounds[0], bounds[1] + step, step))
                    indiv.append(gene)
            elif "swap" in self.mutagen:
                bounds = self.bounds[0]
                step = bounds[2]
                indiv = range(bounds[0], bounds[1] + step, step)
                shuffle(indiv)
            fitness = self.evaluator(indiv)
            population.append((indiv, fitness))
            newborns.append(indiv)
        return population

    def survive(self, population):
        num_survivors = int(self.population_limit * self.survive_coef)
        best = sorted(population, key=lambda i: -i[1])[:num_survivors]
        return best

    def crossover(self, best):
        newborns = []
        for _ in range(len(best) * self.productivity):
            dad, mom = sample(best, 2)
            dad, mom = dad[0], mom[0]  # только геном без фитнеса
            child = []
            if self.cross_type == "uniq_split":
                split = len(dad) // 2
                child = dad[:split] + mom[split:]
                bounds = self.bounds[0]
                step = bounds[2]
                for i, gene in enumerate(child):
                    # если ген с таким значением уже есть, генерируем случайное значение
                    if gene in child[:i]:
                        while True:
                            gene = choice(range(bounds[0], bounds[1] + step, step))
                            if gene not in child:
                                child[i] = gene
                                break

            newborns.append(child)
        return newborns

    def mutate(self, indiv, mutagen, mutate_genes=None):
        if mutagen == "1_step":
            gene_ids = [randint(0, len(indiv) - 1) for _ in range(mutate_genes)]
            for gene_id in gene_ids:
                gene_id = randint(0, len(indiv) - 1)
                while True:  # TODO: сделать, чтобы цикл не был бесконечным
                    step = self.bounds[gene_id][2]
                    step = choice([-step, step])
                    if self.bounds[gene_id][0] <= indiv[gene_id] + step <= self.bounds[gene_id][1]:
                        indiv[gene_id] += step
                        break
        return indiv

    def cataclysm(self, population):
        post_population = []
        for indiv, _fitness in population:
            post_population.append(self.mutate(indiv, self.cata_mutagen, self.cata_mutate_genes))
        if self.verbose >= 1:
            print("- Cataclysm occured because of stagnation {} steps: {} ({} genes)".
                  format(self.stagnation, self.cata_mutagen, self.cata_mutate_genes))
        return post_population


def example_diophante(x):
    a, b, c, d, e = x
    z = a + 2 * b + 3 * c + 4 * d + e
    ans = 663
    print("a={:3.0f} b={:3.0f} c={:3.0f} d={:3.0f} e={:3.0f} z={:3.0f}".format(a, b, c, d, e, z),
          "- Solved!" if z == ans else "")
    return -abs(ans - z)


if __name__ == '__main__':
    ga = GA(example_diophante, bounds=(20, 100, 1), num_genes=5, steps=40, stop_fitness=0, stagnation=3,
            population_limit=10, survive_coef=0.2, productivity=4, mutagen="1_step", cata_mutagen="full_step")
    result = ga.evolve()
    print("Best solution:", result)