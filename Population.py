from collections import deque
from Chrom import Chrom
import numpy as np

class Population:
    def __init__(self, N: int, value: int, length: int, 
                 mutate: int, crossover: int, score: 'function', duplicate = False) -> None:

        '''
        Descrition: The Population class that contain multiple Chrom
        Input:
            N: the number of instance in the population
            value: number of distinct value in the chrom
            length: length of the chrom
            mutate: number of mutation in the population
            crossover: number of crossover in the population
            score: the evaluation funciton
        '''
        self.population = [Chrom(value = value, length = length) for _ in range(N)]
        self.population_num= N
        self.mutate = mutate
        self.crossover = crossover
        self.max_score = float('-inf')
        self.score = score
        self.duplicate = duplicate

    def set_evaluation(self, score: 'function') -> 'Population':
        '''
        Description: method to set the evaluation function to score
        Input:
            score: new evaluation function
        '''
        return self
    
    def get_score(self) -> float:
        '''
        Description: method to get the maximum performance in the population
        '''
        return self.max_score
        
    def evaluate(self):
        pass

    def filter(self):
        '''
        Description: method to eliminate the poor performed chrom 
        '''
        '''
        might write evaluate as a function of population instead
        '''
        self.evaluate()
        keep = np.argsort(self.population)[-self.population_num:]
        population = [self.population[i] for i in keep]
        self.population = population
        self.population_num = len(population)
        return self

    def evolve(self, robust = False) -> 'Population':
        '''
        Description: method for 1 step evolution: create mutation + crossover => filter
        '''

        for _ in range(self.mutate):
            chrom = self.population[np.random.randint(0, self.population_num)]
            new_chrom = chrom.mutate()
            self.population.append(new_chrom)

        for _ in range(self.crossover):
            idx1 = np.random.randint(0, self.population_num)
            idx2 = np.random.randint(0, self.population_num)

            while idx1 == idx2:
                idx1 = np.random.randint(0, self.population_num)
                idx2 = np.random.randint(0, self.population_num)

            chrom_list = self.population[idx1].crossover(self.population[idx2].arr)
            self.population += chrom_list

        # self.population = list(set(self.population)) if not self.duplicate else self.population
        self.filter()
        self.view() if robust else None
        return self

    def evolve_convergence(self, callback: int = None, robust = False) -> int:
        '''
        Description: method to find the converge number of epoch
        Input:
            condition: function to see if the population converged
            callback: number of epoch to stop if the score didn't change
        Output:
            epoch: number of step to converge
        '''
        self.view() if robust else None
        epoch = 0
        record = deque()
        while self.max_score < 0:
            print(epoch)
            epoch += 1
            self.evolve()
            if len(record) < callback:
                record.append(self.max_score)
            else:
                s = record.popleft()
                if self.max_score <= s:
                    return epoch
                record.append(self.max_score)
        return epoch

    def view(self) -> None:
        '''
        Description: method to see the population content
        '''
        for chrom in self.population:
            print(chrom.arr)
        print('-'*10)