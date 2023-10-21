import numpy as np

class Chrom:
    def __init__(self, value: int, length: int, arr: list = None) -> None:
        '''
        Desctiption: class that represent a chrom
        Input:
            value: number of distinct value in the chrom
            length: length of the chrom
            arr (optional): if arr is given, the Chrom will not generate random value at initialize
        '''
        self.value = value
        self.length = length
        self.score = None
        if arr is not None:
            self.arr = arr
        else:
            self.arr = np.array(np.random.randint(0, value, size=(length)))

    def mutate(self) -> 'Chrom':
        '''
        Desctiption: method will mutate the chrom
        Output:
            a new Chrom with new array list
        '''
        mask = np.random.randint(0, 2, size = (self.length))
        new_chrom = np.where(mask == 1, np.random.randint(0, self.value), self.arr)
        return Chrom(self.value, self.length, new_chrom)
    
    def crossover(self, chrom: list) -> 'list[Chrom]':
        '''
        Desctiption: method will crossover the chrom with another list
        Input:
            chrom: a list that the chrom will crossover with
        Output:
            list of 2 new Chroms
        '''
        p = np.random.randint(0, self.length)
        new_chrom1 = np.concatenate((chrom[:p], self.arr[p:]))
        new_chrom2 = np.concatenate((self.arr[:p], chrom[p:]))

        chrom1 = Chrom(self.value, self.length, new_chrom1)
        chrom2 = Chrom(self.value, self.length, new_chrom2)

        return [chrom1, chrom2]

    def set_score(self, s: float) -> None:
        '''
        Desctiption: method will set the score of the chrom on its performance
        Input:
            s: the score of the chrom, the greater the better
        '''
        self.score = s

    def get_score(self) -> float:
        '''
        Desctiption: method will get the score of the chrom
        Output:
            the score of the chrom, the greater the better
        '''
        return self.score

    def evaluate(self, evalutation: 'function') -> None:
        '''
        Desctiption: method evaluate the Chrom on its performance
                     Since once the chrom performed, we can store the score
                     and retrive it instead of compute again
        Input:
            evaluation: the function that we will use to evaluate
        Output:
            the score of the performance
        '''
        if self.score is None:
            score = evalutation(self.arr)
            self.set_score(score)
        return self.get_score()
    
    def get_length(self):
        '''
        Desctiption: method get the length of the Chrom
        Output:
            Length of the Chrom
        '''
        return self.length
    
    def __eq__(self, chrom: 'Chrom') -> bool:
        '''
        Desctiption: method to compare a Chrom equals to another Chrom
        '''
        return all(np.equal(chrom.arr, self.arr))
    
    def __hash__(self) -> int:
        '''
        Desctiption: method to hash the Chrom
        '''
        return int("".join(map(str, self.arr)))

class Population:
    def __init__(self, N: int, value: int, length: int, 
                 mutate: int, crossover: int, score: 'function') -> None:

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
        self.population_num = N
        self.mutate = mutate
        self.crossover = crossover
        self.max_score = 0
        self.score = score

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

    def filter(self) -> 'Population':
        '''
        Description: method to eliminate the poor performed chrom 
        '''
        evaluation = [chrom.evaluate(self.score) for chrom in self.population]
        self.max_score = max(evaluation)
        keep = np.argsort(evaluation)[-self.population_num:]
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

        self.population = list(set(self.population))
        self.filter()
        self.view() if robust else None
        return self

    def evolve_multiple_step(self, epoch: int, robust = False) -> 'Population':
        '''
        Description: method for multiple steps evolution: create mutation + crossover => filter
        Input:
            epoch: number of steps
        '''
        self.view() if robust else None
        for _ in range(epoch):
            self.evolve(robust = robust)
        return self

    def evolve_convergence(self, condition: 'function', callback: int = None, robust = False, **kwargs) -> int:
        '''
        Description: method to find the converge number of epoch
        Input:
            condition: function to see if the population converged
            callback: number of epoch to stop if the score didn't change
            **kwargs: any parameter the condition function need
        Output:
            epoch: number of step to converge
        '''
        self.view() if robust else None
        epoch = 0
        record = self.max_score
        while not condition(self.max_score, **kwargs):
            epoch += 1
            self.evolve()
            if callback and epoch%callback == 0 and record >= self.max_score:
                break
            elif epoch%callback == 0:
                record = self.max_score
        return epoch

    def view(self) -> None:
        '''
        Description: method to see the population content
        '''
        for chrom in self.population:
            print(chrom.arr)
        print('-'*10)


if __name__ == '__main__':
    import itertools
    import time
    import pandas as pd
    from tqdm import tqdm

    def parameter_testing(parameter):

        seed_list = [4444, 12345, 7777, 141414, 2831, 2679, 6583, 8023, 4812, 67410]

        result = {
            'population': [],
            'n_mutation': [],
            'n_crossover': [],
            'length': [],
            'callback': [],
            'time': [],
            'epoch': [],
            'max_score': [],
        }

        def ceiling(score, x):
            return score/x == 1.0

        for N, n_mutation, n_crossover, l, callback in tqdm(itertools.product(*parameter.values()), desc='progress'):
            epoch_list = []
            time_list = []
            score_list = []
            target = np.array([0, 1, 2, 3]*(l//4) + [0, 1, 2, 3][:l%4])
            score = lambda chrom: np.sum(np.equal(chrom, target))
            for seed in seed_list:
                np.random.seed(seed)
                population = Population(
                    N = N, value = 4, length = l, 
                    mutate = n_mutation, crossover = n_crossover, score = score
                    )
                start_time = time.time()
                epoch = population.evolve_convergence(condition = ceiling, callback = callback, robust = False, x = l)
                end_time = time.time()
                epoch_list.append(epoch)
                time_list.append(end_time-start_time)
                score_list.append(population.get_score())
            result['population'].append(N)
            result['n_mutation'].append(n_mutation)
            result['n_crossover'].append(n_crossover)
            result['length'].append(l)
            result['callback'].append(callback)
            result['max_score'].append(np.mean(score_list)/l)
            result['epoch'].append(np.mean(epoch_list))
            result['time'].append(np.mean(time_list))

        return pd.DataFrame(result)

    parameter = {
        'N': [25, 50, 75, 100, 200],
        'mutation': [25, 50, 75, 100, 200], 
        'crossover': [25, 50, 75, 100, 200],
        'l': [20, 30, 40, 50, 60],
        'callback': [25, 50, 75, 100]
    }

    table = parameter_testing(parameter)
    table.to_csv(f"./summary_{'_'.join(map(str, parameter['N']))}.csv", index=False)


