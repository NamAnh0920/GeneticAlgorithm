import numpy as np
import pandas as pd
import process_data
import regression
import os
from Population import Population
import pickle

RECORD = './track_chrom'

class GeneChoice(Population):
    def __init__(self, args, N: int, mutate: int, crossover: int, duplicate = False, one_hot = True, remove_nas = True):
        
        attributes, labels, self.all_categorical = process_data.read_data(remove_nas, args.outcome)

        if one_hot:
            attributes, self.categorical_column_map = process_data.create_all_onehot(attributes, self.all_categorical)

        self.train_X, self.train_y, self.test_X, self.test_y, _ = process_data.split_data(attributes, labels, args.week, args.lag, args.window, args.prior_weeks)        
        self.args = args
        length = self.train_X.shape[1]

        self.record_file = RECORD + '_' + args.model + '_' + str(args.seed) + '.pkl'

        self.seen = {}
        self.seenkey = set()

        if os.path.exists(self.record_file):
            with open(self.record_file, 'rb') as file:
                self.seen = pickle.load(file)
                self.seenkey = set(self.seen.keys())
        else:
            open(self.record_file, 'x')
        
        process_data.scale_test_train(self.train_X, self.test_X, self.all_categorical)

        np.random.seed(args.seed) if args.seed else None

        distinct = 4

        '''
        set constant formula
        '''

        super(GeneChoice, self).__init__(N, distinct, length, mutate, crossover, duplicate)

    def evaluate(self):
        for chrom in self.population:
            if chrom.get_score():
                continue
            h = str(chrom)
            if h in self.seenkey:
                performance = self.seen[h]
                chrom.set_score(performance)
                if self.max_score < performance:
                    self.max_score = performance
                    self.candidate = chrom
                continue

            arr = chrom.arr

            train_X = process_data.process_chromosome(self.train_X, arr)
            test_X = process_data.process_chromosome(self.test_X, arr)

            process_data.scale_test_train(train_X, test_X, self.all_categorical)

            model, _ = regression.create_model(self.args, self.args.seed, train_X, self.train_y)
            y_hat = model.predict(test_X)
            performance = regression.calculate_mse(y_hat, self.test_y)

            chrom.set_score(-performance)

            self.seen[h] = -performance
            self.seenkey.add(h)

            if self.max_score < -performance:
                self.max_score = -performance
                self.candidate = chrom
    
    def __str__(self):
        for chrom in self.population:
            print(chrom)
        return "-"*10
    
    def __del__(self):
        with open(self.record_file, 'wb') as file:
            pickle.dump(self.seen, file)
            print('record save!')
    
def parameter_testing(parameter):
    import itertools
    import time

    class Args:
        def __init__(self, model, seed):
            self.week = 20
            self.lag = 1
            self.window = 0
            self.prior_weeks = 10
            self.model = model
            self.k = 5
            self.outcome = "cases_inc"
            self.seed = seed

    seed_list = [4444, 12345, 7777, 141414, 2831, 2679, 6583, 8023, 4812, 67410]

    result = {
        'population': [],
        'n_mutation': [],
        'n_crossover': [],
        'model': [],
        'callback': [],
        'time': [],
        'epoch': [],
        'max_score': [],
        'candidate': [],
    }

    for N, n_mutation, n_crossover, callback, model in itertools.product(*parameter.values()):
        print("Testing: ", N, n_mutation, n_crossover, callback)
        epoch_list = []
        time_list = []
        score_list = []
        result['population'].append(N)
        result['n_mutation'].append(n_mutation)
        result['n_crossover'].append(n_crossover)
        result['callback'].append(callback)
        result['model'].append(model)
        for seed in seed_list:
            args = Args(model, seed)
            population = GeneChoice(args, N = N, mutate = n_mutation, crossover = n_crossover)
            start_time = time.time()
            epoch = population.evolve_convergence(callback = callback, robust = False)
            end_time = time.time()
            epoch_list.append(epoch)
            time_list.append(end_time-start_time)
            score_list.append(population.get_score())
        result['max_score'].append(population.max_score)
        result['epoch'].append(np.mean(epoch_list))
        result['time'].append(np.mean(time_list))
        result['candidate'].append(str(population.candidate))

    return pd.DataFrame(result)

    
if __name__ == '__main__':
    parameter = {
        'N': [2],
        'mutation': [0, 1], 
        'crossover': [0, 1],
        'callback': [2],
        'model': ['linear']
    }

    table = parameter_testing(parameter)
    table.to_csv(f"./summary_{'_'.join(map(str, parameter['N']))}.csv", index=False)

