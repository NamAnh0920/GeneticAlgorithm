import numpy as np

class Chrom:
    constant = None

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
        if Chrom.constant is not None:
            self.arr = np.where(Chrom.constant == 1, 1, self.arr)

    def set_constant(constant: list = None):
        Chrom.constant = constant

    def mutate(self) -> 'Chrom':
        '''
        Desctiption: method will mutate the chrom
        Output:
            a new Chrom with new array list
        '''
        mask = np.random.randint(0, 2, size = (self.length))
        new_chrom = np.where(mask == 1, np.random.randint(0, self.value), self.arr)
        if self.constant is not None:
            self.arr = np.where(self.constant == 1, 1, new_chrom)
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
        return chrom.score == self.score
    
    def __lt__(self, chrom: 'Chrom') -> bool:
        '''
        Desctiption: method to compare a Chrom equals to another Chrom
        '''
        return chrom.score > self.score
    
    def __gt__(self, chrom: 'Chrom') -> bool:
        '''
        Desctiption: method to compare a Chrom equals to another Chrom
        '''
        return chrom.score < self.score
    
    def __le__(self, chrom: 'Chrom') -> bool:
        '''
        Desctiption: method to compare a Chrom equals to another Chrom
        '''
        return chrom.score >= self.score
    
    def __ge__(self, chrom: 'Chrom') -> bool:
        '''
        Desctiption: method to compare a Chrom equals to another Chrom
        '''
        return chrom.score <= self.score
    
    def __hash__(self) -> int:
        '''
        Desctiption: method to hash the Chrom
        '''
        return int("".join(map(str, self.arr)))
    
    def __str__(self):
        return "".join(map(str, self.arr))
    
if __name__ == "__main__":
    x = Chrom(2, 4)
    y = Chrom(2, 4)
    z = Chrom(2, 4)

    # x.set_score(10)
    # y.set_score(9)
    # z.set_score(100)

    # a = [x, y, z]

    # print(np.argsort(a))

    print(x)
    print(y)
    print(z)