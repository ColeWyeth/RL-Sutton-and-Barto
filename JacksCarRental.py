import random
import math
import itertools

def poisson(n, lam):
    return (lam**n)/(math.factorial(n))*math.exp(-lam)

def get_S():
    return itertools.product(range(21), range(21))

def main():
    # 1: Initialization

    discount = 0.9

    # State value function
    V = dict()
    for i,j in get_S():
        V[(i,j)] = random.random()

    # A(s)
    A = dict()
    for i,j in get_S():
        # We can move at most 5 cars
        # A positive number represents movement from 1st to second location
        # It's clearly suboptimal to move cars in both directions
        A[(i,j)] = range(-min(5, j), min(5, i)+1)


    def bootstrapEstimate(s, a, Vals):
        i, j = s
        cost = -2*abs(a)

        # this does not depend on requests/returns
        val_sum = cost

        # cars at each location in the morning
        cars_1, cars_2 = i-a, j+a

        # probabilities drop to about 1% outside of this range
        n_vals = [i for i in range(10)]

        # it would be much easier/faster to sample the four poisson 
        # distributions for requests/returns at each location, but that
        # would not be a literal implementation of policy iteration...
        for event in itertools.product(n_vals, n_vals, n_vals, n_vals):
            req_1, ret_1, req_2, ret_2 = event
            p = poisson(req_1, 3)*poisson(req_2, 4)*poisson(ret_1, 3)*poisson(ret_2, 2)
            r = 10*(min(req_1, cars_1) + min(req_2, cars_2))

            # cars at the end of the next day
            f_cars_1 = min(cars_1 - min(req_1, cars_1) + ret_1, 20)
            f_cars_2 = min(cars_2 - min(req_2, cars_2) + ret_2, 20)

            val_sum += p*(r + discount*Vals[(f_cars_1, f_cars_2)])
        return val_sum
    
    # Policy
    pi = dict()
    for s in get_S():
        pi[s] = random.choice(A[s])

    policyStable = False
    while not policyStable:
        # Policy Evaluation
        print("Evaluating policy")
        delta = 1000 
        while abs(delta) > 1:
            delta = 0
            for i,j in get_S():
                v = V[(i,j)]
                a = pi[(i,j)]
                V[(i,j)] = bootstrapEstimate((i,j), a, V)
                delta = max(delta, abs(v - V[(i,j)]))
            
            print("Delta: " + str(delta))

        print("V(0,0): %f:" % V[(0,0)])
        print("V(20,20): %f:" % V[(20,20)])
    
        # Policy Improvement
        policyStable = True
        for s in get_S():
            oldAction = pi[s]
            bestVal = 0
            bestA = oldAction
            for a in A[s]:
                estimate = bootstrapEstimate(s, a, V)
                if estimate > bestVal:
                    bestA = a
                    bestVal = estimate
            pi[s] = bestA
            if not bestA == oldAction:
                policyStable = False
        

    print(V)
    print(pi)
    print(pi[(18,3)])

                    
if __name__ == "__main__":
    main() 