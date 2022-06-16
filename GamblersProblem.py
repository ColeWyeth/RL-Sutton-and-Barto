import sys
import random
import matplotlib.pyplot as plt
import numpy as np

def main():
    if len(sys.argv) > 1:
        theta = float(sys.argv[1])
    else:
        theta = 0.00001

    if len(sys.argv) > 2:
        p_h = float(sys.argv[2])
    else:
        p_h = 0.4
    
    V = [random.random() for i in range(101)]
    # The following values are fixed to the correct payoffs for convenience
    V[0] = 0 # Loss
    V[100] = 1 # Win

    while True:
        d = 0 
        for s in range(1,100):
            v = V[s]
            expected_vals = [
                p_h*V[min(s+bet, 100)]+(1-p_h)*V[max(s-bet,0)] for bet in range(1,s+1)
            ]
            V[s] = max(expected_vals)
            d = max(d, abs(v - V[s]))
        if d < theta:
            break

    plt.plot(V)
    plt.title("V(s)")
    plt.ylabel("Value estimates")
    plt.xlabel("Capital")
    plt.show()
    plt.clf()

    # Calculate optimal policy
    pi = [0 for i in range(1,100)]
    for s in range(1,100):
        expected_vals = [
                p_h*V[min(s+bet, 100)]+(1-p_h)*V[max(s-bet,0)] for bet in range(1,s+1)
        ]
        pi[s-1] = np.argmax(expected_vals)+1
    plt.bar(range(1,100), pi)
    plt.ylabel("Final policy (stake)")
    plt.xlabel("Capital")
    plt.show()

if __name__ == "__main__":
    main()