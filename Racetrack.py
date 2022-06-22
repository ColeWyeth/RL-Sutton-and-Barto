from itertools import product
import random
import turtle
import time

# class State:
#     def __init__(self, x, y, vel_x, vel_y):
#         self.x = x
#         self.y = y
#         self.vel_x = vel_x
#         self.vel_y = vel_y 


def main():
    episodes = 50000
    eps = 0.2
    payoffs = []

    sidelength = 20
    scale = 10 # for visualizations
    
    def inTrackBounds(x,y):
        return (x < sidelength//2 or y > sidelength//2) and y < sidelength

    def crossedFinishLine(x,y):
        """Assuming the track is large compared to the maximum velocity,
           there is no way to cut out of bounds and across the finish line.
        """
        return x >= sidelength and y < sidelength

    print("Calculating States")
    S = set()
    for x, y in product(range(sidelength), range(sidelength)):
        if inTrackBounds(x,y):
            for vel_x, vel_y in product(range(5), range(5)):
                if (not y == 0) and (vel_x == 0 and vel_y == 0):
                    continue
                #S.append(State(x,y,vel_x,vel_y))
                S.add((x,y,vel_x,vel_y))

    print("Calculating Actions")   
    A = dict()
    for s in S:
        actions = []
        for acc_x, acc_y in product((-1,0,1),(-1,0,1)):
            x, y, vel_x, vel_y = s
            # check if the resulting velocity is valid for this position
            if (vel_x + acc_x > 0 or vel_y + acc_y > 0) and (x,y,vel_x+acc_x,vel_y+acc_y) in S:
                actions.append((acc_x, acc_y))
        A[s] = actions

    print("Initializing state/action values")
    Q = dict()
    for s in S:
        for a in A[s]:
            Q[(s,a)] = -10

    print("Initializing weight normalization")
    C = dict()
    for s in S:
        for a in A[s]:
            C[(s,a)] = 0

    def getRandomStart():
        starts = []
        for x in range(sidelength):
            if inTrackBounds(x,0):
                starts.append((x,0))
        x, y = random.choice(starts)
        vel_x, vel_y = 0, 0
        s = (x, y, vel_x, vel_y)
        return s

    def generateEpisode(Q, eps, noiseOn, visualize=False):

        """Together, Q and eps specify the behavior policy.
           Returns a history of tuples S,A,R and b(A|S)
           R is always -1 but is included to match pseudocode
        """
        history = []

        s = getRandomStart()

        if visualize:
            turtle.pendown()
            turtle.tracer(1, 8)
            turtle.showturtle()
            turtle.color(random.choice(['red', 'blue', 'green']) )
            x,y,_,_ = s
            turtle.penup()
            turtle.goto(10*x, 10*y)
            turtle.pendown()

        while True:
            x, y, vel_x, vel_y = s

            # Calculate S_t, A_t, and b(A_t|S_t)
            actions = A[s]
            maxVal = -float('inf')
            bestA = None
            for option in actions:
                if Q[(s, option)] > maxVal:
                    maxVal = Q[(s,option)]
                    bestA = option
            if random.random() < eps:
                a = random.choice(actions)
            else:
                a = bestA
            if a == bestA:
                bAS = 1 - eps + eps/len(actions)
            else:
                bAS = eps/len(actions)
            history.append((s, a, -1, bAS))

            # Calculate the next state
            if random.random() > 0.1 or not noiseOn:
                vel_x, vel_y = vel_x + a[0], vel_y + a[1]
            x, y = x + vel_x, y + vel_y 
            if crossedFinishLine(x,y):
                if visualize:
                    turtle.speed(int((vel_x**2 + vel_y**2)**0.5))
                    turtle.goto(scale*x,scale*y)
                break
            elif not inTrackBounds(x,y):
                s = getRandomStart()
            else:
                s = (x, y, vel_x, vel_y)
            if visualize:
                    turtle.speed(int((vel_x**2 + vel_y**2)**0.5))
                    turtle.goto(scale*x,scale*y)
        return history 
            
    print("Running episodes")
    for i in range(episodes):
        history = generateEpisode(Q, eps, True)
        G = 0
        W = 1
        for t in reversed(range(len(history))):
            s, a, R, bAS = history[t]
            G += R
            C[(s,a)] += W
            Q[(s,a)] += (W/C[(s,a)])*(G - Q[(s,a)])
            endOfTail = False
            for option in A[s]:
                if Q[(s,option)] > Q[(s,a)]:
                    endOfTail = True
            if endOfTail:
                break
            W /= bAS 
        payoffs.append(-len(history))
        if i%1000 == 0:
            print("episode " + str(i) + ", " + "average payoff: " + str(sum(payoffs)/len(payoffs)))

    # Draw racetrack
    t = turtle.Turtle()
    turtle.tracer(sidelength, 1)
    t.speed(0)
    t.hideturtle()
    for x, y in product(range(sidelength), range(sidelength)):
        if inTrackBounds(x,y):
            t.penup()
            t.setpos(scale*x, scale*y)
            t.pendown()
            for i in range(4):
                t.forward(scale)
                t.left(90)
    turtle.update()

    screen = turtle.getscreen()
    screen.onscreenclick(lambda x,y: generateEpisode(Q,0,False,visualize=True))
    screen.mainloop()    

if __name__ == "__main__":
    main()