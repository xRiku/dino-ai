import pygame
import os
import random
import time
import math
from sklearn.neighbors import KNeighborsClassifier
from sys import exit

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed, obType):
        pass


def first(x):
    return x[0]


class KeySimplestClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state

    def keySelector(self, distance, obHeight, game_speed, obType):
        self.state = sorted(self.state, key=first)
        for s, d in self.state:
            if game_speed < s:
                limDist = d
                break
        if distance <= limDist:
            if isinstance(obType, Bird) and obHeight > 50:
                return "K_DOWN"
            else:
                return "K_UP"
        return "K_NO"


class KeyKNNClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state
        self.model = KNeighborsClassifier(n_neighbors=5)
        X = [s[:-1] for s in self.state]
        y = [s[-1] for s in self.state]
        self.model.fit(X, y)

    def keySelector(self, distance, obHeight, game_speed, obType):
        prediction = self.model.predict([[game_speed, distance]])

        if prediction == "K_UP":
            if isinstance(obType, Bird) and obHeight > 50:
                return "K_DOWN"
        
        return prediction
        


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0
    spawn_dist = 0

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        SCREEN.fill((255, 255, 255))

        distance = 1500
        obHeight = 0
        obType = 2
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            print(f'Distance {distance}, obHeight {obHeight}, game_speed {game_speed}, obType {obType}')
            userInput = aiPlayer.keySelector(distance, obHeight, game_speed, obType)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)
        player.draw(SCREEN)

        for obstacle in list(obstacles):
            obstacle.update()
            obstacle.draw(SCREEN)

        background()

        cloud.draw(SCREEN)
        cloud.update()

        score()

        # clock.tick(10)
        pygame.display.update()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(2000)
                death_count += 1
                return points


# Change State Operator

def change_state(state, position, vs, vd):
    aux = state.copy()
    # s, d = state[position]
    s, d, l = state[position]
    s = float(s)
    d = float(d)
    ns = s + vs
    nd = d + vd
    if ns < 15 or nd > 1000:
        return []
    # return aux[:position] + [(ns, nd)] + aux[position + 1:]
    # print(aux)
    # print(type(aux))
    # print(type(position))
    # print(aux[:position] )
    return aux[:position] + [[ns, nd, l]] + aux[position + 1:]

# Neighborhood

def generate_neighborhood(state):
    neighborhood = []
    state_size = len(state)
    for i in range(state_size):
        ds = random.randint(1, 10) 
        dd = random.randint(1, 100) 
        new_states = [change_state(state, i, ds, 0), change_state(state, i, (-ds), 0), change_state(state, i, 0, dd),
                      change_state(state, i, 0, (-dd))]
        for s in new_states:
            if s != []:
                neighborhood.append(s)
    return neighborhood


# Gradiente Ascent

def gradient_ascent(state, max_time):
    start = time.process_time()
    res, max_value = manyPlaysResults(3)
    better = True
    end = 0
    while better and end - start <= max_time:
        neighborhood = generate_neighborhood(state)
        better = False
        for s in neighborhood:
            aiPlayer = KeySimplestClassifier(s)
            res, value = manyPlaysResults(3)
            if value > max_value:
                state = s
                max_value = value
                better = True
        end = time.process_time()
    return state, max_value


def change_probability(value,best_value,t):
    p = 1/(math.exp(1)**((best_value-value)/t))
    r = random.uniform(0,1)
    if r < p:
        return True
    else:
        return False

def random_state(states):
    index = random.randint(0,len(states)-1)
    return states[index]


# Simulated Anealing

def simulated_annealing(state,t,alfa,iter_max,max_time):
    solution = state
    res, max_value = manyPlaysResults(2)
    start = time.process_time()
    end = 0
    
    while t >= 1 and end-start <= max_time:        
        
        for _ in range(iter_max):    
            neighborhood = generate_neighborhood(state)
            if neighborhood == []:
                return solution,max_value               
            aux = random_state(neighborhood)
            # auxAiPlayer = KeySimplestClassifier(aux)
            auxAiPlayer = KeyKNNClassifier(aux)
            res, aux_value = manyPlaysResults(2)
            # aiPlayer = KeySimplestClassifier(state)
            aiPlayer = KeyKNNClassifier(state)
            res, state_value = manyPlaysResults(2)
            if aux_value > state_value:
                state = aux
                if aux_value > max_value:
                    solution = aux
                    max_value = aux_value
            else:
                if change_probability(aux_value,state_value,t):
                    state = aux
        t = t*alfa
        end = time.process_time()

    return solution, max_value

from scipy import stats
import numpy as np


def manyPlaysResults(rounds):
    results = []
    for round in range(rounds):
        results += [playGame()]
    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())


def main():
    global aiPlayer

    # initial_state = [(15, 250), (18, 350), (20, 450), (1000, 550)]
    # aiPlayer = KeySimplestClassifier(initial_state)
    # best_state, best_value = gradient_ascent(initial_state, 5000) 
    # aiPlayer = KeySimplestClassifier(initial_state)
    initial_state = [
        [10, 1500, 'K_NO'],
        [12, 200, 'K_NO'],
        [12, 150, 'K_UP'],
        [13, 50, 'K_UP'],
        [12, 50, 'K_UP'],
        [11, 50, 'K_UP'],
        [15, 100, 'K_UP'],
        [13, 100, 'K_NO'],
        [15, 500, 'K_NO'],
        [17, 1500, 'K_NO'],
        [19, 400, 'K_NO'],
        [19, 250, 'K_UP'],
        [21, 700, 'K_NO'],
        [21, 500, 'K_UP'],
        [21, 300, 'K_NO'],
        [21, 200, 'K_UP'],
        [21, 100, 'K_NO'],
        [21, 50, 'K_UP'],
        [21, 0,'K_DOWN'],
        [21, -50, 'K_DOWN'],
     ]
    initial_state = [[10, 1500, 'K_NO'], [12, 200, 'K_NO'], [12, 150, 'K_UP'], [13, 50, 'K_UP'], [12, 50, 'K_UP'], [11, 50, 'K_UP'], [15, 100, 'K_UP'], [13, 100, 'K_NO'], [15.0, 579.0, 'K_NO'], [17, 1500, 'K_NO'], [19, 400, 'K_NO'], [19, 250, 'K_UP'], [21, 700, 'K_NO'], [21.0, 452.0, 'K_UP'], [21, 300, 'K_NO'], [21, 200, 'K_UP'], [21, 100, 'K_NO'], [21, 50, 'K_UP'], [21, 0, 'K_DOWN'], [21, -50, 'K_DOWN']]
    # initial_state = np.array(initial_state, dtype=object)
    aiPlayer = KeyKNNClassifier(initial_state)
    best_state, best_value = simulated_annealing(initial_state,200,0.1,10,120) 
    aiPlayer = KeyKNNClassifier(best_state)
    res, value = manyPlaysResults(2)
    print(f'Best State:\n{best_state}')
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)


main()
