import matplotlib.pyplot as plt
import pygame
import pickle
import random
import copy
import math
import neat
import time
import os

pygame.font.init()

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.txt')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

visual = False
fitness_values = []


def get_random_angle(angle_limit):
    return random.choice([math.radians(random.randrange(-angle_limit, -1)), math.radians(random.randrange(1, angle_limit))])


class Game:
    def __init__(self):
        self.width = 700
        self.height = 500
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pong Game")

        self.score = [None,
                      [0, 0],
                      [0, 0]
                      ]

        self.player_width = 20
        self.player_height = 100
        self.player_speed = 6

        self.player = [None,
                       [self.width - 2 - self.player_width, self.height // 2 - self.player_height // 2],
                       [2, self.height // 2 - self.player_height // 2]
                       ]

        self.ball_y = self.height // 2
        self.ball_x = self.width // 2

        self.ball_max_speed = 5
        self.ball_bonus_speed = 1
        self.ball_radius = 7
        self.ball_move_dir = 1
        if random.random() > 0.5:
            self.ball_move_dir = -1

        self.ball_move_angle = get_random_angle(30)
        self.ball_vel_x = self.ball_move_dir * abs(math.cos(self.ball_move_angle) * self.ball_max_speed)
        self.ball_vel_y = math.sin(self.ball_move_angle) * self.ball_max_speed

        self.font = pygame.font.SysFont("arial", 50)

    def draw(self):
        self.window.fill((0, 0, 0))

        pygame.draw.rect(self.window, (255, 255, 255), (self.width // 2 - 2, 0, 4, self.height))

        pygame.draw.rect(self.window, (255, 255, 255), (self.player[-1][0], self.player[-1][1], self.player_width, self.player_height))
        pygame.draw.rect(self.window, (255, 255, 255), (self.player[1][0], self.player[1][1], self.player_width, self.player_height))

        pygame.draw.circle(self.window, (255, 255, 255), (self.ball_x, self.ball_y), self.ball_radius)

    def reset_ball(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_move_angle = get_random_angle(30)
        self.ball_vel_x = self.ball_move_dir * abs(math.cos(self.ball_move_angle) * self.ball_max_speed)
        self.ball_vel_y = math.sin(self.ball_move_angle) * self.ball_max_speed

    def reset(self):
        self.score = [None,
                      [0, 0],
                      [0, 0]
                      ]

        self.player = [None,
                       [self.width - 2 - self.player_width, self.height // 2 - self.player_height // 2],
                       [2, self.height // 2 - self.player_height // 2]
                       ]

        self.reset_ball()

    def move_paddle(self, player, direction):
        self.player[player][1] += (self.player_speed * direction)
        if self.player[player][1] < 0:
            self.player[player][1] = 0
        elif self.player[player][1] + self.player_height > self.height:
            self.player[player][1] = self.height - self.player_height

    def move_ball(self):
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

    def check_collision(self):
        if self.ball_y - self.ball_radius <= 0:
            self.ball_vel_y *= -1
            self.ball_y = 0 + self.ball_radius
        elif self.ball_y + self.ball_radius >= self.height:
            self.ball_vel_y *= -1
            self.ball_y = self.height - self.ball_radius

        cur_dir = -1
        if self.ball_vel_x > 0:
            cur_dir = 1

        if self.player[cur_dir][1] <= self.ball_y <= self.player[cur_dir][1] + self.player_height + 5:
            if not (self.player[-1][0] + self.player_width <= self.ball_x + (self.ball_radius * cur_dir) <= self.player[1][0]):
                self.ball_move_dir *= -1
                if random.random() > 0.75:
                    self.ball_bonus_speed = random.uniform(0.75, 1.5)
                self.ball_vel_x = abs(self.ball_vel_x * self.ball_bonus_speed)
                self.ball_vel_x = 5 if self.ball_vel_x < 5 else 13 if self.ball_vel_x > 13 else self.ball_vel_x
                self.ball_vel_x *= self.ball_move_dir
                self.ball_vel_y = -1 * (((self.player[cur_dir][1] + self.player_height / 2) - self.ball_y) / self.ball_max_speed)
                self.score[cur_dir][0] += 1

    def loop(self):
        self.move_ball()
        self.check_collision()

        if self.ball_x < 0:
            self.reset()
            self.score[1][1] += 1
        elif self.ball_x > self.width:
            self.reset()
            self.score[-1][1] += 1

        return copy.deepcopy(self.score)


class Simulation:
    def __init__(self):
        self.game = Game()
        self.font = pygame.font.SysFont("arial", 50)

    def play(self, net):
        clock = pygame.time.Clock()
        while True:
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            output = net.activate((self.game.player[1][1],
                                   abs(self.game.player[1][0] - self.game.ball_x),
                                   self.game.ball_y,
                                   self.game.ball_vel_x,
                                   self.game.ball_vel_y
                                   ))
            move = output.index(max(output))
            self.game.move_paddle(1, -1 if move == 1 else 1)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(-1, -1)
            elif keys[pygame.K_s]:
                self.game.move_paddle(-1, 1)

            self.game.loop()
            self.game.draw()
            pygame.display.update()

    def play_ai(self, nets):
        clock = pygame.time.Clock()
        while True:
            clock.tick(120)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            for i in [-1, 1]:
                output = nets[i].activate((self.game.player[i][1],
                                           abs(self.game.player[i][0] - self.game.ball_x),
                                           self.game.ball_y,
                                           self.game.ball_vel_x,
                                           self.game.ball_vel_y
                                           ))
                move = output.index(max(output))
                self.game.move_paddle(i, -1 if move == 1 else 1)

            self.game.loop()
            self.game.draw()
            pygame.display.update()

    def train(self, genomes, is_visual=False):
        time_s = time.time()
        net = [None,
               neat.nn.FeedForwardNetwork.create(genomes[-1], config),
               neat.nn.FeedForwardNetwork.create(genomes[1], config)
               ]

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            score = self.game.loop()

            for i in [-1, 1]:
                output = net[i].activate((self.game.player[i][1],
                                          abs(self.game.player[i][0] - self.game.ball_x),
                                          self.game.ball_y,
                                          self.game.ball_vel_x,
                                          self.game.ball_vel_y
                                          ))
                move = output.index(max(output))
                self.game.move_paddle(i, -1 if move == 1 else 1)

            if is_visual:
                self.game.draw()

            pygame.display.update()

            game_time = time.time() - time_s
            if score[-1][1] == 1 or score[1][1] == 1 or score[1][0] >= 50:
                score_left = score[-1][0] + game_time
                if score_left < 200:
                    genomes[-1].fitness += score_left
                score_right = score[1][0] + game_time
                if score_right < 200:
                    genomes[1].fitness += score_right
                break

        return False


def plot_fitness():
    plt.plot(fitness_values, 'o-')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness Progress')
    plt.grid(True)
    plt.show()


def evaluate(genomes, _):
    best_fitness = -1
    for _, genome in genomes:
        genome.fitness = 0

    for index, (_, genome_1) in enumerate(genomes):
        print("\r", "{" + ("=" * index) + (" " * (len(genomes)-index)) + "}", end="")
        for _, genome_2 in genomes[min(index + 1, len(genomes) - 1):]:
            sim = Simulation()
            if sim.train([None, genome_2, genome_1], is_visual=visual):
                quit()
        if genome_1.fitness > best_fitness:
            best_fitness = genome_1.fitness
    fitness_values.append(best_fitness)
    plot_fitness()


def neat_train(limit, checkpoint_step=1, load_index=None, is_visual=False):
    global visual
    visual = is_visual

    if load_index is None:
        population = neat.Population(config)
    else:
        population = neat.Checkpointer.restore_checkpoint("neat-checkpoint-" + str(load_index))

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(checkpoint_step))

    for i in range(limit):
        best = population.run(evaluate, 1)
        with open("ais/best-gen-" + str(i) + ".pickle", "wb") as f:
            pickle.dump(best, f)


def neat_play_ai(ai_1, ai_2):
    with open(ai_1, "rb") as f:
        ai_1 = pickle.load(f)

    with open(ai_2, "rb") as g:
        ai_2 = pickle.load(g)

    ai_1_net = neat.nn.FeedForwardNetwork.create(ai_1, config)
    ai_2_net = neat.nn.FeedForwardNetwork.create(ai_2, config)

    sim = Simulation()
    sim.play_ai([None, ai_1_net, ai_2_net])


def neat_play_human(ai):
    with open(ai, "rb") as f:
        ai = pickle.load(f)
    ai_net = neat.nn.FeedForwardNetwork.create(ai, config)

    sim = Simulation()
    sim.play(ai_net)


# neat_train(10, checkpoint_step=1, load_index=44, is_visual=True)
# neat_play_human("ais/best-gen-44.pickle")
neat_play_ai("ais/best-gen-31.pickle", "ais/best-gen-44.pickle")
# 6, 18, 31, 44
