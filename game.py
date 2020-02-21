import pygame
import numpy as np
import time

from car import Car
from control import get_trajectory, filter_output

kDraw = False
kSave = True

class Game:
    def __init__(self):
        # pygame.init()
        pygame.display.set_caption("Car tutorial")
        self.width = 1024*3
        self.height = 1024*3
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    def run(self):

        ppu = 32
        T = 200

        # Initialize agents
        N_a = np.random.randint(3, 15)
        agents = [0] * N_a
        agents_ctrl = [0] * N_a
        # Set start point
        for i in range(N_a):
            start_collision = True
            while (start_collision):
                start_collision = False
                x1, y1 = 60*np.random.rand(), 60*np.random.rand()
                for j in range(i):
                    dist = np.sqrt((x1 - agents[j].position[0])**2 + (y1 - agents[j].position[1])**2)
                    if (dist < 8.0):
                        start_collision = True
                        break
                agents[i] = Car(x1, y1)
        # Set goal point
        for i in range(N_a):
            start_collision = True
            while (start_collision):
                start_collision = False
                x1, y1 = 60*np.random.rand(), 60*np.random.rand()
                for j in range(i):
                    dist = np.sqrt((x1 - agents[j].goal[0])**2 + (y1 - agents[j].goal[1])**2)
                    if (dist < 8.0):
                        start_collision = True
                        break
                agents[i].goal = np.array([x1, y1])

        # Collect Data
        data = []
        data_u = []

        # Set barrier for each agent
        horizon_set=[0, 0, 0, 6, 7, 8]
        agents_avoid = [False] * N_a
        for i in range(1, N_a):
            agents[i].Ds = horizon_set[np.random.randint(len(horizon_set))]
            # agents_avoid[i] = horizon_set[np.random.randint(len(horizon_set))]

        for i in range(T):
        # while not self.exit:
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True


            start_time = time.time()
            u, x_path, _ = get_trajectory(agents[0], N=10)
            u  = filter_output(0, agents, x_path)
            agents_ctrl[0] = u
            # print("Solve time: " + str(time.time() - start_time))
            for j in range(1, N_a):
                u2, x2_path, _ = get_trajectory(agents[j])
                if (agents[j].Ds > 0):
                    u2 = filter_output(j, agents, x2_path)
                agents_ctrl[j] = u2

            # Add noise to agents' actions
            noise_a = 0.1
            for j in range(N_a):
                agents[j].update(agents_ctrl[j] + noise_a *
                                 (np.random.rand(2) - 0.5))
            
            # Collect data
            states = np.zeros(4*N_a)
            for i in range(len(agents)):
                states[4*i + 0] = agents[i].position[0]
                states[4*i + 1] = agents[i].position[1]
                states[4*i + 2] = agents[i].velocity[0]
                states[4*i + 3] = agents[i].velocity[1]
            data.append(states)
            data_u.append(u)

            # Drawing
            if (kDraw):
                self.screen.fill((0, 0, 0))

                agent1_img = pygame.image.load('agents.jpg')
                rect = agent1_img.get_rect()
                self.screen.blit(agent1_img, agents[0].position *
                                ppu - (rect.width / 2, rect.height / 2))

                agent1_goal = pygame.image.load('agents.jpg')
                rect = agent1_goal.get_rect()
                self.screen.blit(agent1_goal, agents[0].goal * ppu -
                                (rect.width / 2, rect.height / 2))

                for j in range(1, N_a):
                    agent2_img = pygame.image.load('user.PNG')
                    rect = agent2_img.get_rect()
                    self.screen.blit(
                        agent2_img, agents[j].position * ppu - (rect.width / 2, rect.height / 2))

                pygame.display.flip()
            # time.sleep(0.1)
            self.clock.tick(self.ticks)
        # pygame.quit()
        return data, data_u


if __name__ == '__main__':
    game = Game()
    print("Game Initialized")
    data = []
    data_u = []
    trials = 500
    start_time = time.time()
    for i in range(trials):
        dat_trial, dat_u_trial = game.run()
        data.append(dat_trial)
        data_u.append(dat_u_trial)
        # data.append(np.array(dat_trial))
        print("Trial " + str(i) + " finished in " + str(round(time.time() - start_time, 1)) + " sec")
        if (i % 5 == 0 and kSave):
            np.save('train_data_i6.npy', data)
            np.save('train_data_u_i6.npy', data_u)

    # data = np.load('testnp.npy', allow_pickle=True)