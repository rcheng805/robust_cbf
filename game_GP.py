import pygame
import numpy as np
import time

from car import Car
from control import get_trajectory, filter_output
from GP_predict import GP

kDraw = True
kSave = False

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
        agents[0].max_acceleration = 8.0
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

        # Initialize GP
        all_gp = [0] * N_a
        paths = [0] * N_a
        all_gp[0] = GP(None, None, omega = np.eye(4), l = 60.0, sigma = 8.5, noise = 0.01, horizon=40)      # Robot GP
        all_gp[0].load_parameters('hyperparameters_robot.pkl')
        for i in range(1, N_a):
            all_gp[i] = GP(None, None, omega = np.eye(4), l = 60.0, sigma = 8.5, noise = 0.01, horizon=40)   # Human GP
            all_gp[i].load_parameters('hyperparameters_human.pkl')

        # Set barrier for each agent
        horizon_set=[0, 0, 0, 6, 7, 8]
        for i in range(1, N_a):
            agents[i].Ds = horizon_set[np.random.randint(len(horizon_set))]

        for i in range(T):
        # while not self.exit:
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
            
            # Get trajectory for robot
            u, x_path, _ = get_trajectory(agents[0], N=10)
            u, x_next  = filter_output(0, agents, x_path)
            agents_ctrl[0] = u
            paths[0] = x_path

            # Simulate trajectory for other agents
            for j in range(1, N_a):
                u2, x2_path, _ = get_trajectory(agents[j])
                # Get agent's states (for inference)
                x = np.concatenate((x_path[:,0], x2_path[:,0]))
                # Infer agent's next state and get uncertainty polytope
                if (all_gp[j].N_data > 0):
                    m_d, cov_d = all_gp[j].predict(x)
                    G, g = all_gp[j].extract_box(m_d, cov_d)
                # Obtain CBF controller given uncertainty polytope
                if (agents[j].Ds > 0):
                    u2, x2_next = filter_output(j, agents, x2_path)
                agents_ctrl[j] = u2
                paths[j] = x2_path

            # Add noise to agents' actions
            noise_a = 0.1
            for j in range(N_a):
                agents[j].update(agents_ctrl[j] + noise_a *
                                 (np.random.rand(2) - 0.5))
            
            # Update GPs
            for j in range(1, N_a):
                x = np.concatenate((paths[0][:,0], paths[j][:,0]))
                # y = np.concatenate((paths[0][:,1], paths[j][:,1]))
                y = paths[j][:,1]
                all_gp[j].add_data(x, y)
                all_gp[j].update_obs_covariance()

            # Drawing
            if (kDraw):
                self.screen.fill((220, 220, 220))

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
        print(i)
        dat_trial, dat_u_trial = game.run()
        data.append(dat_trial)
        data_u.append(dat_u_trial)
        # data.append(np.array(dat_trial))
        print("Trial " + str(i) + " finished in " + str(round(time.time() - start_time, 1)) + " sec")
        if (i % 5 == 0 and kSave):
            np.save('train_data_i9.npy', data)
            np.save('train_data_u_i9.npy', data_u)