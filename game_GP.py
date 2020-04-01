import pygame
import numpy as np
import time

from car import Car
from control import get_trajectory, filter_output, filter_output_primal
from GP_predict import GP

from video import make_video

kDraw = True
kSave = False
kVideo = False

class Game:
    def __init__(self):
        pygame.display.set_caption("CBF Test")
        self.width = 1024*3
        self.height = 1024*3
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.ticks = 60

    def run(self, seed=None, robust=True, video=False, iteration=-1):
        ppu = 32
        T = 150     # Max simulation time

        # Initiate video generator
        if (video):
            if (robust):
                save_screen = make_video(self.screen, iteration, "robust")
            else:
                save_screen = make_video(self.screen, iteration, "primal")

        # Random Initialization Seed
        if (seed is not None):
            np.random.seed(seed)

        # Initialize settings
        eps = 0.001
        p_threshold = 1 - 0.95
        dist_threshold = 1.0
        coll_threshold = 4.9
        success = False
        collision_flag = False
        min_dist = np.inf

        # Initialize agents
        N_a = np.random.randint(3, 10)
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
        all_gp = []
        state = np.zeros((N_a, 4))
        next_state = np.zeros((N_a, 4))
        x_state = np.zeros((N_a, 4))
        d_state = np.zeros((N_a, 4))
        all_gp.append(GP(None, None, omega = np.eye(4), l = 60.0, sigma = 8.5, noise = 0.01, horizon=15))     # Robot GP
        all_gp[0].load_parameters('hyperparameters_robot.pkl')
        for i in range(1, N_a):
            all_gp.append(GP(None, None, omega = np.eye(4), l = 60.0, sigma = 8.5, noise = 0.01, horizon=15))    # Human GP
            all_gp[i].load_parameters('hyperparameters_human.pkl')
        G_all = np.zeros((N_a-1, 8*2, 8))
        g_all = np.zeros((N_a-1, 8*2))
        m_all = np.zeros((N_a, 4))
        z_all = np.zeros((N_a, 2))
        G_base = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]])
        g_base = np.array([eps, eps, eps, eps, eps, eps, eps, eps])

        # Set barrier for each agent
        horizon_set=[0, 0, 0, 0, 7, 8]
        for i in range(1, N_a):
            agents[i].Ds = horizon_set[np.random.randint(len(horizon_set))]

        for i in range(T):
            # Get nominal trajectory for our robot
            u, x_path, x0 = get_trajectory(agents[0], N=10)
            state[0,:] = x0

            # Simulate trajectory for other agents
            for j in range(1, N_a):
                # Obtain (CBF) controller for other agent (if applicable)
                u2, x2_path, x2_0 = get_trajectory(agents[j])
                if (agents[j].Ds > 0):
                    u2 = filter_output_primal(j, agents, x2_path)
                agents_ctrl[j] = u2
                # Get agent's states (for inference)
                state[j,:] = x2_0 - x0
                if (np.linalg.norm(x2_0[0:2] - x0[0:2]) < min_dist):
                    min_dist = np.linalg.norm(x2_0[0:2] - x0[0:2])

            if (robust):
                # Infer uncertainty polytope for robot and other agents
                if (all_gp[0].N_data > 0):
                    start_time = time.time()
                    m_d, cov_d = all_gp[0].predict(state[0,:])
                    G_r, g_r = all_gp[0].extract_box(cov_d, p_threshold=p_threshold)       # G: 8x4, g: 8x1
                    m_all[0,:] = m_d
                    z_all[0,0], z_all[0,1] = all_gp[0].extract_norms(cov_d, p_threshold=p_threshold)
                for j in range(1, N_a):
                    if (all_gp[j].N_data > 0):
                        m_d, cov_d = all_gp[j].predict(state[j,:])
                        G_h, g_h = all_gp[j].extract_box(cov_d, p_threshold=p_threshold)
                        m_all[j,:] = m_d
                        z_all[j,0], z_all[j,1] = all_gp[j].extract_norms(cov_d, p_threshold=p_threshold)
                        G_all[j-1,0:8,0:4] = G_r
                        g_all[j-1,0:8] = g_r
                        if (np.linalg.norm(next_state[j,2:4]) < eps):
                            G_all[j-1,8:16,4:8] = G_base
                            g_all[j-1,8:16] = g_base
                        else:
                            G_all[j-1,8:16,4:8] = G_h
                            g_all[j-1,8:16] = g_h

                # Obtain safe control given uncertainty polytopes
                if (all_gp[0].N_data > 0):
                    start_time = time.time()
                    u = filter_output(0, agents, x_path, G_all=G_all, g_all=g_all, m=m_all, z=z_all)
                else:
                    u = filter_output(0, agents, x_path)
            else:
                u = filter_output_primal(0, agents, x_path)
            agents_ctrl[0] = u

            # Add noise to agents' actions and collect data
            noise_a = 0.001
            for j in range(N_a):
                if (j == 0):
                    x0 = np.array([agents[j].position[0], agents[j].position[1], agents[j].velocity[0], agents[j].velocity[1]])
                else:
                    xj = np.array([agents[j].position[0], agents[j].position[1], agents[j].velocity[0], agents[j].velocity[1]])
                agents[j].update(agents_ctrl[j] + noise_a * (np.random.rand(2) - 0.5)) 
                next_state[j,:] = np.array([agents[j].position[0], agents[j].position[1],
                   agents[j].velocity[0], agents[j].velocity[1]])
                if (j == 0):
                    p, v = agents[j].f_err(x0, agents_ctrl[j])
                    x_state[j,:] = x0
                    d_state[j,:] = next_state[j,:] - np.concatenate((p, v))
                else:
                    p, v = agents[j].fh_err(xj)
                    x_state[j,:] = xj - x0
                    d_state[j,:] = next_state[j,:] - np.concatenate((p, v))
            
            # Update GPs
            if (robust):
                start_time = time.time()
                all_gp[0].add_data(x_state[0,:], d_state[0,:])
                all_gp[0].get_obs_covariance()
                for j in range(1, N_a):
                    all_gp[j].add_data(x_state[j,:], d_state[j,:])
                    all_gp[j].get_obs_covariance()

            # Draw screen
            self.screen.fill((220, 220, 220))
            # Draw other agents
            for j in range(1, N_a):
                pygame.draw.circle(self.screen, [200, 0, 0], (agents[j].position*ppu).astype(int), 80)
            # Draw our goal
            agent1_goal = pygame.image.load('star.png')
            rect = agent1_goal.get_rect()
            self.screen.blit(agent1_goal, agents[0].goal * ppu -
                            (rect.width / 2, rect.height / 2))
            # Draw our agent
            pygame.draw.circle(self.screen, [0, 0, 200], (agents[0].position*ppu).astype(int), 80)
            if (kDraw):
                pygame.display.flip()
            
            if (kVideo):
                next(save_screen)

            # Check if there is collision, or if goal is reached
            for j in range(1, N_a):
                if (np.linalg.norm(agents[0].position - agents[j].position) < coll_threshold):
                    success = False
                    collision_flag = True 
            if (np.linalg.norm(agents[0].position - agents[0].goal) < dist_threshold and collision_flag):
                success = False
                break
            elif (np.linalg.norm(agents[0].position - agents[0].goal) < dist_threshold and not collision_flag):
                success = True
                break
            else:
                pass
            
            self.clock.tick(self.ticks)
            time.sleep(0.1)
        
        return data, data_u, success, collision_flag, i, min_dist


if __name__ == '__main__':
    game = Game()
    print("Game Initialized")
    repeat_flag = False
    data = []
    data_u = []
    trials = 1000
    start_time = time.time()
    # Initialize list for storing results
    result_success_robust = []
    result_success_primal = []
    result_collision_robust = []
    result_collision_primal = []
    result_L_robust = []
    result_L_primal = []
    result_dist_robust = []
    result_dist_primal = []
    for i in range(trials):
        if (repeat_flag):
            kVideo = False #True
        else:
            # Set seed for random environment
            seed = np.random.randint(4e9)
        # Run Robust CBF with uncertainty prediction
        _, _, success_robust, collision_robust, L_robust, dist_robust = game.run(seed=seed, robust=True, video=kVideo, iteration=i)
        # Run Nominal CBF without uncertainty prediction
        _, _, success_primal, collision_primal, L_primal, dist_primal = game.run(seed=seed, robust=False, video=kVideo, iteration=i)

        # Store results
        result_success_robust.append(success_robust)
        result_collision_robust.append(collision_robust)
        result_L_robust.append(L_robust)
        result_dist_robust.append(dist_robust)
        result_success_primal.append(success_primal)
        result_collision_primal.append(collision_primal)
        result_L_primal.append(L_primal)
        result_dist_primal.append(dist_primal)
        if (not collision_robust and collision_primal and not repeat_flag):
            repeat_flag = False #True
        else:
            kVideo = False
            repeat_flag = False
        print("Trial " + str(i) + " Completed")

        # Save results comparing Robust vs. Nominal CBF (if kSave flag is set)
        if (i % 10 == 0 and kSave):
            np.save('comparison_results_' + str(start_time) + '.npy', [result_success_robust, result_success_primal, result_collision_robust, result_collision_primal, result_L_robust, result_L_primal, result_dist_robust, result_dist_primal])
