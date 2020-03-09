import os
import pygame
import numpy as np
import time

from control import get_trajectory

class Car:
    def __init__(self, x, y, max_acceleration=4.0):
        self.position = np.array([x, y])
        self.velocity = np.array([0., 0.])
        self.max_acceleration = max_acceleration
        self.max_velocity = 6
        self.dt = 0.1
        
        self.kd = 0.06
        self.ka = 0.08

        self.kd_err = 0.04
        self.ka_err = 0.1

        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[0, 0], [0, 0], [self.dt, 0], [0, self.dt]])

        # Barrier
        self.Ds = 5.0
        self.gamma = 0.8
        self.eps = 1e9

    # Update linearization based on current state
    def update_linearization(self):
        self.A[2,2] = 1 - self.kd*self.velocity[0]*self.dt
        self.A[3,3] = 1 - self.kd*self.velocity[1]*self.dt
        self.B[2,0] = (self.ka * np.linalg.norm(self.velocity) + 1) * self.dt
        self.B[3,1] = (self.ka * np.linalg.norm(self.velocity) + 1) * self.dt
        return self.A, self.B

    # Update linearization based on current state
    def update_linearization_err(self):
        self.A[2,2] = 1 - self.kd_err*self.velocity[0]*self.dt
        self.A[3,3] = 1 - self.kd_err*self.velocity[1]*self.dt
        self.B[2,0] = (self.ka_err * np.linalg.norm(self.velocity) + 1) * self.dt
        self.B[3,1] = (self.ka_err * np.linalg.norm(self.velocity) + 1) * self.dt
        return self.A, self.B

    # Project robot dynamics
    def get_dynamics(self, x):
        pos = x[0:2]
        vel = x[2:4]
        fp = pos + vel*self.dt
        gp = np.array([[0., 0.], [0., 0.]])
        fv = vel - self.kd_err * vel**2 * self.dt
        gv = (self.ka_err * np.linalg.norm(vel) + 1) * self.dt * np.eye(2)
        return fp, gp, fv, gv

    # Predict human dynamics
    def get_dynamics_human(self, x, t=1):
        pos = x[0:2]
        vel = x[2:4]
        fp = pos + vel*self.dt*t
        gp = np.array([[0., 0.], [0., 0.]])
        fv = vel
        gv = np.array([[self.dt, 0.], [0., self.dt]])
        return fp, gp, fv, gv

    # Project human dynamics (with error)
    def fh_err(self,x):
        pos = x[0:2]
        vel = x[2:4]
        p = pos + vel*self.dt
        v_nom = vel
        v = v_nom
        v[0] = max(-self.max_velocity,
                              min(v[0], self.max_velocity))
        v[1] = max(-self.max_velocity,
                              min(v[1], self.max_velocity))
        return p, v
    
    # Project robot dynamics (with error)
    def f_err(self,x,u):
        pos = x[0:2]
        vel = x[2:4]
        p = pos + vel*self.dt
        v_nom = vel - self.kd_err*vel**2 * self.dt
        v = v_nom + (self.ka_err*np.linalg.norm(vel) + 1) * u * self.dt
        v[0] = max(-self.max_velocity,
                              min(v[0], self.max_velocity))
        v[1] = max(-self.max_velocity,
                              min(v[1], self.max_velocity))
        return p, v

    # Project true dynamics
    def f(self,x,u):
        pos = x[0:2]
        vel = x[2:4]
        p = pos + vel*self.dt
        v_nom = vel - self.kd*vel**2 * self.dt
        v = v_nom + (self.ka*np.linalg.norm(vel) + 1) * u * self.dt
        v[0] = max(-self.max_velocity,
                              min(v[0], self.max_velocity))
        v[1] = max(-self.max_velocity,
                              min(v[1], self.max_velocity))
        return p, v

    # Update true dynamics
    def update(self, u):
        x = np.concatenate([self.position, self.velocity], axis=0)
        p, v = self.f(x, u)
        self.position = p
        self.velocity = v

    def project(self, u):
        state = np.zeros(4)
        state[0] += self.velocity[0] * self.dt
        state[1] += self.velocity[1] * self.dt

        state[2] += self.dt*u[0]
        state[3] += self.dt*u[1]
        state[2] = max(-self.max_velocity, min(state[2], self.max_velocity))
        state[3] = max(-self.max_velocity, min(state[3], self.max_velocity))
        return state

if __name__ == '__main__':
    car = Car(0,0)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "user.PNG")
    width = 1024*3
    height = 1024*3
    ppu = 32
    screen = pygame.display.set_mode((width, height))
    for i in range(1000):
        u, _, _ = get_trajectory(car)
        # print(u)
        car.update(u)
        screen.fill((0, 0, 0))
        agent1_img = pygame.image.load('agents.jpg')
        rect = agent1_img.get_rect()
        screen.blit(agent1_img, car.position * ppu - (rect.width / 2, rect.height / 2))
        pygame.display.flip()
        time.sleep(0.05)
    print("Game Initialized")
