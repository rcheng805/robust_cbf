from video import make_video

import os
import pygame
import numpy as np
import time

import osqp
import cvxpy
import scipy.sparse as sparse
import scipy as sp

kRandom = True
kDebug = False
kDraw = False
kVideo = False

def filter_output(agent_idx, agents, x_nom, T=1, G_all=None, g_all=None, m=None, z=None):
    N_a = len(agents)
    x0 = np.array([agents[agent_idx].position[0], agents[agent_idx].position[1],
                   agents[agent_idx].velocity[0], agents[agent_idx].velocity[1]])
    amax = agents[agent_idx].max_acceleration

    # Get dynamics (with error)
    Ad, Bd = agents[agent_idx].update_linearization_err()
    Av = Ad[2:4,:]
    Bv = Bd[2:4,:]

    eps_m = 0.01
    zp_max = 0.001
    zv_max = 0.001
    zph_max = 0.001
    zvh_max = 0.001

    for t in range(T):
        # MPC Problem (u(0), x(1), lambda, eps)
        # Minimize actuation deviation
        dual_f = 16
        duals = dual_f*(N_a-1)
        P = np.zeros((7 + duals, 7 + duals))
        P[2, 2] = 1
        P[3, 3] = 1
        P[4, 4] = 1
        P[5, 5] = 1
        P[6, 6] = agents[agent_idx].eps
        P = sparse.csc_matrix(P)
        q = np.hstack([0.0, 0.0, -x_nom[:, t+1], 0.0, np.zeros(duals)])

        # Velocity/Acceleration Constraints
        up = agents[agent_idx].max_velocity - np.matmul(Av, x0)
        up = np.hstack((up, agents[agent_idx].max_acceleration,
                        agents[agent_idx].max_acceleration))
        lp = -agents[agent_idx].max_velocity - np.matmul(Av, x0)
        lp = np.hstack((lp, -agents[agent_idx].max_acceleration, -
                        agents[agent_idx].max_acceleration))
        A_np = np.vstack([np.hstack([Bv, np.zeros((2, 4)), np.zeros((2, 1)), np.zeros((2, duals))]),
                          np.hstack([np.eye(2), np.zeros((2, 4)), np.zeros((2, 1)), np.zeros((2, duals))])])

        # Dynamics Constraint
        A_np = np.vstack([A_np, np.hstack([Bd, -np.eye(4), np.zeros((4, 1)), np.zeros((4, duals))])])
        lp = np.hstack((lp, -np.matmul(Ad, x0)))
        up = np.hstack((up, -np.matmul(Ad, x0)))

        # Robust CBC Constraint
        pr, vr = x0[0:2], x0[2:4]
        fp, _, fv, gv = agents[agent_idx].get_dynamics(x0)     # Get robot dynamics
        if (m is not None):
            fp = fp + m[0,0:2]
            fv = fv + m[0,2:4]

        idx = 0
        for j in range(len(agents)):
            if (j == agent_idx):
                continue
            xh = np.concatenate([agents[j].position, agents[j].velocity], axis=0)
            ph, _, vh, _ = agents[j].get_dynamics_human(xh, t)
            xh = np.concatenate([ph, vh], axis=0)
            fp_h, _, fv_h, _ = agents[j].get_dynamics_human(xh, t+1)  # Project "human" dynamics (t+1 steps)
            if (m is not None):
                fp_h = fp_h + m[j,0:2]
                fv_h = fv_h + m[j,2:4]

            if (z is not None):
                zp_max = z[0,0]
                zv_max = z[0,1]
                zph_max = z[j,0]
                zvh_max = z[j,1]

            den_p = max(np.linalg.norm(fp - fp_h) + zp_max + zph_max, eps_m)
            den_m = max(np.linalg.norm(fp - fp_h) - zp_max - zph_max, eps_m)

            H1 = np.hstack([-(fv - fv_h) / den_m, -(fp - fp_h) / den_m, (fv - fv_h) / den_m, (fp - fp_h) / den_m ])
            H2 = np.hstack([ -gv / den_m , np.zeros((2,2)), gv / den_m, np.zeros((2,2)) ])
            H3 = -np.matmul((fp - fp_h), gv) / den_p
            if (G_all is None):
                G = np.kron(np.eye(2), np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]]))
            else:
                G = G_all[j-1,:,:]
            if (g_all is None):
                g = np.kron(np.ones(2), np.array([zp_max, zp_max, zv_max, zv_max, zph_max, zph_max, zvh_max, zvh_max]))
            else:
                g = g_all[j-1,:]

            kc = min(np.dot(fp - fp_h, fv - fv_h) / den_p, np.dot(fp - fp_h, fv - fv_h) / den_m) + np.sqrt(2*amax*(max(den_m - agents[agent_idx].Ds, 0.))) + (agents[agent_idx].gamma - 1)*np.sqrt(2*amax*(max(np.linalg.norm(pr - ph) - agents[agent_idx].Ds, 0.))) + (agents[agent_idx].gamma-1)*np.dot(pr - ph, vr - vh)/(max(np.linalg.norm(pr - ph), eps_m))
            h_l1 = np.expand_dims(np.hstack([H3, np.zeros(4), -1.0, np.zeros(dual_f*idx), g, np.zeros(dual_f*(len(agents)-2-idx))]), axis=0)
            h_l2 = np.hstack([np.transpose(H2), np.zeros((8, 4)), np.zeros((8,1)), np.zeros((8,dual_f*idx)), -np.transpose(G), np.zeros((8, dual_f*(len(agents)-2-idx)))])
            h_r2 = -H1
            A_np = np.vstack([A_np, h_l1, h_l2])
            lp = np.hstack((lp, -np.inf, h_r2 - 0.001*np.ones(8)))
            up = np.hstack((up, kc, h_r2 + 0.001*np.ones(8)))
            idx += 1

        h_l3 = np.hstack([np.zeros((duals, 7)), -np.eye(duals)])
        A_np = np.vstack([A_np, h_l3])
        lp = np.hstack((lp, -np.inf*np.ones(duals)))
        up = np.hstack((up, np.zeros(duals)))
        Ap = sparse.csc_matrix(A_np)

        # Setup workspace and change alpha parameter
        if (t >= 0):
            prob = osqp.OSQP()
            prob.setup(P, q, Ap, lp, up, verbose=False)
        else:
            prob.update(q=q, l=lp, u=up)
            prob.update(Ax=Ap)
        # Solve problem
        res = prob.solve()
        if (t == 0):
            ctrl = res.x[0:2]
            pn, vn = agents[agent_idx].f(x0, ctrl)
            x_next = np.concatenate([pn, vn])
        # pn, vn = agents[agent_idx].f_err(x0, ctrl)
        # x0 = np.concatenate([pn, vn], axis=0)

    return ctrl #, x0

def filter_output_primal(agent_idx, u_nom, agents, x_nom, T_bar=4):
    N_a = len(agents)
    x0 = np.array([agents[agent_idx].position[0], agents[agent_idx].position[1],
                   agents[agent_idx].velocity[0], agents[agent_idx].velocity[1]])
    amax = agents[agent_idx].max_acceleration

    # Get dynamics (with error)
    Ad, Bd = agents[agent_idx].update_linearization_err()
    Av = Ad[2:4,:]
    Bv = Bd[2:4,:]

    for t in range(x_nom.shape[1]-1):    
        # MPC Problem (u(0), x(1), eps)
        # Minimize position deviation
        P = np.eye(7)
        P[0, 0] = 0
        P[1, 1] = 0
        P[6, 6] = agents[agent_idx].eps
        P = sparse.csc_matrix(P)
        q = np.hstack([0.0, 0.0, -x_nom[:, t+1], 0.0])

        # Velocity/Acceleration Constraints
        mult = 1
        up = mult*agents[agent_idx].max_velocity - np.matmul(Av, x0)
        up = np.hstack((up, mult*agents[agent_idx].max_acceleration,
                        mult*agents[agent_idx].max_acceleration))
        lp = -mult*agents[agent_idx].max_velocity - np.matmul(Av, x0)
        lp = np.hstack((lp, -mult*agents[agent_idx].max_acceleration, -
                        mult*agents[agent_idx].max_acceleration))
        A_np = np.vstack([np.hstack([Bv, np.zeros((2, 4))]),
                          np.hstack([np.eye(2), np.zeros((2, 4))])])
        
        # Dynamics Constraint
        A_np = np.vstack([A_np, np.hstack([Bd, -np.eye(4)])])
        lp = np.hstack((lp, -np.matmul(Ad, x0)))
        up = np.hstack((up, -np.matmul(Ad, x0)))
        
        # Barrier Constraint
        for j in range(len(agents)):
            if (j == agent_idx):
                continue
            pd = np.array([x0[0] - (agents[j].position[0] + agents[j].velocity[0]*agents[j].dt*t),
                           x0[1] - (agents[j].position[1] + agents[j].velocity[1]*agents[j].dt*t)])
            vd = np.array([x0[2] - agents[j].velocity[0],
                           x0[3] - agents[j].velocity[1]])
            c = pd + vd*dt
            h_const = np.dot(c, vd) / np.linalg.norm(c) + np.sqrt(2*abs(umax)*(max(np.linalg.norm(c) - agents[agent_idx].Ds, 0))) - (1 - agents[agent_idx].gamma)*np.dot(
                pd, vd)/np.linalg.norm(pd) - (1 - agents[agent_idx].gamma)*np.sqrt(2*abs(umax)*(max(np.linalg.norm(pd) - agents[agent_idx].Ds, 0)))  # Ignore u_{user}
            h_u = c*dt/np.linalg.norm(c)

            ub = h_const
            Ab = -h_u
            A_np = np.vstack(
                [A_np, np.hstack([np.expand_dims(Ab, axis=0), np.zeros((1, 4))])])
            lp = np.hstack((lp, -np.inf))
            up = np.hstack((up, ub))

        # Slack variable
        A_np = np.vstack([A_np, np.zeros((1, A_np.shape[1]))])
        ab = np.zeros((A_np.shape[0], 1))
        ab[-1, 0] = 1
        for j in range(len(agents)-1):
            ab[-2-j, 0] = -1
        A_np = np.hstack([A_np, ab])
        Ap = sparse.csc_matrix(A_np)
        lp = np.hstack((lp, 0.0))
        up = np.hstack((up, np.inf))

        # Create an OSQP object
        # prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        if (t >= 0):
            prob = osqp.OSQP()
            prob.setup(P, q, Ap, lp, up, verbose=False)
        else:
            prob.update(q=q, l=lp, u=up)
            prob.update(Ax=Ap)
        # Solve problem
        res = prob.solve()
        if (t == 0):
            ctrl = res.x[0:2]
        u_out = res.x[0:2]
        u_new.append(u_out)
        u_diff[t, :] = u_out - u_nom[t, :]
        x0 = np.matmul(A, x0) + np.matmul(B, u_out)
        path_new.append(x0)
        if (np.linalg.norm(u_diff[t, :]) > 1.0):
            bar_flag = True

    # print(h_const + np.dot(h_u, u_out) + res.x[2])
    # print("Residual: " + str(res.x[2]))
    return np.array(u_new), np.transpose(path_new), u_diff, bar_flag


def get_trajectory(agent, goal=None, N=12, agents=None, agent_idx=None):
    Ad, Bd = agent.update_linearization()

    # Sizes
    nx = 4
    nu = 2

    # Constraints
    umin = -agent.max_acceleration*np.ones(nu)
    umax = agent.max_acceleration*np.ones(nu)
    xmin = np.array([-20, -20, -agent.max_velocity, -agent.max_velocity])
    xmax = np.array([100, 100, agent.max_velocity, agent.max_velocity])

    # Objective function
    Q = sparse.diags([10., 10., 1., 1., ])
    QN = Q
    R = 0.02*sparse.eye(nu)

    # Initial and reference states
    x_init = np.array([agent.position[0], agent.position[1], agent.velocity[0], agent.velocity[1]])
    x0 = np.array([agent.position[0], agent.position[1], agent.velocity[0], agent.velocity[1]])
    if (goal is None):
        xg = np.array([agent.goal[0], agent.goal[1], 0.0, 0.0])
    else:
        xg = np.array([goal[0], goal[1], 0.0, 0.0])

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1), eps)
    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN, sparse.kron(sparse.eye(N), R)]).tocsc()
    # - linear objective
    q = np.hstack([np.kron(np.ones(N), -Q.dot(xg)), -QN.dot(xg), np.zeros((N)*nu)])

    # - linear dynamics
    Ax = np.kron(np.eye(N+1), -np.eye(nx)) + np.kron(np.eye(N+1, k=-1), Ad)
    Bu = np.kron(np.vstack(
        [np.zeros((1, N)), np.eye(N)]), Bd)

    # Dynamics constraints
    Aeq = np.hstack([Ax, Bu])
    leq = np.hstack([-x0[0], -x0[1], -x0[2], -x0[3], np.zeros((N)*nx)])
    # leq = np.zeros((N+1)*nx)
    ueq = leq

    # - input and state constraints
    Aineq = np.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin),
                       np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax),
                       np.kron(np.ones(N), umax)])
    A_vel = np.hstack([np.kron(np.eye(N+1), np.array([[0, 0, 1, 0], [
        0, 0, 0, 1]])), np.zeros((nu*(N+1), nu*N))])
    Aineq = np.vstack([Aineq, A_vel])
    lineq = np.hstack([lineq, np.kron(np.ones(N+1), np.array(
        [-agent.max_velocity, -agent.max_velocity]))])
    uineq = np.hstack([uineq, np.kron(np.ones(N+1), np.array(
        [agent.max_velocity, agent.max_velocity]))])

    A = np.vstack([Aeq, Aineq])
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    A = sparse.csc_matrix(A)

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u, warm_start=True, verbose=False)

    # Simulate in closed loop
    nsim = 1

    # Store data Init
    xst = np.zeros((nx, nsim))
    ust = np.zeros((nu, nsim))

    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    if (agents is not None):
        ctrl = res.x[-N*nu-1:-(N-1)*nu-1]
    else:
        ctrl = res.x[-N*nu:-(N-1)*nu]
    x0 = Ad.dot(x0) + Bd.dot(ctrl)

    i = 0
    # Store Data
    xst[:, i] = x0
    ust[:, i] = ctrl
    x_path = np.transpose(np.reshape(res.x[0:N*nx], (N, nx)))
    if (agents is not None):
        u_path = np.reshape(res.x[-N*nu-1:-1], (N, nu))
    else:
        u_path = np.reshape(res.x[-N*nu:], (N, nu))

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

    return ctrl, x_path, x_init #, u_path

if __name__ == '__main__':
    print("Control Program Run")