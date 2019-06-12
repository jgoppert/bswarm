
import numpy as np
import matplotlib.pyplot as plt

def time_scaling(t, t0, T):
    """
    @param t: elapsed time
    @param t0: initial time for start of leg
    @param T: duration for leg
    """
    return (t - t0)/T
    
def trajectory_coefficients(beta, m, n, T):
    """
    @param beta: scaled time
    @param m: derivative order, 0, for no derivative
    @param n: number of coeff in polynomial (order + 1)
    @param T: period
    """
    p = np.zeros(n)
    for k in range(m, n):
        p[n - k - 1] = (np.math.factorial(k)*beta**(k-m))/((np.math.factorial(k-m))*T**m)
    return p


def find_trajectory(x_list, v_list, T, plot=False):
    n = 4 # this will be fixed to support 3rd order polynomials
    if not len(x_list) == len(T) + 1:
        raise ValueError('x_list must be 1 longer than T')
    if not len(x_list) == len(v_list):
        raise ValueError('x_list and v_list must be same length')

    S = np.hstack([[0], np.cumsum(T)])
    n_legs = len(T)
    n_coeff = n_legs*n
    A = np.zeros((n_coeff, n_coeff))
    b = np.zeros(n_coeff)

    eq_num = 0
    for i in range(n_legs):        
        # p1(0) = x[0]
        A[eq_num, n*i:n*(i+1)] = trajectory_coefficients(beta=0, m=0, n=n, T=T[i])
        b[eq_num] = x_list[i]
        eq_num += 1
    
        # p1'(0) = v[0]
        A[eq_num, n*i:n*(i+1)] = trajectory_coefficients(beta=0, m=1, n=n, T=T[i])
        b[eq_num] = v_list[i]
        eq_num += 1
    
        # p1(1) = x[1]
        A[eq_num, n*i:n*(i+1)] = trajectory_coefficients(beta=1, m=0, n=n, T=T[i])
        b[eq_num] = x_list[i + 1]
        eq_num += 1
    
        # p1'(1) = v[1]
        A[eq_num, n*i:n*(i+1)] = trajectory_coefficients(beta=1, m=1, n=n, T=T[i])
        b[eq_num] = v_list[i + 1]
        eq_num += 1

    rank = np.linalg.matrix_rank(A)
    if rank < n_coeff:
        print('Matrix A not full rank, check constraints, rank: ', rank, '/', n_coeff)
        c = np.linalg.pinv(A).dot(b)
    else:
        c = np.linalg.inv(A).dot(b)
    
    t = []
    x = []
    for i in range(0, len(T)):
        ti = np.linspace(S[i], S[i+1])
        beta = time_scaling(ti, S[i], T[i])
        xi = np.polyval(c[n*i:n*(i+1)], beta)
        t.append(ti)
        x.append(xi)
    x = np.hstack(x)
    t = np.hstack(t)
    return locals()

def plan_trajectory(T, waypoints):
    """
    Plan a trajectory and plot it.
    @param T: leg duraction vector [s],
    @param waypoints: position and velocity waypoint 2D array, each row: [px, py, vx, vy]
    """
    trajx = find_trajectory(x_list=waypoints[:, 0], v_list=waypoints[:, 2], T=T, plot=True)
    trajy = find_trajectory(x_list=waypoints[:, 1], v_list=waypoints[:, 3], T=T, plot=True)
    
    plt.figure()
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'rx', label='waypoint')
    plt.plot(trajx['x'], trajy['x'], label='trajectory')
    plt.axis('equal')
    plt.grid()
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    plt.legend(loc='best')
    
    plt.figure()
    plt.plot(trajx['t'], trajx['x'], label='x')
    plt.plot(trajx['t'], trajy['x'], label='y')
    plt.legend()
    plt.grid()
    return locals()

waypoints = np.array([
    # px, py, vx, vy
    [0, 0, 1, -1],
    [1, 0, 1, 1],
    [1, 1, -1, 1],
    [0, 1, -1, -1],
    [0, 0, 1, -1]
])
traj1 = plan_trajectory(T=[1, 1, 1, 1], waypoints=waypoints);
plt.title('An example trajectory')

plt.figure()
traj2 = plan_trajectory(T=[0.1, 0.1, 0.1, 0.1], waypoints=waypoints);
plt.title('The same trajectory with 1/10 time scaling')
