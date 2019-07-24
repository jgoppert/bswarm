#!/usr/bin/env python

"""
These functions implement the SE2 Lie Group see (http://ethaneade.com/lie.pdf)
"""

import numpy as np
import matplotlib.pyplot as plt

def SE2_log(M):
    """
    Matrix logarithm for SE2 Lie group
    """
    theta = np.arctan2(M[1, 0], M[0, 0])
    if np.abs(theta) < 1e-6:
        A = 1 - theta**2/6 + theta**4/120
        B = theta/2 - theta**3/24 + theta**5/720
    else:
        A = np.sin(theta)/theta
        B = (1 - np.cos(theta))/theta
    V_inv = np.array([[A, B], [-B, A]])/(A**2 + B**2)
    t = M[:2, 2]
    u = V_inv.dot(t)
    return np.array([theta, u[0], u[1]])

def SE2_from_param(v):
    """
    Create SE2 from paramters, [theta, x, y]
    """
    theta, x, y = v
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

def SE2_to_param(M):
    """
    From matrix to [theta, x, y]
    """
    theta = np.arctan2(M[1, 0], M[0, 0])
    x = M[0, 2]
    y = M[1, 2]
    return np.array([theta, x, y])

def SE2_inv(M):
    """
    SE2 inverse
    """
    R = M[:2, :2]
    t = np.array([M[:2, 2]]).T
    return np.block([
        [R.T, -R.T.dot(t)],
        [0, 0, 1]
    ])

def SE2_exp(v):
    """
    SE2 matrix exponential
    """
    theta, x, y = v
    if np.abs(theta) < 1e-6:
        A = 1 - theta**2/6 + theta**4/120
        B = theta/2 - theta**3/24 + theta**5/720
    else:
        A = np.sin(theta)/theta
        B = (1 - np.cos(theta))/theta
    V = np.array([[A, -B], [B, A]])
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    u = np.array([[x, y]]).T
    return np.block([
        [R, V.dot(u)],
        [0, 0,  1]])

def test_SE2():
    v = np.random.randn(3)
    X = SE2_from_param(v)
    assert np.allclose(X.dot(SE2_inv(X)), np.eye(3))
    assert np.allclose(SE2_log(SE2_exp(v)) - v, np.zeros(3))
    assert np.allclose(SE2_to_param(SE2_from_param(v)) - v, np.zeros(3))

    v = np.zeros(3)
    X = SE2_from_param(v)
    assert np.allclose(X.dot(SE2_inv(X)), np.eye(3))
    assert np.allclose(SE2_log(SE2_exp(v)) - v, np.zeros(3))
    assert np.allclose(SE2_to_param(SE2_from_param(v)) - v, np.zeros(3))

def sample(X_goal, box):
    """
    Sample from planning area [10, 10], [-5, 5],
    with 10% probability, return X_goal

    @param X_goal: The goal position
    @param box: [xmin, xmax, ymin, ymax]

    @return: (sample, is_goal) (bool)
    """
    if np.random.rand() < 0.1:
        return X_goal, True
    else:
        xmin, xmax, ymin, ymax = box
        x = xmin + (xmax - xmin)*np.random.rand()
        y = ymin + (ymax - ymin)*np.random.rand()
        theta = 2*np.pi*np.random.rand()
        XS = SE2_from_param([theta, x, y])

        # if close to goal in euclidean distance, set to goal
        theta_g, x_g, y_g = SE2_to_param(X_goal)
        d = np.sqrt((x - x_g)**2 + (y - y_g)**2)
        if d < 1:
            return X_goal, True
        else:
            return XS, False

def find_u_R_d(X0, X1):
    """
    Compute arch length, radius, and euclidean distance
    """
    dtheta, dx, dy = SE2_to_param(SE2_inv(X0).dot(X1))
    d = np.sqrt(dx**2 + dy**2) # hypotenuse
    if np.abs(dx) < 1e-6:
        theta = np.pi/2
    else:
        theta = np.arctan(dy/dx)
    if np.abs(theta) > 1e-3:
        R = d/(2*np.sin(theta))
        u = R*2*theta
    else:
        R = np.infty
        u = d
    return u, R, d

def distance(X0, X1):
    """
    Compute the distance from X0 to X1. There
    is a higher weight placed on cross-track movement (dy),
    as this is harder for the vehicle than turning (dtheta)
    and formward movingment (dx)
    """
    u, R, d = find_u_R_d(X0, X1)
    cost = np.abs(u)
    if d > 0.1 and np.abs(R) < 0.5:
        cost = np.infty
    return cost

def local_path_planner(X0, X1, dist):
    """
    Plan a path from X0 to X1. If the norm of the lie algebra
    is greather than 1, limit it to 1.

    @para X0: start
    @param X1: goal
    @dist: the distance to travel
    """
    u, R, d = find_u_R_d(X0, X1)
    if np.abs(u) > dist:
        u = dist*np.sign(u)
    if np.abs(R) < 1e-3:
        omega = 0
    else:
        omega = u/R
    v = np.array([omega, u, 0])
    return X0.dot(SE2_exp(v))

def collision(collision_points, vehicle_radius, X0, X1, box, steps):
    """
    Check that the points along the trajectory from X0 to
    X1 do not collide with the circular collision points defined
    as a list of [x, y, r], where (x, y) is the center, and
    r is the radius. Evaluate the trajectory at (steps) discrete
    points.

    @param collision_points: list of [x, y, radius]
    @param vehicle_radius: radius of vehicle
    @param X0: start point
    @param X1: end point
    @param box: [xmin, xmax, ymin, ymax]
    @param steps: number of steps along trajectory
    """
    v = SE2_log(SE2_inv(X0).dot(X1))
    for t in np.linspace(0, 1, steps):
        X = X0.dot(SE2_exp(v*t))
        theta, x, y = SE2_to_param(X)
        # check bounds of environment
        rv = vehicle_radius
        if (x < rv or x > 10 - rv or y < -5 + rv or y > 5 - rv):
            return True
        for xc, yc, r in collision_points:
            if (xc - x)**2 + (yc - y)**2 < (r + rv)**2:
                return True
    return False

class Tree:
    """
    This is a tree data structure that is used for RRT.
    Each tree has one parent, and a list of children.
    """

    def __init__(self, position):
        self.position = position
        self.parent = None
        self.children = []

    def add(self, child):
        """
        Add the child node to the tree.
        """
        child.parent = self
        self.children.append(child)

    def closest(self, position, black_list):
        """
        Find the node closest to the given position.

        @param position to get closest to
        @param black_list: black list of children to ignore
         (so can avoid duplicate collision checks)
        """
        closest_node = self
        if self in black_list:
            dist_min = np.infty
        else:
            dist_min = distance(self.position, position)
        for child in self.children:
            child_closest, dist_child = child.closest(position, black_list)
            if dist_child < dist_min:
                closest_node = child_closest
                dist_min = dist_child
        return closest_node, dist_min

    def get_leaves(self):
        """
        Get all nodes (leaves) in the tree.
        """
        leaves = [self]
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def path(self):
        """
        Find the path from the root of the tree to the current node.
        """
        ret = [self]
        if self.parent is not None:
            ret = self.parent.path()
            ret.append(self)
        return ret

def rrt(X_start, X_goal, vehicle_radius, box, collision_points, plot, max_iterations, dist_plan, tolerance):
    """
    Rapidly exploring random tree planner

    @param X_start: start SE2 element
    @param X_goal: goal SE2 element
    @param vehicle_radius: radius of vehicle
    @param box: [xmin, xmax, ymin, ymax]
    @param collision_points: list of [x, y, radius]
    @param plot (bool), controls plotting
    @param max_iteration, maximum iterations
    @param dist_plan, local_path_planner plan distance
    @param tolerance, goal accpetance distance
    @return path as a list of Tree nodes
    """

    root = Tree(X_start)
    node_best = root
    dist_best = distance(X_start, X_goal)
    assert dist_best != 0


    if plot:
        fig = plt.figure(figsize=(10, 10))

    i = 0
    goal_black_list = []
    success = False

    while True:

        # check if we have exeeded max iterations
        i += 1
        if i > max_iterations:
            print('max iterations exceeded')
            break

        # draw a random sample
        XS, goal_sampled = sample(X_goal, box)

        if plot:
            ps = SE2_to_param(XS)
            plt.plot(ps[1], ps[2], 'r+', markersize=10)

        # find the closest node to the sample
        if goal_sampled:
            node, dist = root.closest(XS, goal_black_list)
            if node in goal_black_list:
                continue
        else:
            node, dist = root.closest(XS, [])

        # plan a path towards the sample from the closest node
        X0 = node.position
        X1 = local_path_planner(X0, XS, dist=dist_plan)

        # if the path has a collision, skip
        if collision(collision_points, vehicle_radius, X0, X1, box, 10):
            if goal_sampled:
                goal_black_list.append(node)
            continue

        # add the end of the local_path_planner path to the tree
        new_node = Tree(X1)
        node.add(new_node)

        # plot the tree
        p0 = SE2_to_param(node.position)
        p1 = SE2_to_param(new_node.position)

        if plot:
            # plt.plot([p0[1], p1[1]], [p0[2], p1[2]], 'k-', alpha=0.1)
            v = SE2_log(SE2_inv(X0).dot(X1))
            p_path = []
            for t in np.linspace(0, 1, 10):
                X = X0.dot(SE2_exp(v*t))
                theta, x, y = SE2_to_param(X)
                p_path.append([x, y])
            p_path = np.array(p_path)
            plt.plot(p_path[:, 0], p_path[:, 1],  'r-', alpha=0.1)

        dist_new = distance(X1, X_goal)
        if dist_new < dist_best:
            dist_best = dist_new
            node_best = new_node

        # check if we have reached the goal
        if dist_new < tolerance:
            success = True
            break

    # build the path
    node_path = node_best.path()
    path = np.array([SE2_to_param(n.position)[1:] for n in node_path])

    path_full = []
    for i in range(len(node_path) - 1):
        X0 = node_path[i].position
        X1 = node_path[i+1].position
        v = SE2_log(SE2_inv(X0).dot(X1))
        for t in np.linspace(0, 1, 10):
            X = X0.dot(SE2_exp(v*t))
            theta, x, y = SE2_to_param(X)
            path_full.append([x, y])
    path_full = np.array(path_full)

    # set the limits for the plot
    if plot:
        # plot all nodes
        for leaf in root.get_leaves():
            p = SE2_to_param(leaf.position)
            plt.plot(p[1], p[2], 'bo', alpha=0.3)

        # plot the collision points
        for x, y, r in collision_points:
            circle1 = plt.Circle((x, y), r, color='r')
            plt.gca().add_artist(circle1)

        # plot the start and goal positions
        xs = SE2_to_param(X_start)
        xg = SE2_to_param(X_goal)
        plt.plot(xs[1], xs[2], 'rs', markersize=15, alpha=0.3, label='start')
        plt.plot(xg[1], xg[2], 'gs', markersize=15, alpha=0.3, label='goal')


        if len(path_full) > 0:
            plt.plot(path_full[:, 0], path_full[:, 1], 'g-', linewidth=10, alpha=0.3, label='path')

        plt.gca().set_xlim([0, 10])
        plt.gca().set_ylim([-5, 5])
        plt.grid()

        plt.xlabel('x, m')
        plt.ylabel('y, m')
        plt.title('RRT')
        plt.legend()
        plt.show()

    return {
        'success': success,
        'path': path,
        'path_full': path_full
    }


if __name__ == "__main__":
    import argparse

    test_SE2()

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    X_start = SE2_from_param([0, 1, 0])  # theta=0, x=0, y=0
    X_goal = SE2_from_param([0, 8, 0])  # theta=0, x=10, y=0

    # this is a list of all obstacles
    # x, y, radius
    collision_points = [
        [5, 0, 1],
        [3, 3, 0.2],
        [5, -5, 1],
        [10, 5, 2],
        [10, -5, 1],
    ]

    ret = rrt(X_start=X_start, X_goal=X_goal, vehicle_radius=1,
            box=[0, 10, -5, 5], collision_points=collision_points,
            plot=args.plot, max_iterations=200, dist_plan=3.0, tolerance=0.5)
    print('success', ret['success'])
