from prrt.vehicle import Car
from prrt.primitive import PointR2, PoseR2S1
from prrt.ptg import CPTG
import numpy as np
import prrt.helper as helper
from prrt.planner import Planner
import os
import time

# configurations
rebuild_tables = False
plot_tree = True
plot_trajectory_frames = True

car = Car(2., np.deg2rad(60), 2., 2.)
car_vertices = (PointR2(-1.5, -1.), PointR2(1.5, -1.), PointR2(1.5, 1.), PointR2(-1.5, 1))
car.set_vertices(car_vertices)

if rebuild_tables or not os.path.isfile('./prrt.pkl'):
    ptgs = []  # type: List[PTG]
    fwd_circular_ptg = CPTG(5.0, car, 0.1, 1)
    fwd_circular_ptg.name = 'Forward Circular PTG'
    fwd_circular_ptg.build_cpoints()
    fwd_circular_ptg.build_cpoints_grid()
    fwd_circular_ptg.build_obstacle_grid()
    # fwd_circular_ptg.plot_cpoints()
    ptgs.append(fwd_circular_ptg)

    bwd_circular_ptg = CPTG(5.0, car, 0.1, -1)
    bwd_circular_ptg.name = 'Backward Circular PTG'
    bwd_circular_ptg.build_cpoints()
    bwd_circular_ptg.build_cpoints_grid()
    bwd_circular_ptg.build_obstacle_grid()
    ptgs.append(bwd_circular_ptg)

    planner = Planner(ptgs)
    planner.load_world_map('./lot_caseStudy.png', 117.6, 68.3)  # this should not be done here !
    helper.save_object(planner, './prrt.pkl')
else:
    planner = helper.load_object('./prrt.pkl')  # type: Planner

# Hack: exclude the backward cptg for quicker results
planner._ptgs.pop()

# Set initial pose and goal pose
init_pose = PoseR2S1(80, 60, 0.0 * np.pi)
goal_pose = PoseR2S1(55, 10, -3. / 4 * np.pi)

print(" SOLVING ")
start = time.time()
planner.solve(init_pose, goal_pose)
end = time.time()
print("time elapsed in s = ")
print(end - start)

# Plot as configured
if plot_tree:
    planner.tree.plot_nodes(planner.world, goal_pose)
if plot_trajectory_frames:
    # delete existing frames (if any)
    file_list = [f for f in os.listdir('./out') if f.endswith('.png') and f.startswith('frame')]
    for f in file_list:
        os.remove('./out/' + f)
    planner.trace_solution(car, goal_pose)
