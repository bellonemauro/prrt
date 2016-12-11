from prrt.vehicle import Car, ArticulatedVehicle
from prrt.primitive import PointR2, PoseR2S1
from prrt.ptg import CPTG, ACPTG
import numpy as np
import prrt.helper as helper
from prrt.planner import Planner
import os
import time

# configurations
rebuild_tables = False   # allows to force to rebuild the PTG lookup table
plot_tree = True
plot_trajectory_frames = True

# car configuration parameters  -- TODO input via *.ini configuration file
car_v_max = 2. # max velocity
car_alpha_max = np.deg2rad(60) # maximum absolute steering angle
car_a_max = 2. #  max velocity --- difference with v_max?
car_w_max = 2. #  max rotational velocity  -- if this is [rad/s] may be too much
car_length = 3. # car length
car_width = 2. # car width

#car = Car(2., np.deg2rad(60), 2., 2.) # commented and replaced with symbolic names
car = Car(car_v_max, car_alpha_max, car_a_max, car_w_max)
#car_vertices = (PointR2(-1.5, -1.), PointR2(1.5, -1.), PointR2(1.5, 1.), PointR2(-1.5, 1))
car_vertices = (PointR2(-car_length/2, -car_width/2.),
                PointR2( car_length/2, -car_width/2.),
                PointR2( car_length/2,  car_width/2.),
                PointR2(-car_length/2,  car_width/2))
car.set_vertices(car_vertices)

# articulated vehicle configuration parameters  -- TODO input via *.ini configuration file
av_v_max = 2. # max velocity
av_alpha_max = np.deg2rad(60) # maximum absolute steering angle
av_a_max = 2. #  max velocity --- difference with v_max?
av_w_max = 2. #  max rotational velocity
av_tractor_length = 2. # tractor length
av_tractor_width = 1.  # tractor width
av_link_length = 1.    # link legth
av_trailer_length = 3. # trailer length
av_trailer_width = 2.  # trailer width

#av = ArticulatedVehicle(2., np.deg2rad(60), 2., 2., 2., 1., 1., 2., 3.)
av = ArticulatedVehicle(av_v_max, av_alpha_max, av_a_max, av_w_max,
                        av_tractor_length, av_tractor_width,                 #NOTE: I have changed the order legth/width
                        av_link_length, av_trailer_length, av_trailer_width) #NOTE: I have changed the order legth/width
av.phi = 0 # the initial condition for the trailer is to be set with the state vector


# PTG configuration parameters  -- TODO input via *.ini configuration file
PTG_phi_lower_bound = np.deg2rad(-45.)  # this will accelerate the execution due to less number of conversions
PTG_phi_upper_bound = np.deg2rad(45.01)
PTG_phi_resolution = np.deg2rad(3.)
PTG_size = 5.
PTG_resolution = 0.1
PTG_K = 1.

# the lookup table have to be rebuild if the geometry change but this is already implemented in MRPT
if rebuild_tables or not os.path.isfile('./prrt.pkl'):
    print(" NO PTG FOUND - The generation may take a while ! ")
    ptgs = []  # type: List[PTG]
    # fwd_circular_ptg = CPTG(5.0, car, 0.1, 1)
    # for each phi in the range [-45, +45]deg, with a step of 3deg
    #for phi in np.arange(np.deg2rad(-45), np.deg2rad(45) + 0.01, np.deg2rad(3)):
    for phi in np.arange(PTG_phi_lower_bound, PTG_phi_upper_bound, PTG_phi_resolution):
        #fwd_circular_ptg = ACPTG(5.0, av, 0.1, 1, phi)
        fwd_circular_ptg = ACPTG(PTG_size, av, PTG_resolution, PTG_K, phi)
        #fwd_circular_ptg.name = 'Forward ACPTG @phi = {0:.0f}'.format(np.rad2deg(phi))
        fwd_circular_ptg.name = 'Forward ACPTG @phi = {0:.0f}'.format(phi)  # now there is no necessity of conversion
        fwd_circular_ptg.build_cpoints()
        fwd_circular_ptg.build_cpoints_grid()
        fwd_circular_ptg.build_obstacle_grid()
        # fwd_circular_ptg.plot_cpoints()
        ptgs.append(fwd_circular_ptg)

    # TODO: fix the backward motion
    #  bwd_circular_ptg = CPTG(5.0, car, 0.1, -1)
    # bwd_circular_ptg.name = 'Backward Circular PTG'
    # bwd_circular_ptg.build_cpoints()
    # bwd_circular_ptg.build_cpoints_grid()
    # bwd_circular_ptg.build_obstacle_grid()
    # ptgs.append(bwd_circular_ptg)

    planner = Planner(ptgs)
    planner.load_world_map('./lot_caseStudy.png', 117.6, 68.3)  # this should not be done here !
    helper.save_object(planner, './prrt.pkl')
else: #this should not be a if - else, but if no lookup table exists then build it, no else !
    planner = helper.load_object('./prrt.pkl')  # type: Planner
    # planner.load_world_map('./lot.png', 117.6, 68.3)  # this should not be done here !


# Set initial pose and goal pose TODO: input via *.ini configuration file
x_init = 80 #unit? px, m?
y_init = 60 #unit? px, m?
theta_init = 0.0 * np.pi  #unit? rad, deg?
x_goal = 55 #unit? px, m?
y_goal = 10 #unit? px, m?
theta_goal = -3. / 4 * np.pi  #unit? rad, deg?

#init_pose = PoseR2S1(80, 60, 0.0 * np.pi)
init_pose = PoseR2S1(x_init, y_init, theta_init)
#goal_pose = PoseR2S1(55, 10, -3. / 4 * np.pi)
goal_pose = PoseR2S1(x_goal, y_goal, theta_goal)
#TODO: when we define the goal pose, we also want the orientation of the trailer to be fixed

print(" SOLVING ")
start = time.time()
planner.solve_av(init_pose, goal_pose, 7.)  #7. is the goal_dist_tolerance ?
end = time.time()
print("time elapsed in s = ")
print(end - start)
print("Path following frames will be printed now, ")
input("Press Enter to continue...")

# Plot as configured
if plot_tree:
    planner.tree.plot_nodes(planner.world, goal_pose)
if plot_trajectory_frames:
    # delete existing frames (if any)
    file_list = [f for f in os.listdir('./out') if f.endswith('.png') and f.startswith('frame')]
    for f in file_list:
        os.remove('./out/' + f)
    planner.trace_solution(av, goal_pose)
