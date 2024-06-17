import numpy as np

from auto_robot_design.description.builder import jps_graph2pinocchio_robot
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory
from auto_robot_design.optimization.problems import get_optimizing_joints
from auto_robot_design.optimization.saver import (
    ProblemSaver, )
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator, visualize_constrains
from auto_robot_design.description.builder import jps_graph2pinocchio_robot
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths, get_optimizing_joints, CalculateMultiCriteriaProblem
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import get_steped_round_trajectory, convert_x_y_to_6d_traj_xz, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory,get_workspace_trajectory,get_horizontal_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward, PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ForceEllipsoidReward, ZRRReward, MinForceReward,MinManipulabilityReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward
from auto_robot_design.description.actuators import TMotor_AK10_9, TMotor_AK60_6, TMotor_AK70_10, TMotor_AK80_64, TMotor_AK80_9
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot, MIT_CHEETAH_PARAMS_DICT


class KinematicDataset:
    def __init__(self, graph, builder, jp2limits) -> None:
        self.graph = graph
        self.builder = builder
        self.jp2limits = jp2limits
        self.opt_joints = list(self.jp2limits.keys())
        self.initial_xopt, self.upper_bounds, self.lower_bounds = self.convert_joints2x_opt()

    def convert_joints2x_opt(self):
        x_opt = np.zeros(len(self.opt_joints) * 2)
        upper_bounds = np.zeros(len(x_opt))
        lower_bounds = np.zeros(len(x_opt))
        i = 0
        for jp in self.opt_joints:
            lims = self.jp2limits[jp]
            x_opt[i: i + 2] = np.array([jp.r[0], jp.r[2]])
            upper_bounds[i: i + 2] = np.array(lims[2:]) + x_opt[i: i + 2]
            lower_bounds[i: i + 2] = np.array(lims[:2]) + x_opt[i: i + 2]
            i += 2

        return x_opt, upper_bounds, lower_bounds

    def mutate_JP_by_xopt(self, x_opt):
        num_params_one_jp = len(x_opt) // len(self.opt_joints)

        for id, jp in zip(range(0, len(x_opt), num_params_one_jp), self.opt_joints):
            xz = x_opt[id: (id + num_params_one_jp)]
            list_nodes = list(self.graph.nodes())
            id = list_nodes.index(jp)
            list_nodes[id].r = np.array([xz[0], 0, xz[1]])
    
    def sample_and_rank(self, sample_size, ranking_step, max_ranking_steps = 10,grid_step:int = 100):
        rnd_gen = np.random.default_rng()
        generated_nums= rnd_gen.choice(int(grid_step+1), size=(sample_size, len(self.initial_xopt)))
        sampled_values = self.initial_xopt + self.lower_bounds + generated_nums*(self.upper_bounds - self.lower_bounds)*0.01 
        for x_opt in sampled_values:
            self.mutate_JP_by_xopt(x_opt)
            fixed_robot, free_robot = jps_graph2pinocchio_robot(self.graph, self.builder)
            trajectory = get_steped_round_trajectory([0,-0.3], 0.01, n=10, angle_steps=100)
            dict_trajectory_criteria = {}
            # criteria calculated for each point on the trajectory
            dict_point_criteria = {}
            crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
            point_criteria_vector, trajectory_criteria, res_dict_fixed = crag.get_criteria_data(
                    fixed_robot, free_robot, trajectory)
            pass
        return res_dict_fixed



if __name__ == "__main__":
    generator = TwoLinkGenerator()
    all_graphs = generator.get_standard_set(-0.105, shift=-0.10)
    graph, constrain_dict = all_graphs[0]

    thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    density = MIT_CHEETAH_PARAMS_DICT["density"]
    body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]


    builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
                                density={"default": density, "G":body_density},
                                thickness={"default": thickness, "EE":0.033},
                                actuator={"default": actuator},
                                size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
                                offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"]
    )
    optimizing_joints = get_optimizing_joints(graph, constrain_dict)
    KinematicDataset(graph,  builder, optimizing_joints).sample_and_rank(sample_size = 3, ranking_step=2)
