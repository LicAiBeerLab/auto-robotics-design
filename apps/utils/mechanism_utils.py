import numpy as np
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, MIT_CHEETAH_PARAMS_DICT

def get_builder():
    """Creates builder for 

    Returns:
        _type_: _description_
    """
    thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    density = MIT_CHEETAH_PARAMS_DICT["density"]
    body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]

    builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
                                density={"default": density, "G": body_density},
                                thickness={"default": thickness, "EE": 0.003},
                                actuator={"default": actuator},
                                size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
                                offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"]
                                )
    return builder