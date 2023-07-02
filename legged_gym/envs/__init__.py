# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgSAC, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .base.legged_robot_config import LeggedRobotCfg,LeggedRobotCfgAlg
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgSAC

from .littledog_terrain_v2.littledog_terrain import LittledogTerrain
from .littledog_terrain_v2.littledog_terrain_config import LittledogTerrainCfg,LittledogTerrainCfgPPO


from .littledog_ik.littledog_ik import LittledogIK
from .littledog_CPG.littledog_CPG import LittledogCPG
from .littledog.littledog_config import LittledogRoughCfg, LittledogRoughCfgPPO
from .littledog_ik.littledog_ik_config import LittledogIKRoughCfg, LittledogIKRoughCfgPPO
# from .littledog_CPG.littledog_CPG_config import LittledogCPGRoughCfg, LittledogCPGRoughCfgPPO
# from .littledog_terrain.littledog_terrain_config import LittledogTerrainCfg,LittledogTerrainCfgPPO
# from .littledog_terrain.littledog_terrain import LittledogTerrain

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgSAC() )
task_registry.register( "a1_ppo", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "littledog", LittledogTerrain, LittledogTerrainCfg(), LittledogTerrainCfgPPO())
# task_registry.register( "littledog_ik", LittledogIK, LittledogIKRoughCfg(), LittledogIKRoughCfgPPO())
# task_registry.register( "littledog_CPG", LittledogCPG, LittledogCPGRoughCfg(), LittledogCPGRoughCfgPPO() )
# task_registry.register( "littledog_terrain", LittledogTerrain, LittledogTerrainCfg(),LittledogTerrainCfgPPO())