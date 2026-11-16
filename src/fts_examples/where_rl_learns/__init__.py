# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Where does reinforcement learning learn?
========================================

Shared machinery for reproducing part of two papers on Qwen3-1.7B-Base, which ask which parameters RL post-training
actually needs: `Is One Layer Enough? <https://www.alphaxiv.org/abs/2607.01232>`_ and `Understanding LLM Parameter
Update Sparsity through the Lens of Fisher <https://www.alphaxiv.org/abs/2609.36262>`_.

The recipe is the Fisher paper's (its Appendix B.2): Dr. GRPO over SimpleRL-Zoo's Abel-format MATH levels 3 to 5, with
binary answer checks on MATH-500.
"""
