# Copyright 2025-present the HuggingFace Inc. team.
#
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
"""
Dimensional reduction amounts for different wavelet families during wavelet transforms Each tuple (rows, cols)
represents the reduction in matrix dimensions that occurs when applying wavelet decomposition/reconstruction due to
boundary effects and filter sizes. These values are used to pre-pad matrices before wavelet processing to ensure the
reconstructed matrix maintains the original target dimensions.
"""

WAVELET_REDUCTIONS = {
    "db1": (0, 0),
    "db2": (2, 2),
    "db3": (4, 4),
    "db4": (6, 6),
    "db5": (8, 8),
    "db6": (10, 10),
    "db7": (12, 12),
    "db8": (14, 14),
    "db9": (16, 16),
    "db10": (18, 18),
    "db11": (20, 20),
    "db12": (22, 22),
    "db13": (24, 24),
    "db14": (26, 26),
    "db15": (28, 28),
    "db16": (30, 30),
    "db17": (32, 32),
    "db18": (34, 34),
    "db19": (36, 36),
    "db20": (38, 38),
    "db21": (40, 40),
    "db22": (42, 42),
    "db23": (44, 44),
    "db24": (46, 46),
    "db25": (48, 48),
    "db26": (50, 50),
    "db27": (52, 52),
    "db28": (54, 54),
    "db29": (56, 56),
    "db30": (58, 58),
    "db31": (60, 60),
    "db32": (62, 62),
    "db33": (64, 64),
    "db34": (66, 66),
    "db35": (68, 68),
    "db36": (70, 70),
    "db37": (72, 72),
    "db38": (74, 74),
    "sym2": (2, 2),
    "sym3": (4, 4),
    "sym4": (6, 6),
    "sym5": (8, 8),
    "sym6": (10, 10),
    "sym7": (12, 12),
    "sym8": (14, 14),
    "sym9": (16, 16),
    "sym10": (18, 18),
    "sym11": (20, 20),
    "sym12": (22, 22),
    "sym13": (24, 24),
    "sym14": (26, 26),
    "sym15": (28, 28),
    "sym16": (30, 30),
    "sym17": (32, 32),
    "sym18": (34, 34),
    "sym19": (36, 36),
    "sym20": (38, 38),
    "coif1": (4, 4),
    "coif2": (10, 10),
    "coif3": (16, 16),
    "coif4": (22, 22),
    "coif5": (28, 28),
    "coif6": (34, 34),
    "coif7": (40, 40),
    "coif8": (46, 46),
    "coif9": (52, 52),
    "coif10": (58, 58),
    "coif11": (64, 64),
    "coif12": (70, 70),
    "coif13": (76, 76),
    "coif14": (82, 82),
    "coif15": (88, 88),
    "coif16": (94, 94),
    "coif17": (100, 100),
}
