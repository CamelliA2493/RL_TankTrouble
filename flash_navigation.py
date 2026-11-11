"""只从公开墙体计算四邻域图距离；不计算坦克运动、碰撞或按键。"""
from collections import deque
from functools import lru_cache

import numpy as np


NAVIGATION_SIZE = 33
DISTANCE_SCALE = 16.0
LOCAL_DISTANCE_SCALE = 4.0
OFFSETS = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=np.int64)


@lru_cache(maxsize=8)
def maze_distances(width, height, wall_bytes):
    """返回 (H*W, H*W)。横墙在 y 网格线上，竖墙在 x 网格线上。"""
    walls = np.frombuffer(wall_bytes, dtype=np.uint8).reshape(2, height + 1, width + 1)
    adjacent = [[] for _ in range(width * height)]
    for y in range(height):
        for x in range(width):
            node = y * width + x
            for nx, ny, blocked in ((x - 1, y, walls[1, y, x]), (x + 1, y, walls[1, y, x + 1]),
                                    (x, y - 1, walls[0, y, x]), (x, y + 1, walls[0, y + 1, x])):
                if 0 <= nx < width and 0 <= ny < height and not blocked:
                    adjacent[node].append(ny * width + nx)
    distances = np.full((width * height, width * height), np.inf, dtype=np.float32)
    for start in range(width * height):
        distances[start, start] = 0
        pending = deque([start])
        while pending:
            node = pending.popleft()
            for target in adjacent[node]:
                if not np.isfinite(distances[start, target]):
                    distances[start, target] = distances[start, node] + 1
                    pending.append(target)
    distances.setflags(write=False)
    return distances


def navigation_numpy(obs, max_width=12, max_height=10):
    """33 维：两个目标的局部距离差18、有效格9、绝对距离2、存在标记2、箱子相对坐标2。"""
    width = int(round(float(obs['global'][1]) * max_width))
    height = int(round(float(obs['global'][2]) * max_height))
    if width < 1 or height < 1 or width > max_width or height > max_height:
        raise ValueError('导航特征的地图尺寸无效')
    grid = np.asarray(obs['map'], dtype=np.float32).reshape(3, max_height + 1, max_width + 1)
    walls = np.ascontiguousarray(grid[1:, :height + 1, :width + 1] > .5, dtype=np.uint8)
    distances = maze_distances(width, height, walls.tobytes())
    tanks = np.asarray(obs['tanks'], dtype=np.float32).reshape(2, 32)
    positions = tanks[:, 1:3] * np.array([max_width, max_height], dtype=np.float32)
    tiles = np.floor(positions).astype(np.int64).clip([0, 0], [width - 1, height - 1])
    own = tiles[0, 1] * width + tiles[0, 0]
    local = tiles[0] + OFFSETS
    valid = ((local >= 0) & (local < [width, height])).all(1)
    clipped = local.clip([0, 0], [width - 1, height - 1])
    local_ids = clipped[:, 1] * width + clipped[:, 0]
    # 这只是局部格子的图连通性；不会判断坦克当前姿态下能否通过墙角。
    valid &= distances[own, local_ids] <= np.abs(OFFSETS).sum(1)
    fields, present, relative_crate = [], [], np.zeros(2, dtype=np.float32)
    enemy = tiles[1, 1] * width + tiles[1, 0]
    fields.append(distances[:, enemy])
    present.append(float(tanks[1, 0] > .5))
    entities = np.asarray(obs['entities'], dtype=np.float32).reshape(-1, 16)
    crates = entities[(entities[:, 0] > .5) & (np.abs(entities[:, 1] - 9 / 13) < 1e-5)]
    if len(crates):
        crate_positions = crates[:, 2:4] * np.array([max_width, max_height], dtype=np.float32)
        crate_tiles = np.floor(crate_positions).astype(np.int64).clip([0, 0], [width - 1, height - 1])
        crate_ids = crate_tiles[:, 1] * width + crate_tiles[:, 0]
        fields.append(distances[:, crate_ids].min(1))
        nearest = int(np.argmin(distances[own, crate_ids]))
        dx, dy = crate_positions[nearest] - positions[0]
        cos, sin = tanks[0, 3:5]
        relative_crate[:] = np.clip([(dx * sin - dy * cos) / 4, (dx * cos + dy * sin) / 4], -2, 2)
        present.append(1.)
    else:
        fields.append(np.zeros(width * height, dtype=np.float32))
        present.append(0.)
    patches, absolute = [], []
    for field, exists in zip(fields, present):
        center = field[own]
        values = np.full(9, 2., dtype=np.float32)
        finite = np.isfinite(field[local_ids])
        if np.isfinite(center):
            values[finite] = np.clip((field[local_ids[finite]] - center) / LOCAL_DISTANCE_SCALE, -2, 2)
        patches.append(values * valid * exists)
        absolute.append(min(float(center) / DISTANCE_SCALE, 8.) * exists)
    return np.concatenate([*patches, valid.astype(np.float32), np.array(absolute, dtype=np.float32),
                           np.array(present, dtype=np.float32), relative_crate]).astype(np.float32)
