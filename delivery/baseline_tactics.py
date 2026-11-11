"""混合战术基础模块。路线来自公开墙体，运动由原版SWF执行。"""
import math
from copy import deepcopy

import numpy as np

from flash_env import encode_action
from flash_navigation import maze_distances


TEMPLATES = ("Assault", "Sniper", "Defensive", "Guerrilla", "Hunter", "Flanking", "Control")
NAV_CONFIG = dict(arrival_radius=.12, turn_deadband=.06, move_angle_limit=.22)
AIM_CONFIG = dict(muzzle_offset=4.5/16, target_radius=.15, self_radius=.22,
                  max_bounces=2, max_distance=12., turn_deadband=.09,
                  reposition_distance=.2, reposition_limit=40, trigger_mode="hold")
DODGE_CONFIG = dict(horizon=32, hold_frames=8, body_radius=.26,
                    trigger_clearance=.07, minimum_gain=.03, collision_radius=.21)


def control(obs, memory, enable_dodge=True, template=0, aggression=.5):
    """紧急躲弹覆盖当前动作；只有实际执行普通行为时才提交其内部进度。"""
    tank = np.asarray(obs["tanks"]).reshape(2,32)[0]
    if not tank[0]:
        return 8, dict(reason="dead",intervention=False)
    if template not in range(len(TEMPLATES)) or not 0 <= aggression <= 1:
        raise ValueError("模板或攻击性参数无效")
    tentative = deepcopy(memory)
    baseline, ordinary = template_action(obs,tentative,template,aggression)
    if enable_dodge:
        action, emergency = dodge(obs,baseline)
        if emergency["intervention"]:
            return action, dict(reason="emergency_dodge",intervention=True,
                                baseline_action=baseline,ordinary=ordinary,emergency=emergency)
    memory.clear()
    memory.update(tentative)
    return baseline, dict(reason=ordinary["reason"],intervention=False,ordinary=ordinary)


def template_goal(obs, template, aggression):
    """沿用七种模板职责。距离与评分为原版适配初值，须由专项与实战检验。"""
    width, height, walls, distances = maze_geometry(obs)
    tanks = np.asarray(obs["tanks"]).reshape(2,32)
    own, enemy = tanks[:,1:3]*[12,10]
    tiles = np.floor([own,enemy]).astype(int).clip([0,0],[width-1,height-1])
    own_id = int(tiles[0,1]*width+tiles[0,0])
    enemy_id = int(tiles[1,1]*width+tiles[1,0])
    ids = np.flatnonzero(np.isfinite(distances[own_id]))
    centers = np.stack([ids%width+.5,ids//width+.5],axis=1)
    travel = distances[own_id,ids]
    separation = distances[enemy_id,ids]
    desired = max(.5,4.-3.*aggression)
    if template == 1:  # Sniper，保持较远的交战距离
        cost = np.abs(separation-(desired+2))+.12*travel
    elif template == 2:  # Defensive，靠近掩体并拉开距离
        cover = []
        for x,y in np.floor(centers).astype(int):
            cover.append(walls[0,y,x]+walls[0,y+1,x]+walls[1,y,x]+walls[1,y,x+1])
        cost = np.maximum(desired+1-separation,0)*2+.18*travel-.2*np.asarray(cover)
    elif template == 3:  # Guerrilla，射后短时撤出交战距离
        cost = np.abs(separation-(desired+1))+.08*travel
    elif template == 5:  # Flanking，选择敌方朝向的侧面
        lateral = np.array([tanks[1,3],tanks[1,4]])
        offsets = centers-enemy
        side = np.abs(offsets@lateral)
        cost = np.abs(np.linalg.norm(offsets,axis=1)-desired)-.5*side+.1*travel
    elif template == 6:  # Control，优先占据多个通路连接点
        degrees = (distances[ids]==1).sum(axis=1)
        cost = -.8*degrees+.1*travel+.15*separation
    else:
        cost = separation+.1*travel
    cost = np.where(ids==enemy_id,np.inf,cost)
    return centers[int(np.argmin(cost))].tolist()


def template_action(obs, memory, template, aggression):
    if template == 0 and aggression >= .35:
        return aim_and_navigate(obs,memory)
    tanks = np.asarray(obs["tanks"]).reshape(2,32)
    enemy = tanks[1,1:3]*[12,10]
    frame = int(round(float(obs["global"][0])*1500))
    prior = memory.get("enemy_history")
    velocity = np.zeros(2)
    if prior is not None and frame > prior["frame"]:
        velocity = (enemy-np.asarray(prior["position"]))/(frame-prior["frame"])
    memory["enemy_history"] = dict(frame=frame,position=enemy.tolist())
    action, detail = aim(obs)
    retreating = template == 3 and frame < memory.get("retreat_until",-1)
    if detail["reason"] in ("fire","release_trigger") and not retreating:
        if template == 3:
            memory["retreat_until"] = frame+25
        return action, dict(detail,template=TEMPLATES[template])
    if template == 4:
        width, height, _, _ = maze_geometry(obs)
        goal = np.clip(enemy+velocity*12,[.5,.5],[width-.5,height-.5]).tolist()
    else:
        goal = template_goal(obs,2 if template==0 else template,aggression)
    # 非撤退状态下保留有明确射击路径时的对准行为。
    if detail["reason"]=="align" and not retreating:
        return action, dict(detail,template=TEMPLATES[template],goal=goal)
    action, navigation = navigate(obs,goal,memory.setdefault("template_navigation",{}))
    return action, dict(navigation,template=TEMPLATES[template],goal=goal,
                        reason="template_retreat" if retreating else "template_navigate")


def forecast_bullets(obs, segments):
    """普通炮弹的公开速度和厚墙反射近似，输出(H,N,2)格子坐标。"""
    entities = np.asarray(obs["entities"]).reshape(-1,16)
    bullets = entities[(entities[:,0]>.5)&(np.abs(entities[:,1]-2/13)<1e-5)]
    positions = bullets[:,2:4]*[12,10]
    velocities = bullets[:,6:8]*7
    result = []
    for _ in range(DODGE_CONFIG["horizon"]):
        for index in range(len(bullets)):
            speed = float(np.linalg.norm(velocities[index]))
            if speed < 1e-10:
                continue
            direction = velocities[index]/speed
            distance, axis, inside = wall_contact(positions[index],direction,segments)
            if inside:
                velocities[index] *= -1
            elif distance < speed:
                positions[index] += direction*distance
                velocities[index,axis] *= -1
                direction = velocities[index]/speed
                positions[index] += direction*(speed-distance+1e-5)
                continue
            positions[index] += velocities[index]
        result.append(positions.copy())
    return np.asarray(result)


def dodge(obs, baseline_action=8):
    """按32帧几何净空选择短动作，每个真实帧可重新规划。"""
    segments = wall_segments(obs)
    future = forecast_bullets(obs,segments)
    if future.shape[1] == 0:
        return baseline_action, dict(reason="no_bullets",intervention=False)
    tank = np.asarray(obs["tanks"]).reshape(2,32)[0]
    initial = tank[1:3]*[12,10]
    heading = math.atan2(float(tank[4]),float(tank[3]))
    actions = list(dict.fromkeys([baseline_action//2*2]+list(range(0,18,2))))
    scores = []
    for action in actions:
        position = initial.copy()
        angle = heading
        path = []
        move, turn = action//6-1, action//2%3-1
        for frame in range(DODGE_CONFIG["horizon"]):
            if frame < DODGE_CONFIG["hold_frames"]:
                angle += math.radians(float(tank[13])*10*turn)
                speed = float(tank[14])*.08 if move>0 else -float(tank[15])*.05 if move<0 else 0.
                direction = np.array([math.sin(angle),-math.cos(angle)])
                candidate = position+direction*speed
                blocked = False
                for axis, boundary, lower, upper, half in segments:
                    normal = 1-axis
                    gap_normal = max(abs(candidate[normal]-boundary)-half,0.)
                    gap_side = max(lower-half-candidate[axis],candidate[axis]-upper-half,0.)
                    if math.hypot(gap_normal,gap_side) < DODGE_CONFIG["collision_radius"]:
                        blocked = True
                        break
                if not blocked:
                    position = candidate
            path.append(position.copy())
        clearance = float(np.linalg.norm(np.asarray(path)[:,None,:]-future,axis=2).min())-DODGE_CONFIG["body_radius"]
        scores.append(clearance)
    best = int(np.argmax(scores))
    intervene = scores[0]<DODGE_CONFIG["trigger_clearance"] and scores[best]>scores[0]+DODGE_CONFIG["minimum_gain"]
    return (actions[best] if intervene else baseline_action), dict(reason="dodge" if intervene else "continue",
        intervention=intervene,baseline_clearance=scores[0],chosen_clearance=scores[best],candidate_scores=scores)


def wall_segments(obs, max_width=12, max_height=10):
    width, height, walls, _ = maze_geometry(obs, max_width, max_height)
    scale = float(obs["global"][3])*100
    half_width = math.floor(scale/16)/scale
    segments = []
    for y, x in np.argwhere(walls[0]):
        if x < width:
            segments.append((0, math.floor(y*scale)/scale, math.floor(x*scale)/scale,
                             math.floor((x+1)*scale)/scale, half_width))
    for y, x in np.argwhere(walls[1]):
        if y < height:
            segments.append((1, math.floor(x*scale)/scale, math.floor(y*scale)/scale,
                             math.floor((y+1)*scale)/scale, half_width))
    return segments


def ray_target_distance(origin, direction, center, radius):
    delta = center-origin
    projection = float(delta@direction)
    perpendicular2 = float(delta@delta)-projection*projection
    if perpendicular2 > radius*radius:
        return math.inf
    entry = projection-math.sqrt(max(0.,radius*radius-perpendicular2))
    return max(entry,0.) if projection >= 0 else math.inf


def wall_contact(origin, direction, segments):
    wall_distance, hit_axis = math.inf, None
    for axis, boundary, lower, upper, half in segments:
        bounds = [(lower-half,upper+half),(boundary-half,boundary+half)]
        if axis == 1:
            bounds.reverse()
        enter, leave, normal = -math.inf, math.inf, None
        inside = True
        for component, (low,high) in enumerate(bounds):
            inside = inside and low <= origin[component] <= high
            if abs(direction[component]) < 1e-10:
                if not low <= origin[component] <= high:
                    leave = -math.inf
                continue
            a = (low-origin[component])/direction[component]
            b = (high-origin[component])/direction[component]
            near, far = min(a,b), max(a,b)
            if near > enter:
                enter, normal = near, component
            leave = min(leave,far)
        if inside:
            return 0., None, True
        if 1e-6 < enter <= leave and enter < wall_distance:
            wall_distance, hit_axis = enter, normal
    return wall_distance, hit_axis, False


def predict_shot(position, heading, enemy, segments):
    """厚墙反射的几何近似，厚度和像素取整来自原版drawMaze。"""
    direction = np.array([math.sin(heading),-math.cos(heading)])
    origin = position+direction*AIM_CONFIG["muzzle_offset"]
    traveled = 0.
    path = []
    for bounce in range(AIM_CONFIG["max_bounces"]+1):
        wall_distance, hit_axis, inside = wall_contact(origin,direction,segments)
        if inside:
            return dict(result="wall_origin",bounces=bounce,distance=traveled,path=path)
        enemy_distance = ray_target_distance(origin,direction,enemy,AIM_CONFIG["target_radius"])
        own_distance = ray_target_distance(origin,direction,position,AIM_CONFIG["self_radius"])
        limit = min(wall_distance,AIM_CONFIG["max_distance"]-traveled)
        if min(enemy_distance,own_distance) <= limit:
            result = "enemy" if enemy_distance < own_distance else "self"
            return dict(result=result,bounces=bounce,distance=traveled+min(enemy_distance,own_distance),path=path)
        if hit_axis is None or wall_distance > limit:
            break
        end = origin+direction*wall_distance
        path.append([origin.tolist(),end.tolist()])
        traveled += wall_distance
        direction[hit_axis] *= -1
        origin = end+direction*1e-5
    return dict(result="clear",bounces=len(path),distance=traveled,path=path)


def aim(obs, max_width=12, max_height=10):
    tanks = np.asarray(obs["tanks"]).reshape(2,32)
    position, enemy = tanks[:,1:3]*[max_width,max_height]
    heading = math.atan2(float(tanks[0,4]),float(tanks[0,3]))
    segments = wall_segments(obs,max_width,max_height)
    current = predict_shot(position,heading,enemy,segments)
    if tanks[0,18] > .5 and tanks[1,0] > .5 and current["result"] == "enemy":
        released = tanks[0,6] > .5 or AIM_CONFIG["trigger_mode"]=="hold"
        return encode_action(fire=bool(released)), dict(reason="fire" if released else "release_trigger",prediction=current)
    targets = [enemy]
    reflected_planes = set()
    for axis, boundary, _, _, _ in segments:
        if (axis,boundary) in reflected_planes:
            continue
        reflected_planes.add((axis,boundary))
        reflected = enemy.copy()
        reflected[1-axis] = 2*boundary-reflected[1-axis]
        targets.append(reflected)
    candidates = []
    for target in targets:
        delta = target-position
        angle = math.atan2(float(delta[0]),-float(delta[1]))
        prediction = predict_shot(position,angle,enemy,segments)
        if prediction["result"] == "enemy":
            error = (angle-heading+math.pi)%(2*math.pi)-math.pi
            candidates.append((prediction["bounces"],abs(error),error))
    if candidates:
        _, _, error = min(candidates)
        turn = int(np.sign(error)) if abs(error)>AIM_CONFIG["turn_deadband"] else 0
        return encode_action(turn=turn), dict(reason="align" if turn else "alignment_quantized",heading_error=error)
    return encode_action(), dict(reason="no_shot",prediction=current)


def aim_and_navigate(obs, memory):
    """无法形成可执行射角时，完成一次短移位再重新瞄准，避免两个控制器逐帧抢转向。"""
    tanks = np.asarray(obs["tanks"]).reshape(2,32)
    position = tanks[0,1:3]*[12,10]
    start = memory.get("reposition_start")
    if start is not None:
        moved = np.linalg.norm(position-start)
        if moved < AIM_CONFIG["reposition_distance"] and memory["reposition_frames"] < AIM_CONFIG["reposition_limit"]:
            memory["reposition_frames"] += 1
            return navigate(obs,tanks[1,1:3]*[12,10],memory.setdefault("navigation",{}))
        memory.pop("reposition_start")
    action, detail = aim(obs)
    if detail["reason"] in ("no_shot","alignment_quantized"):
        memory.update(reposition_start=position.copy(),reposition_frames=1)
        return navigate(obs,tanks[1,1:3]*[12,10],memory.setdefault("navigation",{}))
    return action, detail


def maze_geometry(obs, max_width=12, max_height=10):
    width = int(round(float(obs["global"][1]) * max_width))
    height = int(round(float(obs["global"][2]) * max_height))
    grid = np.asarray(obs["map"]).reshape(3, max_height + 1, max_width + 1)
    walls = np.ascontiguousarray(grid[1:, :height + 1, :width + 1] > .5, dtype=np.uint8)
    distances = maze_distances(width, height, walls.tobytes())
    return width, height, walls, distances


def navigate(obs, goal, memory, max_width=12, max_height=10):
    """goal使用格子坐标，memory仅保存当前路点。朝向0度朝上，正角顺时针。"""
    width, height, walls, distances = maze_geometry(obs, max_width, max_height)
    tank = np.asarray(obs["tanks"]).reshape(2, 32)[0]
    position = tank[1:3] * [max_width, max_height]
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (2,) or not ((goal >= 0).all() and (goal < [width, height]).all()):
        raise ValueError("导航目标超出地图")
    if np.linalg.norm(position - goal) < NAV_CONFIG["arrival_radius"]:
        memory.clear()
        return encode_action(), dict(reason="arrived", distance=float(np.linalg.norm(position-goal)))
    tile = np.floor(position).astype(int).clip([0, 0], [width-1, height-1])
    target = np.floor(goal).astype(int)
    destination = int(target[1] * width + target[0])
    current = int(tile[1] * width + tile[0])
    if not np.isfinite(distances[current, destination]):
        return encode_action(), dict(reason="unreachable")
    waypoint = memory.get("waypoint")
    if waypoint is None or memory.get("goal") != goal.tolist() or np.linalg.norm(position-waypoint) < NAV_CONFIG["arrival_radius"]:
        if current == destination:
            waypoint = goal
        else:
            x, y = tile
            neighbors = [(x-1,y,walls[1,y,x]), (x+1,y,walls[1,y,x+1]),
                         (x,y-1,walls[0,y,x]), (x,y+1,walls[0,y+1,x])]
            candidates = [(distances[ny*width+nx,destination],nx,ny) for nx,ny,blocked in neighbors
                          if 0 <= nx < width and 0 <= ny < height and not blocked]
            _, nx, ny = min(candidates)
            waypoint = np.array([nx+.5, ny+.5])
        memory.update(waypoint=np.asarray(waypoint), goal=goal.tolist())
    delta = waypoint-position
    desired = math.atan2(float(delta[0]), -float(delta[1]))
    heading = math.atan2(float(tank[4]), float(tank[3]))
    error = (desired-heading+math.pi) % (2*math.pi)-math.pi
    turn = int(np.sign(error)) if abs(error) > NAV_CONFIG["turn_deadband"] else 0
    move = int(abs(error) < NAV_CONFIG["move_angle_limit"])
    return encode_action(move,turn), dict(reason="navigate", waypoint=np.asarray(waypoint).tolist(),
        heading_error=error, graph_distance=float(distances[current,destination]))
