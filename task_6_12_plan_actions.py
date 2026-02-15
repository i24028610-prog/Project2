def plan_actions(cur_x: int, cur_rot: int, target_x: int, target_rot: int,
                 A_LEFT, A_RIGHT, A_ROTATE, A_HARD_DROP):
    actions = []

    # 先旋转到目标朝向（假设 A_ROTATE 是“顺时针旋转一次”）
    rot_steps = (target_rot - cur_rot) % 4
    for _ in range(rot_steps):
        actions.append(A_ROTATE)

    # 再左右移动到目标列
    dx = target_x - cur_x
    if dx < 0:
        actions += [A_LEFT] * (-dx)
    elif dx > 0:
        actions += [A_RIGHT] * dx

    # 最后硬降
    actions.append(A_HARD_DROP)
    return actions
