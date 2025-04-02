{
    'Objects': ['bin', 'rubbish', 'tomato1', 'tomato2'], 
    'Query': 'pick up the rubbish and leave it in the trash can', 
    'Planner': ['grasp the rubbish', 'back to default pose', 'move to the top of the bin', 'open gripper'], 
    'Action': 'grasp the rubbish', 
    'movable': 'gripper', 
    'affordable': {
        'object':'rubbish',
        'x': "object['position'][0]",
        'y': "object['position'][1]",
        'z': "object['position'][2]",
        'target_affordance': 1, 
        'set': 'affordance_map[x, y, z] = target_affordance'
        }, 
    'avoid': {
        'object':'bin', 
        'x': "object['position'][0]", 
        'y': "object['position'][1]", 
        'z': "object['position'][2]", 
        'radius_cm': 5,
        'set': 'set_voxel_by_radius(avoidance_map, [x,y,z], radius_cm=radius_cm, value=1)'
        }, 
    'gripper': {
        'object':'rubbish', 
        'x': "object['position'][0]", 
        'y': "object['position'][1]", 
        'z': "object['position'][2]", 
        'radius_cm': 1,
        'set': 'set_voxel_by_radius(gripper_map, [x,y,z], radius_cm=radius_cm, value=0)'
        }, 
    'rotation': {
        'object':'rubbish', 
        'target_rotation': "vec2quat(-object['normal'])", 
        'set': 'rotation_map[:, :, :] = target_rotation'
        }, 
    'velocity': {
        'set': 'default'
        }
 }

a="3"
b=eval(a)
print(b)