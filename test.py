import json

str1 = '''{
    "affordable": {
        "x": "rubbish['position'][0]",
        "y": "rubbish['position'][1]",
        "z": "rubbish['position'][2]",
        "target_affordance": 1,
        "set": "affordance_map[x, y, z] = target_affordance"
    },
    "avoid" : {
        "x" : "mug['position'][0]",
        "y" : "mug['position'][1]",
        "z" : "mug['position'][2]",
        "set" : "set_voxel_by_radius(avoidance_map, [x,y,z], radius_cm=5, value=1)"
    }
}'''

# Now it should load properly
print(json.loads(str1))
