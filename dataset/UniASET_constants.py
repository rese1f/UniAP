
# Task Type Grouping

TASKS_KEYPOINTS2D = ['keypoints2d_0']
TASKS_MASK = ['mask_0']
TASKS_CLS = ['cls_0']

TASKS = TASKS_KEYPOINTS2D + TASKS_MASK + TASKS_CLS
# Train and Test Tasks - can be splitted in other ways
N_SPLITS = 3

TASKS_GROUP_NAMES = ["keypoints2d", "mask", "cls",]
TASKS_GROUP_LIST = [TASKS_KEYPOINTS2D, TASKS_MASK, TASKS_CLS]
TASKS_GROUP_DICT = {name: group for name, group in zip(TASKS_GROUP_NAMES, TASKS_GROUP_LIST)}

N_TASK_GROUPS = len(TASKS_GROUP_NAMES)
GROUP_UNIT = N_TASK_GROUPS // N_SPLITS

TASKS_GROUP_TRAIN = {}
TASKS_GROUP_TEST = {}
for split_idx in range(N_SPLITS):
    TASKS_GROUP_TRAIN[split_idx] = TASKS_GROUP_NAMES[:-GROUP_UNIT*(split_idx+1)] + (TASKS_GROUP_NAMES[-GROUP_UNIT*split_idx:] if split_idx > 0 else []) 
    TASKS_GROUP_TEST[split_idx] = TASKS_GROUP_NAMES[-GROUP_UNIT*(split_idx+1):-GROUP_UNIT*split_idx] if split_idx > 0 else TASKS_GROUP_NAMES[-GROUP_UNIT*(split_idx+1):] 

TASKS_GROUP_TRAIN[0] = ['keypoints2d']
TASKS_GROUP_TRAIN[1] = ['mask']
TASKS_GROUP_TRAIN[2] = ['cls']
TASKS_GROUP_TRAIN[3] = ['keypoints2d', 'mask', 'cls']
TASKS_GROUP_TRAIN[4] = ['keypoints2d', 'mask']
N_TASKS = len(TASKS)

