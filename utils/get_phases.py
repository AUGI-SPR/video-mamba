def get_phases(mapping_file):
    file_ptr = open(mapping_file, "r")
    actions = file_ptr.read().split("\n")[:-1]
    file_ptr.close()
    phases_dict = dict()
    for a in actions:
        phases_dict[a.split()[1]] = int(a.split()[0])
    index2label = dict()
    for k, v in phases_dict.items():
        index2label[v] = k
    num_classes = len(phases_dict)

    return phases_dict, num_classes
