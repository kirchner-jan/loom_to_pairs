import json, jsonlines

# recursively process children and rank them by the size of their subtree
def process_children(history, children):
    comparisons = []
    if len(children) == 0: # no children
        return comparisons
    stubs = []
    for child in children: # for each child
        child_comparisons = process_children(history + child['text'], child['children'])
        comparisons += child_comparisons
        stubs += [(history + child['text'], len(child_comparisons))] # collect history and size of subtree

    comparisons += [stubs]
    return comparisons



# load data from json file and process it
if __name__ == '__main__':
    for fName in ['Sample_ARCADIA', 'Sample_Lawrence Builds A Computer', 'Sample_Phaedrus', 'Sample_Pen']:
        with open('data/' + fName + '.json', 'r') as f:
            loom_tree = json.load(f)
        comparisons = process_children( loom_tree['text'] ,  loom_tree['children'] )
        with jsonlines.open('data/' + fName + '.jsonl', mode='w') as writer:
            for comparison in comparisons:
                writer.write(comparison)