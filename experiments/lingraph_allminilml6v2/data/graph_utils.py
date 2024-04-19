def linearize_graph(graph_dict, sep_token='[SEP]'):
    """Borrowed from baseline, needs to be modified..."""
    
    nodes = sorted((node_dict for node_dict in graph_dict["nodes"]), key=lambda d:d["id"])
    for n_id, node_dict in enumerate(nodes):
        assert n_id == node_dict["id"]
    src_node_id2links = {}
    
    for link_dict in graph_dict["links"]:
        link_src =  link_dict["source"]
        if src_node_id2links.get(link_src) is None:
            src_node_id2links[link_src] = []
        src_node_id2links[link_src].append(link_dict)
    
    graph_s = ""
    for n_id, node_dict in enumerate(nodes):
        links = src_node_id2links.get(n_id, list())
        start_label = node_dict["label"]
        if node_dict["type"] == "ANSWER_CANDIDATE_ENTITY":
            start_label = f"{sep_token} {start_label} {sep_token}"
        for link_dict in links:
            target_label = nodes[link_dict["target"]]["label"]
            if nodes[link_dict["target"]]["type"] == "ANSWER_CANDIDATE_ENTITY":
                target_label = f"{sep_token} {target_label} {sep_token}"
            # pqlet - changed format, separation of triplets with ';'
            link_s = f" {start_label}, {link_dict['label']}, {target_label} ;"
            graph_s += link_s
    return graph_s