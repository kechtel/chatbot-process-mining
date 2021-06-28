import os

from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri.exporter import exporter
from pm4py.visualization.petrinet import visualizer as pn_visualizer


def apply_alpha_miner(log, path, filename):
    net, initial_marking, final_marking = alpha_miner.apply(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    if filename is not None:
        exporter.apply(net, initial_marking, os.path.join(path, filename.format(algorithm='alpha_miner') + '.pnml'), final_marking)
        pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='alpha_miner') + '.png'))
    return net, initial_marking, final_marking


def apply_inductive_miner(log, path, filename):
    net, initial_marking, final_marking = inductive_miner.apply(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    if filename is not None:
        exporter.apply(net, initial_marking, os.path.join(path, filename.format(algorithm='inductive_miner') + '.pnml'), final_marking)
        pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='inductive_miner') + '.png'))
    return net, initial_marking, final_marking


def apply_inductive_miner_imf(log, path, filename):
    net, initial_marking, final_marking = inductive_miner.apply(log, variant=inductive_miner.Variants.IMf)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    if filename is not None:
        exporter.apply(net, initial_marking, os.path.join(path, filename.format(algorithm='inductive_miner_infrequent') + '.pnml'), final_marking)
        pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='inductive_miner_infrequent') + '.png'))
    return net, initial_marking, final_marking


def apply_inductive_miner_imd(log, path, filename):
    net, initial_marking, final_marking = inductive_miner.apply(log, variant=inductive_miner.Variants.IMd)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    if filename is not None:
        exporter.apply(net, initial_marking, os.path.join(path, filename.format(algorithm='inductive_miner_dfg') + '.pnml'), final_marking)
        pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='inductive_miner_dfg') + '.png'))
    return net, initial_marking, final_marking


def apply_heuristics_miner(log, path, filename):
    net, initial_marking, final_marking = heuristics_miner.apply(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, variant=pn_visualizer.Variants.FREQUENCY, log=log)
    if filename is not None:
        exporter.apply(net, initial_marking, os.path.join(path, filename.format(algorithm='heuristics_miner') + '.pnml'), final_marking)
        pn_visualizer.save(gviz, os.path.join(path, filename.format(algorithm='heuristics_miner') + '.png'))
    return net, initial_marking, final_marking
