import argparse
import os

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.conformance.logs_alignments import algorithm as logs_alignments
from pm4py.algo.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.algo.evaluation.precision import evaluator as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.algo.evaluation.simplicity import evaluator as simplicity_evaluator
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.petri.importer import importer
from pm4py.statistics.traces.log import case_statistics

from process_mining import apply_alpha_miner, apply_inductive_miner, apply_inductive_miner_imf, \
    apply_inductive_miner_imd, apply_heuristics_miner

discovery_algorithms = [
    ('alpha_miner', lambda x: apply_alpha_miner(x[0], x[1], x[2])),
    ('inductive_miner', lambda x: apply_inductive_miner(x[0], x[1], x[2])),
    ('inductive_miner_infrequent', lambda x: apply_inductive_miner_imf(x[0], x[1], x[2])),
    ('inductive_miner_dfg', lambda x: apply_inductive_miner_imd(x[0], x[1], x[2])),
    ('heuristics_miner', lambda x: apply_heuristics_miner(x[0], x[1], x[2]))
]

log_to_model_metrics = [
    ('replay_fitness', lambda x: replay_fitness(x[0], x[1], x[2], x[3])),
    ('precision', lambda x: precision(x[0], x[1], x[2], x[3])),
    ('generalization', lambda x: generalization(x[0], x[1], x[2], x[3])),
    ('simplicity', lambda x: simplicity(x[0], x[1], x[2], x[3])),
]


def replay_fitness(log, net, im, fm):
    return replay_fitness_evaluator.apply(log, net, im, fm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)['average_trace_fitness']


def precision(log, net, im, fm):
    return precision_evaluator.apply(log, net, im, fm, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)


def generalization(log, net, im, fm):
    return generalization_evaluator.apply(log, net, im, fm)


def simplicity(log=None, net=None, im=None, fm=None):
    return simplicity_evaluator.apply(net)


def variants_sorted(log):
    variants_count_log = case_statistics.get_variant_statistics(log)
    return sorted(variants_count_log, key=lambda x: x['count'], reverse=True)


def conformance_checking_alignment(log, replayed_log):
    if len(log) == 0:
        return -1
    alignments = logs_alignments.apply(replayed_log, log, parameters={})
    return np.mean([alignment['fitness'] for alignment in alignments])


def extract_customer_start_activities(log, company):
    tracefilter_log_neg = attributes_filter.apply_events(log, [company],
                                                         parameters={
                                                             attributes_filter.Parameters.ATTRIBUTE_KEY: "org:resource",
                                                             attributes_filter.Parameters.POSITIVE: False})
    start_activities = start_activities_filter.get_start_activities(tracefilter_log_neg).keys()
    return start_activities


def conformance_checking_alignment_per_start_activity(log, replayed_log, start_activities, filter):
    result = []
    for start_activity in start_activities:
        filtered_by_start_activity_log = start_activities_filter.apply(log, start_activity)
        filtered_by_start_activity_simulated_log = start_activities_filter.apply(replayed_log, start_activity)
        filtered_by_start_activity_log = filter(filtered_by_start_activity_log)
        variants_count_log = variants_sorted(filtered_by_start_activity_log)
        variants_count_simulated_log = variants_sorted(filtered_by_start_activity_simulated_log)
        if len(variants_count_log) > 0 and len(variants_count_simulated_log):
            result.append((start_activity, variants_count_log, variants_count_simulated_log,
                           conformance_checking_alignment(filtered_by_start_activity_log, filtered_by_start_activity_simulated_log)))
    return result


def import_xes_logs(training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log):
    training_event_log = pm4py.read_xes(training_event_log)
    training_replayed_event_log = pm4py.read_xes(training_replayed_event_log)
    test_event_log = pm4py.read_xes(test_event_log)
    test_replayed_event_log = pm4py.read_xes(test_replayed_event_log)
    return training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log


def filter_classified_start_activities(log, company):
    tracefilter_log_neg = attributes_filter.apply_events(log, [company],
                                                         parameters={
                                                             attributes_filter.Parameters.ATTRIBUTE_KEY: "org:resource",
                                                             attributes_filter.Parameters.POSITIVE: False})
    topics = start_activities_filter.get_start_activities(tracefilter_log_neg).keys()
    return start_activities_filter.apply(log, topics)


def conformance_checking_proxy(training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log,
                               variants_filter_name, company, start_activity):
    filter = parse_variants_filter_arg(variants_filter_name)

    training_event_log_filtered = filter(training_event_log)
    training_event_log_variants = variants_filter.get_variants(training_event_log_filtered)
    test_event_log_filtered = variants_filter.apply(test_event_log, training_event_log_variants)

    proxy_results = []
    for name, discovery_algorithm in discovery_algorithms:
        training_model = discovery_algorithm((training_event_log_filtered, 'models',
                                              'training-{}-{}-{}-{}'.format(company, start_activity, variants_filter_name, name)))
        training_net, training_initial_marking, training_final_marking = training_model

        for metric_name, metric in log_to_model_metrics:
            metric_training = metric((training_event_log_filtered, training_net, training_initial_marking, training_final_marking))
            metric_training_replayed = metric((training_replayed_event_log, training_net, training_initial_marking, training_final_marking))
            metric_test = metric((test_event_log_filtered, training_net, training_initial_marking, training_final_marking))
            metric_test_replayed = metric((test_replayed_event_log, training_net, training_initial_marking, training_final_marking))
            proxy_results.append((name, metric_name, metric_training, metric_training_replayed, metric_test, metric_test_replayed))

    alignment_training = conformance_checking_alignment(training_event_log_filtered, training_replayed_event_log)
    alignment_test = conformance_checking_alignment(test_event_log_filtered, test_replayed_event_log)
    return proxy_results, alignment_training, alignment_test, len(training_event_log_filtered), len(test_event_log_filtered)


def conformance_checking_model(training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log,
                               variants_filter_name, net, initial_marking, final_marking):
    filter = parse_variants_filter_arg(variants_filter_name)

    training_event_log_filtered = filter(training_event_log)
    training_event_log_variants = variants_filter.get_variants(training_event_log_filtered)
    test_event_log_filtered = variants_filter.apply(test_event_log, training_event_log_variants)

    proxy_results = []
    for metric_name, metric in log_to_model_metrics:
        metric_training = metric((training_event_log_filtered, net, initial_marking, final_marking))
        metric_training_replayed = metric((training_replayed_event_log, net, initial_marking, final_marking))
        metric_test = metric((test_event_log_filtered, net, initial_marking, final_marking))
        metric_test_replayed = metric((test_replayed_event_log, net, initial_marking, final_marking))
        proxy_results.append((metric_name, metric_training, metric_training_replayed, metric_test, metric_test_replayed))

    alignment_training = conformance_checking_alignment(training_event_log_filtered, training_replayed_event_log)
    alignment_test = conformance_checking_alignment(test_event_log_filtered, test_replayed_event_log)
    return proxy_results, alignment_training, alignment_test, len(training_event_log_filtered), len(test_event_log_filtered)


def parse_variants_filter_arg(variants_filter_name):
    if variants_filter_name.startswith('variants_percentage'):
        percentage = variants_filter_name.split('_')[-1]
        return lambda x: variants_filter.filter_log_variants_percentage(x, percentage=float(percentage))
    elif variants_filter_name.startswith('variants_auto'):
        percentage = variants_filter_name.split('_')[-1]
        return lambda x: variants_filter.apply_auto_filter(
            x, parameters={attributes_filter.Parameters.DECREASING_FACTOR: float(percentage)})
    elif variants_filter_name.startswith('variants_top'):
        k = variants_filter_name.split('_')[-1]
        return lambda x: variants_filter.filter_variants_top_k(x, int(k))


def chatbot_evaluation_proxy(training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log,
                             variants_filter, company):
    results = []

    training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log = import_xes_logs(
        training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log)

    start_activities = extract_customer_start_activities(training_event_log, company)
    training_event_log = start_activities_filter.apply(training_event_log, start_activities)
    training_replayed_event_log = start_activities_filter.apply(training_replayed_event_log, start_activities)
    test_event_log = start_activities_filter.apply(test_event_log, start_activities)
    test_replayed_event_log = start_activities_filter.apply(test_replayed_event_log, start_activities)

    conformance_checking = conformance_checking_proxy(training_event_log, training_replayed_event_log,
                                                      test_event_log, test_replayed_event_log,
                                                      variants_filter, company, '')
    for result in conformance_checking[0]:
        results.append(('',) + result + (conformance_checking[3], len(training_replayed_event_log),
                                         conformance_checking[4], len(test_replayed_event_log),
                                         conformance_checking[1], conformance_checking[2]))

    for start_activity in start_activities:
        training_event_log_sa = start_activities_filter.apply(training_event_log, start_activity)
        training_replayed_event_log_sa = start_activities_filter.apply(training_replayed_event_log, start_activity)
        test_event_log_sa = start_activities_filter.apply(test_event_log, start_activity)
        test_replayed_event_log_sa = start_activities_filter.apply(test_replayed_event_log, start_activity)

        conformance_checking = conformance_checking_proxy(training_event_log_sa, training_replayed_event_log_sa,
                                                          test_event_log_sa, test_replayed_event_log_sa,
                                                          variants_filter, company, start_activity.replace('/', '_'))
        for result in conformance_checking[0]:
            results.append((start_activity,) + result +
                           (conformance_checking[3], len(training_replayed_event_log_sa), conformance_checking[4],
                            len(test_replayed_event_log_sa), conformance_checking[1], conformance_checking[2]))

    df = pd.DataFrame(results, columns=['Start Activity', 'Discovery Algorithm', 'Metric',
                                        'Training Log', 'Training Log Replayed', 'Test Log', 'Test Log Replayed',
                                        'Training Log - Traces', 'Training Log Replayed - Traces',
                                        'Test Log - Traces', 'Test Log Replayed - Traces',
                                        'Alignment - Training', 'Alignment - Test'])

    writer = pd.ExcelWriter(company + '_' + variants_filter + '_chatbot_evaluation_proxy.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    for column in ['E', 'F', 'G', 'H', 'M', 'N']:
        worksheet.conditional_format('{}2:{}{}'.format(column, column, str(len(df.index) + 1)),
                                     {'type': '3_color_scale'})
    writer.save()


def chatbot_evaluation_normative(training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log,
                             variants_filter, company, model_file):
    training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log = import_xes_logs(
        training_event_log, training_replayed_event_log, test_event_log, test_replayed_event_log)

    start_activities = extract_customer_start_activities(training_event_log, company)
    training_event_log = start_activities_filter.apply(training_event_log, start_activities)
    training_replayed_event_log = start_activities_filter.apply(training_replayed_event_log, start_activities)
    test_event_log = start_activities_filter.apply(test_event_log, start_activities)
    test_replayed_event_log = start_activities_filter.apply(test_replayed_event_log, start_activities)

    results = []

    net, initial_marking, final_marking = importer.apply(model_file)
    conformance_checking = conformance_checking_model(training_event_log, training_replayed_event_log,
                                                      test_event_log, test_replayed_event_log,
                                                      variants_filter, net, initial_marking, final_marking)
    for result in conformance_checking[0]:
        results.append(('',) + result + (conformance_checking[3], len(training_replayed_event_log),
                                         conformance_checking[4], len(test_replayed_event_log),
                                         conformance_checking[1], conformance_checking[2]))

    for start_activity in start_activities:
        training_event_log_sa = start_activities_filter.apply(training_event_log, start_activity)
        training_replayed_event_log_sa = start_activities_filter.apply(training_replayed_event_log, start_activity)
        test_event_log_sa = start_activities_filter.apply(test_event_log, start_activity)
        test_replayed_event_log_sa = start_activities_filter.apply(test_replayed_event_log, start_activity)

        conformance_checking = conformance_checking_model(training_event_log_sa, training_replayed_event_log_sa,
                                                          test_event_log_sa, test_replayed_event_log_sa,
                                                          variants_filter, net, initial_marking, final_marking)
        for result in conformance_checking[0]:
            results.append(
                (start_activity,) + result + (conformance_checking[3], len(training_replayed_event_log_sa),
                                              conformance_checking[4], len(test_replayed_event_log_sa),
                                              conformance_checking[1], conformance_checking[2]))

    df = pd.DataFrame(results, columns=['Start Activity', 'Metric',
                                        'Training Log', 'Training Log Replayed', 'Test Log', 'Test Log Replayed',
                                        'Training Log - Traces', 'Training Log Replayed - Traces',
                                        'Test Log - Traces', 'Test Log Replayed - Traces',
                                        'Alignment - Training', 'Alignment - Test'])

    writer = pd.ExcelWriter(company + '_' + variants_filter + '_chatbot_evaluation_normative.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    for column in ['D', 'E', 'F', 'G', 'L', 'M']:
        worksheet.conditional_format('{}2:{}{}'.format(column, column, str(len(df.index) + 1)),
                                     {'type': '3_color_scale'})
    writer.save()


if __name__ == '__main__':

    if not os.path.exists('models'):
        os.mkdir('models')

    parser = argparse.ArgumentParser()
    parser.add_argument('--training-event-log', help='File of training_event_log.')
    parser.add_argument('--training-replayed-event-log', help='File of training_replayed_event_log.')
    parser.add_argument('--test-event-log', help='File of test_event_log.')
    parser.add_argument('--test-replayed-event-log', help='File of test_replayed_event_log.')
    parser.add_argument('--variants-filter',
                        help='PM4Py variants filter (variants_percentage_x.x or variants_auto_x.x or variants_top_k)')
    parser.add_argument('--company', default='', help='Company whose chatbot is analyzed.')
    parser.add_argument('--model-file', help='.pnml file of normative process model.')
    args = parser.parse_args()

    if args.model_file and args.company and args.training_event_log and args.training_replayed_event_log \
            and args.test_event_log and args.test_replayed_event_log:
        chatbot_evaluation_normative(args.training_event_log, args.training_replayed_event_log,
                                     args.test_event_log, args.test_replayed_event_log,
                                     'variants_percentage_1.0', args.company, args.model_file)

    else:
        filters = ['variants_percentage_1.0', 'variants_percentage_0.75', 'variants_percentage_0.5',
                   'variants_percentage_0.25', 'variants_auto_0.4', 'variants_auto_0.5', 'variants_auto_0.6',
                   'variants_auto_0.7', 'variants_top_5', 'variants_top_10', 'variants_top_15']
        if args.training_event_log and args.training_replayed_event_log \
                and args.test_event_log and args.test_replayed_event_log and args.company:
            if args.variants_filter:
                chatbot_evaluation_proxy(args.training_event_log, args.training_replayed_event_log,
                                         args.test_event_log, args.test_replayed_event_log, args.variants_filter,
                                         args.company)
            else:
                for filter in filters:
                    chatbot_evaluation_proxy(args.training_event_log, args.training_replayed_event_log,
                                             args.test_event_log, args.test_replayed_event_log, filter, args.company)
        else:
            if os.path.exists('xes'):
                for company in os.listdir('xes'):
                    training_event_log_filename = [os.path.join('xes', company, file)
                                                   for file in os.listdir(os.path.join('xes', company))
                                                   if file.endswith('-training.xes')][0]
                    training_replayed_event_log_filename = [os.path.join('xes', company, file)
                                                            for file in os.listdir(os.path.join('xes', company))
                                                            if file.endswith('-training-replayed.xes')][0]
                    test_event_log_filename = [os.path.join('xes', company, file)
                                               for file in os.listdir(os.path.join('xes', company))
                                               if file.endswith('-test.xes')][0]
                    test_replayed_event_log = [os.path.join('xes', company, file)
                                               for file in os.listdir(os.path.join('xes', company))
                                               if file.endswith('-test-replayed.xes')][0]
                    if args.variants_filter:
                        chatbot_evaluation_proxy(training_event_log_filename, training_replayed_event_log_filename,
                                                 test_event_log_filename, test_replayed_event_log,
                                                 args.variants_filter, company)
                    else:
                        for filter in filters:
                            chatbot_evaluation_proxy(training_event_log_filename, training_replayed_event_log_filename,
                                                     test_event_log_filename, test_replayed_event_log,
                                                     filter, company)
