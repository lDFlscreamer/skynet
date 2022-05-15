import argparse
import json
import multiprocessing

import torch

from skynet.DeepCoder.dsl.DeepCoder_program import Program
from skynet.DeepCoder.dsl.DeepCoder_value import Value
from skynet.DeepCoder.env.env import ProgramState
from skynet.PCCoder.env.search import cab, dfs
from skynet.PCCoder.model_tf.PCCoder import PCCoder
from skynet.PCCoder.params import pcCoder_params as params
from skynet.dsl.example import Example
from skynet.env.env import ProgramEnv


def load_problems(path):
    problems = []
    with open(path) as fh:
        for line in fh:
            problems.append(json.loads(line.rstrip()))
    return problems


def init_worker(*args):
    global method, counter, fail_counter, model, timeout, max_program_len, max_beam_size
    method, counter, fail_counter, model, timeout, max_program_len, max_beam_size = args


def solve_problems(problems, method, model, timeout, max_program_len, max_beam_size, num_workers):
    """
    Attempts to predict programs for the given I/O sample sets.
    """

    counter = multiprocessing.Value('i', 0)
    fail_counter = multiprocessing.Value('i', 0)

    if num_workers is None or num_workers > 1:
        pool = multiprocessing.Pool(processes=num_workers, initializer=init_worker,
                                    initargs=(method, counter, fail_counter, model, timeout, max_program_len,
                                              max_beam_size))
        return pool.map(solve_problem_worker, problems)
    else:
        # Don't run in pool to enable debugging
        init_worker(method, counter, fail_counter, model, timeout, max_program_len, max_beam_size)
        return [solve_problem_worker(data) for data in problems]


def solve_problem_worker(data):
    examples = Example.from_line(data,Value)
    env = ProgramEnv(examples,ProgramState)

    if method == 'beam':
        solution = cab(env, max_program_len, model, params.cab_beam_size, params.cab_width,
                       params.cab_width_growth, timeout, max_beam_size=max_beam_size)
    elif method == 'dfs':
        solution = dfs(env, max_program_len, model, params.dfs_max_width, timeout)

    counter.value += 1
    print("\rSolving problems... %d (failed: %d)" % (counter.value, fail_counter.value), end="")

    if solution['result'] is False:
        solution['result'] = "Failed"
        fail_counter.value += 1
    else:
        values = [Value.construct(x) for x in data['examples'][0]['inputs']]
        value_types = [x.type for x in values]
        solution['result'] = Program(value_types, solution['result']).encode()
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('model_config_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('timeout', type=int)
    parser.add_argument('max_program_len', type=int)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--max_beam_size', type=int, default=819200)
    parser.add_argument('--search_method', choices=['beam', 'dfs'], default='beam')

    args = parser.parse_args()

    problems = load_problems(args.input_path)

    config = PCCoder.load_config(args.model_config_path)
    tf_model = PCCoder.load_model(args.model_path)
    model_pcc = PCCoder.from_config_and_model(config=config, model=tf_model)


    res = solve_problems(problems, args.search_method, model_pcc, args.timeout, args.max_program_len,
                         args.max_beam_size, args.num_workers)
    print("")

    solved = len([x for x in res if x['result'] != 'Failed'])
    print("Solved: %d\\%d:" % (solved, len(res)), str(100.0 * solved / len(res)) + '%')

    open(args.output_path, 'w').write('\n'.join([json.dumps(x) for x in res]))


if __name__ == '__main__':
    main()
