import sys
import math
import random
import itertools
import functools
from collections import namedtuple, defaultdict
from typing import *

import tqdm

Rule = namedtuple("Rule", ['lhs', 'rhs'])

ROOT = 'ROOT'
NONE = '-NONE-'
TERMINAL_MARKER = '_'

def safelog(x):
    if x == 0:
        return -float('inf')
    else:
        return math.log(x)

def is_close(x, y, eps=10**-5):
    return abs(x-y) < eps

def splits(n: int, k: int) -> Iterator:
    assert k >= 1
    if k > n:
        return
    else:
        for cuts in itertools.combinations(range(1, n), k - 1):
            indices = (0, *cuts, n)
            yield tuple((indices[i], indices[i+1]) for i in range(k))

class PCFG:
    """ PCFG, not necessarily in Chomsky Normal Form. """
    def __init__(self, rules, root):
        self.rules = rules
        self.root = root

    def score(self, xs):
        @functools.cache
        def f(lhs, start_index, span_size):
            if span_size == 1 and lhs == xs[start_index]:
                return 1.0
            else:
                result = 0.0
                for rule, p in self.rules.items():
                    if rule.lhs == lhs:
                        for split in splits(span_size, len(rule.rhs)): 
                            sub_result = 1.0
                            for rhs_part, (start, end) in zip(rule.rhs, split):
                                sub_result *= f(rhs_part, start_index + start, end - start)
                            result += p * sub_result
                return result
        result = f(self.root, 0, len(xs))
        return math.log(result)

def gensym(_state=itertools.count()):
    return 'X' + str(next(_state))

def preterminal_for(x):
    return 'P' + x

def nonterminal_for_sequence(xs, rules, _suffix_dict={}):
    if xs in _suffix_dict:
        return _suffix_dict[xs]
    elif len(xs) == 2:
        new_nt = gensym()
        rule = Rule(new_nt, (xs[0], xs[1]))
        rules[rule] = 1.0
        _suffix_dict[xs] = new_nt
        return new_nt
    else:
        new_nt = gensym()
        first, *rest = xs
        next_nt = nonterminal_for_sequence(tuple(rest), rules)
        rule = Rule(new_nt, (first, next_nt))
        rules[rule] = 1.0
        _suffix_dict[xs] = new_nt
        return new_nt

def is_terminal(symbol):
    return symbol.startswith(TERMINAL_MARKER) or symbol == NONE

def convert_to_cnf(rules):
    # remove unary productions
    nonterminals = {rule.lhs for rule in rules.keys()}
    unit_paths = defaultdict(Counter)
    for A in nonterminals:
        unit_paths[A][A] = 1.0
        queue = [A]
        while queue:
            x = queue.pop()
            for rule, p in rules.items():
                if rule.lhs == x and len(rule.rhs) == 1:
                    y, = rule.rhs
                    new_prob = unit_paths[A][x] * p
                    if y not in unit_paths[A]:
                        unit_paths[A][y] = new_prob
                        queue.append(y)
    deunarized_rules = Counter()
    for A in nonterminals:
        for B in unit_paths[A]:
            for rule, p in rules.items():
                if rule.lhs == B and (len(rule.rhs) > 1 or is_terminal(rule.rhs[0])):
                    new_prob = unit_paths[A][B] * p
                    new_rule = Rule(A, rule.rhs)
                    deunarized_rules[new_rule] += new_prob
        
    # add preterminals
    pt_rules = {}
    for (lhs, rhs), p in deunarized_rules.items():
        new_rhs = []
        for symbol in rhs:
            assert len(rhs) > 1 or is_terminal(rhs[0])
            if is_terminal(symbol) and len(rhs) > 1:
                preterminal = preterminal_for(symbol)
                new_preterminal_rule = Rule(preterminal, (symbol,))
                pt_rules[new_preterminal_rule] = 1.0
                new_rhs.append(preterminal)
            else:
                new_rhs.append(symbol)
        pt_rules[Rule(lhs, tuple(new_rhs))] = p

    # binarize by introducing new nonterminals
    nt_rules = {}
    t_rules = {}
    for rule, p in pt_rules.items():
        if len(rule.rhs) == 1: # terminal rule
            t_rules[rule] = p
        elif len(rule.rhs) == 2: # binary rule
            nt_rules[rule] = p
        else: # ternary+ rule
            first, *rest = rule.rhs
            new_nt = nonterminal_for_sequence(tuple(rest), nt_rules)
            rule = Rule(rule.lhs, (first, new_nt))
            nt_rules[rule] = p

    return nt_rules, t_rules

def test_anbn():
    for i in range(100):
        p_continue = random.random() * .4
        rules = {
            Rule('S', ('_a', '_b')) : 1 - p_continue,
            Rule('S', ('_a', 'S', '_b')) : p_continue,
        }
        pcfg = PCFG(rules, 'S')
        assert is_close(pcfg.score(['_a', '_b']), math.log(1 - p_continue))
        assert is_close(pcfg.score(['_a', '_a', '_b', '_b']), math.log(p_continue * (1 - p_continue)))
        assert is_close(pcfg.score(['_a', '_a', '_a', '_b', '_b', '_b']), math.log(p_continue **2 * (1 - p_continue)))

def test_binarize():
    for i in range(100):
        p_continue = random.random() * .4
        rules = {
            Rule('S', ('_a', '_b')) : 1 - p_continue,
            Rule('S', ('_a', 'S', '_b')) : p_continue,
        }
        pcfg = PCFG(rules, 'S')
        nt_rules, t_rules = convert_to_cnf(rules)
        cnf_rules = t_rules | nt_rules
        cnf_pcfg = PCFG(cnf_rules, 'S')
        assert is_close(pcfg.score(['_a', '_b']), cnf_pcfg.score(['_a', '_b']))
        assert is_close(pcfg.score(['_a', '_a', '_b', '_b']), cnf_pcfg.score(['_a', '_a', '_b', '_b']))
        assert is_close(pcfg.score(['_a', '_a', '_a', '_b', '_b', '_b']), cnf_pcfg.score(['_a', '_a', '_a', '_b', '_b', '_b']))
    
        
def read_grammar(grammar_filename):
    rules = {}
    with open(grammar_filename) as infile:
        for line in infile:
            if line.startswith('#'):
                continue
            else:
                logprob, lhs, *rhs = line.strip().split()
                if rhs:
                    rule = Rule(lhs, tuple(rhs))
                    rules[rule] = math.exp(float(logprob))
    return PCFG(rules, ROOT)

def main(grammar_filename, text_filename):
    print("Processing grammar...", file=sys.stderr)
    grammar = read_grammar(grammar_filename)
    #print("Built CNF grammar with %d nonterminal rules." % len(grammar.nt_rules), file=sys.stderr)
    print("Calculating inside probabilities...", file=sys.stderr)    
    with open(text_filename) as infile:
        lines = infile.readlines()
    for line in tqdm.tqdm(lines):
        terminals = [
            "".join([TERMINAL_MARKER, terminal]) if terminal != NONE else terminal
            for terminal in line.strip().split()
        ]
        score = grammar.score(terminals)
        print(line.strip(), "\t", score, sep="")

if __name__ == '__main__':
    main(*sys.argv[1:])

