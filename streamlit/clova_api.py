import base64
import json
import http.client
import math
from collections import Counter
import argparse
import sys
import os
import scipy.stats
import numpy as np
import random


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/tasks/haxwkc9o/completions/LK-C', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        while True:
            res = self._send_request(completion_request)

            if res['status']['code'] == '20000':
                break

        return res['result']['text']
                


'''
https://github.com/soyoung97/Standard_Korean_GEC
Modified MIT License

Software Copyright (c) 2022 Soyoung Yoon

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
'''
# Courtney Napoles
# <napoles@cs.jhu.edu>
# 21 June 2015
# ##
# gleu.py
# 
# This script calculates the GLEU score of a sentence, as described in
# our ACL 2015 paper, Ground Truth for Grammatical Error Correction Metrics
# by Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault.
# 
# For instructions on how to get the GLEU score, call "compute_gleu -h"
#
# Updated 2 May 2016: This is an updated version of GLEU that has been
# modified to handle multiple references more fairly.
#
# Updated 6 9 2017: Fixed inverse brevity penalty
# 
# This script was adapted from bleu.py by Adam Lopez.
# <https://github.com/alopez/en600.468/blob/master/reranker/>


class GLEU :

    def __init__(self,n=4) :
        self.order = 4

    def load_hypothesis_sentence(self,hypothesis) :
        self.hlen = len(hypothesis)
        self.this_h_ngrams = [ self.get_ngram_counts(hypothesis,n)
                               for n in range(1,self.order+1) ]

    def load_sources(self,spath) :
        self.all_s_ngrams = [ [ self.get_ngram_counts(line.split(),n)
                                for n in range(1,self.order+1) ]
                              for line in open(spath, encoding='utf-8') ]

    def load_references(self,rpaths) :
        self.refs = [ [] for i in range(len(self.all_s_ngrams)) ]
        self.rlens = [ [] for i in range(len(self.all_s_ngrams)) ]
        for rpath in rpaths :
            for i,line in enumerate(open(rpath, encoding='utf-8')) :
                self.refs[i].append(line.split())
                self.rlens[i].append(len(line.split()))

        # count number of references each n-gram appear sin
        self.all_rngrams_freq = [ Counter() for i in range(self.order) ]

        self.all_r_ngrams = [ ]
        for refset in self.refs :
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1,self.order+1) :
                ngrams = self.get_ngram_counts(refset[0],n)
                all_ngrams.append(ngrams)

                for k in ngrams.keys() :
                    self.all_rngrams_freq[n-1][k]+=1

                for ref in refset[1:] :
                    new_ngrams = self.get_ngram_counts(ref,n)
                    for nn in new_ngrams.elements() :
                        if new_ngrams[nn] > ngrams.get(nn,0) :
                            ngrams[nn] = new_ngrams[nn]

    def get_ngram_counts(self,sentence,n) :
        return Counter([tuple(sentence[i:i+n])
                        for i in range(len(sentence)+1-n)])

    # returns ngrams in a but not in b
    def get_ngram_diff(self,a,b) :
        diff = Counter(a)
        for k in (set(a) & set(b)) :
            del diff[k]
        return diff

    def normalization(self,ngram,n) :
        return 1.0*self.all_rngrams_freq[n-1][ngram]/len(self.rlens[0])

    # Collect BLEU-relevant statistics for a single hypothesis/reference pair.
    # Return value is a generator yielding:
    # (c, r, numerator1, denominator1, ... numerator4, denominator4)
    # Summing the columns across calls to this function on an entire corpus
    # will produce a vector of statistics that can be used to compute GLEU
    def gleu_stats(self,i,r_ind=None):

      hlen = self.hlen
      try:
        rlen = self.rlens[i][r_ind]
      except:
          import pdb; pdb.set_trace()
      yield hlen
      yield rlen

      for n in range(1,self.order+1):
        h_ngrams = self.this_h_ngrams[n-1]
        s_ngrams = self.all_s_ngrams[i][n-1]
        r_ngrams = self.get_ngram_counts(self.refs[i][r_ind],n)

        s_ngram_diff = self.get_ngram_diff(s_ngrams,r_ngrams)

        yield max([ sum( (h_ngrams & r_ngrams).values() ) - \
                    sum( (h_ngrams & s_ngram_diff).values() ), 0 ])

        yield max([hlen+1-n, 0])

    # Compute GLEU from collected statistics obtained by call(s) to gleu_stats
    def gleu(self,stats,smooth=False):
        # smooth 0 counts for sentence-level scores
        if smooth :
            stats = [ s if s != 0 else 1 for s in stats ]
        if stats.count(0) > 0:
            return 0
        (c, r) = stats[:2]
        log_gleu_prec = sum([math.log(float(x)/y)
                             for x,y in zip(stats[2::2],stats[3::2])]) / 4
        return math.exp(min([0, 1-float(r)/c]) + log_gleu_prec)


'''
https://github.com/soyoung97/Standard_Korean_GEC
Modified MIT License

Software Copyright (c) 2022 Soyoung Yoon

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
'''
#!/usr/bin/env python

# Courtney Napoles
# <courtneyn@jhu.edu>
# 21 June 2015
# ##
# compute_gleu
#
# This script calls gleu.py to calculate the GLEU score of a sentence, as
# described in our ACL 2015 paper, Ground Truth for Grammatical Error
# Correction Metrics by Courtney Napoles, Keisuke Sakaguchi, Matt Post,
# and Joel Tetreault.
#
# For instructions on how to get the GLEU score, call "compute_gleu -h"
#
# Updated 2 May 2016: This is an updated version of GLEU that has been
# modified to handle multiple references more fairly.
#
# This script was adapted from compute-bleu by Adam Lopez.
# <https://github.com/alopez/en600.468/blob/master/reranker/>


def get_gleu_stats(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    res = ['%f' % mean, '%f' % std]
    if len(scores) != 1:
        ci = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
        res.append('(%.3f,%.3f)' % (ci[0], ci[1]))
    return res


def run_gleu(reference='', source='', hypothesis=''):
    args = {
        'n': 4,
        'l': 0.0,
        'reference': reference,
        'source': source,
        'hypothesis': hypothesis,
        'debug': False
    }

    # if there is only one reference, just do one iteration, which is always our case.
    num_iterations = 1

    gleu_calculator = GLEU(args['n'])

    gleu_calculator.load_sources(args['source'])
    gleu_calculator.load_references([args['reference']])

    for hpath in [args['hypothesis']]:
        instream = open(hpath, encoding='utf-8')
        hyp = [line.split() for line in instream]

        if args['debug']:
            print(os.path.basename(hpath))

        # first generate a random list of indices, using a different seed
        # for each iteration
        indices = []
        for j in range(num_iterations):
            random.seed(j * 101)
            indices.append([random.randint(0, 0)
                            for _ in range(len(hyp))])

        if args['debug']:
            print('')
            print('===== Sentence-level scores =====')
            print('SID Mean Stdev 95%CI GLEU')

        iter_stats = [[0 for i in range(2 * args['n'] + 2)]
                      for j in range(num_iterations)]

        for i, h in enumerate(hyp):

            gleu_calculator.load_hypothesis_sentence(h)
            # we are going to store the score of this sentence for each ref
            # so we don't have to recalculate them 500 times

            stats_by_ref = [None for r in range(1)]

            for j in range(num_iterations):
                ref = indices[j][i]
                this_stats = stats_by_ref[ref]

                if this_stats is None:
                    this_stats = [s for s in gleu_calculator.gleu_stats(
                        i, r_ind=ref)]
                    stats_by_ref[ref] = this_stats

                iter_stats[j] = [sum(scores)
                                 for scores in zip(iter_stats[j], this_stats)]

            if args['debug']:
                # sentence-level GLEU is the mean GLEU of the hypothesis
                # compared to each reference
                for r in range(1):
                    if stats_by_ref[r] is None:
                        stats_by_ref[r] = [s for s in gleu_calculator.gleu_stats(
                            i, r_ind=r)]

                print(i)
                print(' '.join(get_gleu_stats([gleu_calculator.gleu(stats, smooth=True)
                                               for stats in stats_by_ref])))

        if args['debug']:
            print('\n==== Overall score =====')
            print('Mean Stdev 95%CI GLEU')
            print(' '.join(get_gleu_stats([gleu_calculator.gleu(stats)
                                           for stats in iter_stats])))
        else:
            return (get_gleu_stats([gleu_calculator.gleu(stats)
                                  for stats in iter_stats])[0])
