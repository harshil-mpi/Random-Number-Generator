#Copyright 2019 D - Wave Systems, Inc.
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http: // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

#-- -- -- Import necessary packages -- --
from collections import defaultdict

from dwave.system.samplers import DWaveSampler
import numpy as np
from matplotlib import pyplot as plt
import neal
import pandas as pd
from pyqubo import Array, Placeholder, solve_qubo, Constraint
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dimod import AdjVectorBQM

#import dwave.minorminer.find_embedding

N = 1000
isQPU = 1
Advantage = 1
if(isQPU):
else:
    print()
if isQPU:
    print("QPU")
    if Advantage:
        print("Advantage")
        qpu = DWaveSampler(solver={'qpu': True, 'topology__type': 'pegasus'})
    else:
        print("Advantage")
        qpu = DWaveSampler(solver={'qpu': True, 'topology__type': 'chimera'})
    sampler = EmbeddingComposite(qpu)       
else :
    print("SA")
    sampler = neal.SimulatedAnnealingSampler()

# // average = {}

# // for itr in range(-100,101,1):

#if (itr == 10 or itr == 15 or itr == 20) :
#continue
    bias = itr/100
    if(bias ==0):
        continue
#for bias in[0.1, 0.15, 0.2]:

    x = Array.create("vector", N, "BINARY")
    H = x[0] - x[0]

    for i in range(N):
        H += bias*(x[i])

    l1 = []
    model = H.compile()
    qubo,offset = model.to_qubo()

response = sampler.sample_qubo(qubo, num_reads = 1, annealing_time = 1,       \
                                return_embedding = True)

#embedding = response.info['embedding_context']['embedding']
#print(embedding)

#fixed_sampler = FixedEmbeddingComposite(qpu, embedding = embedding)

#for i in range(1000):
#print(i)
response = fixed_sampler.sample_qubo(qubo, num_reads = 1, annealing_time = 1, \
                                      return_embedding = True)
    response = sampler.sample_qubo(qubo, num_reads = 100,annealing_time = 1,auto_scale = False)
    df = response.to_pandas_dataframe()
#print(df)

    cnt = [[0]*2]*N
    cnt = np.array(cnt)

    for id,row in df.iterrows():
        for i in range(N):
            if(int(row['vector[{}]'.format(i)]) == 0 ):
                cnt[i][0] += row['num_occurrences']
            else:
                cnt[i][1] += row['num_occurrences']

    pd.DataFrame(cnt).to_csv("bias_{}.csv".format(bias,isQPU))
    print(np.mean(cnt,axis = 0))
    average[bias] = np.mean(cnt,axis = 0)[0]
    del cnt
    del df
    del response

pd.DataFrame(average,index = [0]).to_csv("final.csv")
