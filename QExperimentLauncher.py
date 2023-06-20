# Модуль подготовки СКММ и запуска Q# кода, симулирующего модель.
import numpy as np
import math
import cmath
from PDB_import import *
import qsharp
from HQMM import  StartExperiment #SayHello, CreateQuantumRNG, DoCoinFlips,


def stochastic_to_unitary(matrix):
    """
    Converts a stochastic matrix to a unitary matrix.
    """
    n = matrix.shape[0]
    unitary = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            unitary[i][j] = np.sqrt(matrix[i][j])
    q, r = np.linalg.qr(unitary)
    diagonal = np.diag(np.diag(r) / np.abs(np.diag(r)))
    unitary = np.dot(q, diagonal)
    return unitary


def stochastic_to_block_diagonal(matrix):
    """
    input: stochastic matrix
    [a,b]
    [c,d]
    :return: block-diagonal matrix:
    [a,b,0,0]
    [b,a,0,0]
    [0,0,c,d]
    [0,0,d,c]
    """
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    return np.array([
        [a, b, 0, 0],
        [b, a, 0, 0],
        [0, 0, c, d],
        [0, 0, d, c]
    ])

def vec_to_matrix(vec):
    """:param vec: vector [a,b]
        :return: matrix [a,b]
                        [b,a]
    """
    return np.array([
        [vec[0],vec[1]],
        [vec[1],vec[0]]
    ])



def Run_HQMM(model, sequence):
    #model = load_model(model_file)
    transmat = model.transmat_
    startprob = model.startprob_
    emissionprob = model.emissionprob_

    uni_startprob = stochastic_to_unitary(stochastic_to_block_diagonal(vec_to_matrix(startprob)))
    uni_transmat = stochastic_to_unitary(stochastic_to_block_diagonal(transmat))

    pred = []
    for i in range(len(sequence)):
        exp_uni_transmat = np.linalg.matrix_power(uni_transmat, i+1)


        em0 = emissionprob[0][sequence[i]]
        em1 = emissionprob[1][sequence[i]]
        """
        emission_matrix = np.array([
            [em0, 1-em0],
            [em1,1-em1]
        ])
        """
        uni_emission = stochastic_to_unitary(stochastic_to_block_diagonal(vec_to_matrix([em0,em1])))

        result = StartExperiment.simulate(transmat=np.real(exp_uni_transmat),startprob=np.real(uni_startprob),emissionprob=np.real(uni_emission))
        pred.append(result)
        #print(result)
        #add_to_sequence(result)
    #print(pred)
    return pred


if __name__ == "__main__":
    model = load_model('helix_modelnew.pkl')
    # MNLTELKNTPVSELITLGENMGLENLARMRKQDIIFAILKQHAKSGE
    # -EHHHHH---HHHHHHHHH-----------HHHHHHHHHHHHHH---
    sequence = 'KRYRALLEKVDPNKIYTIDEAAHLVKELATAKFDETVEVHAKLGIDPRRSDQNVRGTVSLPHGLGKQVRVLAIAKGEKIKEAEEAGADYVGGEEIIQKILDGWMDFDAVVATPDVMGAVGSKLGRILGPRGLLPNPKAGTVGFNIGEIIREIKAGRIEFRNDKTGAIHAPVGKACFPPEKLADNIRAFIRALEAHKPEGAKGTFLRSVYVTTTMGPSVRINPHS'
    ready_seq = convert_aa_to_num(sequence)

    pred = Run_HQMM(model, ready_seq)
    helix_assumption = ''.join([('H' if i > 0 else '-') for i in pred])
    print('---------------E-HHHHHHHHH---------EEEEEEEE-----------EEEEE----------EEEE---HHHHHHHH----EEE----HHHHH-------EEEE----HHHHHHHHHHHHHHH----------E---HHHHHHHHH--EEEEE-----EEEEEEEE----HHHHHHHHHHHHHHHH-----------EEEEEEE------EEE----')
    print(helix_assumption)
