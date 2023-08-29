import re
import numpy as np
from numpy import random
import scipy.stats
import pandas as pd
import argparse
from Bio.SeqUtils import GC
from Bio import SeqIO
from itertools import cycle

# 分配完整的密码子表，根据用户输入操作密码子选择
codon_table = {'A': ['GCT', 'GCC', 'GCA', 'GCG'],
               'C': ['TGT', 'TGC'],
               'D': ['GAT', 'GAC'],
               'E': ['GAA', 'GAG'],
               'F': ['TTT', 'TTC'],
               'G': ['GGT', 'GGC', 'GGA', 'GGG'],
               'H': ['CAT', 'CAC'],
               'I': ['ATT', 'ATC', 'ATA'],
               'K': ['AAA', 'AAG'],
               'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
               'M': ['ATG'],
               'N': ['AAT', 'AAC'],
               'P': ['CCT', 'CCC', 'CCA', 'CCG'],
               'Q': ['CAA', 'CAG'],
               'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
               'S': ['AGT', 'AGC', 'TCT', 'TCC', 'TCA', 'TCG'],
               'T': ['ACT', 'ACC', 'ACA', 'ACG'],
               'V': ['GTT', 'GTC', 'GTA', 'GTG'],
               'W': ['TGG'],
               'Y': ['TAT', 'TAC'],
               '*': ['TAG', 'TAA', 'TGA']}


class P2N:
    def __init__(self, cfg=None, codon_table_full=codon_table):
        self.my_prot = cfg.seqs  # 目的蛋白
        self.seq_Vars = cfg.seq_Vars  # 生成的不同基因数量
        self.host = cfg.host  # 表达宿主
        self.rar_thr = cfg.rar_thr  # 输入 RSCU 值，低于该值的密码子将被丢弃
        self.codon_table_full = codon_table_full
        self.host_path = self.host + '_release.csv'
        print('host:', self.host_path)
        self.codon_table = self.generate_codon_table()

    """
    Generates a codon table based on the RSCU (Relative Synonymous Codon Usage) values.
    Returns:
        codon_table (dict): A dictionary where the keys are amino acid symbols and the values are lists of codons.
    """

    def generate_codon_table(self):
        rsc = pd.read_csv(self.host_path)

        sym = rsc['AmOneLet']
        codon_table = {s: [] for s in sym}

        cod = rsc['Codon']
        val = rsc['RSCU']

        for i in range(len(sym)):
            if val[i] > self.rar_thr:
                codon_table[sym[i]].append(cod[i])

        bund = [(sym[i], cod[i], val[i]) for i in range(len(cod))]

        for key in codon_table:
            if not codon_table[key]:
                if key != 'W' and key != 'M':
                    print('Using only most highly expressed codon for amino acid %s' % key)
                pool = [b for b in bund if b[0] == key]
                c = max(pool, key=lambda item: item[2])[1]
                codon_table[key].append(c)

        return codon_table

    """
    Reads a CSV file at the given path and creates a dictionary of codon usage 
    values (RSCU) with the codon as the key and the RSCU value as the value.
    Parameters:
    - path (str): The path of the CSV file to read.
    Returns:
    - RSCU_dict (dict): A dictionary containing the codon as the key and the RSCU 
        value as the value.
    """

    def rscu(self, path):
        codon_usage = pd.read_csv(path)
        rscu_dict = dict(zip(codon_usage['Codon'], codon_usage['RSCU']))
        return rscu_dict

    """
    Calculates the relative adaptiveness of each codon in a given codon table.
    Args:
        codon_table_full (dict): A dictionary representing the codon table, where each key is an amino acid and the value is a list of codons that encode that amino acid.
        path (str): The path to the file containing the codon usage data.
    Returns:
        dict: A dictionary where each key is a codon and the value is the relative adaptiveness of that codon.
    """

    def relative_adaptiven(self, codon_table_full, path):
        rsc_dict = self.rscu(path)
        max_rsc_dict = {cod: max([rsc_dict[c] for c in codon_table_full[aa]]) for aa in codon_table_full for cod in
                        codon_table_full[aa]}
        relative_adhesiveness_dict = {cod: rsc_dict[cod] / max_rsc_dict[cod] for cod in rsc_dict}
        return relative_adhesiveness_dict

    """
    Calculates the Codon Adaptation Index (CAI) for a given DNA sequence.
    Args:
        codon_table_full (dict): A dictionary containing the codon table with codons as keys and their corresponding relative adaptiveness as values.
        path (str): The path to the file containing the codon table.
        seq (str): The DNA sequence for which the CAI needs to be calculated.
    Raises:
        ValueError: If the length of the sequence is not a multiple of three.
    Returns:
        float: The calculated Codon Adaptation Index (CAI) value.
    """

    def CAI_calculator(self, codon_table_full, path, seq):
        relative_adaptiven_dict = self.relative_adaptiven(codon_table_full, path)
        if len(seq) % 3 != 0:
            raise ValueError('length of sequence is not a multiple of three')

        seq_adaptiven = [relative_adaptiven_dict[seq[i:i + 3]] for i in range(0, len(seq), 3)]
        CAI = scipy.stats.gmean(seq_adaptiven)

        return CAI

    """
    Find the indices of identical elements in a list.
    Args:
        l (list): A list of elements.
    Returns:
        dict: A dictionary where the keys are the unique elements in the list and the values are lists of indices where those elements occur.
    """

    @staticmethod
    def find_identities(l):
        ident_dict = {}
        for index, item in enumerate(l):
            if item not in ident_dict:
                ident_dict[item] = []
            ident_dict[item].append(index)

        return ident_dict

    """
    Generates an infinite iterator that alternates between 0 and 1.
    Returns:
        An infinite iterator that alternates between 0 and 1.
    """

    @staticmethod
    def alternate():
        return cycle([0, 1])

    """
    Generates a library of sequences based on the provided protein sequence.
    """

    def generator(self):
        my_prot = self.my_prot
        seqVars = self.seq_Vars
        codon_table = self.codon_table
        N = len(my_prot)

        mat = np.zeros((seqVars, N), dtype='S3')
        codVars = [codon_table[aa] for aa in my_prot]

        if len(codVars) != mat.shape[1]:
            raise ValueError('list of codon vars and matrix incompatible')

        if codVars[0] != ['ATG']:
            raise ValueError('first codon is not ATG')
        mat[:, 0] = codVars[0]

        for cur in range(seqVars):
            codIndex = cur % len(codVars[1])
            mat[cur, 1] = codVars[1][codIndex]

        for k in range(2, len(codVars)):
            finPos = k
            iniPos = k - 1
            idents = []
            switch = 1
            while switch == 1:
                jointCods = ["".join([mat[x, p].decode() for p in range(iniPos, finPos)]) for x in range(seqVars)]
                identDict = self.find_identities(jointCods)
                if iniPos == k - 1 and max([len(x) for x in identDict.values()]) == 1:
                    idents.append(identDict)
                    switch = 0

                if max([len(x) for x in identDict.values()]) == 1 or iniPos == 0:
                    switch = 0
                elif max([len(x) for x in identDict.values()]) > 1:
                    idents.append(identDict)
                    iniPos = iniPos - 1
                else:
                    raise ValueError('problem with identDict evaluation or logic')

            if len(codVars[k]) == 1:
                cod = codVars[k][0]
                mat[:, k] = cod
            else:
                if len(idents) == 1:
                    outerDict_temp = {key: [idents[0][key]] for key in idents[0]}
                if len(idents) > 1:
                    outerDict = idents[0]
                    outerDict_temp = {item: [] for item in outerDict}
                    for cur in range(1, len(idents) + 1):
                        index = -cur
                        longestHom = [x for x in idents[index] if len(idents[index][x]) > 1]
                        for item in longestHom:
                            currentKey = item[-3:]
                            l = [x for x in idents[index][item]]
                            if outerDict_temp[currentKey] == []:
                                outerDict_temp[currentKey].append(l)
                            else:
                                all_items = set(x for sublist in outerDict_temp[currentKey] for x in sublist)

                                if set(l).isdisjoint(all_items):
                                    outerDict_temp[currentKey].append(l)
                                else:
                                    for sublist in outerDict_temp[currentKey]:
                                        if not set(l).isdisjoint(set(sublist)):
                                            dif = set(l).difference(set(sublist))
                                            for e in dif:
                                                if e not in all_items:
                                                    sublist.append(e)
                                                    all_items.add(e)
                for key in outerDict_temp:
                    if outerDict_temp[key] == []:
                        outerDict_temp[key] = [idents[0][key]]
                alternator = self.alternate()
                codList = [x for x in codVars[k]]
                random.shuffle(codList)

                x1 = codList[0:len(codList) // 2]
                x2 = codList[len(codList) // 2:len(codList)]
                pool = [x1, x2]
                if k == 2:
                    state = alternator.__next__()
                if state == 1:
                    state = alternator.__next__()

                for item in outerDict_temp:
                    workingList = outerDict_temp[item]

                    for sub in workingList:
                        for id in sub:
                            if pool[state] == []:
                                state = alternator.__next__()
                                pool[state - 1] = [x for x in codVars[k] if x not in pool[state]]

                            if pool[state] == []:
                                state = alternator.__next__()
                            cod = random.choice(pool[state])
                            mat[id, k] = cod
                            pool[state].remove(cod)

                        if state == 0:
                            x2 = pool[state] + pool[state - 1]
                            x1 = [x for x in codVars[k] if x not in x2]
                            pool = [x1, x2]
                            state = alternator.__next__()
                        if state == 1:
                            x2 = pool[state]
                            x1 = [x for x in codVars[k] if x not in x2]
                            pool = [x1, x2]

            if k == 2 and len(codVars[k]) == 1:
                alternator = self.alternate()
                state = alternator.__next__()

        my_seqs = []
        for ind in range(seqVars):
            s = ''
            for cod in mat[ind, :]:
                s = s + cod.decode()
            my_seqs.append(s)
        return my_seqs

    """
    Calculate the Hamming distance between two strings.
    Args:
        s1 (str): The first string.
        s2 (str): The second string.
    Raises:
        ValueError: If the input strings have unequal length.
    Returns:
        int: The Hamming distance between the two strings.
    """

    def hamming_distance(self, s1, s2):
        if len(s1) != len(s2):
            raise ValueError("Undefined for sequences of unequal length")
        return sum(1 for ch1, ch2 in zip(s1, s2) if ch1 != ch2)

    """
    Generates a Hamming distance matrix for a given pool of elements.
    Parameters:
        pool (list): A list of elements for which the Hamming distance matrix needs to be generated.
    Returns:
        numpy.ndarray: A 2-dimensional numpy array representing the Hamming distance matrix.
    """

    def hamming_matrix(self, pool):
        if not isinstance(pool, list):
            raise ValueError('Pool is not a list')

        dim_len = len(pool)
        hamming_matrix = np.zeros((dim_len, dim_len))

        for i, ele_1 in enumerate(pool):
            for j in range(i):
                mis_matches = self.hamming_distance(ele_1, pool[j])
                hamming_matrix[i, j] = mis_matches
                hamming_matrix[j, i] = mis_matches

        return hamming_matrix

    """
    Calculate the Hamming statistics of a given pool.
    Args:
        pool (list): A list of strings representing the pool of items.
    Returns:
        tuple: A tuple containing the mean, minimum, and maximum Hamming percentages.
    Example:
        >>> pool = ["abc", "def", "ghi"]
        >>> hamming(pool)
        (33.333333333333336, 0.0, 66.66666666666666)
    """

    def hamming(self, pool):
        div_array = self.hamming_matrix(pool)
        up_tri = div_array[np.triu_indices(div_array.shape[0])]
        mean_ham_per = np.mean(up_tri) / len(pool[0]) * 100
        min_ham_per = np.min(up_tri) / len(pool[0]) * 100
        max_ham_per = np.max(up_tri) / len(pool[0]) * 100
        return mean_ham_per, min_ham_per, max_ham_per

    """
    Calculates the length of the longest continuous substring that is the same in two given strings.
    """

    def longest_cont(self, s1, s2):
        if len(s1) != len(s2):
            raise ValueError("Undefined for sequences of unequal length")
        res = ''.join(['1' if ch1 == ch2 else '0' for ch1, ch2 in zip(s1, s2)])
        return np.max([len(x) for x in re.findall("(1+1)*", res)])

    """
    Calculate the length of the longest continuous matrix in a given pool.
    """

    def longest_cont_matrix(self, pool):
        if not isinstance(pool, list):
            raise ValueError('Pool is not a list')

        dim_len = len(pool)

        longest_matrix = np.zeros((dim_len, dim_len))
        for i, ele_1 in enumerate(pool):
            for j in range(i):
                idents = self.longest_cont(ele_1, pool[j])
                longest_matrix[i, j] = idents
                longest_matrix[j, i] = idents

        return longest_matrix

    """
    Calculate the absolute value of the longest sequence from the given pool.
    Parameters:
        pool (list): A list of elements representing the pool of sequences.
    Returns:
        int: The absolute value of the longest sequence.
    """

    def abs_longest(self, pool):

        mat = np.triu(self.longest_cont_matrix(pool))
        store_seq = []
        # dump non-zero elements into store_seq
        for row in mat:
            for ele in row:
                if ele != 0:
                    store_seq.append(ele)
        # return max
        return np.max(store_seq), mat

    """
    translate the function.
    This function generates a list of sequences using the `generator` method and assigns it to the variable `mySeqs`. It then creates a list of sequence IDs using a list comprehension and assigns it to the variable `IDlist`. The IDs are generated in the format 'seq{i}', where `i` is the index of the sequence in `mySeqs`.
    Next, it calculates the Codon Adaptation Index (CAI) for each sequence in `mySeqs` using a list comprehension and assigns the results to the variable `CAIlist`. The `CAI_calculator` method is used to perform the calculations, passing in the `codon_table_full`, `host_path`, and `seq` as parameters.
    It also calculates the GC percentage for each sequence in `mySeqs` using a list comprehension and assigns the results to the variable `GClist`. The `GC` function is used to perform the calculations, passing in each sequence as a parameter.
    The function then creates a dictionary `output_data` with the keys 'seq ID', 'Sequence', 'CAI', and 'GC %' and the corresponding values `IDlist`, `mySeqs`, `CAIlist`, and `GClist`, respectively. 
    It creates a pandas DataFrame `df` using the `output_data` dictionary and saves it as a CSV file named 'codingSequenceVariants.csv' using the `to_csv` method.
    """

    def translate(self):
        my_seqs = self.generator()

        i_dlist = [f'seq{i}' for i in range(len(my_seqs))]
        ca_ilist = [self.CAI_calculator(codon_table_full=self.codon_table_full, path=self.host_path, seq=seq) for seq in
                    my_seqs]
        g_clist = [GC(seq) for seq in my_seqs]

        output_data = {'seq id': i_dlist, 'sequence': my_seqs, 'CAI': ca_ilist, 'GC %': g_clist}
        df = pd.DataFrame(output_data)

        #
        ## Communicate hamming distance stats and longest stretch of homology
        print('Stats: Mean, minimum and maximum hamming distances in the sequence set are (per cent):',
              self.hamming(my_seqs))
        lentgh, matrix = self.abs_longest(my_seqs)
        print('Stats: Longest stretch of homology between any two sequences (in bp):', lentgh)
        print(matrix)
        np.savetxt('matrix.txt', matrix, fmt='%d')

        df.to_csv('dna_sequence.csv', sep=',')


class Parameter:
    def __init__(self, my_prot='', seq_vars=10, host='CLIB', description='用户自定义参数', rar_thr=0.5):
        self.my_prot = my_prot
        self.seq_Vars = seq_vars
        self.host = host
        self.rar_thr = rar_thr
        self.description = description
        self.seqs = self.get_seq()

    def get_seq(self):
        with open(self.my_prot) as file:
            seqrecord = next(SeqIO.parse(file, "fasta"))
            return str(seqrecord.seq)

    def __str__(self):
        return '{}={}'.format(self.name, self.value)


def main():
    parser = argparse.ArgumentParser(description='用户自定义参数')
    parser.add_argument('-p', dest='my_prot', default='pro.fasta', help='氨基酸序列的fasta文件')
    parser.add_argument('-n', dest='seq_Vars', type=int, default=10, help='基因序列数量')
    parser.add_argument('-os', dest='host', default='CLIB', help='物种[CLIB\coli\cerevisiae]')

    # 解析参数
    args = parser.parse_args()
    cfg = Parameter(args.my_prot, args.seq_Vars, args.host)
    instance = P2N(cfg, codon_table)
    instance.translate()


if __name__ == "__main__":
    main()
