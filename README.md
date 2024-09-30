Given an immunoPETE dataset, find features in the immune repertoire, for machine learning clustering or classification.

The immunoPETE sequencing protocol amplifies cdr3 regions of rearranged immunoglobin heavy chain genes from B cells
and T cell receptor genes from $\delta$ chains of T $\gamma:\delta$ cells and $\beta$ chains from $\alpha:\beta$ T cells. Each B or T cell contributes one (functional)
or two (one nonfunctional, one functional) such cdr3 sequences. Raw sequencing data is processed by the 
Daedalus bioinformatics pipeline (Roche) and annotated with (among other things):
* cdr3_AA: the amino acid sequence that the nucleotide sequence encodes
* functional/nonfunctional: nonfunctional sequences contain a stop codon or a frame shift
* Jgene: a guess at which J gene segment is in the rearrangement
* Vgene: a guess at which V gene segment is in the rearrangement

Data is exported by Daedalus into .csv files, one file per sample and one file per replicate (each sample is divided into 
several replicates, plated in separate wells). These files contain one row per cdr3 sequence/UMI pairing. If we see multiple
UMIs with the same cdr3 sequence, this says that there were multiple cells in the source material with the same cdr3 sequence.
Thus we can get some idea of _clonality_.

From this we would like to extract features of the immune repertoire. A naive approach would be to consider each distinct 
sequence a feature, and the number of observed clones (that is, the number of distinct UMI sequences paired with that 
sequence) to be the numeric value of the feature. However, exact sequence overlap between samples is extremely low:
even people with the same infection are unlikely to produce identical antibodies/receptors. Presumably, however, 
_similar_ sequences may be responding to the same epitopes. How do we quantify _similar_ clones? 

From [Leinster and Cobbald](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/10-2402.1) we adopt the idea of 
_effective_ species counts (here each sequence is a "species"): given a similarity matrix __Z__, and an abundance vector 
__p__, the matrix multiplication product __Zp__ is a vector of effective species counts. See 
https://github.com/ArnaoutLab/diversity for more about similarity matrices. Thus, a similarity-aware set of features could 
be the _effective_ counts of each sequence. 

Let's leave aside for the moment the observation that this yields _way too many_ features, a number of features that would
choke any ML algorithm I know of. (Let's say each blood sample contributes about 20,000 distinct IGH cdr3 sequences&mdash;
I'm spit-balling here, but it's something in this vicinity&mdash;and that overlap between sequences is low enough to be 
negligible. If we process 10 samples, there's something like 200,000 species&mdash;an unmanageable number. So, we will
get to some stage of vetting features for relevance... later.)

How do we make the calculation of __Zp__ computationally tractable? Since the exact-sequence overlap between samples
is so low, the number of species is almost linear in the number of samples. Thus, given n is the number of samples,
the size of __Z__ is $O(n^2)$. The length of __p__, the abundance vector for one "community" (sample) will be $O(n)$, 
because we have to represent all species in the vector, not just those in the community; and, of course, there are
$n$ communities (samples). Thus we have the problem of creating and multiplying two $O(n^2)$ matrices to yield the
effective species counts for all communities. 

In this codebase I explore some ideas for making this computationally tractable.

First off, and this is just the low-hanging fruit: Only functional sequences are considered in this analysis. Although
Braun et al. claimed to observe shifts in V- and J-gene usage in nonfunctional genes with disease state 
(which may be plausible, if temperature and/or intercellular biochemical milieu influence gene rearrangement kinetics), 
I would expect such effects would be swamped by clonal selection effects in functional genes. If we detect features of
relevance to disease state, presumably these would be because particular antibody shapes bound to certain epitopes and
thus were selected for in the disease state. This dynamic would, of course, only affect clone size for the _functional_
genes. To follow up on Braun et al., nonfunctional genes should be examined using some entirely different analysis
(probably very simple, i.e. V and J gene usage frequencies and mean length).

Secondly, B-cell and T-cell sequences should be analyzed independently. B-cell antibodies bind to antigens, whereas
T-cell receptors bind to antigens complexed with MHC molecules, and thus we shouldn't expect to see the same binding
sites selected for across both types of cells.

But this still leaves a huge problem.

It's important that we use a fast and effiecient calculation of similarity between any two sequences, given that we have
to do $O(n^2)$ such calculations. A task for the future is to look at motif-extraction algorithms, and basing similarity
on the presence or absence of such motifs. For now, I am exploring the use of kmer count vectors, with k=3. Rather than 
using the traditional definition of kmer distance, I'm using cosine similarity, because it is very quick to 
calculate dot products of all vectors in one set with all vectors in another set, as a matrix multiplication. 

Something to note is that these kmer vectors are _very sparse_. There are $20^3 = 8,000$ 3-mers of amino acids; most
cdr3 sequences are less than 20 amino acids in length; thus, density is well under 1%. Key to keeping this computationally
tractable is keeping the data structures small enough to fit into RAM, because swapping slows down processing so very much.
Thus, the kmer vectors are stacked into a sparse matrix.

A measure that Braun et al. took to make the calculation of a similarity matrix tractable was to consider the similarity
between two cdr3 sequences to be zero if they did not derive from the same V and J genes. Given that there are 6 J genes,
and about 40 V genes, any one cdr3 has the same V and J assignments as only about 0.4% of the cdr3 sequences in the 
population. By only calculating similarity between cdr3 sequences that share V and J genes, we should be able to create
a similarity matrix that has only about 0.4% density. (Admittedly, this will lose some information: there can be cdr3
sequences that are exactly the same in spite of differing germ line segments.)

Furthermore, the abundance matrix (the compilation of abundance vectors for each community) is extremely sparse. 
Given that there is very little overlap in sequences observed across samples, most of any one abundance vector is 
devoted to noting 0 instances of a sequence found in some other sample. In a metacommunity of n communities, with no
overlap, each abundance vector would have an expected density of 1/n. 

Through the use of `scipy.sparse` sparse array data structures, I have been so far able to process IGH sequences from 
8 (?) samples on my laptop&mdash;that is, to create the abundance and similarity matrices, and multipy these to produce the 
resulting effective species counts, without swapping. Without the use of sparse matrices, the data would 
enormously exceed the size of RAM.
of RAM 
