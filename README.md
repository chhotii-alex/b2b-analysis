Given an immunoPETE dataset, find features in the immune repertoire, for machine learning clustering or classification.

The immunoPETE sequencing protocol amplifies cdr3 regions of rearranged immunoglobin heavy chain genes from B cells
and T cell receptor genes from both $\gamma:\delta$ and $\alpha:\beta$ T cells. Each B or T cell contributes one (functional)
or two (one nonfunctional, one functional) cdr3 sequences. Raw sequencing data is processed by the 
Daedalus bioinformatics pipeline (Roche) and annotated with (among other things):
* cdr3_AA: the amino acid sequence that the nucleotide sequence encodes
* functional/nonfunctional: nonfunctional sequences contain a stop codon or a frame shift
* Jgene: a guess at which J gene segment is in the rearrangement
* Vgene: a guess at which V gene segment is in the rearrangement

Data is exported by Daedalus into .csv files, one file per sample and one file per replicate (each sample is divided into 
several replicates, plated in separate wells). These files contain one row per cdr3 sequence/UMI pairing. If we see multiple
UMIs with the same cdr3 sequence, this says that there were multiple cells in the source material with the same cdr3 sequence.
Thus we can get some idea of _clonality_.

