# jetRNA - RNA SECONDARY STRUCTURE PREDICTOR

Goal: Predict RNA structure from its sequence using machine learning.

This project was initially created as part of my senior capstone project. The original paper can be found here: https://docs.google.com/document/d/e/2PACX-1vR7L3FTyzXdoT_OViWmghlCYueMxhMV_0dzefUTLI5Y7zJ4D3RGOtTXRZy_R-5Fo1JhHpQv8k5yyeLw/pub

I have since then touched up the code and paper for publication.

PAPER: https://docs.google.com/document/d/1hgje3w5C8stdhpRdfTPJNT4l8M9k2shDKNOXLaQT7M8/edit?usp=sharing

<br/>

## GETTING STARTED

To run the model, simply clone the repo and run the command 'python jetrna <sequence>', where <sequence> is any RNA sequence. A list of sequences can be found here: [INSERT LINK].

Currently, the model is not capable of predicting pseudoknots.

## ABOUT

DISCLAIMER: This section is not a comprehensive explanation on RNA. For more in-depth context, read my paper at [INSERT LINK].

In this study, the goal is to predict RNA secondary structures from its primary nucleotide sequence using machine learning models. Such a problem can be thought of as a binary classification problem: either two nucleotides form a base pair, or they do not.

A more formal explanation of the problem can be made. Let $n$ be a primary nucleotide sequence represented by a list $(n_1, n_2, …, n_k)$ where any entry can be one of the 4 nucleotides $A$, $U$, $G$, $C$, and $k$ be the length of the sequence. A set of pairings must be predicted which define the secondary structure according to three base pairing rules (Figure 1B) (Singh et al., 2019):

1. A base pairing is defined by the Watson-Crick and non-canonical pair types (A, U), (U, A), (U, G), (G, U), (G, C), (C, G) for six pairings.
2. Each nucleotide may only be part of one pairing unless it is unpaired.
3. Bases paired must be at least 3 bases apart from each other.

## References:

NOTE: This section cites code references only. For a full list of references, visit [INSERT LINK].

Booy, M. S., Ilin, A., & Orponen, P. (2022). RNA secondary structure prediction with convolutional neural networks. BMC Bioinformatics, 23(1). https://doi.org/10.1186/s12859-021-04540-7

Hendrixlab. (n.d.). GitHub - hendrixlab/bpRNA: bpRNA: Large-scale Annotation and Analysis of RNA Secondary Structure. GitHub. https://github.com/hendrixlab/bpRNA

Kunzmann, P., & Hamacher, K. (2018). Biotite: a unifying open source computational biology framework in Python. BMC Bioinformatics, 19(1). https://doi.org/10.1186/s12859-018-2367-z

Singh, J., Hanson, J., Paliwal, K. K., & Zhou, Y. (2019). RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning. Nature Communications, 10(1). https://doi.org/10.1038/s41467-019-13395-9
