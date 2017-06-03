# TRN-MRF
A trust-region Newton method for MAP inference in Markov Random Fields (MRFs).

Kindly cite the following paper in case you are using the code.

H. Kannan, N. Komodakis, N. Paragios 
Newton-type Methods for Inference in Higher-Order Markov Random Fields (poster) 
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

The additional libraries required are:

1. opengm.
2. eigen.
3. openmp.

The executable (newtonTest) is created by running the Makefile.

The executable is used along with a config file. An example config file, config.txt is present in the project.

The fields in the config file are explained below:

1. Input type

There are three ways to specify the input graphical model. Specified as hdf5 or uai or code.

i.e, using a hdf5 file or an uai file or by using energy functions (defined in the file variousEnergies.hpp).

The preferable way to do it is hdf5 because it can represent sparse clique energies, shared clique energies etc.

e.g., Input type = hdf5

2. Input name

It can be the name of the hdf5 or uai file or the input image (as a txt file) for code based input.

e.g., Input name = graph_match_1.h5

3. Stereo right image

Useful for code based input only. Used for specifying the other image (as a txt file) in a stereo pair.

e.g., Stereo right image = tsukubaRight.txt

4. No. of rows

5. No. of cols

No need to specify if hdf5 based input.

No. of rows and columns of a grid topology based graphical model. 

e.g.,
No. of rows = 144
No. of cols = 192

6. No. of labels

No need to specify if hdf5 based input.

No. of labels at one node. Assuming all nodes have same number of labels.

e.g., No. of labels = 16

7. Clique row size

8. Clique col size

No need to specify if hdf5 based input.

Clique dimensions in a grid topology based graphical model.

e.g.,
Clique row size = 1
Clique col size = 3

9. Clique stride

No need to specify if hdf5 based input.

The stride with which the cliques are formed within a grid topology based graphical model.

e.g.,
Clique stride = 1

10. tau

The smoothing parameter.

The value specified here will be either the beginning value within an annealing schedule or the fixed value of the smoothing, if annealing is disabled.

e.g,
tau = 1

11. Max. iterations

Maximum number of iterations for which the algorithms are run.

e.g.,
Max. iterations = 1000

12. Algorithm

Name of the algorithm.

The options are trn, fista, scd, ad3, dd.

Note: ad3 and dd are called from opengm.

e.g.,
Algorithm = trn

13. Sparse

Flag indicating whether the clique energies are sparse or not.

This is important when the number of labels is large, to save space.

e.g.,
Sparse = true

14. Anneal = true

Flag indicating whether smoothing is annealed or not.

e.g.,
Anneal = true

15. Log label

A string that is appended to the output name.

e.g.,
Log label = opengm_dd

16. One clique table

Flag indicating whether all the cliques share one table of values or each of them have unique tables.

e.g.,
One clique table = false

17. MRF model

Useful only when specifying code based input.

String indicating the type of clique energy. Please, look into newtonTest.cpp for understanding this point.

e.g.,
MRF model = highstereo
