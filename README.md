# SparseMatFac.jl
Matrix factorization for rectangular array data with very few observations.

This is potentially useful for a variety of situations where you want to model pairwise interactions between entities:

* [Game scores between sports teams](https://github.com/dpmerrell/Bracketology.jl)
* Uses of products by customers (as in a recommender system)
* Interactions between drugs and biological samples (as in a drug discovery setting)
* etc.

**WARNING:** this package is under development -- its API is almost sure to change!

See [Bracketology.jl](https://github.com/dpmerrell/Bracketology.jl) for example usage of this package in a sports analytics application.

SparseMatFac.jl is intended to be the "sparse analog" of [MatFac.jl](https://github.com/dpmerrell/MatFac.jl).
