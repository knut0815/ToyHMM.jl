# ToyHMM.jl
A simple Hidden Markov Model implementation in Julia. Intended mostly for educational purposes. Only supports discrete emission probabilities

I am developing [HMM.jl](https://github.com/ahwillia/HMM.jl) for a more general-purpose module.

### Installation

```julia
Pkg.clone("https://github.com/ahwillia/ToyHMM.jl.git")
```

### Simple Example

```julia
using ToyHMM

n_states = 2
n_outputs = 3
hmm = dHMM(n_states,n_outputs)

println(hmm.A) # state-transition matrix (randomly initialized, rows sum to 1)
println(hmm.B) # emmission matrix (randomly initialized, rows sum to 1)
println(hmm.p) # initial state probabilities (randomly initialized)

o = [1,1,2,1,1,2,1,2,1,3,3,3,3,2,2,3,3,3] # example observation sequence

ch = baum_welch!(hmm,o) # fit model using Expectation-Maximization

println(ch) # log-likelihood values, convergence history

println(hmm.A) # fitted values of the hmm model
println(hmm.B)
println(hmm.p)

println(viterbi(hmm,o)) # most likely state sequence given hmm params
```

(also see `test/runtests.jl` for some examples)

### How fast is it?

```julia
using ToyHMM

n_states = 2
n_outputs = 3

# create a very long output sequence
true_model = dHMM(n_states,n_outputs)
(s,o) = generate(true_model,100_000)

# try to recover similar params by fitting new model
fit_model = dHMM(n_states,n_outputs)
@time ch = baum_welch!(fit_model,o)
```

`elapsed time: 7.958006041 seconds (4140814448 bytes allocated, 26.85% gc time)`

### References and Acknowledgements:

Michael Hamilton's implementation (python): http://www.cs.colostate.edu/~hamiltom/code.html

Guy Zyskind's implementation (python): https://github.com/guyz/HMM

Rabiner, Lawrence R. ["A tutorial on hidden Markov models and selected applications in speech recognition."](http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf) Proceedings of the IEEE 77.2 (1989): 257-286.

### To Do:

MacKay DJC (1997). [Ensemble Learning for Hidden Markov Models](http://www.inference.phy.cam.ac.uk/mackay/ensemblePaper.pdf) *Technical report, Cavendish
Laboratory, University of Cambridge*
