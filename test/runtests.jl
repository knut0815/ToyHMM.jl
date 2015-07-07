using Base.Test
using toyHMM

n_states = 2
n_outputs = 3
hmm = dHMM(n_states,n_outputs)

# for j = 1:n_states
# 	@test(sum(hmm.A[:,j]) == 1)
# end
# for i = 1:n_states
# 	@test(sum(hmm.B[i,:]) == 1)
# end
# @test(sum(hmm.p) == 1)

# simple hmm with testable output
hmm.A = eye(2)
hmm.B = [0 1 0; 1 0 0]
hmm.p = vec([1 0])

# state sequence s, observation sequence o
n_obs = 10
s,o = generate(hmm,n_obs)
@test(length(s) == length(o) == n_obs)
@test(all(s .== 1))
@test(all(o .== 2))

# check forward algorithm on perfect sequence
p_obs,alpha = forward(hmm,o)
@test(p_obs == 1)
@test(all(alpha[1,:] .== 1))
@test(all(alpha[2,:] .== 0))

# check backward algorithm on perfect sequence
beta = backward(hmm,o)
@test(all(beta[:,end] .== 1))
@test(all(beta[1,:] .== 1))

# check viterbi algorithm on perfect sequence
s_est = viterbi(hmm,o)
@test(all(s_est .== s))

# check forward-backward algorithm on impossible sequence
p_obs,alpha = forward(hmm,o+1)
beta = backward(hmm,o+1)
@test(p_obs == 0)
@test(all(alpha .== 0))
@test(all(beta[:,1:end-1] .== 0))

######
# Compare output to Michael Hamilton's "dishonest casino" example
# see Hamilton's HMM implementation in python:
# http://www.cs.colostate.edu/~hamiltom/code.html#python-hidden-markov-model

A = [ 0.99 0.01 ;
      0.01 0.99 ]
B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
      1/10  1/10  1/10  1/10  1/10  1/2 ]
p = [0.5, 0.5]
hmm = dHMM(A,B,p)

@test(hmm.n == 2)
@test(hmm.m == 6)
@test(length(hmm.p) == hmm.n)

o = [1, 2, 1, 6, 6] # observation sequence

# check forward/backward
true_alpha = [ 8.3333e-02   1.3833e-02   2.2909e-03   3.7885e-04   6.295e-05 ;
               5.0000e-02   5.0333e-03   5.1213e-04   2.6496e-04   1.3305e-04 ]

true_beta = [ 8.9630e-04   5.2841e-03   3.0533e-02   1.7000e-01   1.0000e+00 ;
              2.4262e-03   2.4418e-02   2.4613e-01   4.9667e-01   1.0000e+00 ]

p_obs,alpha = forward(hmm,o)
beta = backward(hmm,o)

@test(round(p_obs,6) == 0.000196)
@test(all(round(alpha,6) .== round(true_alpha,6)))
@test(all(round(beta,5) .== round(true_beta,5)))

# check viterbi
true_vit = [2,2,2,2,2]
@test(all(viterbi(hmm,o) .== true_vit))
@test(all(viterbi(hmm,[1,2,3,4,5]) .== ones(Int,5)))

# another check for viterbi
hmm.A = [ 0.7 0.3 ;
          0.3 0.7 ]
o = [ 1, 2, 1, 6, 6, 3, 3, 2, 1 ]
true_vit = [ 1, 1, 1, 2, 2, 1, 1, 1, 1 ]
@test(all(viterbi(hmm,o) .== true_vit))

# check Baum-Welch algorithm
o = [ 1, 2, 1, 6, 6, 3, 3, 2, 1 ]
expected_result_A = [ 0.75  0.25 ;
                      0.25  0.75 ]
expected_result_B = [ 0.6  0.4  0.0  0.0  0.0  0.0 ;
                      0.0  0.0  0.5  0.0  0.0  0.5 ]
expected_result_p = [1.0, 0.0]

baum_welch!(hmm, o; n_epochs=60)
@test(all(round(hmm.A,4) .== expected_result_A))
@test(all(round(hmm.B,4) .== expected_result_B))
@test(all(round(hmm.p,4) .== expected_result_p))

# reinitialize hmm
hmm.A = [ 0.7 0.3 ;
          0.3 0.7 ]
hmm.B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
          1/10  1/10  1/10  1/10  1/10  1/2 ]
hmm.p = [0.5, 0.5]

# check Baum-Welch algorith on two sequences
#o1 = [ 1, 2, 1, 6, 6, 3, 3, 2, 1 ]
#o2 = [ 1, 2, 1, 6, 6, 3, 3, 2, 1 ]
o1 = [ 2, 1, 3, 5, 6, 6, 3, 2, 1 ]
o2 = [ 2, 1, 3, 5, 6, 6, 3, 2, 1 ]
seq = Vector{Int}[o1,o2]

expected_result_A = [ 0.7143  0.2857 ; # expected results to 4 decimal places
                      0.2222  0.7778 ]
expected_result_B = [ 0.5556  0.4444  0.      0.      0.      0.     ;
                      0.      0.      0.4444  0.      0.1111  0.4444 ]
expected_result_p = [1.0, 0.0]

baum_welch!(hmm, seq; n_epochs=200)
@test(all(round(hmm.A,4) .== expected_result_A))
@test(all(round(hmm.B,4) .== expected_result_B))
@test(all(round(hmm.p,4) .== expected_result_p))

# TODO: tests without a symmetric A