module toyHMM

using StatsBase:sample,WeightVec
export dHMM, generate, forward, backward, viterbi, baum_welch!

# Discrete HMM -- HMM with discrete emission probabilities
type dHMM
	n::Int             # Number of hidden states
	m::Int 			   # Alphabet length (observations are discrete for now)
	A::Matrix{Float64} # Estimated state-transition matrix
	B::Matrix{Float64} # Estimated emission probabilities (discrete for now)
	p::Vector{Float64} # Estimiated initial state probabilities
	
	# Notes:
	#   "A" is a NxN matrix, columns sum to one
	#   "B" is a NxM matrix, rows sum to one

	# To do:
	#    Allow B to depend on other observables, for observation o and param c, B(o|c)
end

function dHMM(n::Int,m::Int)
	# Randomize state-transition matrix
	A = rand(n,n)
	A ./= repmat(sum(A,1),n,1) # normalize columns
	
	# Randomize emission probability matrix
	B = rand(n,m)
	B ./= repmat(sum(B,2),1,m) # normalize rows
	
	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return dHMM(n,m,A,B,p)
end

function dHMM(A::Matrix{Float64},B::Matrix{Float64})
	# Check dimensions of matrices
	n,m = size(B)
	assert(size(A,1) == size(A,2) == n)

	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return dHMM(n,m,A,B,p)
end

function dHMM(A::Matrix{Float64},B::Matrix{Float64},p::Vector{Float64})
	assert(sum(p) == 1)
	hmm = dHMM(A,B)
	hmm.p = p
	return hmm
end

function generate(hmm::dHMM, n_obs::Int)
	# Generate a sequence of n_obs observations from an HMM.

	# Sequence of states and observations
	s = zeros(Int,n_obs) # states
	o = zeros(Int,n_obs) # observations

	# Choose initial state with probabilities weighted by "init_state"
	s[1] = sample(WeightVec(hmm.p))         # hmm.p are the initial state probabilities
	o[1] = sample(WeightVec(vec(hmm.B[s[1],:]))) # draw obs given initial state

	# Iterate drawing observations and updating state
	for i = 2:n_obs
		s[i] = sample(WeightVec(hmm.A[:,s[i-1]])) # Pr(s[i+1]==j|s[i]) = A[j,s[i]]
		o[i] = sample(WeightVec(vec(hmm.B[s[i],:])))   # Pr(o==k|s) = B[s,k]
	end

	# return sequence of states and observations
	return (s,o)
end

function forward(hmm::dHMM, o::Vector{Int})
	n_obs = length(o)

	# alpha[i,j] = Pr(o[1:i] | state j at step i)
	alpha = zeros(hmm.n, n_obs) 

	# base case (initialize at start)
	alpha[:,1] = hmm.p .* hmm.B[:,o[1]]

	# induction step
	for i = 2:n_obs
		alpha[:,i] = (hmm.A*alpha[:,i-1]) .* hmm.B[:,o[i]]
	end

	# Pr(observations), given hmm parameters (sum over all states at last point)
	p_obs = sum(alpha[:,end]) 
	return (p_obs, alpha)
end

function backward(hmm::dHMM, o::Vector{Int})
	n_obs = length(o)

	# beta[i,j] = Pr(o[i+1:end] | state j at step i)
	beta = zeros(hmm.n, n_obs)

	# base case (initialize at end)
	beta[:,end] += 1

	# induction step
	for i = reverse(1:n_obs-1)
		beta[:,i] = hmm.A * (hmm.B[:,o[i+1]].*beta[:,i+1])
	end

	return beta
end

function viterbi(hmm::dHMM, o::Vector{Int})
	n_obs = length(o)

	# delta[i,j] = highest probability of state sequence ending in state j on step i
	# psi[i,j] = most likely state on step i-1 given state j on step i (argmax of deltas)
	delta = zeros(hmm.n, n_obs)
	psi = zeros(Int, hmm.n, n_obs)

	# base case, psi[:,1] is ignored so don't initialize
	delta[:,1] = hmm.p .* hmm.B[:,o[1]]

	# recursion
	for i = 2:n_obs
		for j = 1:hmm.n
			delta[j,i],psi[j,i] = findmax(hmm.A[:,j].*delta[:,i-1].*hmm.B[j,o[i]])
		end
	end

	# backtrack to uncover the most likely path / state sequence
	q = zeros(Int,n_obs) # vector holding state sequence
	q[end] = indmax(delta[:,end])

	# backtrack recursively
	for i = reverse(1:n_obs-1)
		q[i] = psi[q[i+1],i+1]
	end
	return q
end
	
function baum_welch!(hmm::dHMM, o::Vector{Int}; n_epochs=20)
	# if user provides a single observation sequence, convert input appropriately
	return baum_welch!(hmm,(Vector{Int})[o];n_epochs=n_epochs)
end

function baum_welch!(hmm::dHMM, sequences::Vector{Vector{Int}}; n_epochs=20)
	# Fit hmm parameters given set of observation sequences
	n_seq = length(sequences)

    # convergence history of the fit, log-liklihood
    ch = zeros(n_epochs) 

    for k = 1:n_epochs
    	A_new = zeros(hmm.n, hmm.n)
    	B_new = zeros(hmm.n, hmm.m)
    	p_new = zeros(hmm.n)
    	for o in sequences
    		n_obs = length(o)

    		# Calculate forward/backward probabilities
			log_p_obs, alpha = forward(hmm, o)  # Calculate forward probs, log-likelihood
		    beta = backward(hmm, o)             # Calculate backward probs
		    ch[k] += log_p_obs

			# x[i,j,t] = expected transition from state i at time t to state j on the next step
			x = zeros(hmm.n, hmm.n, n_obs-1)
			for t = 1:(n_obs-1)
				for i = 1:hmm.n
					for j = 1:hmm.n
						x[i,j,t] = alpha[i,t] * hmm.A[i,j] * hmm.B[j,o[t+1]] * beta[j,t+1]
					end
				end
				x[:,:,t] ./= sum(x[:,:,t]) # normalize to achieve probabilities
			end

			# g[i,t] = estimated probability of being in state i at time/step t
			g = alpha .* beta
			g ./= sum(g,1)   # normalize across states
			gsum2 = sum(g,2) # est proportion of steps spent in each state

			# Update parameter estimates
			p_new += g[:,1]
			A_new += (squeeze(sum(x,3),3) ./ sum(g[:,1:end-1],2))'
			for z = 1:hmm.m
				B_new[:,z] += sum(g[:,o.==z],2) ./ gsum2
			end
		end
		# renormalize across sequences and update parameters
		hmm.p = p_new ./ n_seq # enforces sum(p_new) == 1
		hmm.A = A_new ./ n_seq
		hmm.B = B_new ./ n_seq
	end

	return ch
end

end
