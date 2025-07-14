using LinearAlgebra

function concept(;N=4, bs=4, cs=2)
  Ai = [rand(bs, bs) for i in 1:N]
  Ci = [rand(bs, cs) for i in 1:N]
  Bi = [rand(cs, bs) for i in 1:N]
  D = rand(cs, cs)
  bi = [rand(bs) for i in 1:N]
  c = rand(cs)

  lhs = zeros(N * bs + cs, N * bs + cs)
  for i in 1:N
    lhs[(i-1)*bs+1:i*bs, (i-1)*bs + 1:i*bs] .= Ai[i]
    lhs[(i-1)*bs+1:i*bs, (N*bs+1):(N*bs+cs)] .= Ci[i]
    lhs[(N*bs+1):(N*bs+cs), (i-1)*bs+1:i*bs] .= Bi[i]
  end
  lhs[(N*bs+1):(N*bs+cs), (N*bs+1):(N*bs+cs)] .= D
  rhs = vcat(bi..., c)
  @assert size(lhs, 2) == size(rhs, 1)

  # 1. LU decompose A
  luAi = [lu!(A) for A in Ai] # each done locally on communicator
  # 2a. Calculate partial solution of A \ C
  C̃i = [luAi[i] \ Ci[i] for i in 1:N] 
  # 2b. Calculate partial solution of A \ b
  b̃i = [luAi[i] \ bi[i] for i in 1:N]
  # 3a. Calculate Schur complement
  S = D - sum(Bi[i] * C̃i[i] for i in 1:N)
  # 3b. Calculate right hand side for Schur complement solve
  c̃ = c - sum(Bi[i] * b̃i[i] for i in 1:N)
  # 4. Solve Schur complement to get lower part of x vector
  y = S \ c̃
  # 5. Solve for upper parts of x vector
  x = [b̃i[i] - C̃i[i] * y for i in 1:N]
  return vcat(x..., y), lhs \ rhs
end

@show result, expected = concept(N=5, bs=3, cs=2)
@show result ./ expected
